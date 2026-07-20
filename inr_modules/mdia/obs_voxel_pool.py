"""
ObsVoxelPool: FY 卫星观测 4D 体素池化模块（无可学习参数）

Plan9 实现：将 FY 训练观测预处理为 4D 体素网格，为每个训练 batch 提供
时间-空间邻域观测上下文（observation context），实现推断期观测条件化。

体素分辨率: 2°lat × 5°lon × 30km alt × 0.5h time

LOO 策略：Time-Bin Block Exclusion
    所有 batch 点来自同一 bin_size_hours（默认3h）时间窗口（TimeBinSampler）
    → 排除该窗口内的所有体素（整体 LOO），防止数据泄漏
    → 只使用邻近时间段（非当前窗口）的体素作为观测上下文

Kp 自适应时间窗口：
    σ_time(Kp) = σ_time_quiet × exp(-Kp / kp_scale)
    平静期 Kp≈0: σ_time ≈ 1.5h（利用较大时间邻域）
    磁暴期 Kp≈6: σ_time ≈ 0.5h（紧缩到近期轨道）

极区密度归一化：
    rho(q) = Σ dist_weight_v（所有候选体素）
    归一化权重 = dist_weight / max(rho, 1.0)
    防止极区体素密集导致 h_obs_context 饱和
"""

import numpy as np
import torch
import torch.nn.functional as F


class ObsVoxelPool:
    """
    FY 观测体素池化预处理模块（不含可学习参数）

    输入: fy_train_data [N, 5+] = [lat_geo, lon_geo, alt_km, rel_hour, ne_log10, ...]
    输出（per batch）:
        voxel_feats  [B, K_max, 5]   归一化相对坐标 + Ne
        dist_weights [B, K_max]      软距离权重（极区归一化后）
        valid_mask   [B, K_max]      bool，True=有效体素
    """

    # 地理常量
    KM_PER_DEG_LAT = 111.0   # km per degree latitude

    def __init__(self, fy_train_data, config, device):
        """
        Args:
            fy_train_data: numpy [N, 5+] — 仅含训练集 FY 观测（不含验证集）
            config: CONFIG_MDIA dict
            device: torch.device
        """
        self.device = device

        # 体素网格分辨率
        self.lat_res  = config.get('voxel_lat_res',  2.0)   # degrees
        self.lon_res  = config.get('voxel_lon_res',  5.0)   # degrees
        self.alt_res  = config.get('voxel_alt_res',  30.0)  # km
        self.time_res = config.get('voxel_time_res', 0.5)   # hours

        # 各向异性距离参数
        self.sigma_horiz      = config.get('obs_sigma_horiz',   800.0)  # km
        self.sigma_vert       = config.get('obs_sigma_vert',    50.0)   # km
        self.sigma_time_quiet = config.get('obs_sigma_time',    1.5)    # hours
        self.kp_scale         = config.get('obs_kp_scale',      3.0)

        # 查找参数
        self.K_max          = config.get('obs_K_max',          8)
        self.max_ctx_voxels = config.get('obs_max_ctx_voxels', 5000)
        self.n_sigma        = config.get('obs_n_sigma',        3.0)   # 上下文半径（σ 倍数）
        self.bin_size_hours = config.get('bin_size_hours',     3.0)   # 训练 batch 时间窗口

        self._build(fy_train_data)

    # ------------------------------------------------------------------
    # 构建阶段（训练开始前一次性执行）
    # ------------------------------------------------------------------

    def _build(self, fy_data):
        """
        构建体素存储（按时间 bin 排序，支持高效时间范围切片）

        时间复杂度: O(N log N)（argsort 主导）
        空间复杂度: O(V)（V = 有效体素数 ≪ N）
        """
        N = len(fy_data)
        lat  = fy_data[:, 0].astype(np.float32)
        lon  = fy_data[:, 1].astype(np.float32)
        alt  = fy_data[:, 2].astype(np.float32)
        time = fy_data[:, 3].astype(np.float32)
        ne   = fy_data[:, 4].astype(np.float32)

        # ---- 过滤 NaN/Inf 观测（防止 NaN 体素污染 voxel_data）----
        valid = (np.isfinite(lat) & np.isfinite(lon) & np.isfinite(alt)
                 & np.isfinite(time) & np.isfinite(ne))
        n_bad = int((~valid).sum())
        if n_bad > 0:
            print(f'  [ObsVoxelPool] 过滤 {n_bad}/{N} 个 NaN/Inf 观测')
            lat, lon, alt, time, ne = (lat[valid], lon[valid], alt[valid],
                                       time[valid], ne[valid])
            N = int(valid.sum())
        if N == 0:
            raise ValueError('[ObsVoxelPool] 过滤后无有效 FY 观测，请检查数据')

        # ---- 计算体素 bin 索引 ----
        lat_bin  = np.floor((lat  +  90.0) / self.lat_res ).astype(np.int64).clip(0, 90)
        lon_bin  = np.floor((lon  + 180.0) / self.lon_res ).astype(np.int64).clip(0, 72)
        alt_bin  = np.floor((alt  - 120.0) / self.alt_res ).astype(np.int64).clip(0, 20)
        time_bin = np.floor(time            / self.time_res).astype(np.int64).clip(0, 9999)

        # ---- 构造唯一整数 key（各维度不重叠）----
        # lat_bin ≤ 90, lon_bin ≤ 72, alt_bin ≤ 20, time_bin ≤ 1440
        voxel_key = (lat_bin  * 100_000_000
                   + lon_bin  * 1_000_000
                   + alt_bin  * 10_000
                   + time_bin)

        # ---- 排序 → 分组 ----
        sort_idx = np.argsort(voxel_key, kind='stable')
        lat_s    = lat[sort_idx];   lon_s    = lon[sort_idx]
        alt_s    = alt[sort_idx];   time_s   = time[sort_idx]
        ne_s     = ne[sort_idx];    tbin_s   = time_bin[sort_idx]
        key_s    = voxel_key[sort_idx]

        # ---- 找唯一体素及其起始位置 ----
        unique_keys, unique_start = np.unique(key_s, return_index=True)
        V      = len(unique_keys)
        counts = np.diff(np.append(unique_start, N)).astype(np.float64)  # 避免 int/float 混合

        # ---- 向量化均值聚合（O(N)，无 Python 循环）----
        centers_lat  = (np.add.reduceat(lat_s.astype(np.float64),  unique_start) / counts).astype(np.float32)
        centers_lon  = (np.add.reduceat(lon_s.astype(np.float64),  unique_start) / counts).astype(np.float32)
        centers_alt  = (np.add.reduceat(alt_s.astype(np.float64),  unique_start) / counts).astype(np.float32)
        centers_time = (np.add.reduceat(time_s.astype(np.float64), unique_start) / counts).astype(np.float32)
        ne_means     = (np.add.reduceat(ne_s.astype(np.float64),   unique_start) / counts).astype(np.float32)
        time_bins    = tbin_s[unique_start].astype(np.int32)

        # ---- 按 time_bin 排序（支持二分查找切片）----
        time_sort    = np.argsort(time_bins, kind='stable')
        centers_lat  = centers_lat[time_sort]
        centers_lon  = centers_lon[time_sort]
        centers_alt  = centers_alt[time_sort]
        centers_time = centers_time[time_sort]
        ne_means     = ne_means[time_sort]
        time_bins    = time_bins[time_sort]

        # ---- 存储为设备张量（排序后的 [V, 5]）----
        voxel_arr = np.stack(
            [centers_lat, centers_lon, centers_alt, centers_time, ne_means], axis=1
        )  # [V, 5]
        self.voxel_data      = torch.tensor(voxel_arr, dtype=torch.float32, device=self.device)
        self._time_bins_np   = time_bins          # numpy，用于二分查找

        # ---- 构建 time_bin → 连续切片索引（累积索引）----
        unique_tb, tb_start_idx = np.unique(time_bins, return_index=True)
        tb_end_idx = np.append(tb_start_idx[1:], V)
        self._tb_start = dict(zip(unique_tb.tolist(), tb_start_idx.tolist()))
        self._tb_end   = dict(zip(unique_tb.tolist(), tb_end_idx.tolist()))
        self._min_time_bin = int(time_bins.min())
        self._max_time_bin = int(time_bins.max())

        print(f'  [ObsVoxelPool] 构建完成: {V:,} 个有效体素 | '
              f'time_bin 范围 [{self._min_time_bin}, {self._max_time_bin}] | '
              f'σ_horiz={self.sigma_horiz:.0f}km σ_vert={self.sigma_vert:.0f}km '
              f'σ_time={self.sigma_time_quiet:.1f}h (quiet)')

    # ------------------------------------------------------------------
    # 上下文时间窗口索引（LOO 过滤）
    # ------------------------------------------------------------------

    def _get_context_slices(self, batch_time_center, kp_current=0.0):
        """
        计算当前 batch 可用的上下文体素切片列表（已做 LOO 过滤）

        Args:
            batch_time_center: float — batch 中心时间 (hours)
            kp_current:        float — 当前 Kp 指数 (0–9, approximate)

        Returns:
            slices: list of (start_idx, end_idx) into self.voxel_data
            sigma_t: float — 自适应时间 σ (hours)
        """
        # Kp 自适应 σ_time
        sigma_t = self.sigma_time_quiet * float(np.exp(-max(kp_current, 0.0) / self.kp_scale))
        sigma_t = max(sigma_t, 0.25)   # 最小 15min

        # LOO 排除窗口：整个 batch 时间 bin（含 ±buffer）
        loo_half  = self.bin_size_hours / 2.0 + 0.25   # buffer = 15min
        loo_start = batch_time_center - loo_half
        loo_end   = batch_time_center + loo_half
        loo_tbin_s = int(np.floor(loo_start / self.time_res))
        loo_tbin_e = int(np.ceil( loo_end   / self.time_res))

        # 上下文搜索窗口
        ctx_start  = batch_time_center - self.n_sigma * sigma_t
        ctx_end    = batch_time_center + self.n_sigma * sigma_t
        ctx_tbin_s = int(np.floor(ctx_start / self.time_res))
        ctx_tbin_e = int(np.ceil( ctx_end   / self.time_res))

        # 限制到有效 time_bin 范围
        ctx_tbin_s = max(ctx_tbin_s, self._min_time_bin)
        ctx_tbin_e = min(ctx_tbin_e, self._max_time_bin)

        slices = []
        for tb in range(ctx_tbin_s, ctx_tbin_e + 1):
            if loo_tbin_s <= tb <= loo_tbin_e:
                continue   # LOO: 跳过当前 batch 所在 time bin
            if tb in self._tb_start:
                slices.append((self._tb_start[tb], self._tb_end[tb]))

        return slices, sigma_t

    # ------------------------------------------------------------------
    # Per-batch 特征计算（训练循环中调用）
    # ------------------------------------------------------------------

    def compute_features(self, query_coords, batch_time_center, kp_current=0.0):
        """
        为当前 batch 计算观测上下文特征（全张量运算，无 Python 循环）

        Args:
            query_coords:       torch.Tensor [B, 4+] on device — [lat, lon, alt, time, ...]
            batch_time_center:  float — batch 时间中心 (hours)，用于 LOO 过滤
            kp_current:         float — 当前 Kp（用于 σ_time 自适应）

        Returns:
            voxel_feats:  [B, K_max, 5]  归一化相对坐标 [Δlat_n,Δlon_n,Δalt_n,Δtime_n,Ne_n]
            dist_weights: [B, K_max]     软距离权重（极区归一化）
            valid_mask:   [B, K_max]     bool，True=有效体素（False=padding）

        所有输出在 self.device 上（训练中无需额外移动）。
        """
        B      = query_coords.shape[0]
        K      = self.K_max
        device = self.device
        dtype  = torch.float32

        # 空返回（全 invalid）
        def _zeros():
            return (torch.zeros(B, K, 5,           device=device, dtype=dtype),
                    torch.zeros(B, K,               device=device, dtype=dtype),
                    torch.zeros(B, K, dtype=torch.bool, device=device))

        # ---- 获取上下文体素切片 ----
        ctx_slices, sigma_t = self._get_context_slices(batch_time_center, kp_current)
        if not ctx_slices:
            return _zeros()

        # ---- 拼接候选体素（zero-copy tensor slice 后 cat）----
        parts = [self.voxel_data[s:e] for s, e in ctx_slices]
        ctx   = torch.cat(parts, dim=0)   # [V_c, 5]
        V_c   = ctx.shape[0]

        if V_c == 0:
            return _zeros()

        # ---- 若候选体素过多，随机子采样（限制内存占用）----
        if V_c > self.max_ctx_voxels:
            perm = torch.randperm(V_c, device=device)[:self.max_ctx_voxels]
            ctx  = ctx[perm]
            V_c  = self.max_ctx_voxels

        # ---- 确保 query_coords 在同一设备 ----
        q = query_coords[:, :4].to(device=device, dtype=dtype)   # [B, 4]

        q_lat  = q[:, 0:1]   # [B, 1]
        q_lon  = q[:, 1:2]
        q_alt  = q[:, 2:3]
        q_time = q[:, 3:4]

        v_lat  = ctx[None, :, 0]   # [1, V_c]
        v_lon  = ctx[None, :, 1]
        v_alt  = ctx[None, :, 2]
        v_time = ctx[None, :, 3]

        # ---- 各向异性距离 [B, V_c] ----
        cos_lat  = torch.cos(q_lat * (3.14159265 / 180.0))        # [B, 1]
        d_lat_km = (q_lat  - v_lat)  * self.KM_PER_DEG_LAT        # [B, V_c]
        d_lon_km = (q_lon  - v_lon)  * self.KM_PER_DEG_LAT * cos_lat
        d_alt_km = (q_alt  - v_alt)                                 # [B, V_c]
        d_time_h = (q_time - v_time)                                # [B, V_c]

        dist2 = ((d_lat_km ** 2 + d_lon_km ** 2) / (self.sigma_horiz ** 2)
               + d_alt_km ** 2                    / (self.sigma_vert  ** 2)
               + d_time_h ** 2                    / (sigma_t         ** 2))   # [B, V_c]

        dist_w_all = torch.exp(-0.5 * dist2)                       # [B, V_c]

        # ---- 极区密度归一化（防止极区体素密集导致 context 过强）----
        rho          = dist_w_all.sum(dim=1, keepdim=True).clamp(min=1.0)   # [B, 1]
        dist_w_norm  = dist_w_all / rho                                      # [B, V_c]

        # ---- Top-K 体素 ----
        K_actual = min(K, V_c)
        topk_w, topk_idx = torch.topk(dist_w_norm, K_actual, dim=1)         # [B, K_actual]

        # ---- 构建相对坐标特征（归一化）----
        ctx_sel = ctx[topk_idx]   # [B, K_actual, 5]

        sigma_deg = self.sigma_horiz / self.KM_PER_DEG_LAT   # ≈ 7.2°

        q_lat_e  = q_lat.unsqueeze(2)    # [B, 1, 1]
        q_lon_e  = q_lon.unsqueeze(2)
        q_alt_e  = q_alt.unsqueeze(2)
        q_time_e = q_time.unsqueeze(2)

        delta_lat  = (ctx_sel[..., 0:1] - q_lat_e)  / sigma_deg          # [B, K_actual, 1]
        delta_lon  = (ctx_sel[..., 1:2] - q_lon_e)  / sigma_deg
        delta_alt  = (ctx_sel[..., 2:3] - q_alt_e)  / self.sigma_vert
        delta_time = (ctx_sel[..., 3:4] - q_time_e) / sigma_t
        ne_norm    = (ctx_sel[..., 4:5] - 11.0)     / 2.0

        feats = torch.cat([delta_lat, delta_lon, delta_alt, delta_time, ne_norm], dim=-1)
        # [B, K_actual, 5]

        # 安全网：将任何残余 NaN/Inf 特征置零并标记为 invalid
        bad_feat = ~torch.isfinite(feats).all(dim=-1)   # [B, K_actual]
        feats = feats.clone()
        feats[bad_feat] = 0.0

        valid = (topk_w > 1e-8) & (~bad_feat)   # [B, K_actual] bool

        # ---- Pad 到 K_max（若 K_actual < K）----
        if K_actual < K:
            pad = K - K_actual
            feats  = F.pad(feats,  (0, 0, 0, pad))   # [B, K, 5]
            topk_w = F.pad(topk_w, (0, pad))          # [B, K]
            valid  = F.pad(valid,  (0, pad))           # [B, K]  False=invalid

        return feats, topk_w, valid
