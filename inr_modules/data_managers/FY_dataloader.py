

import torch
from torch.utils.data import Dataset, Sampler, DataLoader
import numpy as np
import os
from typing import List, Iterator


def _observation_payload_from_cached(cached, dlat, dlon, dt, source_code):
    """Flatten selected profiles into physical log10Ne observations."""
    values = cached['sel_abs']
    valid = cached['valid_prof'][:, :, None] & cached['sel_vmask']
    query_lat = cached['lat_q'][:, None, None]
    query_lon = cached['lon_q'][:, None, None]
    query_time = cached['t_q'][:, None, None]
    dlat_n = np.abs(values[..., 0] - query_lat) / dlat
    dlon_n = np.abs(
        (values[..., 1] - query_lon + 180.0) % 360.0 - 180.0) / dlon
    dt_n = np.abs(values[..., 3] - query_time) / dt
    # Max-norm matches the rectangular hard window and reaches one on every
    # boundary, allowing the ETKF precision to taper continuously to zero.
    rho_squared = np.maximum.reduce([dlat_n, dlon_n, dt_n]) ** 2
    profile_ids = np.broadcast_to(
        cached['sel_ids'][:, :, None], valid.shape).copy()
    profile_ids[~valid] = -1
    flat_valid = valid.reshape(valid.shape[0], -1)
    payload = {
        'coords': values[..., :4].reshape(values.shape[0], -1, 4).copy(),
        'value': values[..., 4].reshape(values.shape[0], -1).copy(),
        'valid_mask': flat_valid,
        'profile_id': profile_ids.reshape(values.shape[0], -1),
        'source': np.full(flat_valid.shape, source_code, dtype=np.int8),
        'rho_squared': rho_squared.reshape(values.shape[0], -1).astype(
            np.float32, copy=False),
    }
    payload['coords'][~flat_valid] = 0.0
    payload['value'][~flat_valid] = 0.0
    payload['rho_squared'][~flat_valid] = 1.0
    return payload


def _load_profile_index(index_path, row_count):
    """Expand profile-level QC boundaries to one profile ID per NPY row."""
    with np.load(index_path, allow_pickle=False) as index:
        required = {'profile_id', 'pass_profile', 'output_start', 'output_end'}
        missing = required.difference(index.files)
        if missing:
            raise ValueError(f'profile index missing arrays: {sorted(missing)}')
        profile_ids = np.asarray(index['profile_id'], dtype=np.int64)
        passed = np.asarray(index['pass_profile'], dtype=bool)
        starts = np.asarray(index['output_start'], dtype=np.int64)
        ends = np.asarray(index['output_end'], dtype=np.int64)
    if not (len(profile_ids) == len(passed) == len(starts) == len(ends)):
        raise ValueError('profile index arrays must have equal length')
    row_ids = np.full(row_count, -1, dtype=np.int64)
    for profile_id, is_pass, start, end in zip(
            profile_ids, passed, starts, ends):
        if not is_pass:
            if start != -1 or end != -1:
                raise ValueError('failed profile must use output boundary -1')
            continue
        if start < 0 or end <= start or end > row_count:
            raise ValueError('invalid passing-profile output boundary')
        if np.any(row_ids[start:end] != -1):
            raise ValueError('overlapping profile output boundaries')
        row_ids[start:end] = profile_id
    if row_count and np.any(row_ids < 0):
        raise ValueError('profile index does not cover every NPY row')
    return row_ids, profile_ids


class FY3D_Dataset(Dataset):
    """
    FY3D 卫星电离层数据 Dataset。

    [Time-Aware Strategy Step 1]:
    在初始化时，根据 'bin_size_hours' 将数据按时间分组。
    这建立了一个 'Bin ID -> Sample Indices' 的映射表。

    [Memory-Efficient Loading]:
    支持两种加载模式：
    - use_memmap=False: 一次性加载到内存（原始行为，速度快但内存占用大）
    - use_memmap=True: 使用内存映射按需加载（内存友好，速度略慢）
    """
    def __init__(self, npy_path: str, mode: str = 'train', val_days: List[int] = None,
                 bin_size_hours: float = 3.0, use_memmap: bool = False,
                 profile_path: str = None, val_ratio: float = None,
                 split_seed: int = 42, profile_index_path: str = None):
        super().__init__()

        if val_days is None:
            val_days = []

        self.use_memmap = use_memmap
        self.npy_path = npy_path

        print(f"Loading data from {npy_path}...")
        print(f"  内存模式: {'Memory-Mapped (按需加载)' if use_memmap else '全量加载到内存'}")

        try:
            if use_memmap:
                # 使用memmap按需加载（节省内存）
                raw_data = np.load(npy_path, mmap_mode='r')
                print(f"  OK Memory-mapped加载成功，形状: {raw_data.shape}")
            else:
                # 全量加载到内存（原始行为）
                raw_data = np.load(npy_path).astype(np.float32)
        except FileNotFoundError:
            workspace_file = os.path.join(os.getcwd(), os.path.basename(npy_path))
            if os.path.exists(workspace_file):
                print(f"Warning: {npy_path} not found. Using {workspace_file} instead.")
                if use_memmap:
                    raw_data = np.load(workspace_file, mmap_mode='r')
                else:
                    raw_data = np.load(workspace_file).astype(np.float32)
            else:
                raise FileNotFoundError(f"Could not find file at {npy_path} or {workspace_file}")

        # Profile IDs are metadata only and never enter the model features.
        if profile_index_path is not None:
            profile_id_raw, split_profile_ids = _load_profile_index(
                profile_index_path, len(raw_data))
        elif profile_path is not None:
            profile_raw = np.load(profile_path, mmap_mode='r')
            if (profile_raw.ndim != 2 or profile_raw.shape[0] != raw_data.shape[0]
                    or profile_raw.shape[1] <= 6):
                raise ValueError('FY profile metadata must align row-wise and contain column 7')
            profile_id_raw = profile_raw[:, 6]
            split_profile_ids = np.asarray(profile_id_raw)
        elif raw_data.shape[1] > 5:
            profile_id_raw = raw_data[:, 5]
            split_profile_ids = np.asarray(profile_id_raw)
        else:
            profile_id_raw = np.arange(len(raw_data), dtype=np.int64)
            split_profile_ids = np.asarray(profile_id_raw)
        if (not np.isfinite(profile_id_raw).all()
                or not np.array_equal(profile_id_raw, np.rint(profile_id_raw))):
            raise ValueError('profile_id must contain finite integers')
        self.raw_profile_ids = np.rint(profile_id_raw).astype(np.int64)
        self.split_profile_ids = np.unique(
            np.rint(split_profile_ids).astype(np.int64))
        if profile_index_path is None and profile_path is not None:
            del profile_id_raw, profile_raw

        # --- Filter NaNs ---
        # 注意：memmap模式下，需要先创建索引再过滤
        print(f"  检查NaN值...")
        isnan_mask = np.isnan(raw_data).any(axis=1)
        if isnan_mask.any():
            print(f"  Warning: Found {np.sum(isnan_mask)} samples with NaN values. Filtering them out...")
            valid_indices = np.where(~isnan_mask)[0]
            if use_memmap:
                # memmap模式：只存储有效索引，不复制数据
                self.data = raw_data
                self.valid_indices = valid_indices
            else:
                # 全量模式：复制有效数据
                self.data = raw_data[~isnan_mask]
                self.valid_indices = None
        else:
            self.data = raw_data
            self.valid_indices = None if not use_memmap else np.arange(len(raw_data))

        # 获取工作索引（考虑NaN过滤）
        if self.valid_indices is not None:
            working_indices = self.valid_indices
            working_data = self.data[working_indices]
        else:
            working_indices = np.arange(len(self.data))
            working_data = self.data

        # --- Sanity Check: 验证时间格式 ---
        print(f"  验证时间格式...")
        relative_hours_check = working_data[:, 3]
        max_hour = np.max(relative_hours_check)
        if max_hour <= 48:
            raise ValueError(f"Input data appears to use Daily Hours (0-24). Max hour: {max_hour:.2f}. Expected Continuous Hours (0-720).")

        # --- 划分 Train/Val ---
        print(f"  划分训练/验证集 (mode={mode})...")
        relative_hours = working_data[:, 3]
        working_profile_ids = self.raw_profile_ids[working_indices]
        if val_ratio is not None:
            unique_profiles = self.split_profile_ids
            rng = np.random.default_rng(split_seed)
            shuffled = rng.permutation(unique_profiles)
            n_val_profiles = max(1, int(round(len(shuffled) * val_ratio)))
            val_profile_ids = shuffled[:n_val_profiles]
            is_val = np.isin(working_profile_ids, val_profile_ids)
        else:
            days = np.floor(relative_hours / 24.0).astype(int)
            is_val = np.isin(days, val_days)

        if mode == 'val':
            self.selected_indices = working_indices[is_val]
        else:
            self.selected_indices = working_indices[~is_val]

        self.profile_ids = self.raw_profile_ids[self.selected_indices]
        print(f"Mode '{mode}': {len(self.selected_indices)} samples selected.")

        # --- [关键步骤] 预计算时间分箱 (Bin Indexing) ---
        print(f"  计算时间分箱 (bin_size={bin_size_hours}h)...")
        # 1. 获取过滤后数据的 relative_hours
        filtered_hours = self.data[self.selected_indices, 3]

        # 2. 计算 Bin ID (例如 3小时一个Bin, 0-3h -> Bin 0)
        # 这保证了同一个 Bin 内的数据对应的 IRI 背景场帧索引是相同的 (Frame t, Frame t+1)
        self.bin_ids = np.floor(filtered_hours / bin_size_hours).astype(int)

        # 3. 构建高效索引表: {bin_id: [indices...]}
        # 使用 argsort 避免循环，极大加速初始化
        sorted_indices = np.argsort(self.bin_ids)
        sorted_bin_ids = self.bin_ids[sorted_indices]
        unique_bins, split_points = np.unique(sorted_bin_ids, return_index=True)
        grouped_indices = np.split(sorted_indices, split_points[1:])

        self.indices_by_bin = {}
        for bin_id, indices in zip(unique_bins, grouped_indices):
            self.indices_by_bin[bin_id] = indices

        print(f"  OK Data binned into {len(self.indices_by_bin)} time bins")
        if use_memmap:
            print(f"  OK Memory-mapped模式：数据将按需从磁盘读取，大幅节省内存")

    def __len__(self):
        return len(self.selected_indices)

    def __getitem__(self, idx):
        # 映射到实际数据索引
        actual_idx = self.selected_indices[idx]

        # 按需读取数据
        if self.use_memmap:
            data_sample = self.data[actual_idx].astype(np.float32)
        else:
            data_sample = self.data[actual_idx]

        # 同时返回数据集索引（供 P2 预计算邻域查询使用）
        return (torch.from_numpy(data_sample),
                torch.tensor(idx, dtype=torch.long),
                torch.tensor(self.profile_ids[idx], dtype=torch.long))


class TimeBinSampler(Sampler):
    """
    [Time-Aware Strategy Step 2]: 自定义 Sampler
    
    执行逻辑:
    1. 每次迭代时，随机打乱 Bin 的顺序 (Bin-Level Shuffle)。
    2. 锁定一个 Bin 后，取出该 Bin 所有数据，并在内部打乱 (Intra-Bin Shuffle)。
    3. 按照 batch_size 切分该 Bin 的数据。
    
    结果: 
    yield 出的每一个 batch_indices 列表，其所有索引都严格属于同一个 Time Bin。
    """
    def __init__(self, dataset: FY3D_Dataset, batch_size: int, shuffle: bool = True, drop_last: bool = False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        
    def __iter__(self) -> Iterator[List[int]]:
        # 获取所有存在的 Bin ID
        bin_ids = list(self.dataset.indices_by_bin.keys())
        
        # 1. Bin Level Shuffle: 随机决定先学哪个时间段 (例如先学 Day 5, 再学 Day 1)
        if self.shuffle:
            np.random.shuffle(bin_ids)
            
        for bin_id in bin_ids:
            # 获取该 Bin 下的所有样本索引
            # 注意: 这里使用 copy() 确保不影响原始索引顺序，虽然在当前逻辑下非必要，但为了安全
            indices = self.dataset.indices_by_bin[bin_id].copy()
            
            # 2. Batch Level Shuffle: 打乱该时间段内的空间点顺序
            if self.shuffle:
                np.random.shuffle(indices)
                
            # 3. 生成 Batches (严格限制在当前 Bin 内)
            num_samples = len(indices)
            for i in range(0, num_samples, self.batch_size):
                batch_indices = indices[i : i + self.batch_size]
                
                # 处理最后一个不足 batch_size 的包
                if len(batch_indices) < self.batch_size and self.drop_last:
                    continue
                    
                # yield 给 DataLoader
                yield batch_indices.tolist()

    def __len__(self):
        # 准确计算 Batch 总数
        count = 0
        for indices in self.dataset.indices_by_bin.values():
            n = len(indices)
            if self.drop_last:
                count += n // self.batch_size
            else:
                count += (n + self.batch_size - 1) // self.batch_size
        return count


class ProfileTimeBinSampler(Sampler):
    """Keep complete profile blocks together and give each profile equal weight."""

    def __init__(self, dataset: FY3D_Dataset, batch_size: int,
                 points_per_profile: int = None, shuffle: bool = True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.points_per_profile = points_per_profile
        self.shuffle = shuffle
        self.profiles_by_bin = {}

        profile_ids = dataset.profile_ids
        order = np.argsort(profile_ids, kind='stable')
        boundaries = np.flatnonzero(np.diff(profile_ids[order])) + 1
        for indices in np.split(order, boundaries):
            bin_id = int(np.median(dataset.bin_ids[indices]))
            self.profiles_by_bin.setdefault(bin_id, []).append(indices)

        max_profile = max(
            (len(indices) if points_per_profile is None else points_per_profile)
            for profiles in self.profiles_by_bin.values()
            for indices in profiles
        )
        if max_profile > batch_size:
            raise ValueError('batch_size must fit one complete sampled profile')

    def _sample_profile(self, indices):
        n = self.points_per_profile
        if n is None:
            return indices
        if self.shuffle:
            return np.random.choice(indices, n, replace=len(indices) < n)
        positions = np.linspace(0, len(indices) - 1, n).round().astype(int)
        return indices[positions]

    def __iter__(self):
        bin_ids = list(self.profiles_by_bin)
        if self.shuffle:
            np.random.shuffle(bin_ids)
        for bin_id in bin_ids:
            profiles = list(self.profiles_by_bin[bin_id])
            if self.shuffle:
                np.random.shuffle(profiles)
            batch = []
            for indices in profiles:
                selected = self._sample_profile(indices).tolist()
                if batch and len(batch) + len(selected) > self.batch_size:
                    yield batch
                    batch = []
                batch.extend(selected)
            if batch:
                yield batch

    def __len__(self):
        total = 0
        for profiles in self.profiles_by_bin.values():
            used = 0
            for indices in profiles:
                n = len(indices) if self.points_per_profile is None else self.points_per_profile
                if used and used + n > self.batch_size:
                    total += 1
                    used = 0
                used += n
            total += int(used > 0)
        return total


def get_dataloaders(
    npy_path: str,
    val_days: List[int] = None,
    batch_size: int = 1024,
    bin_size_hours: float = 3.0,
    num_workers: int = 0,
    use_memmap: bool = False,
    profile_path: str = None,
    profile_index_path: str = None,
    val_ratio: float = 0.1,
    split_seed: int = 42,
    points_per_profile: int = 8,
):
    """
    工厂函数: 组装 Dataset 和 Time-Aware Sampler

    Args:
        npy_path: 数据文件路径
        val_days: 验证集日期列表
        batch_size: 批次大小
        bin_size_hours: 时间分箱大小（小时）
        num_workers: DataLoader工作进程数
        use_memmap: 是否使用内存映射按需加载（节省内存）

    Returns:
        train_loader, val_loader
    """
    train_dataset = FY3D_Dataset(
        npy_path, mode='train', val_days=val_days, bin_size_hours=bin_size_hours,
        use_memmap=use_memmap, profile_path=profile_path,
        val_ratio=val_ratio, split_seed=split_seed,
        profile_index_path=profile_index_path)
    val_dataset = FY3D_Dataset(
        npy_path, mode='val', val_days=val_days, bin_size_hours=bin_size_hours,
        use_memmap=use_memmap, profile_path=profile_path,
        val_ratio=val_ratio, split_seed=split_seed,
        profile_index_path=profile_index_path)
    
    # 训练集开启 Shuffle (Time-Aware Shuffle)
    train_sampler = ProfileTimeBinSampler(
        train_dataset, batch_size=batch_size,
        points_per_profile=points_per_profile, shuffle=True)
    # 验证集关闭 Shuffle (顺序评估)
    val_sampler = ProfileTimeBinSampler(
        val_dataset, batch_size=batch_size,
        points_per_profile=None, shuffle=False)
    
    # DataLoader 必须使用 batch_sampler 参数
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=num_workers, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_sampler=val_sampler, num_workers=num_workers, pin_memory=False)
    
    return train_loader, val_loader

# ======================== FYNeighborhoodIndex（run61 剖面级版本）========================

class FYNeighborhoodIndex:
    """
    FY 掩星剖面邻域索引（run61 profile-level 聚合版）

    核心设计：
        FY EDP 数据 = GNSS 掩星剖面，每次掩星事件在 (lat₀,lon₀,t₀) 产生
        ~200-400 个垂直高度层行。直接索引原始点时 M~3000，即使分块仍慢。

        解决方案：把时空索引从「逐测量点」升级为「逐掩星剖面」：
            1. __init__:  使用 clean3 第 7 列 profile_id 检测真实剖面边界
                          每条剖面均匀预采样 n_alt=8 个高度点存入 prof_abs_data
                          时间分箱建立在剖面代表点上 → M_prof ≈ 5-15（vs 原始 ~3000）
            2. query_observation_batch:
                          CHUNK_G=64 分块广播 [Gc, M_prof] 找 K_prof=8 最近剖面
                          查表 prof_abs_data → 物理坐标、log10Ne和有效掩码
                          无 Python 内层循环 → ~2ms/batch

    输出为物理观测payload；ETKF观测始终保持在log10Ne空间。
    """

    _CHUNK_G = 64

    def __init__(self, fy_data_path, config=None):
        if config is None:
            config = {}
        raw = np.load(fy_data_path, mmap_mode='r')
        profile_index_path = config.get('fy_profile_index_path')
        if profile_index_path:
            profile_ids_raw, _ = _load_profile_index(
                profile_index_path, len(raw))
        else:
            profile_path = config.get('fy_profile_path')
            if not profile_path:
                profile_path = str(fy_data_path).replace(
                    '_clean1.npy', '_clean3.npy')
            profile_raw = np.load(profile_path, mmap_mode='r')
            if (raw.ndim != 2 or profile_raw.ndim != 2 or raw.shape[1] < 5
                    or raw.shape[0] != profile_raw.shape[0]
                    or profile_raw.shape[1] <= 6):
                raise ValueError(
                    'FY clean1/clean3 rows are not aligned or profile_id is missing')
            for start in range(0, len(raw), 1_000_000):
                end = min(start + 1_000_000, len(raw))
                if not np.array_equal(
                        raw[start:end, :5], profile_raw[start:end, :5],
                        equal_nan=True):
                    raise ValueError(
                        f'FY clean1/clean3 physical columns differ at {start}:{end}')
            profile_ids_raw = profile_raw[:, 6]
        if raw.ndim != 2 or raw.shape[1] < 5:
            raise ValueError('FY physical data must have at least five columns')
        valid = np.isfinite(raw[:, :5]).all(axis=1)
        if not np.isfinite(profile_ids_raw).all() or not np.array_equal(
                profile_ids_raw, np.rint(profile_ids_raw)):
            raise ValueError('FY profile_id must contain finite integers')
        data = np.array(raw[valid, :5], dtype=np.float32)
        profile_ids = np.rint(profile_ids_raw[valid]).astype(np.int64)
        if len(data) == 0:
            raise ValueError('FY contains no finite physical rows')

        self.dt       = float(config.get('fy_nb_dt',       1.5))
        self.dlat     = float(config.get('fy_nb_dlat',     5.0))
        self.dlon     = float(config.get('fy_nb_dlon',    15.0))
        self.k_prof   = int(config.get('fy_nb_k_prof',     8))
        self.n_alt    = int(config.get('fy_nb_n_alt',      8))
        # ---- 1. 以 clean3 profile_id 聚合，保留剖面内原始点顺序 ----
        sort_idx          = np.argsort(profile_ids, kind='stable')
        self.sorted_data  = data[sort_idx]
        sorted_pids       = profile_ids[sort_idx]

        # ---- 2. profile_id 变化即新剖面 ----
        breaks           = np.where(np.diff(sorted_pids) != 0)[0] + 1
        starts           = np.concatenate([[0], breaks]).astype(np.int32)
        ends             = np.concatenate([breaks, [len(self.sorted_data)]]).astype(np.int32)
        self.prof_starts = starts
        self.prof_ends   = ends
        self.prof_ids    = sorted_pids[starts]
        N_prof           = len(starts)
        counts           = (ends - starts).astype(np.float64)
        print(f'[FYNeighborhoodIndex] 原始点={len(self.sorted_data):,}  '
              f'掩星剖面={N_prof:,}  压缩比={len(self.sorted_data)/N_prof:.0f}×')

        # ---- 3. 剖面代表点（均值中心）[N_prof, 3]: lat_c, lon_c, t_c ----
        lat_c = np.add.reduceat(self.sorted_data[:, 0].astype(np.float64), starts) / counts
        lon_c = np.add.reduceat(self.sorted_data[:, 1].astype(np.float64), starts) / counts
        t_c   = np.add.reduceat(self.sorted_data[:, 3].astype(np.float64), starts) / counts
        self.prof_meta = np.stack([lat_c, lon_c, t_c], axis=1).astype(np.float32)

        # ---- 4. 每剖面均匀预采样 n_alt 个高度点 → [N_prof, n_alt, 5] ----
        frac     = np.linspace(0.0, 1.0, self.n_alt)                          # [n_alt]
        row_idx  = (starts[:, None].astype(np.float64)
                    + frac[None, :] * (counts[:, None] - 1))                  # [N_prof, n_alt]
        row_idx  = np.round(row_idx).astype(np.int32)
        row_idx  = np.clip(row_idx, starts[:, None], ends[:, None] - 1)
        self.prof_abs_data   = self.sorted_data[row_idx.ravel()].reshape(N_prof, self.n_alt, 5)
        # 有效性：counts[p] < n_alt 时末尾槽是重复点，标为无效
        self.prof_valid_mask = (np.arange(self.n_alt)[None, :] <
                                counts[:, None].astype(int))                   # [N_prof, n_alt]

        # ---- 5. 时间分箱索引建立在剖面代表点上 ----
        prof_sort_t              = np.argsort(self.prof_meta[:, 2])
        self.prof_sorted_idx     = prof_sort_t.astype(np.int32)
        self.prof_sorted_meta    = self.prof_meta[prof_sort_t]                 # [N_prof, 3]
        self.prof_sorted_abs     = self.prof_abs_data[prof_sort_t]             # [N_prof, n_alt, 5]
        self.prof_sorted_vmask   = self.prof_valid_mask[prof_sort_t]           # [N_prof, n_alt]
        self.prof_sorted_ids     = self.prof_ids[prof_sort_t]

        self.bin_size = self.dt
        self.t_min    = float(self.prof_sorted_meta[0, 2])
        t_max_val     = float(self.prof_sorted_meta[-1, 2])
        self.n_bins   = int(np.ceil((t_max_val - self.t_min) / self.bin_size)) + 2
        bin_edges     = (self.t_min
                         + np.arange(self.n_bins + 1, dtype=np.float64) * self.bin_size)
        self.bin_starts = np.searchsorted(
            self.prof_sorted_meta[:, 2].astype(np.float64), bin_edges).astype(np.int32)

    # ------------------------------------------------------------------
    def _cand_slice(self, blo: int, bhi: int):
        b0 = int(np.clip(blo,     0, self.n_bins))
        b1 = int(np.clip(bhi + 1, 0, self.n_bins))
        return int(self.bin_starts[b0]), int(self.bin_starts[b1])

    # ------------------------------------------------------------------
    def query_profiles_only(self, coords_np, exclude_profile_ids=None):
        """
        Phase 1：查找每个查询点最近的 K 个掩星剖面（仅依赖 lat/lon/time，忽略高度）。

        同一站点不同高度共享同一套 top-K 剖面。

        Args:
            coords_np: [B, 4+] float32 — col0=lat_geo, col1=lon_geo, col3=rel_hour
                       col2 (alt) 完全忽略

        Returns:
            dict:
                'sel_abs'    : [B, K_p, n_alt, 5]  预采样剖面绝对数据
                'sel_vmask'  : [B, K_p, n_alt]      有效性掩码
                'valid_prof' : [B, K_p]             剖面级有效性
                'has_obs'    : [B] float32          是否有有效邻居
                'lat_q'      : [B] float32
                'lon_q'      : [B] float32
                't_q'        : [B] float32
                'K_p'        : int
        """
        B   = coords_np.shape[0]
        K_p = self.k_prof

        sel_abs    = np.zeros((B, K_p, self.n_alt, 5), dtype=np.float32)
        sel_vmask  = np.zeros((B, K_p, self.n_alt),    dtype=bool)
        valid_prof = np.zeros((B, K_p),                dtype=bool)
        has_obs    = np.zeros(B,                        dtype=np.float32)
        sel_ids    = np.full((B, K_p), -1,              dtype=np.int64)
        sel_distance = np.full((B, K_p), np.inf,        dtype=np.float32)

        lats_q  = coords_np[:, 0].astype(np.float32)
        lons_q  = coords_np[:, 1].astype(np.float32)
        times_q = coords_np[:, 3].astype(np.float32)

        bin_lo = np.floor(
            (times_q - self.dt - self.t_min) / self.bin_size).astype(np.int32)
        bin_hi = np.floor(
            (times_q + self.dt - self.t_min) / self.bin_size).astype(np.int32)

        keys, inv = np.unique(
            np.stack([bin_lo, bin_hi], axis=1), axis=0, return_inverse=True)

        for g in range(len(keys)):
            grp_idx      = np.where(inv == g)[0]
            G            = len(grp_idx)
            blo_g, bhi_g = int(keys[g, 0]), int(keys[g, 1])
            lo, hi       = self._cand_slice(blo_g, bhi_g)
            if lo >= hi:
                continue

            cands_meta = self.prof_sorted_meta[lo:hi]
            M_prof     = len(cands_meta)
            K_p_g      = min(K_p, M_prof)
            c_lat      = cands_meta[:, 0]
            c_lon      = cands_meta[:, 1]
            c_time     = cands_meta[:, 2]
            local_abs   = self.prof_sorted_abs[lo:hi]
            local_vmask = self.prof_sorted_vmask[lo:hi]
            local_ids   = self.prof_sorted_ids[lo:hi]

            for g_start in range(0, G, self._CHUNK_G):
                chunk_idx = grp_idx[g_start:g_start + self._CHUNK_G]
                Gc = len(chunk_idx)

                gc_lat = lats_q [chunk_idx, None]   # [Gc, 1]
                gc_lon = lons_q [chunk_idx, None]
                gc_t   = times_q[chunk_idx, None]

                dlat_pm  = c_lat [None, :] - gc_lat
                dt_pm    = c_time[None, :] - gc_t
                dlon_raw = c_lon [None, :] - gc_lon
                dlon_pm  = (dlon_raw + 180.0) % 360.0 - 180.0
                dlon_abs = np.abs(dlon_pm)

                valid_pm = ((np.abs(dlat_pm) <= self.dlat) &
                            (dlon_abs         <= self.dlon) &
                            (np.abs(dt_pm)    <= self.dt))
                if exclude_profile_ids is not None:
                    valid_pm &= (
                        local_ids[None, :]
                        != np.asarray(exclude_profile_ids)[chunk_idx, None])
                if not valid_pm.any():
                    continue

                dist_pm = (np.abs(dlat_pm) / self.dlat
                           + dlon_abs       / self.dlon
                           + np.abs(dt_pm)  / self.dt)
                dist_pm[~valid_pm] = np.inf

                if M_prof <= K_p_g:
                    topk_p = np.argsort(dist_pm, axis=1)[:, :K_p_g]
                else:
                    raw_p  = np.argpartition(dist_pm, K_p_g - 1, axis=1)[:, :K_p_g]
                    d_p    = np.take_along_axis(dist_pm, raw_p, axis=1)
                    topk_p = np.take_along_axis(raw_p, np.argsort(d_p, axis=1), axis=1)

                v_prof = np.isfinite(
                    np.take_along_axis(dist_pm, topk_p, axis=1))           # [Gc, K_p_g]
                selected_distance = np.take_along_axis(
                    dist_pm, topk_p, axis=1)
                selected_ids = local_ids[topk_p]

                c_sel_abs   = local_abs  [topk_p.ravel()].reshape(Gc, K_p_g, self.n_alt, 5)
                c_sel_vmask = local_vmask[topk_p.ravel()].reshape(Gc, K_p_g, self.n_alt)

                sel_abs   [chunk_idx, :K_p_g]  = c_sel_abs
                sel_vmask [chunk_idx, :K_p_g]  = c_sel_vmask
                valid_prof[chunk_idx, :K_p_g]  = v_prof
                sel_ids[chunk_idx, :K_p_g] = np.where(
                    v_prof, selected_ids, -1)
                sel_distance[chunk_idx, :K_p_g] = np.where(
                    v_prof, selected_distance, np.inf)
                has_obs   [chunk_idx[v_prof.any(axis=1)]] = 1.0

        return {
            'sel_abs':    sel_abs,     # [B, K_p, n_alt, 5]
            'sel_vmask':  sel_vmask,   # [B, K_p, n_alt]
            'valid_prof': valid_prof,  # [B, K_p]
            'sel_ids':    sel_ids,     # [B, K_p]
            'sel_distance': sel_distance, # normalized space-time distance
            'has_obs':    has_obs,     # [B]
            'lat_q':      lats_q,      # [B]
            'lon_q':      lons_q,      # [B]
            't_q':        times_q,     # [B]
            'K_p':        K_p,
        }

    def query_observation_batch(self, coords_np, exclude_profile_ids=None):
        """Return actual sampled FY profile points for the density observation operator."""
        cached = self.query_profiles_only(coords_np, exclude_profile_ids)
        return self.observation_payload_from_cached(cached)

    def observation_payload_from_cached(self, cached):
        return _observation_payload_from_cached(
            cached, self.dlat, self.dlon, self.dt, source_code=0)

# ===========================================================================
# run64: COSMIC-2 数据集与邻域索引
# ===========================================================================

class COSMICDataset(FY3D_Dataset):
    """COSMIC-2 数据集 — 去掉第 6 列 profile_id，返回 5 列 [Lat,Lon,Alt,RelHour,Log10Ne]。"""

    def __getitem__(self, idx):
        data_tensor, ds_idx, profile_id = super().__getitem__(idx)
        return data_tensor[:5], ds_idx, profile_id


class COSMICNeighborhoodIndex:
    """
    COSMIC-2 掩星剖面邻域索引。

    与 FYNeighborhoodIndex 的物理观测接口兼容，但以profile_id（col 5）识别剖面边界，
    而非 FY 的 Δt 断点。

    data  shape: (N, 6) — [Lat, Lon, Alt, RelHour, Log10Ne, profile_id]
    输出物理观测payload，与FYNeighborhoodIndex一致。
    """

    def __init__(self, cosmic_path, config=None):
        if config is None:
            config = {}
        self.dt     = float(config.get('cosmic_nb_dt',    1.5))
        self.dlat   = float(config.get('cosmic_nb_dlat',  5.0))
        self.dlon   = float(config.get('cosmic_nb_dlon', 15.0))
        self.k_prof = int(config.get('cosmic_nb_k_prof',  8))
        self.n_alt  = int(config.get('cosmic_nb_n_alt',   8))

        raw   = np.load(cosmic_path, mmap_mode='r')
        profile_index_path = config.get('cosmic_profile_index_path')
        if raw.ndim != 2 or raw.shape[1] < 5:
            raise ValueError('COSMIC physical data must have at least five columns')
        valid = np.isfinite(raw[:, :5]).all(axis=1)
        if profile_index_path:
            pid_raw, _ = _load_profile_index(profile_index_path, len(raw))
        elif raw.shape[1] > 5:
            pid_raw = raw[:, 5]
        else:
            raise ValueError(
                'COSMIC requires column 6 profile_id or cosmic_profile_index_path')
        if not np.isfinite(pid_raw).all() or not np.array_equal(pid_raw, np.rint(pid_raw)):
            raise ValueError('COSMIC profile_id must contain finite integers')
        data  = np.array(raw[valid, :5], dtype=np.float32)   # [N, 5]
        pids  = np.rint(pid_raw[valid]).astype(np.int64)     # profile_id
        if len(data) == 0:
            raise ValueError('COSMIC contains no finite physical rows')

        sort_idx         = np.argsort(pids, kind='stable')
        self.sorted_data = data[sort_idx]
        sorted_pids      = pids[sort_idx]

        # 以 profile_id 变化标记剖面边界
        breaks = np.where(np.diff(sorted_pids) != 0)[0] + 1
        starts = np.concatenate([[0], breaks]).astype(np.int32)
        ends   = np.concatenate([breaks, [len(self.sorted_data)]]).astype(np.int32)
        self.prof_starts = starts
        self.prof_ends   = ends
        self.prof_ids    = sorted_pids[starts]
        N_prof  = len(starts)
        counts  = (ends - starts).astype(np.float64)
        print(f'[COSMICNeighborhoodIndex] 原始点={len(self.sorted_data):,}  '
              f'掩星剖面={N_prof:,}  压缩比={len(self.sorted_data)/N_prof:.0f}×')

        # 剖面中心 meta: [N_prof, 3] (lat_c, lon_c, t_c)
        lat_c = np.add.reduceat(self.sorted_data[:, 0].astype(np.float64), starts) / counts
        lon_c = np.add.reduceat(self.sorted_data[:, 1].astype(np.float64), starts) / counts
        t_c   = np.add.reduceat(self.sorted_data[:, 3].astype(np.float64), starts) / counts
        self.prof_meta = np.stack([lat_c, lon_c, t_c], axis=1).astype(np.float32)

        # 每剖面均匀采样 n_alt 个高度点
        frac    = np.linspace(0.0, 1.0, self.n_alt)
        row_idx = (starts[:, None].astype(np.float64)
                   + frac[None, :] * (counts[:, None] - 1))
        row_idx = np.round(row_idx).astype(np.int32)
        row_idx = np.clip(row_idx, starts[:, None], ends[:, None] - 1)
        self.prof_abs_data   = self.sorted_data[row_idx.ravel()].reshape(N_prof, self.n_alt, 5)
        self.prof_valid_mask = (np.arange(self.n_alt)[None, :]
                                < counts[:, None].astype(int))

        # 按时间排序的剖面序列（用于时间窗口二分查找）
        prof_sort_t              = np.argsort(self.prof_meta[:, 2])
        self.prof_sorted_idx     = prof_sort_t.astype(np.int32)
        self.prof_sorted_meta    = self.prof_meta[prof_sort_t]
        self.prof_sorted_abs     = self.prof_abs_data[prof_sort_t]
        self.prof_sorted_vmask   = self.prof_valid_mask[prof_sort_t]
        self.prof_sorted_ids     = self.prof_ids[prof_sort_t]

        self.bin_size = self.dt
        self.t_min    = float(self.prof_sorted_meta[0, 2])
        t_max_val     = float(self.prof_sorted_meta[-1, 2])
        self.n_bins   = int(np.ceil((t_max_val - self.t_min) / self.bin_size)) + 2
        bin_edges     = (self.t_min
                         + np.arange(self.n_bins + 1, dtype=np.float64) * self.bin_size)
        self.bin_starts = np.searchsorted(
            self.prof_sorted_meta[:, 2].astype(np.float64), bin_edges).astype(np.int32)

    # ------------------------------------------------------------------
    def _cand_slice(self, t_q):
        """按时间返回候选剖面切片 [start, end)。"""
        b0 = max(0, int((t_q - self.dt - self.t_min) / self.bin_size) - 1)
        b1 = min(self.n_bins, int((t_q + self.dt - self.t_min) / self.bin_size) + 2)
        lo = int(self.bin_starts[b0])
        hi = int(self.bin_starts[b1])
        return lo, hi

    # ------------------------------------------------------------------
    def query_profiles_only(self, coords_np, exclude_profile_ids=None):
        """
        [B,4] → cached dict（与 FYNeighborhoodIndex 接口兼容）
        """
        B = len(coords_np)
        K_p = self.k_prof

        lat_q = coords_np[:, 0].astype(np.float32)
        lon_q = coords_np[:, 1].astype(np.float32)
        t_q   = coords_np[:, 3].astype(np.float32)
        has_obs = np.zeros(B, dtype=np.float32)

        sel_meta  = np.zeros((B, K_p, 3),           dtype=np.float32)
        sel_abs   = np.zeros((B, K_p, self.n_alt, 5), dtype=np.float32)
        sel_vmask = np.zeros((B, K_p, self.n_alt),  dtype=bool)
        valid_prof = np.zeros((B, K_p),              dtype=bool)
        sel_ids = np.full((B, K_p), -1,               dtype=np.int64)
        sel_distance = np.full((B, K_p), np.inf,      dtype=np.float32)

        for i in range(B):
            lo, hi = self._cand_slice(t_q[i])
            if lo >= hi:
                continue
            seg_meta  = self.prof_sorted_meta[lo:hi]   # [M, 3]
            seg_abs   = self.prof_sorted_abs[lo:hi]
            seg_vmask = self.prof_sorted_vmask[lo:hi]
            seg_ids   = self.prof_sorted_ids[lo:hi]

            dt_m  = np.abs(seg_meta[:, 2] - t_q[i])
            dlat_m = np.abs(seg_meta[:, 0] - lat_q[i])
            dlon_m = np.abs((seg_meta[:, 1] - lon_q[i] + 180.0) % 360.0 - 180.0)
            in_win = (dt_m <= self.dt) & (dlat_m <= self.dlat) & (dlon_m <= self.dlon)
            if exclude_profile_ids is not None:
                in_win &= seg_ids != np.asarray(exclude_profile_ids)[i]
            idx_in = np.where(in_win)[0]
            if len(idx_in) == 0:
                continue

            dist2 = (dlat_m[idx_in] / self.dlat) ** 2 + (dlon_m[idx_in] / self.dlon) ** 2
            distance_order = np.argsort(dist2)[:K_p]
            top_k = idx_in[distance_order]
            n_found = len(top_k)
            sel_meta[i, :n_found]  = seg_meta[top_k]
            sel_abs[i, :n_found]   = seg_abs[top_k]
            sel_vmask[i, :n_found] = seg_vmask[top_k]
            valid_prof[i, :n_found] = True
            sel_ids[i, :n_found] = seg_ids[top_k]
            sel_distance[i, :n_found] = np.sqrt(dist2[distance_order])
            has_obs[i] = 1.0

        return dict(lat_q=lat_q, lon_q=lon_q, t_q=t_q,
                    sel_meta=sel_meta, sel_abs=sel_abs,
                    sel_vmask=sel_vmask, valid_prof=valid_prof,
                    sel_ids=sel_ids, sel_distance=sel_distance,
                    has_obs=has_obs)

    def query_observation_batch(self, coords_np, exclude_profile_ids=None):
        """Return actual sampled COSMIC profile points for the density operator."""
        cached = self.query_profiles_only(coords_np, exclude_profile_ids)
        return self.observation_payload_from_cached(cached)

    def observation_payload_from_cached(self, cached):
        return _observation_payload_from_cached(
            cached, self.dlat, self.dlon, self.dt, source_code=1)

def get_cosmic_dataloader(cosmic_path, batch_size, bin_size_hours=0.5,
                           num_workers=0, use_memmap=True, val_ratio=0.1,
                           split_seed=42, points_per_profile=8,
                           profile_index_path=None):
    """COSMIC-2 DataLoader 工厂函数（接口与 get_dataloaders 对称）。"""
    train_dataset = COSMICDataset(
        cosmic_path, mode='train', val_days=[], bin_size_hours=bin_size_hours,
        use_memmap=use_memmap, val_ratio=val_ratio, split_seed=split_seed,
        profile_index_path=profile_index_path)
    val_dataset = COSMICDataset(
        cosmic_path, mode='val', val_days=[], bin_size_hours=bin_size_hours,
        use_memmap=use_memmap, val_ratio=val_ratio, split_seed=split_seed,
        profile_index_path=profile_index_path)
    train_sampler = ProfileTimeBinSampler(
        train_dataset, batch_size=batch_size,
        points_per_profile=points_per_profile, shuffle=True)
    val_sampler = ProfileTimeBinSampler(
        val_dataset, batch_size=batch_size,
        points_per_profile=None, shuffle=False)
    train_loader  = DataLoader(train_dataset, batch_sampler=train_sampler,
                               num_workers=num_workers, pin_memory=False)
    val_loader    = DataLoader(val_dataset,   batch_sampler=val_sampler,
                               num_workers=num_workers, pin_memory=False)
    return train_loader, val_loader


if __name__ == "__main__":
    # --- 配置 ---
    NPY_FILE_PATH = r"D:\FYsatellite\EDP_data\fy_202409_clean.npy"
    VAL_DAYS = [5, 15, 25] # 假设这些天作为验证集
    BATCH_SIZE = 512       # CPU: Reduced batch size (was 4096)
    BIN_SIZE = 3.0         # 3小时窗口
    
    print("=== 初始化 DataLoader 管道 ===")
    
    # 检查文件是否存在，如果不存在创建一个假的用于测试（仅用于演示代码可运行性）
    if not os.path.exists(NPY_FILE_PATH):
        print(f"文件 {NPY_FILE_PATH} 未找到。")
        # 尝试在本地生成一个用于演示的 dummy 文件
        print("正在生成 Dummy 数据用于测试...")
        dummy_data = np.zeros((100000, 5), dtype=np.float32)
        # Lat, Lon, Alt
        dummy_data[:, 0] = np.random.uniform(-90, 90, 100000)
        dummy_data[:, 1] = np.random.uniform(-180, 180, 100000)
        dummy_data[:, 2] = np.random.uniform(100, 800, 100000)
        # Relative Hour (0 to 720 hours, ~30 days)
        dummy_data[:, 3] = np.random.uniform(0, 30*24, 100000) 
        # Ne_Log
        dummy_data[:, 4] = np.random.uniform(8, 13, 100000)
        
        # 保存到当前目录
        NPY_FILE_PATH = "dummy_fy_data.npy"
        np.save(NPY_FILE_PATH, dummy_data)
        print(f"Dummy data saved to {NPY_FILE_PATH}")

    # 获取 Loaders
    train_loader, val_loader = get_dataloaders(
        npy_path=NPY_FILE_PATH,
        val_days=VAL_DAYS,
        batch_size=BATCH_SIZE,
        bin_size_hours=BIN_SIZE
    )
    
    print("\n=== 验证 Train Loader ===")
    print(f"Train batches: {len(train_loader)}")

    # 迭代一个 Batch 验证逻辑（__getitem__ 现在返回 (data, idx) 元组）
    for b_idx, batch_item in enumerate(train_loader):
        batch_data, batch_ds_idx = batch_item   # 解包 (data_tensor, idx_tensor)
        relative_hours = batch_data[:, 3]

        min_h = relative_hours.min().item()
        max_h = relative_hours.max().item()

        bin_ids = torch.floor(relative_hours / BIN_SIZE).int()
        unique_bins = torch.unique(bin_ids)

        if len(unique_bins) != 1:
            print(f"Error: Batch {b_idx} contains data from multiple bins: {unique_bins.tolist()}")
            break

        print(f"Batch {b_idx} check passed:")
        print(f"  Shape: {batch_data.shape}  idx range: [{batch_ds_idx.min()},{batch_ds_idx.max()}]")
        print(f"  Bin ID: {unique_bins.item()}")
        print(f"  Hour Range: [{min_h:.2f}, {max_h:.2f}] (Span: {max_h - min_h:.4f}h)")
        break

    print("\n=== 验证 Val Loader ===")
    print(f"Val batches: {len(val_loader)}")
    for _, _ in enumerate(val_loader):
        pass
    print("Val iteration complete.")

