"""
FSIA-INR × ISR 独立验证主程序

功能：
  1. 加载训练好的 FSIA-INR 模型
  2. 读取 Jicamarca 和 Poker Flat ISR 数据
  3. 将 Poker Flat AACGM 坐标转换为地理坐标
  4. 在 ISR 观测坐标上推理模型
  5. 计算 RMSE / MAE / Pearson R / NmF2 / hmF2 指标
  6. 输出时间-高度对比图、NmF2 散点图、文本报告

运行方式：
    cd FSIA_INR
    python isr_evaluation/main_isr_eval.py

配置项在下方 CONFIG 区块修改。
  model_type: 'fsia' → 加载 FSIA_INR_Model  + best_fsia_model.pth（默认）
              'mdia' → 加载 MDIA_INR_Model + best_mdia_model.pth（备用）
"""

import os
import sys
import csv
import json
import datetime
import numpy as np
import torch

# ─────────────────────────────────────────────
# 路径设置：确保 inr_modules 可被导入
# ─────────────────────────────────────────────
_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
_FSIA_DIR    = os.path.dirname(_SCRIPT_DIR)                     # FSIA_INR/
_INR_MODULES = os.path.join(_FSIA_DIR, 'inr_modules')
for _p in [_FSIA_DIR, _INR_MODULES]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ─────────────────────────────────────────────
# ==================== 配置 ====================
# ─────────────────────────────────────────────
CONFIG = {
    # ---- 模型类型：'fsia'（默认）或 'mdia' ----
    'model_type': 'fsia',

    # ---- 检查点路径（None = 根据 model_type 自动推断）----
    'checkpoint_path': None,

    # ---- ISR 数据目录 ----
    # 每个目录下应包含 .hdf5 / .h5 文件（可多个文件，同站同月）
    'jicamarca_dir':  r'D:\ISR\DATA\10jicamarca_is_radar(~12°S,低纬磁赤道)',
    'poker_flat_dir': r'D:\ISR\DATA\61poker_flat_is_radar(lp)\05min',

    # ---- 时间范围（与 MDIA 训练窗口一致）----
    'start_date_str': '2024-09-01 00:00:00',
    'end_date_str':   '2024-10-01 00:00:00',

    # ---- ISR 数据过滤 ----
    'alt_min':         120.0,   # km
    'alt_max':         500.0,   # km
    'err_ratio_max':   0.5,     # dne/ne 阈值，超过则视为质量差

    # ---- 模型推理 ----
    'batch_size':      2048,    # 单次推理点数

    # ---- 输出目录 ----
    'save_dir': os.path.join(_FSIA_DIR, r'isr_validation_outputs\run66-etkf-loss'),

    # ---- 是否处理各站点（可单独关闭）----
    'run_jicamarca':  True,
    'run_poker_flat': True,
}
# ─────────────────────────────────────────────


# ─────────────────────────────────────────────
# 分层指标工具（分高度 × 分昼夜）
# ─────────────────────────────────────────────
_STRAT_ALT_BINS  = [(120.0, 300.0), (300.0, 500.0)]
_STRAT_ALT_NAMES = ['120-300km', '300-500km']
_DAY_LT_RANGE    = (6.0, 18.0)   # 白天：LT 06-18h


def _lt_from_relhour_lon(rel_hour, lon_deg):
    """相对小时（UT）+ 地理经度 → 地方时 LT ∈ [0, 24)"""
    ut_hour = rel_hour % 24.0
    return (ut_hour + lon_deg / 15.0) % 24.0


def _ccc(obs, pred):
    """Lin's Concordance Correlation Coefficient（与 metrics.py 保持一致）。"""
    n = len(obs)
    if n < 2:
        return np.nan
    mu_o, mu_p = obs.mean(), pred.mean()
    cov   = np.mean((obs - mu_o) * (pred - mu_p))
    denom = obs.var(ddof=0) + pred.var(ddof=0) + (mu_o - mu_p) ** 2
    return float(2.0 * cov / denom) if denom > 1e-30 else np.nan


def _strat_metrics_1d(pred_log10, obs_log10):
    """计算单个分层内的统计指标（log10 空间）。"""
    n = len(pred_log10)
    if n < 5:
        return {'n': n, 'rmse': np.nan, 'bias': np.nan,
                'mae': np.nan, 'pearson_r': np.nan, 'ccc': np.nan}
    diff = pred_log10 - obs_log10
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    bias = float(np.mean(diff))
    mae  = float(np.mean(np.abs(diff)))
    if obs_log10.std() < 1e-12 or pred_log10.std() < 1e-12:
        r = np.nan
    else:
        r = float(np.corrcoef(obs_log10, pred_log10)[0, 1])
    ccc = _ccc(obs_log10, pred_log10)
    return {'n': n, 'rmse': rmse, 'bias': bias, 'mae': mae,
            'pearson_r': r, 'ccc': ccc}


def _compute_stratified_metrics(alt_all, lon_all, rh_all,
                                pred_all, obs_all, background_all=None,
                                iri_all=None):
    """
    按高度层（120-300 / 300-500 km）× 昼夜（LT 06-18 / 其余）计算分层指标。

    Parameters
    ----------
    alt_all  : np.ndarray [N]  高度 km
    lon_all  : np.ndarray [N]  地理经度 °
    rh_all   : np.ndarray [N]  相对 UT 小时（自 start_unix 起）
    pred_all : np.ndarray [N]  模型预测 log10(Ne)
    obs_all  : np.ndarray [N]  ISR 观测 log10(Ne)
    background_all : np.ndarray [N] | None  FNDA Background log10(Ne)
    iri_all  : np.ndarray [N] | None  Raw IRI log10(Ne)

    Returns
    -------
    dict  键形如  'model_alt_120-300km_day', 'iri_alt_300-500km_night', ...
          每个值为 {'n', 'rmse', 'bias', 'mae', 'pearson_r'}
    """
    lt = _lt_from_relhour_lon(rh_all, lon_all)
    day_mask = (lt >= _DAY_LT_RANGE[0]) & (lt < _DAY_LT_RANGE[1])

    result = {}
    sources = [('analysis', pred_all)]
    if background_all is not None:
        sources.append(('background', background_all))
    if iri_all is not None:
        sources.append(('iri', iri_all))

    for src_name, src_pred in sources:
        # 全高度分昼夜
        for dn_label, dn_mask in [('day', day_mask), ('night', ~day_mask), ('all', np.ones(len(alt_all), bool))]:
            m = dn_mask
            key = f'{src_name}_all_alt_{dn_label}'
            result[key] = _strat_metrics_1d(
                src_pred[m].astype(np.float64),
                obs_all[m].astype(np.float64),
            )
        # 分高度层 × 分昼夜
        for (lo, hi), alt_name in zip(_STRAT_ALT_BINS, _STRAT_ALT_NAMES):
            alt_mask = (alt_all >= lo) & (alt_all < hi)
            for dn_label, dn_mask in [('day', day_mask), ('night', ~day_mask), ('all', np.ones(len(alt_all), bool))]:
                m = alt_mask & dn_mask
                key = f'{src_name}_alt_{alt_name}_{dn_label}'
                result[key] = _strat_metrics_1d(
                    src_pred[m].astype(np.float64),
                    obs_all[m].astype(np.float64),
                )

    return result


def _print_stratified_table(strat, station_name):
    """控制台打印分层指标表格。"""
    print(f'\n  ── {station_name} 分层统计（分高度 × 分昼夜）──')
    header = (f'  {"层次":<30} {"来源":<10} {"N":>7} {"RMSE":>8} {"Bias":>8} '
              f'{"MAE":>8} {"R":>7} {"CCC":>7}')
    print(header)
    print('  ' + '-' * (len(header) - 2))

    def _fv(v, fmt='8.4f'):
        return f'{v:{fmt}}' if (v is not None and np.isfinite(v)) else ' ' * int(fmt.split('.')[0]) + 'NaN'

    for key, v in sorted(strat.items()):
        parts = key.split('_', 1)
        src   = parts[0]
        layer = parts[1] if len(parts) > 1 else ''
        print(f'  {layer:<30} {src:<10} {v["n"]:>7d} {_fv(v["rmse"]):>8} '
              f'{_fv(v["bias"]):>8} {_fv(v["mae"], "8.4f"):>8} '
              f'{_fv(v["pearson_r"], "7.4f"):>7} {_fv(v["ccc"], "7.4f"):>7}')


def _save_stratified_csv(strat, csv_path):
    """将分层指标写入 CSV。"""
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['key', 'source', 'layer', 'n', 'rmse', 'bias', 'mae',
                    'pearson_r', 'ccc'])
        for key, v in sorted(strat.items()):
            parts = key.split('_', 1)
            src   = parts[0]
            layer = parts[1] if len(parts) > 1 else ''
            w.writerow([key, src, layer,
                        v['n'], v['rmse'], v['bias'], v['mae'],
                        v['pearson_r'], v.get('ccc', np.nan)])


# ─────────────────────────────────────────────

def _parse_unix(date_str):
    dt = datetime.datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
    return dt.replace(tzinfo=datetime.timezone.utc).timestamp()


def _resolve_checkpoint(config, mdia_cfg):
    """若 checkpoint_path 为 None，根据 model_type 自动推断。"""
    if config['checkpoint_path'] is not None:
        return config['checkpoint_path']
    model_type = config.get('model_type', 'fsia')
    if model_type == 'fsia':
        return os.path.join(mdia_cfg['save_dir'], 'best_fsia_model.pth')
    return os.path.join(mdia_cfg['save_dir'], 'best_mdia_model.pth')


def _load_state_compat(model, state_dict):
    """
    带架构兼容的 state_dict 加载。

    处理已知的跨版本不兼容：
    - run13 (7D spatial) → run15+ (6D spatial, 删除 cos_I 列)
      siren_low/siren_high 第一层权重 [hidden, 7] → 截取前 6 列
    - run13 缺少 film_mag_res → 保留模型零初始化权重
    """
    model_sd = model.state_dict()
    adapted  = {}
    skipped  = []

    for k, v in state_dict.items():
        if k not in model_sd:
            skipped.append(f'  extra key (ignored): {k}')
            continue
        target_shape = model_sd[k].shape
        if v.shape == target_shape:
            adapted[k] = v
        elif v.dim() == 2 and target_shape[1] < v.shape[1] and target_shape[0] == v.shape[0]:
            # 输入维度缩减：截取前 target_shape[1] 列（e.g. 7D→6D 去掉最后一列 cos_I）
            adapted[k] = v[:, :target_shape[1]].contiguous()
            print(f'  [compat] {k}: {list(v.shape)} → truncated to {list(target_shape)}')
        else:
            skipped.append(f'  shape mismatch (kept model init): {k} '
                           f'ckpt={list(v.shape)} model={list(target_shape)}')

    # 缺失键保留模型初始化值
    for k in model_sd:
        if k not in adapted:
            adapted[k] = model_sd[k]
            if k not in [s.split(':')[0].strip() for s in skipped]:
                skipped.append(f'  missing key (kept model init): {k}')

    model.load_state_dict(adapted, strict=True)
    if skipped:
        print('[compat] 以下参数未从 checkpoint 加载（兼容性处理）:')
        for msg in skipped:
            print(msg)


def _load_model_and_managers(config, device):
    """加载模型、SpaceWeatherManager 和 IRIPeakManager。

    Returns:
        (model, sw_manager, cfg, model_name, iri_peak_manager,
         fy_nb_index, cosmic_nb_index)
    """
    from inr_modules.config_mdia import get_config_mdia
    from inr_modules.data_managers.space_weather_manager import SpaceWeatherManager
    from inr_modules.data_managers.irinc_neural_proxy import IRINeuralProxy

    cfg = dict(get_config_mdia())
    model_type = config.get('model_type', 'mdia')
    ckpt = _resolve_checkpoint(config, cfg)
    manifest_path = os.path.join(os.path.dirname(ckpt), 'run_manifest.json')
    if model_type == 'fsia' and os.path.isfile(manifest_path):
        with open(manifest_path, encoding='utf-8') as stream:
            trained_config = json.load(stream).get('config', {})
        for key in (
                'fy_path', 'fy_profile_path', 'fy_profile_index_path',
                'cosmic_path', 'cosmic_profile_index_path', 'iri_proxy_path',
                'iri_hmf2_path', 'iri_nmf2_path', 'sw_path'):
            if key in trained_config:
                cfg[key] = trained_config[key]

    # IRI 代理
    iri_proxy = IRINeuralProxy(layers=[4, 128, 128, 128, 128, 1]).to(device)
    proxy_state = torch.load(cfg['iri_proxy_path'], map_location=device)
    iri_proxy.load_state_dict(proxy_state)
    iri_proxy.eval()

    if not os.path.exists(ckpt):
        raise FileNotFoundError(f'模型权重文件不存在: {ckpt}')

    if model_type == 'fsia':
        from inr_modules.mdia.fsia_model import FSIA_INR_Model
        model = FSIA_INR_Model(iri_proxy=iri_proxy, config=cfg).to(device)
        model_name = 'FSIA-INR'
    else:
        from inr_modules.mdia.mdia_model import MDIA_INR_Model
        model = MDIA_INR_Model(iri_proxy=iri_proxy, config=cfg).to(device)
        model_name = 'MDIA-INR'

    state = torch.load(ckpt, map_location=device, weights_only=True)

    # N-adaptive：检查点 N_members 可能与当前 model 不同（如 run56=8, run57=4）
    if model_type == 'fsia':
        _ckpt_P_w1 = state.get('kalman_layer.P_w1')
        if _ckpt_P_w1 is not None and _ckpt_P_w1.shape[0] != model.enkf_n_members:
            _ckpt_n = int(_ckpt_P_w1.shape[0])
            print(f'[main] 检查点 N={_ckpt_n} ≠ 当前 N={model.enkf_n_members}，'
                  f'自适应重建 NeuralETKFLayer(n_members={_ckpt_n})')
            from inr_modules.mdia.fsia_model import NeuralETKFLayer
            _kl = model.kalman_layer
            model.kalman_layer = NeuralETKFLayer(
                d_model    = _kl.d_model,
                b_net_in   = _kl.b_net_in,
                n_members  = _ckpt_n,
                pert_hidden= _kl.pert_hidden,
                r_fy       = _kl.r_fy.item(),
                r_cosmic   = _kl.r_cosmic.item(),
            ).to(device)
            model.enkf_n_members = _ckpt_n

    if model_type == 'fsia':
        model.load_state_dict(state, strict=True)
        if not all(torch.isfinite(value).all() for value in state.values()
                   if torch.is_tensor(value)):
            raise ValueError('FSIA checkpoint contains non-finite values')
    else:
        _load_state_compat(model, state)
    model.eval()
    print(f'[main] {model_name} 模型加载完成: {ckpt}')

    sw_manager = SpaceWeatherManager(
        txt_path=cfg['sw_path'],
        start_date_str=cfg['start_date_str'],
        total_hours=cfg['total_hours'],
        seq_len=cfg['seq_len'],
        device=device,
    )

    # IRI peak parameters provide the FSIA structural reference.
    iri_peak_manager = None
    fy_nb_index = None
    cosmic_nb_index = None
    if model_type == 'fsia':
        try:
            from inr_modules.data_managers.iri_peak_manager import IRIPeakManager
            hmf2_path = cfg.get('iri_hmf2_path', '')
            nmf2_path = cfg.get('iri_nmf2_path', '')
            if hmf2_path and nmf2_path and os.path.exists(hmf2_path) and os.path.exists(nmf2_path):
                iri_peak_manager = IRIPeakManager(
                    hmf2_path=hmf2_path,
                    nmf2_path=nmf2_path,
                    device=device,
                )
                print(f'[main] IRIPeakManager 加载完成')
            else:
                print(f'[main] IRIPeakManager: IRI 峰参数文件未找到，使用中性后备值')
        except Exception as e:
            print(f'[main] IRIPeakManager 初始化失败（{e}），使用中性后备值')

        from inr_modules.data_managers.FY_dataloader import (
            FYNeighborhoodIndex, COSMICNeighborhoodIndex)
        fy_nb_index = FYNeighborhoodIndex(cfg['fy_path'], cfg)
        cosmic_path = cfg.get('cosmic_path', '')
        if not cosmic_path or not os.path.exists(cosmic_path):
            raise FileNotFoundError(f'COSMIC 数据文件不存在: {cosmic_path}')
        cosmic_nb_index = COSMICNeighborhoodIndex(cosmic_path, cfg)
        print('[main] FY/COSMIC 正式局部邻域索引加载完成')

    return (model, sw_manager, cfg, model_name, iri_peak_manager,
            fy_nb_index, cosmic_nb_index)


def _process_station(station_name, day_records, model, sw_manager,
                     start_unix, config, device, model_name='MDIA-INR',
                     iri_peak_manager=None, fy_nb_index=None,
                     cosmic_nb_index=None):
    """
    对单个站点的所有 DayRecord 完成推理、指标计算、绘图。

    Returns:
        dict — 该站点的汇总指标 report
    """
    from isr_evaluation.model_query import query_model_grid, extract_model_nmf2_hmf2
    from isr_evaluation.metrics import (extract_isr_nmf2_hmf2,
                                        compute_nmf2_hmf2_metrics)
    from isr_evaluation.plots import (plot_time_altitude_comparison,
                                      plot_nmf2_scatter,
                                      plot_peak_lt_comparison)

    save_dir    = config['save_dir']
    station_dir = os.path.join(save_dir, station_name)
    os.makedirs(station_dir, exist_ok=True)

    from isr_evaluation.metrics import _valid_pair

    # ---- 累积列表 ----
    # 逐点：分别存 obs/mdia/iri，三者用同一公共有效掩码对齐
    all_obs_log10  = []
    all_pred_log10 = []
    all_bkg_log10  = []
    all_iri_log10  = []

    # 分层指标用坐标（高度、经度、相对UT小时）
    all_strat_alt = []
    all_strat_lon = []
    all_strat_rh  = []

    all_isr_nmf2 = []; all_model_nmf2 = []; all_bkg_nmf2 = []; all_iri_nmf2 = []
    all_isr_hmf2 = []; all_model_hmf2 = []; all_bkg_hmf2 = []; all_iri_hmf2 = []
    all_peak_lt    = []   # 对应 peak 时刻的地方时
    n_valid_days   = 0

    for rec in day_records:
        date_str = rec.get('date_str', 'unknown')
        print(f'  [{station_name}] 处理 {date_str} ...')

        # 一次 forward 同时得到 M11 Analysis、FNDA Background 和 Raw IRI。
        ne_pred, ne_bkg, ne_iri = query_model_grid(
            model, sw_manager, rec, start_unix, device,
            batch_size=config['batch_size'],
            iri_peak_manager=iri_peak_manager,
            fy_nb_index=fy_nb_index,
            cosmic_nb_index=cosmic_nb_index,
        )

        # 六列时间-高度对比图。
        fname = f'{station_name}_{date_str}_comparison.png'
        plot_time_altitude_comparison(
            rec, ne_iri, ne_bkg, ne_pred,
            save_path=os.path.join(station_dir, fname)
        )

        # ISR → log10
        with np.errstate(divide='ignore', invalid='ignore'):
            isr_l = np.where(rec['ne_2d'] > 0,
                             np.log10(rec['ne_2d'].astype(np.float64)), np.nan
                             ).astype(np.float32)

        # 三者公共有效掩码（确保 obs/mdia/iri 长度完全一致）
        common_mask = (
            _valid_pair(isr_l, ne_pred)
            & np.isfinite(ne_bkg) & np.isfinite(ne_iri))
        if common_mask.sum() >= 10:
            all_obs_log10.append(isr_l[common_mask])
            all_pred_log10.append(ne_pred[common_mask])
            all_bkg_log10.append(ne_bkg[common_mask])
            all_iri_log10.append(ne_iri[common_mask])

            # ── 分层指标所需坐标 ──────────────────────────────
            n_alt_r  = len(rec['alt_1d'])
            n_time_r = len(rec['ts_1d'])
            alts_2d_r = np.tile(rec['alt_1d'][:, None], (1, n_time_r))   # [n_alt, n_time]
            rh_2d_r   = np.tile(
                ((rec['ts_1d'] - start_unix) / 3600.0)[None, :],
                (n_alt_r, 1)
            ).astype(np.float32)
            geo_lon_2d_r = rec.get('geo_lon_2d')
            if geo_lon_2d_r is not None:
                lon_2d_r = geo_lon_2d_r
            else:
                lon_2d_r = np.full((n_alt_r, n_time_r),
                                   rec.get('lon', 0.0), dtype=np.float32)
            all_strat_alt.append(alts_2d_r[common_mask].astype(np.float32))
            all_strat_lon.append(lon_2d_r[common_mask].astype(np.float32))
            all_strat_rh.append(rh_2d_r[common_mask].astype(np.float32))
            # ────────────────────────────────────────────────

        # NmF2 / hmF2
        isr_nmf2,   isr_hmf2   = extract_isr_nmf2_hmf2(rec['ne_2d'], rec['alt_1d'])
        model_nmf2, model_hmf2 = extract_model_nmf2_hmf2(ne_pred,    rec['alt_1d'])
        bkg_nmf2,   bkg_hmf2   = extract_model_nmf2_hmf2(ne_bkg,     rec['alt_1d'])
        iri_nmf2,   iri_hmf2   = extract_model_nmf2_hmf2(ne_iri,     rec['alt_1d'])

        all_isr_nmf2.append(isr_nmf2);   all_model_nmf2.append(model_nmf2)
        all_bkg_nmf2.append(bkg_nmf2)
        all_iri_nmf2.append(iri_nmf2)
        all_isr_hmf2.append(isr_hmf2);   all_model_hmf2.append(model_hmf2)
        all_bkg_hmf2.append(bkg_hmf2)
        all_iri_hmf2.append(iri_hmf2)

        # 每个时刻的 LT Unix 时间戳（供 peak-vs-LT 连续时间轴使用）
        lon_1d = float(rec.get('lon') or 0.0)
        if rec.get('geo_lon_2d') is not None:
            lon_1d = float(np.nanmedian(rec['geo_lon_2d'][0, :]))
        lt_unix_1d = rec['ts_1d'] + lon_1d / 15.0 * 3600.0   # UT → LT (Unix s)
        all_peak_lt.append(lt_unix_1d.astype(np.float64))

        n_valid_days += 1

    if n_valid_days == 0:
        print(f'  [{station_name}] 无有效数据，跳过')
        return None

    # ---- 全局逐点统计 ----
    def _point_stats(obs_list, pred_list, prefix=''):
        if not obs_list:
            return {f'{prefix}point_n': 0, f'{prefix}point_rmse': np.nan,
                    f'{prefix}point_mae': np.nan, f'{prefix}point_r': np.nan,
                    f'{prefix}point_ccc': np.nan, f'{prefix}point_bias': np.nan}
        obs_c  = np.concatenate(obs_list).astype(np.float64)
        pred_c = np.concatenate(pred_list).astype(np.float64)
        err    = pred_c - obs_c
        rmse   = float(np.sqrt(np.mean(err ** 2)))
        bias   = float(np.mean(err))
        mae    = float(np.mean(np.abs(err)))
        r = (float(np.corrcoef(obs_c, pred_c)[0, 1])
             if obs_c.std() > 1e-12 and pred_c.std() > 1e-12 else np.nan)
        ccc = _ccc(obs_c, pred_c)
        return {f'{prefix}point_n': len(obs_c), f'{prefix}point_rmse': rmse,
                f'{prefix}point_mae': mae, f'{prefix}point_r': r,
                f'{prefix}point_ccc': ccc, f'{prefix}point_bias': bias}

    model_pt = _point_stats(all_obs_log10, all_pred_log10, prefix='')
    bkg_pt   = _point_stats(all_obs_log10, all_bkg_log10, prefix='background_')
    iri_pt   = _point_stats(all_obs_log10, all_iri_log10,  prefix='iri_')

    # ---- NmF2 / hmF2 全局指标 ----
    isr_nmf2_cat   = np.concatenate(all_isr_nmf2)
    model_nmf2_cat = np.concatenate(all_model_nmf2)
    bkg_nmf2_cat   = np.concatenate(all_bkg_nmf2)
    iri_nmf2_cat   = np.concatenate(all_iri_nmf2)
    isr_hmf2_cat   = np.concatenate(all_isr_hmf2)
    model_hmf2_cat = np.concatenate(all_model_hmf2)
    bkg_hmf2_cat   = np.concatenate(all_bkg_hmf2)
    iri_hmf2_cat   = np.concatenate(all_iri_hmf2)

    model_peak = compute_nmf2_hmf2_metrics(isr_nmf2_cat, isr_hmf2_cat,
                                           model_nmf2_cat, model_hmf2_cat)
    bkg_peak   = compute_nmf2_hmf2_metrics(isr_nmf2_cat, isr_hmf2_cat,
                                           bkg_nmf2_cat, bkg_hmf2_cat)
    iri_peak   = compute_nmf2_hmf2_metrics(isr_nmf2_cat, isr_hmf2_cat,
                                           iri_nmf2_cat,  iri_hmf2_cat)

    # ---- NmF2 散点图 ----
    model_tag = model_name.lower().replace('-', '_').replace(' ', '_')
    plot_nmf2_scatter(
        isr_nmf2_cat, model_nmf2_cat,
        station=f'{station_name} {model_name}',
        save_path=os.path.join(station_dir, f'{station_name}_nmf2_scatter_{model_tag}.png')
    )
    plot_nmf2_scatter(
        isr_nmf2_cat, bkg_nmf2_cat,
        station=f'{station_name} FNDA Background',
        save_path=os.path.join(station_dir, f'{station_name}_nmf2_scatter_background.png')
    )
    plot_nmf2_scatter(
        isr_nmf2_cat, iri_nmf2_cat,
        station=f'{station_name} IRI',
        save_path=os.path.join(station_dir, f'{station_name}_nmf2_scatter_iri.png')
    )

    # ---- hmF2 / NmF2 vs 地方时对比图 ----
    if all_peak_lt:
        lt_cat      = np.concatenate(all_peak_lt)
        plot_peak_lt_comparison(
            lt_all=lt_cat,
            isr_hmf2=isr_hmf2_cat, model_hmf2=model_hmf2_cat,
            background_hmf2=bkg_hmf2_cat, iri_hmf2=iri_hmf2_cat,
            isr_nmf2=isr_nmf2_cat, model_nmf2=model_nmf2_cat,
            background_nmf2=bkg_nmf2_cat, iri_nmf2=iri_nmf2_cat,
            station=station_name, model_name=model_name,
            save_path=os.path.join(station_dir,
                                   f'{station_name}_peak_vs_lt_{model_tag}.png'),
        )

    # 将 IRI peak 指标加 'iri_' 前缀
    iri_peak_prefixed = {f'iri_{k}': v for k, v in iri_peak.items()}
    bkg_peak_prefixed = {f'background_{k}': v for k, v in bkg_peak.items()}

    # ---- 分高度 × 分昼夜 分层指标 ----
    strat_metrics = {}
    if all_strat_alt:
        alt_cat  = np.concatenate(all_strat_alt)
        lon_cat  = np.concatenate(all_strat_lon)
        rh_cat   = np.concatenate(all_strat_rh)
        pred_cat = np.concatenate(all_pred_log10).astype(np.float64)
        bkg_cat  = np.concatenate(all_bkg_log10).astype(np.float64)
        obs_cat  = np.concatenate(all_obs_log10).astype(np.float64)
        iri_cat  = np.concatenate(all_iri_log10).astype(np.float64)
        strat_metrics = _compute_stratified_metrics(
            alt_cat, lon_cat, rh_cat,
            pred_cat, obs_cat, background_all=bkg_cat, iri_all=iri_cat,
        )
        _print_stratified_table(strat_metrics, station_name)

        # CSV 输出
        csv_path = os.path.join(station_dir,
                                f'{station_name}_stratified_metrics.csv')
        _save_stratified_csv(strat_metrics, csv_path)
        print(f'  [{station_name}] 分层指标已保存: {csv_path}')

    report = {
        'station':    station_name,
        'model_name': model_name,
        'n_days':     n_valid_days,
        **model_pt,
        **model_peak,
        **bkg_pt,
        **bkg_peak_prefixed,
        **iri_pt,
        **iri_peak_prefixed,
        'stratified': strat_metrics,
    }
    return report


def main():
    # ==================== 设备 ====================
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[main] 使用设备: {device}')

    # ==================== 时间范围 ====================
    start_unix = _parse_unix(CONFIG['start_date_str'])
    end_unix   = _parse_unix(CONFIG['end_date_str'])
    print(f'[main] 验证时间段: {CONFIG["start_date_str"]} ~ {CONFIG["end_date_str"]}')

    # ==================== 加载模型 ====================
    (model, sw_manager, mdia_cfg, model_name, iri_peak_manager,
     fy_nb_index, cosmic_nb_index) = \
        _load_model_and_managers(CONFIG, device)
    print(f'[main] 模型类型: {model_name}')

    # ==================== 加载 ISR 数据 ====================
    from isr_evaluation.isr_loader import load_jicamarca, load_poker_flat
    from isr_evaluation.coord_convert import convert_day_record_cgm

    station_reports = []
    save_dir = CONFIG['save_dir']
    os.makedirs(save_dir, exist_ok=True)

    # ---------- Jicamarca ----------
    if CONFIG['run_jicamarca']:
        print('\n[main] 读取 Jicamarca ISR 数据 ...')
        jica_records = load_jicamarca(
            data_dir=CONFIG['jicamarca_dir'],
            start_unix=start_unix,
            end_unix=end_unix,
            alt_min=CONFIG['alt_min'],
            alt_max=CONFIG['alt_max'],
            err_ratio_max=CONFIG['err_ratio_max'],
        )
        print(f'  Jicamarca: {len(jica_records)} 天有效数据')

        if jica_records:
            rep = _process_station(
                'Jicamarca', jica_records,
                model, sw_manager, start_unix, CONFIG, device,
                model_name=model_name,
                iri_peak_manager=iri_peak_manager,
                fy_nb_index=fy_nb_index,
                cosmic_nb_index=cosmic_nb_index,
            )
            if rep is not None:
                station_reports.append(rep)

    # ---------- Poker Flat ----------
    if CONFIG['run_poker_flat']:
        print('\n[main] 读取 Poker Flat ISR 数据 ...')
        pf_records = load_poker_flat(
            data_dir=CONFIG['poker_flat_dir'],
            start_unix=start_unix,
            end_unix=end_unix,
            alt_min=CONFIG['alt_min'],
            alt_max=CONFIG['alt_max'],
            err_ratio_max=CONFIG['err_ratio_max'],
        )
        print(f'  Poker Flat: {len(pf_records)} 天有效数据')

        # AACGM → 地理坐标转换
        print('  Poker Flat: 转换 AACGM → 地理坐标 ...')
        for rec in pf_records:
            convert_day_record_cgm(rec)

        if pf_records:
            rep = _process_station(
                'PokerFlat', pf_records,
                model, sw_manager, start_unix, CONFIG, device,
                model_name=model_name,
                iri_peak_manager=iri_peak_manager,
                fy_nb_index=fy_nb_index,
                cosmic_nb_index=cosmic_nb_index,
            )
            if rep is not None:
                station_reports.append(rep)

    # ==================== 汇总报告 ====================
    if station_reports:
        from isr_evaluation.plots import save_metrics_report
        report_path = os.path.join(save_dir, 'isr_validation_report.txt')
        save_metrics_report(station_reports, report_path)

        print('\n' + '=' * 80)
        print('验证完成。汇总指标：')
        print(f'  {"站点":<12}  {"模型":<10}  {"RMSE":>7}  {"MAE":>7}  '
              f'{"CCC":>7}  {"NmF2_CCC":>9}  {"hmF2_MAE":>9}')
        for rep in station_reports:
            def _fv(v, fmt): return f'{v:{fmt}}' if np.isfinite(v) else '   N/A'
            for tag, rmse, mae, pt_ccc, nmf2ccc, hmf2mae in [
                (rep.get('model_name', 'Model'),
                 rep['point_rmse'],              rep['point_mae'],
                 rep.get('point_ccc', np.nan),   rep.get('nmf2_ccc', np.nan),
                 rep['hmf2_mae']),
                ('Background',
                 rep.get('background_point_rmse', np.nan),
                 rep.get('background_point_mae', np.nan),
                 rep.get('background_point_ccc', np.nan),
                 rep.get('background_nmf2_ccc', np.nan),
                 rep.get('background_hmf2_mae', np.nan)),
                ('IRI',
                 rep.get('iri_point_rmse', np.nan), rep.get('iri_point_mae', np.nan),
                 rep.get('iri_point_ccc',  np.nan), rep.get('iri_nmf2_ccc',  np.nan),
                 rep.get('iri_hmf2_mae',   np.nan)),
            ]:
                print(f'  {rep["station"]:12s}  {tag:<10}  '
                      f'{_fv(rmse,".4f"):>7}  {_fv(mae,".4f"):>7}  '
                      f'{_fv(pt_ccc,".4f"):>7}  {_fv(nmf2ccc,".4f"):>9}  '
                      f'{_fv(hmf2mae,".1f"):>8} km')
        print(f'输出目录: {save_dir}')
    else:
        print('\n[main] 无有效站点数据，未生成报告。')


if __name__ == '__main__':
    main()
