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
    python isr_evaluation/main_isr_eval.py --checkpoint <complete v12/v13 checkpoint>

配置项在下方 CONFIG 区块修改。
  model_type: 'fsia' → 加载 FSIA_INR_Model + 显式冻结的M2-V checkpoint（默认）
              'mdia' → 加载 MDIA_INR_Model + best_mdia_model.pth（备用）
"""

import argparse
import hashlib
import os
import sys
import csv
import json
import datetime
import tempfile
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

from inr_modules.mdia.evaluation_stats import paired_group_bootstrap
from isr_evaluation.peak_qa import PeakSearchContract

# ─────────────────────────────────────────────
# ==================== 配置 ====================
# ─────────────────────────────────────────────
CONFIG = {
    # ---- 模型类型：'fsia'（默认）或 'mdia' ----
    'model_type': 'fsia',

    # ---- FSIA必须显式指定完整Analysis checkpoint ----
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
    'save_dir': None,

    # ---- 是否处理各站点（可单独关闭）----
    'run_jicamarca':  True,
    'run_poker_flat': True,
}
_HISTORICAL_EPOCH12_SHA256 = (
    '486ffe73722cde1ff2909da93898e02d4d1e173700c9e4fa9dd2f0990e2fe56a')
# ─────────────────────────────────────────────


# ─────────────────────────────────────────────
# 分层指标工具（分高度 × 分昼夜）
# ─────────────────────────────────────────────
_DAY_LT_RANGE    = (6.0, 18.0)   # 白天：LT 06-18h
_STRATIFIED_SPLIT_KM = 300.0
_STRATIFIED_METRIC_NAMES = ('n', 'rmse', 'bias', 'mae', 'pearson_r', 'ccc')


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
    finite = np.isfinite(pred_log10) & np.isfinite(obs_log10)
    pred_log10 = pred_log10[finite]
    obs_log10 = obs_log10[finite]
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


def _stratified_altitude_masks(altitude, alt_range):
    """Yield the frozen [low, 300), [300, high] ISR reporting strata."""
    domain_min, domain_max = map(float, alt_range)
    if not domain_min < _STRATIFIED_SPLIT_KM < domain_max:
        raise ValueError('ISR reporting domain must straddle 300 km')
    altitude = np.asarray(altitude)
    bins = ((domain_min, _STRATIFIED_SPLIT_KM),
            (_STRATIFIED_SPLIT_KM, domain_max))
    for index, (lower, upper) in enumerate(bins):
        upper_mask = altitude <= upper if index == len(bins) - 1 else altitude < upper
        yield lower, upper, (altitude >= lower) & upper_mask


def _reporting_altitude_masks(altitude, alt_range):
    """Yield the two frozen strata followed by the closed full-domain aggregate."""
    yield from _stratified_altitude_masks(altitude, alt_range)
    domain_min, domain_max = map(float, alt_range)
    altitude = np.asarray(altitude)
    yield (domain_min, domain_max,
           (altitude >= domain_min) & (altitude <= domain_max))


def _grouped_error_metric_bootstrap(observation, prediction, unit_ids,
                                    replicates=2000, seed=42):
    """Profile-grouped CIs for one model's bias, RMSE, and MAE."""
    observation = np.asarray(observation, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    unit_ids = np.asarray(unit_ids)
    finite = np.isfinite(observation) & np.isfinite(prediction)
    error, unit_ids = prediction[finite] - observation[finite], unit_ids[finite]
    _, inverse = np.unique(unit_ids, return_inverse=True)
    count = np.bincount(inverse).astype(np.float64)
    if len(count) < 2:
        return {'status': 'insufficient_data', 'n': int(error.size),
                'sampling_units': int(len(count))}
    stats = np.stack([
        count,
        np.bincount(inverse, weights=error),
        np.bincount(inverse, weights=error ** 2),
        np.bincount(inverse, weights=np.abs(error)),
    ], axis=1)
    rng = np.random.default_rng(seed)
    sampled = stats[rng.integers(0, len(stats), size=(replicates, len(stats)))].sum(axis=1)
    values = {
        'bias': sampled[:, 1] / sampled[:, 0],
        'rmse': np.sqrt(sampled[:, 2] / sampled[:, 0]),
        'mae': sampled[:, 3] / sampled[:, 0],
    }
    point = {
        'bias': float(error.mean()),
        'rmse': float(np.sqrt(np.mean(error ** 2))),
        'mae': float(np.mean(np.abs(error))),
    }
    return {
        'status': 'computed', 'n': int(error.size),
        'sampling_units': int(len(stats)), 'replicates': int(replicates),
        'seed': int(seed),
        **{name: {'estimate': point[name],
                  'ci95': [float(bound) for bound in
                           np.quantile(sampled_values, [0.025, 0.975])]}
           for name, sampled_values in values.items()},
    }


def _analysis_common_mask(observation, analysis, raw_iri):
    """Pair M11 and Raw IRI without consulting diagnostic M00 values."""
    return (np.isfinite(observation) & (observation > 0)
            & np.isfinite(analysis) & np.isfinite(raw_iri))


def _candidate_baseline_common_mask(observation, candidate, baseline,
                                    altitude=None):
    """Strict public mask for a paired 200--500 km checkpoint comparison."""
    mask = (np.isfinite(observation) & (observation > 0)
            & np.isfinite(candidate) & np.isfinite(baseline))
    if altitude is not None:
        mask &= (np.asarray(altitude) >= 200.0) & (np.asarray(altitude) <= 500.0)
    return mask


def _safe_paired_group_bootstrap(observation, candidate, baseline, unit_ids):
    """Keep QA reports serializable when a strict public mask has too few units."""
    finite = (np.isfinite(observation) & np.isfinite(candidate)
              & np.isfinite(baseline))
    n_units = int(np.unique(np.asarray(unit_ids)[finite]).size)
    if finite.sum() < 2 or n_units < 2:
        return {'status': 'insufficient_data', 'n': int(finite.sum()),
                'sampling_units': n_units, 'decision': 'inconclusive'}
    return paired_group_bootstrap(
        np.asarray(observation)[finite], np.asarray(candidate)[finite],
        np.asarray(baseline)[finite], np.asarray(unit_ids)[finite],
        replicates=2000, seed=42)


def _compute_stratified_metrics(alt_all, lon_all, rh_all,
                                pred_all, obs_all, background_all=None,
                                iri_all=None, alt_range=(120.0, 500.0)):
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
    dict  键形如  'analysis_alt_120-300km_day', 'iri_alt_300-500km_night', ...
          每个值为 {'n', 'rmse', 'bias', 'mae', 'pearson_r', 'ccc'}
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
        # 分高度层 × 分昼夜
        for lo, hi, alt_mask in _reporting_altitude_masks(alt_all, alt_range):
            alt_name = f'{lo:g}-{hi:g}km'
            for dn_label, dn_mask in [('day', day_mask), ('night', ~day_mask), ('all', np.ones(len(alt_all), bool))]:
                m = alt_mask & dn_mask
                key = f'{src_name}_alt_{alt_name}_{dn_label}'
                result[key] = _strat_metrics_1d(
                    src_pred[m].astype(np.float64),
                    obs_all[m].astype(np.float64),
                )
                if (lo, hi) == tuple(map(float, alt_range)):
                    result[f'{src_name}_all_alt_{dn_label}'] = result[key]

    return result


def _compute_stratified_bootstrap(altitude, longitude, rel_hour,
                                  observation, analysis, raw_iri, unit_ids,
                                  alt_range):
    """Paired time-profile bootstrap for each M2-W altitude/day stratum."""
    local_time = _lt_from_relhour_lon(rel_hour, longitude)
    day = (local_time >= _DAY_LT_RANGE[0]) & (local_time < _DAY_LT_RANGE[1])
    masks = {}
    for lower, upper, in_altitude in _reporting_altitude_masks(
            altitude, alt_range):
        label = f'alt_{lower:g}-{upper:g}km'
        masks.update({f'{label}_day': in_altitude & day,
                      f'{label}_night': in_altitude & ~day,
                      f'{label}_all': in_altitude})
    result = {}
    for name, selected in masks.items():
        if selected.sum() >= 2 and np.unique(unit_ids[selected]).size >= 2:
            result[name] = paired_group_bootstrap(
                observation[selected], analysis[selected], raw_iri[selected],
                unit_ids[selected], replicates=2000, seed=42)
    full_label = f'alt_{float(alt_range[0]):g}-{float(alt_range[1]):g}km'
    for period in ('day', 'night', 'all'):
        canonical = f'{full_label}_{period}'
        if canonical in result:
            result[f'all_alt_{period}'] = result[canonical]
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


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def _json_text(value):
    """Serialize a public artifact before opening its destination path."""
    return json.dumps(_json_safe(value), ensure_ascii=False, indent=2,
                      allow_nan=False)


def _write_json_atomically(path, value):
    """Write a fully serialized JSON artifact without a partial final pathname."""
    payload = _json_text(value)
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
                mode='w', encoding='utf-8', dir=directory, delete=False,
                prefix=f'.{os.path.basename(path)}.', suffix='.tmp') as stream:
            temporary_path = stream.name
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path and os.path.exists(temporary_path):
            os.unlink(temporary_path)


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


_REPORT_CACHE_KEYS = frozenset({
    'evaluation_cache',
    'peak_cache',
    'paired_evaluation_cache',
    'additional_paired_evaluation_caches',
})


def _public_station_reports(station_reports):
    """Return JSON-facing reports while keeping in-memory cache payloads intact."""
    return [
        {key: value for key, value in report.items()
         if key not in _REPORT_CACHE_KEYS}
        for report in station_reports
    ]


def _artifact_identity(path, save_dir):
    return {
        'relative_path': os.path.relpath(path, save_dir).replace(os.sep, '/'),
        'sha256': _sha256(path),
        'size_bytes': int(os.path.getsize(path)),
    }


def _build_isr_evaluation_contract(candidate_contract, baseline_contract,
                                   isr_input_files, poker_geometry_audit,
                                   config, output_artifacts=None):
    """Build the schema-v2 ISR contract independently of report finalization."""
    unique_isr_files = {
        item['sha256']: item
        for item in isr_input_files
        if isinstance(item, dict) and item.get('sha256')
    }
    peak_contract = PeakSearchContract(
        lower_km=float(config['peak_search_alt_range'][0]),
        upper_km=float(config['peak_search_alt_range'][1]))
    altitude_bins = [[float(lower), float(upper)]
                     for lower, upper, _ in _stratified_altitude_masks(
                         np.empty(0), config.get('alt_range', (120.0, 500.0)))]
    contract = {
        'evaluation_schema_version': 2,
        'stratified_metrics_contract_version': 4,
        'evaluation_code_sha256': _evaluation_code_sha256(),
        'candidate_checkpoint': candidate_contract,
        'baseline_checkpoints': baseline_contract,
        'token_partitions': ['train', 'development'],
        'isr_input_files': sorted(unique_isr_files.values(),
                                  key=lambda item: item['path']),
        'poker_geometry': {
            'semantics': 'wgs84_enu_los_to_ecef_to_geodetic_v1',
            'range_scale_to_km': 1e-3,
            'aacgm_inverse_role': 'diagnostic_only_v1',
            'audit': poker_geometry_audit,
        },
        'quality_thresholds': {
            'finite_observation_required': True,
            'coordinate_mask_separate_from_observation_mask': True,
            'hmf2_public_mask': 'all_compared_fields_status_valid',
            'nmf2_public_mask': 'all_compared_fields_nmf2_valid',
        },
        'peak_search': {
            **peak_contract.as_dict(),
            'alt_range_km': list(config['peak_search_alt_range']),
        },
        'legacy_peak_metrics_semantics': (
            'finite_argmax_with_unbounded_linear_interpolation_v1'),
        'paired_bootstrap': {
            'replicates': 2000, 'seed': 42,
            'group': 'station_time_profile',
            'positive_deltas': [
                'CCC_candidate_minus_baseline',
                'RMSE_baseline_minus_candidate',
                'PearsonR_candidate_minus_baseline'],
        },
        'stratified_metrics': {
            'altitude_bins_km': altitude_bins,
            'interval_semantics': (
                'left_closed_right_open_except_final_right_closed'),
            'full_altitude_range_km': [
                float(config.get('alt_range', (120.0, 500.0))[0]),
                float(config.get('alt_range', (120.0, 500.0))[1]),
            ],
            'full_altitude_interval_semantics': 'closed',
            'full_altitude_aliases': [
                'all_alt_day', 'all_alt_night', 'all_alt_all'],
            'full_altitude_alias_semantics': (
                'exact_aliases_of_closed_full_altitude_range'),
            'day_local_time_range_hours': list(_DAY_LT_RANGE),
            'day_interval_semantics': 'left_closed_right_open',
            'periods': ['day', 'night', 'all'],
            'metrics': list(_STRATIFIED_METRIC_NAMES),
            'source_labels': {
                'analysis': 'M11', 'background': 'M00', 'iri': 'Raw IRI'},
            'low_altitude_diagnostic': {
                'altitude_range_km': [120.0, 200.0],
                'interval_semantics': 'left_closed_right_open',
                'independent_from_stratified_metrics': True,
            },
            'interpretation_warning': (
                '120-300 km mixes 120-200 km without satellite-density targets '
                'and 200-300 km with satellite-density targets; do not treat '
                'the stratum as single-mechanism evidence.'),
        },
    }
    if output_artifacts is not None:
        contract['output_artifacts'] = output_artifacts
    return contract


def _write_isr_final_artifacts(save_dir, station_reports, candidate_contract,
                               baseline_contract, isr_input_files,
                               poker_geometry_audit, config):
    """Persist ISR public reports, caches, and completion contract in that order."""
    from isr_evaluation.plots import save_metrics_report

    public_reports = _public_station_reports(station_reports)
    report_text_path = os.path.join(save_dir, 'isr_validation_report.txt')
    save_metrics_report(public_reports, report_text_path)

    evaluation_cache = {
        'station': [], 'key': [], 'observation_log10': [], 'M11_log10': [],
        'M00_log10': [], 'IRI_log10': [], 'altitude_km': [], 'unit_id': [],
    }
    for report in station_reports:
        cache = report.get('evaluation_cache', {})
        count = len(cache.get('keys', []))
        evaluation_cache['station'].extend([report['station']] * count)
        for output_key, cache_key in (
                ('key', 'keys'), ('observation_log10', 'observation_log10'),
                ('M11_log10', 'M11_log10'), ('M00_log10', 'M00_log10'),
                ('IRI_log10', 'IRI_log10'), ('altitude_km', 'altitude_km'),
                ('unit_id', 'unit_id')):
            evaluation_cache[output_key].extend(cache.get(cache_key, []))
    cache_paths = []
    evaluation_cache_path = os.path.join(save_dir, 'isr_evaluation_cache.npz')
    np.savez_compressed(
        evaluation_cache_path,
        **{key: np.asarray(value) for key, value in evaluation_cache.items()})
    cache_paths.append(evaluation_cache_path)

    peak_cache = {}
    for report in station_reports:
        for key, value in report.get('peak_cache', {}).items():
            peak_cache.setdefault(key, []).extend(value)
    if peak_cache:
        peak_cache_path = os.path.join(save_dir, 'isr_peak_cache.npz')
        np.savez_compressed(
            peak_cache_path,
            **{key: np.asarray(value) for key, value in peak_cache.items()})
        cache_paths.append(peak_cache_path)

    paired_cache = {
        'station': [], 'key': [], 'observation_log10': [],
        'candidate_M11_log10': [], 'candidate_M00_log10': [],
        'candidate_IRI_log10': [], 'baseline_M11_log10': [],
        'baseline_M00_log10': [], 'baseline_IRI_log10': [],
        'altitude_km': [], 'unit_id': [],
    }
    for report in station_reports:
        cache = report.get('paired_evaluation_cache', {})
        count = len(cache.get('keys', []))
        paired_cache['station'].extend([report['station']] * count)
        for output_key, cache_key in (
                ('key', 'keys'), ('observation_log10', 'observation_log10'),
                ('candidate_M11_log10', 'candidate_M11_log10'),
                ('candidate_M00_log10', 'candidate_M00_log10'),
                ('candidate_IRI_log10', 'candidate_IRI_log10'),
                ('baseline_M11_log10', 'baseline_M11_log10'),
                ('baseline_M00_log10', 'baseline_M00_log10'),
                ('baseline_IRI_log10', 'baseline_IRI_log10'),
                ('altitude_km', 'altitude_km'), ('unit_id', 'unit_id')):
            paired_cache[output_key].extend(cache.get(cache_key, []))
    if paired_cache['key']:
        paired_cache_path = os.path.join(
            save_dir, 'isr_paired_evaluation_cache.npz')
        np.savez_compressed(
            paired_cache_path,
            **{key: np.asarray(value) for key, value in paired_cache.items()})
        cache_paths.append(paired_cache_path)

    additional_pair_caches = {}
    for report in station_reports:
        for label, cache in report.get(
                'additional_paired_evaluation_caches', {}).items():
            target = additional_pair_caches.setdefault(
                label, {key: [] for key in cache})
            count = len(cache.get('keys', []))
            target.setdefault('station', []).extend([report['station']] * count)
            for key, value in cache.items():
                target.setdefault(key, []).extend(value)
    for label, cache in additional_pair_caches.items():
        if cache.get('keys'):
            safe_label = ''.join(
                char if char.isalnum() or char in '-_' else '_'
                for char in label)
            cache_path = os.path.join(
                save_dir, f'isr_paired_evaluation_cache_{safe_label}.npz')
            np.savez_compressed(
                cache_path,
                **{key: np.asarray(value) for key, value in cache.items()})
            cache_paths.append(cache_path)

    report_json_path = os.path.join(save_dir, 'isr_validation_report.json')
    _write_json_atomically(report_json_path, public_reports)
    stratified_csv_paths = []
    for report in public_reports:
        csv_path = os.path.join(
            save_dir, report['station'],
            f"{report['station']}_stratified_metrics.csv")
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        _save_stratified_csv(report.get('stratified', {}), csv_path)
        stratified_csv_paths.append(csv_path)
    output_artifacts = {
        'report_text': _artifact_identity(report_text_path, save_dir),
        'report_json': _artifact_identity(report_json_path, save_dir),
        'caches': [_artifact_identity(path, save_dir) for path in cache_paths],
        'stratified_csvs': [
            _artifact_identity(path, save_dir) for path in stratified_csv_paths],
    }
    contract = _build_isr_evaluation_contract(
        candidate_contract, baseline_contract, isr_input_files,
        poker_geometry_audit, config, output_artifacts=output_artifacts)
    contract_path = os.path.join(save_dir, 'isr_evaluation_contract.json')
    _write_json_atomically(contract_path, contract)
    return public_reports, contract


def _evaluation_code_sha256():
    """Hash the P0-A evaluator sources without conflating it with checkpoint format."""
    digest = hashlib.sha256()
    for relative in ('main_isr_eval.py', 'metrics.py', 'model_query.py',
                     'isr_loader.py', 'coord_convert.py', 'peak_qa.py', 'plots.py'):
        path = os.path.join(_SCRIPT_DIR, relative)
        digest.update(relative.encode('utf-8'))
        with open(path, 'rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
    return digest.hexdigest()


def _checkpoint_evaluation_contract(checkpoint, config):
    checkpoint = os.path.abspath(checkpoint)
    run_dir = os.path.dirname(checkpoint)
    with open(os.path.join(run_dir, 'run_manifest.json'), encoding='utf-8') as stream:
        manifest = json.load(stream)
    with open(os.path.join(run_dir, 'training_summary.json'), encoding='utf-8') as stream:
        summary = json.load(stream)
    return {
        'path': checkpoint,
        'sha256': _sha256(checkpoint),
        'checkpoint_format_version': int(config['checkpoint_format_version']),
        'model_domain_semantics': config['model_domain_semantics'],
        'model_alt_range_km': list(map(float, config['alt_range'])),
        'observation_alt_range_km': list(map(float, (
            config.get('observation_alt_range') or config['alt_range']))),
        'peak_search_alt_range_km': list(map(float, (
            config.get('peak_search_alt_range') or config['alt_range']))),
        'date_split': summary.get('date_split'),
        'input_data_identity': manifest.get('data_identity'),
    }


def _resolve_checkpoint(config, mdia_cfg):
    """Resolve the configured or CLI-provided checkpoint."""
    if config['checkpoint_path'] is not None:
        return config['checkpoint_path']
    model_type = config.get('model_type', 'fsia')
    if model_type == 'fsia':
        raise ValueError('M2-V ISR验证缺少显式完整Analysis checkpoint路径')
    return os.path.join(mdia_cfg['save_dir'], 'best_mdia_model.pth')


def _require_m2v_config(config):
    domain = config.get(
        'model_domain_semantics', 'legacy_120_500_domain_v1')
    contracts = {
        'legacy_120_500_domain_v1': (12, (120.0, 500.0)),
        'strict_200_500_domain_v1': (13, (200.0, 500.0)),
        'hybrid_120_500_model_200_500_observation_v1': (
            14, (120.0, 500.0)),
    }
    if domain not in contracts:
        raise ValueError(f'unsupported FSIA model domain: {domain}')
    checkpoint_version, alt_range = contracts[domain]
    expected = {
        'checkpoint_format_version': checkpoint_version,
        'basis_dim': 64,
        'enkf_n_members': 8,
        'enkf_anomaly_parameterization': 'orthogonal_factor',
        'density_basis_semantics': 'endpoint_context_symmetric',
        'r_mode': 'global',
        'use_distance_localization': True,
        'use_physical_localization': True,
        'assimilation_semantics': 'continuous_physical_local_letkf',
        'neighbor_directory_semantics': 'token_exact_positive_support_v1',
        'physical_localization_space_km': 1800.0,
        'physical_localization_time_hours': 1.5,
        'representativeness_floor': 1.0,
        'use_empirical_covariance_loss': False,
    }
    mismatches = {}
    for key, value in expected.items():
        actual = config.get(key)
        if isinstance(value, float):
            if actual is None or not np.isclose(float(actual), value):
                mismatches[key] = (actual, value)
        elif actual != value:
            mismatches[key] = (actual, value)
    if config.get('representativeness_kernel_path') is not None:
        mismatches['representativeness_kernel_path'] = (
            config.get('representativeness_kernel_path'), None)
    configured_alt_range = config.get(
        'alt_range', alt_range if domain == 'legacy_120_500_domain_v1' else ())
    if tuple(map(float, configured_alt_range)) != alt_range:
        mismatches['alt_range'] = (config.get('alt_range'), alt_range)
    expected_observation = (
        (200.0, 500.0) if domain in (
            'strict_200_500_domain_v1',
            'hybrid_120_500_model_200_500_observation_v1') else alt_range)
    expected_peak = expected_observation
    observation = tuple(map(float, config.get('observation_alt_range') or alt_range))
    peak = tuple(map(float, config.get('peak_search_alt_range') or alt_range))
    if observation != expected_observation:
        mismatches['observation_alt_range'] = (observation, expected_observation)
    if peak != expected_peak:
        mismatches['peak_search_alt_range'] = (peak, expected_peak)
    if domain == 'hybrid_120_500_model_200_500_observation_v1':
        if config.get('background_trust_gate_enabled', False):
            mismatches['background_trust_gate_enabled'] = (True, False)
        if config.get('background_seed_ckpt') is not None:
            mismatches['background_seed_ckpt'] = (
                config.get('background_seed_ckpt'), None)
    if config.get('background_trust_gate_enabled', False):
        if config.get('background_trust_gate_semantics') != (
                'fixed_altitude_localtime_dip_smoothstep_v1'):
            mismatches['background_trust_gate_semantics'] = (
                config.get('background_trust_gate_semantics'),
                'fixed_altitude_localtime_dip_smoothstep_v1')
    if mismatches:
        raise ValueError(f'checkpoint不是M2-V推理语义: {mismatches}')


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


def _load_model_and_managers(config, device, checkpoint=None,
                             historical_expected_sha256=None):
    """加载模型、SpaceWeatherManager 和 IRIPeakManager。

    Returns:
        (model, sw_manager, cfg, model_name, iri_peak_manager,
         fy_nb_index, cosmic_nb_index)
    """
    from inr_modules.config_mdia import get_config_mdia
    from inr_modules.data_managers.space_weather_manager import SpaceWeatherManager

    cfg = dict(get_config_mdia())
    model_type = config.get('model_type', 'mdia')
    ckpt = checkpoint or _resolve_checkpoint(config, cfg)
    if model_type == 'fsia':
        from inr_modules.mdia.checkpoint_io import (
            allowed_observation_profile_ids,
            load_fsia_analysis_checkpoint,
        )
        model, cfg, _, _ = load_fsia_analysis_checkpoint(
            ckpt, device,
            allow_historical_epoch=historical_expected_sha256 is not None,
            expected_sha256=historical_expected_sha256)
        _require_m2v_config(cfg)
        allowed_profile_ids = allowed_observation_profile_ids(cfg)
        model_name = 'FSIA-INR'
    else:
        from inr_modules.data_managers.irinc_neural_proxy import IRINeuralProxy
        from inr_modules.mdia.mdia_model import MDIA_INR_Model
        iri_proxy = IRINeuralProxy(layers=[4, 128, 128, 128, 128, 1]).to(device)
        proxy_state = torch.load(cfg['iri_proxy_path'], map_location=device)
        iri_proxy.load_state_dict(proxy_state)
        iri_proxy.eval()
        model = MDIA_INR_Model(iri_proxy=iri_proxy, config=cfg).to(device)
        model_name = 'MDIA-INR'
        state = torch.load(ckpt, map_location=device, weights_only=True)
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

        print('[main] ISR观测token已排除locked-test profile')
    else:
        allowed_profile_ids = None

    return (model, sw_manager, cfg, model_name, iri_peak_manager,
            fy_nb_index, cosmic_nb_index, allowed_profile_ids)


def _process_station(station_name, day_records, model, sw_manager,
                     start_unix, config, device, model_name='MDIA-INR',
                     iri_peak_manager=None, fy_nb_index=None,
                     cosmic_nb_index=None, allowed_profile_ids=None,
                     baseline_context=None):
    """
    对单个站点的所有 DayRecord 完成推理、指标计算、绘图。

    Returns:
        dict — 该站点的汇总指标 report
    """
    from isr_evaluation.model_query import query_model_grid, extract_model_nmf2_hmf2
    from isr_evaluation.metrics import (extract_isr_nmf2_hmf2,
                                        compute_nmf2_hmf2_metrics,
                                        extract_grid_peak_qa,
                                        peak_qc_counts)
    from isr_evaluation.plots import (plot_time_altitude_comparison,
                                      plot_nmf2_scatter,
                                      plot_peak_lt_comparison)

    save_dir    = config['save_dir']
    station_dir = os.path.join(save_dir, station_name)
    os.makedirs(station_dir, exist_ok=True)

    from isr_evaluation.metrics import _valid_pair

    if baseline_context is None:
        baseline_contexts = {}
    elif isinstance(baseline_context, dict):
        baseline_contexts = dict(baseline_context)
    else:
        # Public Python callers before P0-A may still pass a single context.
        baseline_contexts = {'baseline': baseline_context}
    primary_baseline_label = next(iter(baseline_contexts), None)
    extra_baseline_labels = tuple(label for label in baseline_contexts
                                  if label != primary_baseline_label)

    # ---- 累积列表 ----
    # 逐点：分别存 obs/mdia/iri，三者用同一公共有效掩码对齐
    all_obs_log10  = []
    all_pred_log10 = []
    all_bkg_obs_log10 = []
    all_bkg_point_log10 = []
    all_iri_log10  = []

    # 分层指标用坐标（高度、经度、相对UT小时）
    all_strat_alt = []
    all_strat_lon = []
    all_strat_rh  = []
    all_strat_unit = []
    all_bkg_strat_alt = []
    all_bkg_strat_lon = []
    all_bkg_strat_rh = []
    all_cache_obs = []; all_cache_m11 = []; all_cache_m00 = []; all_cache_iri = []
    all_cache_alt = []; all_cache_unit = []; all_cache_key = []
    all_pair_obs = []; all_pair_candidate_m11 = []; all_pair_candidate_m00 = []
    all_pair_candidate_iri = []; all_pair_baseline_m11 = []
    all_pair_baseline_m00 = []; all_pair_baseline_iri = []
    all_pair_alt = []; all_pair_unit = []; all_pair_key = []
    additional_pair_buffers = {
        label: {name: [] for name in (
            'obs', 'candidate_m11', 'candidate_m00', 'candidate_iri',
            'baseline_m11', 'baseline_m00', 'baseline_iri', 'alt', 'unit', 'key')}
        for label in extra_baseline_labels}

    all_isr_nmf2 = []; all_model_nmf2 = []; all_bkg_nmf2 = []; all_iri_nmf2 = []
    all_isr_hmf2 = []; all_model_hmf2 = []; all_bkg_hmf2 = []; all_iri_hmf2 = []
    legacy_isr_nmf2 = []; legacy_model_nmf2 = []; legacy_bkg_nmf2 = []; legacy_iri_nmf2 = []
    legacy_isr_hmf2 = []; legacy_model_hmf2 = []; legacy_bkg_hmf2 = []; legacy_iri_hmf2 = []
    peak_cache = {
        key: [] for key in (
            'station', 'timestamp', 'date',
            'ISR_nmf2_log10', 'ISR_hmf2_km', 'ISR_status', 'ISR_censoring',
            'M11_nmf2_log10', 'M11_hmf2_km', 'M11_status', 'M11_censoring',
            'M00_nmf2_log10', 'M00_hmf2_km', 'M00_status', 'M00_censoring',
            'IRI_nmf2_log10', 'IRI_hmf2_km', 'IRI_status', 'IRI_censoring',
            'ISR_nmf2_valid', 'ISR_hmf2_valid', 'M11_nmf2_valid', 'M11_hmf2_valid',
            'M00_nmf2_valid', 'M00_hmf2_valid', 'IRI_nmf2_valid', 'IRI_hmf2_valid',
            'ISR_prominence_dex', 'M11_prominence_dex', 'M00_prominence_dex',
            'IRI_prominence_dex', 'ISR_max_local_gap_km', 'M11_max_local_gap_km',
            'M00_max_local_gap_km', 'IRI_max_local_gap_km')}
    peak_qc_by_field = {key: [] for key in ('ISR', 'M11', 'M00', 'IRI')}
    all_peak_lt    = []   # 对应 peak 时刻的地方时
    all_peak_unit = []
    all_baseline_nmf2 = []; all_baseline_hmf2 = []
    additional_baseline_peaks = {
        label: {'nmf2': [], 'hmf2': []} for label in extra_baseline_labels}
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
            allowed_profile_ids=allowed_profile_ids,
        )
        baseline_fields = {}
        for label, context in baseline_contexts.items():
            (baseline_model, baseline_sw, _, baseline_iri_peak,
             baseline_fy_index, baseline_cosmic_index,
             baseline_allowed) = context
            baseline_fields[label] = query_model_grid(
                baseline_model, baseline_sw, rec, start_unix, device,
                batch_size=config['batch_size'],
                iri_peak_manager=baseline_iri_peak,
                fy_nb_index=baseline_fy_index,
                cosmic_nb_index=baseline_cosmic_index,
                allowed_profile_ids=baseline_allowed,
            )
        if primary_baseline_label is not None:
            baseline_pred, baseline_bkg, baseline_iri = baseline_fields[
                primary_baseline_label]
        else:
            baseline_pred = baseline_bkg = baseline_iri = None

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

        n_alt_r = len(rec['alt_1d'])
        n_time_r = len(rec['ts_1d'])
        alts_2d_r = np.tile(rec['alt_1d'][:, None], (1, n_time_r))
        rh_2d_r = np.tile(
            ((rec['ts_1d'] - start_unix) / 3600.0)[None, :],
            (n_alt_r, 1)).astype(np.float32)
        geo_lon_2d_r = rec.get('geo_lon_2d')
        lon_2d_r = (geo_lon_2d_r if geo_lon_2d_r is not None else
                    np.full((n_alt_r, n_time_r), rec.get('lon', 0.0),
                            dtype=np.float32))
        ts_2d_r = np.tile(rec['ts_1d'][None, :], (n_alt_r, 1))

        # Analysis有效性只与M11和Raw IRI对齐；M00使用独立诊断掩码。
        common_mask = _analysis_common_mask(isr_l, ne_pred, ne_iri)
        if common_mask.sum() >= 10:
            all_obs_log10.append(isr_l[common_mask])
            all_pred_log10.append(ne_pred[common_mask])
            all_iri_log10.append(ne_iri[common_mask])

            # ── 分层指标所需坐标 ──────────────────────────────
            all_strat_alt.append(alts_2d_r[common_mask].astype(np.float32))
            all_strat_lon.append(lon_2d_r[common_mask].astype(np.float32))
            all_strat_rh.append(rh_2d_r[common_mask].astype(np.float32))
            all_strat_unit.append(ts_2d_r[common_mask].astype(np.int64))
            all_cache_obs.append(isr_l[common_mask])
            all_cache_m11.append(ne_pred[common_mask])
            all_cache_m00.append(ne_bkg[common_mask])
            all_cache_iri.append(ne_iri[common_mask])
            all_cache_alt.append(alts_2d_r[common_mask].astype(np.float32))
            all_cache_unit.append(ts_2d_r[common_mask].astype(np.int64))
            all_cache_key.append(np.asarray([
                f'{station_name}|{date_str}|{int(timestamp)}|{altitude:.3f}'
                for timestamp, altitude in zip(
                    ts_2d_r[common_mask], alts_2d_r[common_mask])
            ]))
            # ────────────────────────────────────────────────

        if baseline_pred is not None:
            paired_mask = _candidate_baseline_common_mask(
                isr_l, ne_pred, baseline_pred, alts_2d_r)
            if paired_mask.sum() >= 10:
                all_pair_obs.append(isr_l[paired_mask])
                all_pair_candidate_m11.append(ne_pred[paired_mask])
                all_pair_candidate_m00.append(ne_bkg[paired_mask])
                all_pair_candidate_iri.append(ne_iri[paired_mask])
                all_pair_alt.append(alts_2d_r[paired_mask].astype(np.float32))
                all_pair_unit.append(ts_2d_r[paired_mask].astype(np.int64))
                all_pair_key.append(np.asarray([
                    f'{station_name}|{date_str}|{int(timestamp)}|{altitude:.3f}'
                    for timestamp, altitude in zip(
                        ts_2d_r[paired_mask], alts_2d_r[paired_mask])
                ]))
                # Store the candidate and baseline fields side-by-side for a
                # reproducible public-mask comparison without re-reading ISR.
                all_pair_baseline_m11.append(baseline_pred[paired_mask])
                all_pair_baseline_m00.append(baseline_bkg[paired_mask])
                all_pair_baseline_iri.append(baseline_iri[paired_mask])

        for label in extra_baseline_labels:
            extra_pred, extra_bkg, extra_iri = baseline_fields[label]
            paired_mask = _candidate_baseline_common_mask(
                isr_l, ne_pred, extra_pred, alts_2d_r)
            if paired_mask.sum() < 10:
                continue
            buffer = additional_pair_buffers[label]
            buffer['obs'].append(isr_l[paired_mask])
            buffer['candidate_m11'].append(ne_pred[paired_mask])
            buffer['candidate_m00'].append(ne_bkg[paired_mask])
            buffer['candidate_iri'].append(ne_iri[paired_mask])
            buffer['baseline_m11'].append(extra_pred[paired_mask])
            buffer['baseline_m00'].append(extra_bkg[paired_mask])
            buffer['baseline_iri'].append(extra_iri[paired_mask])
            buffer['alt'].append(alts_2d_r[paired_mask].astype(np.float32))
            buffer['unit'].append(ts_2d_r[paired_mask].astype(np.int64))
            buffer['key'].append(np.asarray([
                f'{station_name}|{date_str}|{int(timestamp)}|{altitude:.3f}'
                for timestamp, altitude in zip(
                    ts_2d_r[paired_mask], alts_2d_r[paired_mask])
            ]))

        background_mask = _valid_pair(isr_l, ne_bkg)
        if background_mask.sum() >= 10:
            all_bkg_obs_log10.append(isr_l[background_mask])
            all_bkg_point_log10.append(ne_bkg[background_mask])
            all_bkg_strat_alt.append(
                alts_2d_r[background_mask].astype(np.float32))
            all_bkg_strat_lon.append(
                lon_2d_r[background_mask].astype(np.float32))
            all_bkg_strat_rh.append(
                rh_2d_r[background_mask].astype(np.float32))

        # Peak QA-v2: retain the historical finite-argmax arrays, then derive the
        # primary arrays from the shared ISR/GIRO no-extrapolation contract.
        peak_range = tuple(map(float, config.get('peak_search_alt_range')
                               or config.get('alt_range', (120.0, 500.0))))
        legacy_isr_nm, legacy_isr_hm = extract_isr_nmf2_hmf2(
            rec['ne_2d'], rec['alt_1d'], peak_search_alt_range=peak_range)
        legacy_model_nm, legacy_model_hm = extract_model_nmf2_hmf2(
            ne_pred, rec['alt_1d'], peak_search_alt_range=peak_range)
        legacy_bkg_nm, legacy_bkg_hm = extract_model_nmf2_hmf2(
            ne_bkg, rec['alt_1d'], peak_search_alt_range=peak_range)
        legacy_iri_nm, legacy_iri_hm = extract_model_nmf2_hmf2(
            ne_iri, rec['alt_1d'], peak_search_alt_range=peak_range)
        peak_contract = PeakSearchContract(
            lower_km=peak_range[0], upper_km=peak_range[1])
        isr_peak_qa = extract_grid_peak_qa(isr_l, rec['alt_1d'], peak_contract)
        model_peak_qa = extract_grid_peak_qa(ne_pred, rec['alt_1d'], peak_contract)
        bkg_peak_qa = extract_grid_peak_qa(ne_bkg, rec['alt_1d'], peak_contract)
        iri_peak_qa = extract_grid_peak_qa(ne_iri, rec['alt_1d'], peak_contract)
        isr_nmf2 = np.where(isr_peak_qa['nmf2_valid'], isr_peak_qa['nmf2_log10'], np.nan)
        isr_hmf2 = np.where(isr_peak_qa['hmf2_valid'], isr_peak_qa['hmf2_km'], np.nan)
        model_nmf2 = np.where(model_peak_qa['nmf2_valid'], model_peak_qa['nmf2_log10'], np.nan)
        model_hmf2 = np.where(model_peak_qa['hmf2_valid'], model_peak_qa['hmf2_km'], np.nan)
        bkg_nmf2 = np.where(bkg_peak_qa['nmf2_valid'], bkg_peak_qa['nmf2_log10'], np.nan)
        bkg_hmf2 = np.where(bkg_peak_qa['hmf2_valid'], bkg_peak_qa['hmf2_km'], np.nan)
        iri_nmf2 = np.where(iri_peak_qa['nmf2_valid'], iri_peak_qa['nmf2_log10'], np.nan)
        iri_hmf2 = np.where(iri_peak_qa['hmf2_valid'], iri_peak_qa['hmf2_km'], np.nan)
        if baseline_pred is not None:
            legacy_baseline_nm, legacy_baseline_hm = extract_model_nmf2_hmf2(
                baseline_pred, rec['alt_1d'], peak_search_alt_range=peak_range)
            baseline_peak_qa = extract_grid_peak_qa(
                baseline_pred, rec['alt_1d'], peak_contract)
            baseline_nmf2 = np.where(baseline_peak_qa['nmf2_valid'],
                                     baseline_peak_qa['nmf2_log10'], np.nan)
            baseline_hmf2 = np.where(baseline_peak_qa['hmf2_valid'],
                                     baseline_peak_qa['hmf2_km'], np.nan)
        extra_peak_fields = {}
        for label in extra_baseline_labels:
            extra_pred, _, _ = baseline_fields[label]
            extra_qa = extract_grid_peak_qa(
                extra_pred, rec['alt_1d'], peak_contract)
            extra_peak_fields[label] = {
                'nmf2': np.where(extra_qa['nmf2_valid'], extra_qa['nmf2_log10'], np.nan),
                'hmf2': np.where(extra_qa['hmf2_valid'], extra_qa['hmf2_km'], np.nan),
                'qa': extra_qa,
            }

        all_isr_nmf2.append(isr_nmf2);   all_model_nmf2.append(model_nmf2)
        all_bkg_nmf2.append(bkg_nmf2)
        all_iri_nmf2.append(iri_nmf2)
        all_isr_hmf2.append(isr_hmf2);   all_model_hmf2.append(model_hmf2)
        all_bkg_hmf2.append(bkg_hmf2)
        all_iri_hmf2.append(iri_hmf2)
        legacy_isr_nmf2.append(legacy_isr_nm); legacy_model_nmf2.append(legacy_model_nm)
        legacy_bkg_nmf2.append(legacy_bkg_nm); legacy_iri_nmf2.append(legacy_iri_nm)
        legacy_isr_hmf2.append(legacy_isr_hm); legacy_model_hmf2.append(legacy_model_hm)
        legacy_bkg_hmf2.append(legacy_bkg_hm); legacy_iri_hmf2.append(legacy_iri_hm)
        if baseline_pred is not None:
            all_baseline_nmf2.append(baseline_nmf2)
            all_baseline_hmf2.append(baseline_hmf2)
        for label, values in extra_peak_fields.items():
            additional_baseline_peaks[label]['nmf2'].append(values['nmf2'])
            additional_baseline_peaks[label]['hmf2'].append(values['hmf2'])

        peak_qc_by_field['ISR'].append(isr_peak_qa)
        peak_qc_by_field['M11'].append(model_peak_qa)
        peak_qc_by_field['M00'].append(bkg_peak_qa)
        peak_qc_by_field['IRI'].append(iri_peak_qa)
        n_peaks = len(rec['ts_1d'])
        peak_cache['station'].extend([station_name] * n_peaks)
        peak_cache['timestamp'].extend(rec['ts_1d'].astype(np.int64).tolist())
        peak_cache['date'].extend([date_str] * n_peaks)
        for label, arrays in (('ISR', isr_peak_qa), ('M11', model_peak_qa),
                              ('M00', bkg_peak_qa), ('IRI', iri_peak_qa)):
            for source_key, cache_key in (
                    ('nmf2_log10', f'{label}_nmf2_log10'),
                    ('hmf2_km', f'{label}_hmf2_km'),
                    ('status', f'{label}_status'),
                    ('censoring', f'{label}_censoring'),
                    ('nmf2_valid', f'{label}_nmf2_valid'),
                    ('hmf2_valid', f'{label}_hmf2_valid'),
                    ('prominence_dex', f'{label}_prominence_dex'),
                    ('max_local_gap_km', f'{label}_max_local_gap_km')):
                peak_cache[cache_key].extend(np.asarray(arrays[source_key]).tolist())
        if baseline_pred is not None:
            for source_key in ('nmf2_log10', 'hmf2_km', 'status', 'censoring',
                               'nmf2_valid', 'hmf2_valid', 'prominence_dex',
                               'max_local_gap_km'):
                cache_key = f'baseline_{primary_baseline_label}_{source_key}'
                peak_cache.setdefault(cache_key, []).extend(
                    np.asarray(baseline_peak_qa[source_key]).tolist())
        for label, values in extra_peak_fields.items():
            arrays = values['qa']
            for source_key in ('nmf2_log10', 'hmf2_km', 'status', 'censoring',
                               'nmf2_valid', 'hmf2_valid', 'prominence_dex',
                               'max_local_gap_km'):
                cache_key = f'baseline_{label}_{source_key}'
                peak_cache.setdefault(cache_key, []).extend(
                    np.asarray(arrays[source_key]).tolist())

        # 每个时刻的 LT Unix 时间戳（供 peak-vs-LT 连续时间轴使用）
        lon_1d = float(rec.get('lon') or 0.0)
        if rec.get('geo_lon_2d') is not None:
            lon_1d = float(np.nanmedian(rec['geo_lon_2d'][0, :]))
        lt_unix_1d = rec['ts_1d'] + lon_1d / 15.0 * 3600.0   # UT → LT (Unix s)
        all_peak_lt.append(lt_unix_1d.astype(np.float64))
        all_peak_unit.append(rec['ts_1d'].astype(np.int64))

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
    bkg_pt   = _point_stats(
        all_bkg_obs_log10, all_bkg_point_log10, prefix='background_')
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

    nmf2_common = (np.isfinite(isr_nmf2_cat) & np.isfinite(model_nmf2_cat)
                   & np.isfinite(iri_nmf2_cat))
    hmf2_common = (np.isfinite(isr_hmf2_cat) & np.isfinite(model_hmf2_cat)
                   & np.isfinite(iri_hmf2_cat))
    model_peak = compute_nmf2_hmf2_metrics(
        np.where(nmf2_common, isr_nmf2_cat, np.nan),
        np.where(hmf2_common, isr_hmf2_cat, np.nan),
        np.where(nmf2_common, model_nmf2_cat, np.nan),
        np.where(hmf2_common, model_hmf2_cat, np.nan))
    bkg_peak   = compute_nmf2_hmf2_metrics(isr_nmf2_cat, isr_hmf2_cat,
                                           bkg_nmf2_cat, bkg_hmf2_cat)
    iri_peak = compute_nmf2_hmf2_metrics(
        np.where(nmf2_common, isr_nmf2_cat, np.nan),
        np.where(hmf2_common, isr_hmf2_cat, np.nan),
        np.where(nmf2_common, iri_nmf2_cat, np.nan),
        np.where(hmf2_common, iri_hmf2_cat, np.nan))

    # Historical finite-argmax aggregate retained only to show the impact of the
    # new censoring/gap/public-mask contract.
    legacy_isr_nmf2_cat = np.concatenate(legacy_isr_nmf2)
    legacy_model_nmf2_cat = np.concatenate(legacy_model_nmf2)
    legacy_bkg_nmf2_cat = np.concatenate(legacy_bkg_nmf2)
    legacy_iri_nmf2_cat = np.concatenate(legacy_iri_nmf2)
    legacy_isr_hmf2_cat = np.concatenate(legacy_isr_hmf2)
    legacy_model_hmf2_cat = np.concatenate(legacy_model_hmf2)
    legacy_bkg_hmf2_cat = np.concatenate(legacy_bkg_hmf2)
    legacy_iri_hmf2_cat = np.concatenate(legacy_iri_hmf2)
    legacy_common = (np.isfinite(legacy_isr_nmf2_cat)
                     & np.isfinite(legacy_model_nmf2_cat)
                     & np.isfinite(legacy_iri_nmf2_cat))
    legacy_h_common = (np.isfinite(legacy_isr_hmf2_cat)
                       & np.isfinite(legacy_model_hmf2_cat)
                       & np.isfinite(legacy_iri_hmf2_cat))
    legacy_peak_metrics = {
        'analysis': compute_nmf2_hmf2_metrics(
            np.where(legacy_common, legacy_isr_nmf2_cat, np.nan),
            np.where(legacy_h_common, legacy_isr_hmf2_cat, np.nan),
            np.where(legacy_common, legacy_model_nmf2_cat, np.nan),
            np.where(legacy_h_common, legacy_model_hmf2_cat, np.nan)),
        'background': compute_nmf2_hmf2_metrics(
            legacy_isr_nmf2_cat, legacy_isr_hmf2_cat,
            legacy_bkg_nmf2_cat, legacy_bkg_hmf2_cat),
        'iri': compute_nmf2_hmf2_metrics(
            np.where(legacy_common, legacy_isr_nmf2_cat, np.nan),
            np.where(legacy_h_common, legacy_isr_hmf2_cat, np.nan),
            np.where(legacy_common, legacy_iri_nmf2_cat, np.nan),
            np.where(legacy_h_common, legacy_iri_hmf2_cat, np.nan)),
    }

    # ---- NmF2 散点图 ----
    model_tag = model_name.lower().replace('-', '_').replace(' ', '_')
    plot_nmf2_scatter(
        np.where(nmf2_common, isr_nmf2_cat, np.nan),
        np.where(nmf2_common, model_nmf2_cat, np.nan),
        station=f'{station_name} {model_name}',
        save_path=os.path.join(station_dir, f'{station_name}_nmf2_scatter_{model_tag}.png')
    )
    plot_nmf2_scatter(
        isr_nmf2_cat, bkg_nmf2_cat,
        station=f'{station_name} FNDA Background',
        save_path=os.path.join(station_dir, f'{station_name}_nmf2_scatter_background.png')
    )
    plot_nmf2_scatter(
        np.where(nmf2_common, isr_nmf2_cat, np.nan),
        np.where(nmf2_common, iri_nmf2_cat, np.nan),
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
    strat_bootstrap = {}
    if all_strat_alt:
        alt_cat  = np.concatenate(all_strat_alt)
        lon_cat  = np.concatenate(all_strat_lon)
        rh_cat   = np.concatenate(all_strat_rh)
        pred_cat = np.concatenate(all_pred_log10).astype(np.float64)
        obs_cat  = np.concatenate(all_obs_log10).astype(np.float64)
        iri_cat  = np.concatenate(all_iri_log10).astype(np.float64)
        strat_metrics = _compute_stratified_metrics(
            alt_cat, lon_cat, rh_cat,
            pred_cat, obs_cat, iri_all=iri_cat,
            alt_range=config.get('alt_range', (120.0, 500.0)),
        )
        if all_bkg_strat_alt:
            background_strata = _compute_stratified_metrics(
                np.concatenate(all_bkg_strat_alt),
                np.concatenate(all_bkg_strat_lon),
                np.concatenate(all_bkg_strat_rh),
                np.concatenate(all_bkg_point_log10).astype(np.float64),
                np.concatenate(all_bkg_obs_log10).astype(np.float64),
                alt_range=config.get('alt_range', (120.0, 500.0)))
            strat_metrics.update({
                key.replace('analysis_', 'background_', 1): value
                for key, value in background_strata.items()})
        _print_stratified_table(strat_metrics, station_name)
        unit_cat = np.concatenate(all_strat_unit)
        strat_bootstrap = _compute_stratified_bootstrap(
            alt_cat, lon_cat, rh_cat, obs_cat, pred_cat, iri_cat, unit_cat,
            config.get('alt_range', (120.0, 500.0)))

        # CSV 输出
        csv_path = os.path.join(station_dir,
                                f'{station_name}_stratified_metrics.csv')
        _save_stratified_csv(strat_metrics, csv_path)
        print(f'  [{station_name}] 分层指标已保存: {csv_path}')

    overall_bootstrap = (paired_group_bootstrap(
        obs_cat, pred_cat, iri_cat, unit_cat, replicates=2000, seed=42)
        if all_strat_alt else None)
    paired_comparison = None
    if all_pair_obs:
        pair_obs = np.concatenate(all_pair_obs).astype(np.float64)
        pair_candidate = np.concatenate(all_pair_candidate_m11).astype(np.float64)
        pair_baseline = np.concatenate(all_pair_baseline_m11).astype(np.float64)
        pair_units = np.concatenate(all_pair_unit)
        paired_comparison = {
            'public_mask': 'finite(obs,candidate,baseline) & 200<=alt<=500',
            'point_metrics': {
                'candidate_m11': _point_stats([pair_obs], [pair_candidate]),
                'baseline_m11': _point_stats([pair_obs], [pair_baseline]),
            },
            'm11_candidate_vs_baseline_bootstrap': paired_group_bootstrap(
                pair_obs, pair_candidate, pair_baseline, pair_units,
                replicates=2000, seed=42),
        }
    peak_comparison = None
    if all_baseline_nmf2 and all_peak_unit:
        peak_units = np.concatenate(all_peak_unit)
        candidate_nmf2 = np.concatenate(all_model_nmf2)
        baseline_nmf2 = np.concatenate(all_baseline_nmf2)
        candidate_hmf2 = np.concatenate(all_model_hmf2)
        baseline_hmf2 = np.concatenate(all_baseline_hmf2)
        nmf2_mask = (np.isfinite(isr_nmf2_cat) & np.isfinite(candidate_nmf2)
                     & np.isfinite(baseline_nmf2))
        hmf2_mask = (np.isfinite(isr_hmf2_cat) & np.isfinite(candidate_hmf2)
                     & np.isfinite(baseline_hmf2))
        peak_comparison = {
            'NmF2_m11_candidate_vs_baseline_bootstrap': paired_group_bootstrap(
                isr_nmf2_cat[nmf2_mask], candidate_nmf2[nmf2_mask],
                baseline_nmf2[nmf2_mask], peak_units[nmf2_mask],
                replicates=2000, seed=42),
            'hmF2_m11_candidate_vs_baseline_bootstrap': paired_group_bootstrap(
                isr_hmf2_cat[hmf2_mask], candidate_hmf2[hmf2_mask],
                baseline_hmf2[hmf2_mask], peak_units[hmf2_mask],
                replicates=2000, seed=42),
        }
    additional_comparisons = {}
    for label in extra_baseline_labels:
        buffer = additional_pair_buffers[label]
        comparison = {}
        if buffer['obs']:
            pair_obs = np.concatenate(buffer['obs']).astype(np.float64)
            pair_candidate = np.concatenate(buffer['candidate_m11']).astype(np.float64)
            pair_baseline = np.concatenate(buffer['baseline_m11']).astype(np.float64)
            pair_units = np.concatenate(buffer['unit'])
            comparison['point'] = {
                'public_mask': 'finite(obs,candidate,baseline) & 200<=alt<=500',
                'candidate_m11': _point_stats([pair_obs], [pair_candidate]),
                'baseline_m11': _point_stats([pair_obs], [pair_baseline]),
                'bootstrap': _safe_paired_group_bootstrap(
                    pair_obs, pair_candidate, pair_baseline, pair_units),
            }
        if all_peak_unit and additional_baseline_peaks[label]['nmf2']:
            peak_units = np.concatenate(all_peak_unit)
            extra_nmf2 = np.concatenate(additional_baseline_peaks[label]['nmf2'])
            extra_hmf2 = np.concatenate(additional_baseline_peaks[label]['hmf2'])
            candidate_nmf2 = np.concatenate(all_model_nmf2)
            candidate_hmf2 = np.concatenate(all_model_hmf2)
            nmf2_mask = (np.isfinite(isr_nmf2_cat) & np.isfinite(candidate_nmf2)
                         & np.isfinite(extra_nmf2))
            hmf2_mask = (np.isfinite(isr_hmf2_cat) & np.isfinite(candidate_hmf2)
                         & np.isfinite(extra_hmf2))
            comparison['peaks'] = {
                'NmF2_m11_candidate_vs_baseline_bootstrap': _safe_paired_group_bootstrap(
                    isr_nmf2_cat[nmf2_mask], candidate_nmf2[nmf2_mask],
                    extra_nmf2[nmf2_mask], peak_units[nmf2_mask]),
                'hmF2_m11_candidate_vs_baseline_bootstrap': _safe_paired_group_bootstrap(
                    isr_hmf2_cat[hmf2_mask], candidate_hmf2[hmf2_mask],
                    extra_hmf2[hmf2_mask], peak_units[hmf2_mask]),
            }
        additional_comparisons[label] = comparison or {
            'status': 'insufficient_data'}
    peak_qc_summary = {}
    for label, fragments in peak_qc_by_field.items():
        joined = {key: np.concatenate([fragment[key] for fragment in fragments])
                  for key in fragments[0]} if fragments else {}
        peak_qc_summary[label] = peak_qc_counts(joined) if joined else {
            'status': {}, 'n_total': 0, 'nmf2_valid': 0,
            'hmf2_valid': 0, 'boundary_or_censored': 0}
    peak_mask_attrition = {
        'legacy_common_hmf2_n': int(legacy_h_common.sum()),
        'primary_common_hmf2_n': int(hmf2_common.sum()),
        'legacy_common_nmf2_n': int(legacy_common.sum()),
        'primary_common_nmf2_n': int(nmf2_common.sum()),
        'hmf2_removed_by_qa': int(legacy_h_common.sum() - hmf2_common.sum()),
        'nmf2_removed_by_qa': int(legacy_common.sum() - nmf2_common.sum()),
    }

    low_altitude_diagnostic = {
        'status': 'not_applicable',
        'reason': 'model_domain_does_not_include_120_200_km',
        'altitude_range_km': [120.0, 200.0],
        'models': ['M11', 'M00', 'Raw IRI'],
    }
    if float(config.get('alt_range', (200.0, 500.0))[0]) < 200.0:
        low_altitude_diagnostic = {
            'status': 'insufficient_data',
            'reason': 'no_finite_low_altitude_public_samples',
            'altitude_range_km': [120.0, 200.0],
            'models': ['M11', 'M00', 'Raw IRI'],
        }
    if (float(config.get('alt_range', (200.0, 500.0))[0]) < 200.0
            and all_cache_obs):
        low_obs = np.concatenate(all_cache_obs).astype(np.float64)
        low_m11 = np.concatenate(all_cache_m11).astype(np.float64)
        low_m00 = np.concatenate(all_cache_m00).astype(np.float64)
        low_iri = np.concatenate(all_cache_iri).astype(np.float64)
        low_altitude = np.concatenate(all_cache_alt).astype(np.float64)
        low_units = np.concatenate(all_cache_unit)
        low_mask = ((low_altitude >= 120.0) & (low_altitude < 200.0)
                    & np.isfinite(low_obs) & np.isfinite(low_m11)
                    & np.isfinite(low_m00) & np.isfinite(low_iri))
        if (low_mask.sum() >= 2
                and np.unique(low_units[low_mask]).size >= 2):
            low_point_metrics = {}
            low_bootstrap = {}
            for label, prediction in (
                    ('M11', low_m11), ('M00', low_m00), ('Raw IRI', low_iri)):
                point = _strat_metrics_1d(
                    prediction[low_mask], low_obs[low_mask])
                low_point_metrics[label] = {
                    key: point[key] for key in ('n', 'rmse', 'bias', 'mae')}
                low_bootstrap[label] = _grouped_error_metric_bootstrap(
                    low_obs[low_mask], prediction[low_mask],
                    low_units[low_mask], replicates=2000, seed=42)
            low_altitude_diagnostic = {
                'status': 'computed',
                'altitude_range_km': [120.0, 200.0],
                'models': ['M11', 'M00', 'Raw IRI'],
                'point_metrics': low_point_metrics,
                'bootstrap': low_bootstrap,
                'm11_vs_m00_bootstrap': paired_group_bootstrap(
                    low_obs[low_mask], low_m11[low_mask], low_m00[low_mask],
                    low_units[low_mask], replicates=2000, seed=42),
                'm11_vs_raw_iri_bootstrap': paired_group_bootstrap(
                    low_obs[low_mask], low_m11[low_mask], low_iri[low_mask],
                    low_units[low_mask], replicates=2000, seed=42),
            }
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
        'm11_vs_raw_iri_bootstrap': overall_bootstrap,
        'passed_m2w_m11_vs_raw_iri_gate': (
            overall_bootstrap is not None
            and overall_bootstrap['decision'] == 'pass'),
        'stratified_m11_vs_raw_iri_bootstrap': strat_bootstrap,
        'candidate_vs_baseline': paired_comparison,
        'peak_candidate_vs_baseline': peak_comparison,
        'additional_candidate_vs_baselines': additional_comparisons,
        'low_altitude_diagnostic': low_altitude_diagnostic,
        'primary_peak_metrics': {
            'analysis': model_peak,
            'background': bkg_peak,
            'iri': iri_peak,
        },
        'legacy_peak_metrics': legacy_peak_metrics,
        'peak_qc_counts': peak_qc_summary,
        'mask_attrition': peak_mask_attrition,
        'peak_cache': peak_cache,
        'evaluation_cache': {
            'keys': np.concatenate(all_cache_key).tolist() if all_cache_key else [],
            'observation_log10': np.concatenate(all_cache_obs).tolist() if all_cache_obs else [],
            'M11_log10': np.concatenate(all_cache_m11).tolist() if all_cache_m11 else [],
            'M00_log10': np.concatenate(all_cache_m00).tolist() if all_cache_m00 else [],
            'IRI_log10': np.concatenate(all_cache_iri).tolist() if all_cache_iri else [],
            'altitude_km': np.concatenate(all_cache_alt).tolist() if all_cache_alt else [],
            'unit_id': np.concatenate(all_cache_unit).tolist() if all_cache_unit else [],
        },
        'paired_evaluation_cache': {
            'keys': np.concatenate(all_pair_key).tolist() if all_pair_key else [],
            'observation_log10': np.concatenate(all_pair_obs).tolist() if all_pair_obs else [],
            'candidate_M11_log10': np.concatenate(all_pair_candidate_m11).tolist()
            if all_pair_candidate_m11 else [],
            'candidate_M00_log10': np.concatenate(all_pair_candidate_m00).tolist()
            if all_pair_candidate_m00 else [],
            'candidate_IRI_log10': np.concatenate(all_pair_candidate_iri).tolist()
            if all_pair_candidate_iri else [],
            'baseline_M11_log10': np.concatenate(all_pair_baseline_m11).tolist()
            if all_pair_baseline_m11 else [],
            'baseline_M00_log10': np.concatenate(all_pair_baseline_m00).tolist()
            if all_pair_baseline_m00 else [],
            'baseline_IRI_log10': np.concatenate(all_pair_baseline_iri).tolist()
            if all_pair_baseline_iri else [],
            'altitude_km': np.concatenate(all_pair_alt).tolist() if all_pair_alt else [],
            'unit_id': np.concatenate(all_pair_unit).tolist() if all_pair_unit else [],
        },
        'additional_paired_evaluation_caches': {
            label: {
                'keys': np.concatenate(buffer['key']).tolist() if buffer['key'] else [],
                'observation_log10': np.concatenate(buffer['obs']).tolist() if buffer['obs'] else [],
                'candidate_M11_log10': np.concatenate(buffer['candidate_m11']).tolist()
                if buffer['candidate_m11'] else [],
                'candidate_M00_log10': np.concatenate(buffer['candidate_m00']).tolist()
                if buffer['candidate_m00'] else [],
                'candidate_IRI_log10': np.concatenate(buffer['candidate_iri']).tolist()
                if buffer['candidate_iri'] else [],
                'baseline_M11_log10': np.concatenate(buffer['baseline_m11']).tolist()
                if buffer['baseline_m11'] else [],
                'baseline_M00_log10': np.concatenate(buffer['baseline_m00']).tolist()
                if buffer['baseline_m00'] else [],
                'baseline_IRI_log10': np.concatenate(buffer['baseline_iri']).tolist()
                if buffer['baseline_iri'] else [],
                'altitude_km': np.concatenate(buffer['alt']).tolist() if buffer['alt'] else [],
                'unit_id': np.concatenate(buffer['unit']).tolist() if buffer['unit'] else [],
            }
            for label, buffer in additional_pair_buffers.items()
        },
    }
    return report


def _is_historical_epoch_checkpoint(checkpoint):
    parts = os.path.splitext(os.path.basename(checkpoint))[0].split('_')
    return (len(parts) == 3 and parts[0] == 'epoch' and parts[1].isdigit()
            and parts[2] == 'model')


def main(checkpoint=None, save_dir=None, preflight_only=False,
         baseline_checkpoint=None, baseline_checkpoint_sha256=None,
         baseline_labels=None):
    if checkpoint is not None:
        CONFIG['checkpoint_path'] = checkpoint
    if save_dir is not None:
        CONFIG['save_dir'] = save_dir
    if CONFIG['save_dir'] is None and CONFIG['checkpoint_path'] is not None:
        checkpoint_path = os.path.abspath(CONFIG['checkpoint_path'])
        run_name = os.path.basename(os.path.dirname(checkpoint_path))
        epoch_name = os.path.splitext(os.path.basename(checkpoint_path))[0]
        CONFIG['save_dir'] = os.path.join(
            _FSIA_DIR, 'isr_validation_outputs',
            f'{run_name}-{epoch_name}-isr')
    # ==================== 设备 ====================
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[main] 使用设备: {device}')

    # ==================== 时间范围 ====================
    start_unix = _parse_unix(CONFIG['start_date_str'])
    end_unix   = _parse_unix(CONFIG['end_date_str'])
    print(f'[main] 验证时间段: {CONFIG["start_date_str"]} ~ {CONFIG["end_date_str"]}')

    # ==================== 加载模型 ====================
    (model, sw_manager, mdia_cfg, model_name, iri_peak_manager,
     fy_nb_index, cosmic_nb_index, allowed_profile_ids) = \
        _load_model_and_managers(CONFIG, device)
    candidate_contract = _checkpoint_evaluation_contract(
        CONFIG['checkpoint_path'], mdia_cfg)
    if baseline_checkpoint is None:
        baseline_paths = []
    elif isinstance(baseline_checkpoint, (str, os.PathLike)):
        baseline_paths = [str(baseline_checkpoint)]
    else:
        baseline_paths = list(baseline_checkpoint)
    if baseline_checkpoint_sha256 is None:
        baseline_hashes = []
    elif isinstance(baseline_checkpoint_sha256, str):
        baseline_hashes = [baseline_checkpoint_sha256]
    else:
        baseline_hashes = list(baseline_checkpoint_sha256)
    if len(baseline_hashes) > len(baseline_paths):
        raise ValueError('more baseline SHA256 values than baseline checkpoints')
    requested_labels = list(baseline_labels or [])
    if len(requested_labels) > len(baseline_paths):
        raise ValueError('more baseline labels than baseline checkpoints')
    baseline_context = {}
    baseline_contract = []
    for index, path in enumerate(baseline_paths):
        expected_sha = baseline_hashes[index] if index < len(baseline_hashes) else None
        if _is_historical_epoch_checkpoint(path) and not expected_sha:
            raise ValueError('historical ISR baseline requires --baseline-checkpoint-sha256')
        if (os.path.basename(path) == 'epoch_12_model.pth'
                and expected_sha != _HISTORICAL_EPOCH12_SHA256):
            raise ValueError('historical epoch12 ISR baseline SHA256 is not approved')
        (baseline_model, baseline_sw, baseline_cfg, _, baseline_iri_peak,
         baseline_fy_index, baseline_cosmic_index,
         baseline_allowed) = _load_model_and_managers(
            CONFIG, device, checkpoint=path,
            historical_expected_sha256=expected_sha)
        if (os.path.basename(path) == 'epoch_12_model.pth'
                and baseline_cfg.get('background_trust_gate_enabled', False)):
            raise ValueError('historical epoch12 ISR baseline must be gate-off')
        item_contract = _checkpoint_evaluation_contract(path, baseline_cfg)
        if (candidate_contract['observation_alt_range_km'] !=
                item_contract['observation_alt_range_km']
                or candidate_contract['peak_search_alt_range_km'] !=
                item_contract['peak_search_alt_range_km']):
            raise ValueError(
                'paired ISR checkpoints require equal observation and peak domains')
        for key in ('fy_path', 'cosmic_path', 'sw_path', 'iri_hmf2_path',
                    'iri_nmf2_path'):
            if mdia_cfg.get(key) != baseline_cfg.get(key):
                raise ValueError(
                    f'paired ISR checkpoints require identical input {key}')
        default_label = ('historical_epoch12' if _is_historical_epoch_checkpoint(path)
                         else f"checkpoint_v{baseline_cfg.get('checkpoint_format_version', index)}")
        label = requested_labels[index] if index < len(requested_labels) else default_label
        if label in baseline_context:
            raise ValueError(f'duplicate baseline label: {label}')
        baseline_context[label] = (
            baseline_model, baseline_sw, baseline_cfg, baseline_iri_peak,
            baseline_fy_index, baseline_cosmic_index, baseline_allowed)
        baseline_contract.append({'label': label, **item_contract})
    CONFIG['alt_range'] = tuple(map(float, mdia_cfg['alt_range']))
    CONFIG['observation_alt_range'] = tuple(map(float, (
        mdia_cfg.get('observation_alt_range') or mdia_cfg['alt_range'])))
    CONFIG['alt_min'], CONFIG['alt_max'] = CONFIG['alt_range']
    CONFIG['peak_search_alt_range'] = tuple(map(float, (
        mdia_cfg.get('peak_search_alt_range') or mdia_cfg['alt_range'])))
    print(f'[main] 模型类型: {model_name}')

    if preflight_only:
        print('[preflight] FSIA checkpoint、配置和模型依赖加载通过；未读取ISR数据')
        return

    # ==================== 加载 ISR 数据 ====================
    from isr_evaluation.isr_loader import load_jicamarca, load_poker_flat
    from isr_evaluation.coord_convert import convert_day_record_cgm

    station_reports = []
    isr_input_files = []
    poker_geometry_audit = []
    save_dir = CONFIG['save_dir']
    if os.path.isdir(save_dir) and os.listdir(save_dir):
        raise FileExistsError(
            f'ISR output directory already contains artifacts: {save_dir}')
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
        for record in jica_records:
            isr_input_files.extend(record.get('source_file_identity', []))

        if jica_records:
            rep = _process_station(
                'Jicamarca', jica_records,
                model, sw_manager, start_unix, CONFIG, device,
                model_name=model_name,
                iri_peak_manager=iri_peak_manager,
                fy_nb_index=fy_nb_index,
                cosmic_nb_index=cosmic_nb_index,
                allowed_profile_ids=allowed_profile_ids,
                baseline_context=baseline_context,
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
            isr_input_files.extend(rec.get('source_file_identity', []))
            poker_geometry_audit.append({
                'date': rec.get('date_str'),
                'geometry_qc': rec.get('geometry_qc', {}),
            })

        if pf_records:
            rep = _process_station(
                'PokerFlat', pf_records,
                model, sw_manager, start_unix, CONFIG, device,
                model_name=model_name,
                iri_peak_manager=iri_peak_manager,
                fy_nb_index=fy_nb_index,
                cosmic_nb_index=cosmic_nb_index,
                allowed_profile_ids=allowed_profile_ids,
                baseline_context=baseline_context,
            )
            if rep is not None:
                station_reports.append(rep)

    # ==================== 汇总报告 ====================
    if station_reports:
        _write_isr_final_artifacts(
            save_dir, station_reports, candidate_contract, baseline_contract,
            isr_input_files, poker_geometry_audit, CONFIG)
        gate_summary = {
            report['station']: bool(
                report['passed_m2w_m11_vs_raw_iri_gate'])
            for report in station_reports}
        print(f'[M2-W M11 vs Raw IRI] {gate_summary}')

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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--checkpoint', required=True,
        help='必需：完整Analysis阶段的FSIA v12/v13/v14 checkpoint路径')
    parser.add_argument('--save-dir', default=None)
    parser.add_argument('--baseline-checkpoint', action='append', default=None,
                        help='repeatable paired baseline evaluated on the same ISR loop')
    parser.add_argument('--baseline-checkpoint-sha256', action='append', default=None,
                        help='SHA256 aligned by order; required for a historical epoch baseline')
    parser.add_argument('--baseline-label', action='append', default=None,
                        help='optional comparison label aligned by baseline order')
    parser.add_argument(
        '--preflight-only', action='store_true',
        help='只加载并核验模型与配置，不读取ISR数据或生成评估输出')
    args = parser.parse_args()
    main(checkpoint=args.checkpoint, save_dir=args.save_dir,
         preflight_only=args.preflight_only,
         baseline_checkpoint=args.baseline_checkpoint,
         baseline_checkpoint_sha256=args.baseline_checkpoint_sha256,
         baseline_labels=args.baseline_label)
