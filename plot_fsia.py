"""
FSIA-INR 独立绘图脚本

可在训练进行中随时运行，从最佳检查点加载模型并生成可视化图像。
无需等待训练结束。

运行方式：
    cd FSIA_INR
    python plot_fsia.py

可在下方 CONFIG 区块自定义检查点路径、输出目录、绘图时刻和高度层。
"""

import os
import sys
import torch

# ─────────────────────────────────────────────
# 路径设置
# ─────────────────────────────────────────────
_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
_INR_MODULES = os.path.join(_SCRIPT_DIR, 'inr_modules')
for _p in [_SCRIPT_DIR, _INR_MODULES]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ─────────────────────────────────────────────
# ==================== 配置 ====================
# ─────────────────────────────────────────────
CONFIG = {
    # ---- 检查点路径（None = 自动推断 checkpoints_fsia/run6/best_fsia_model.pth）----
    'checkpoint_path': r"D:\code11\IRI01\IRI03\INR1-1\FSIA_INR18\checkpoints_fsia\run40\best_fsia_model.pth",

    # ---- 输出目录（None = 检查点所在目录下的 plots/ 子目录）----
    'save_dir': None,

    # ---- 绘图目标时刻：天数列表 × 小时列表（笛卡尔积）----
    # 天数：相对于 start_date 的偏移天数（0 = 第 1 天）
    'vis_days':  [4, 14, 24],
    'vis_hours': [0, 6, 12, 18],

    # ---- 全球切片高度层 (km) ----
    'alt_levels': [250, 300, 350, 400, 450],

    # ---- EDP 廓线位置（Jicamarca，Sep 5）----
    'edp_lat':  -11.9,   # °N
    'edp_lon':  -76.0,   # °E
    'edp_day':    4,     # 天数（0 = Sep 1，4 = Sep 5）
    'edp_hours': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24],   # 多时刻 UT（每时刻一个面板）

    # ---- Jicamarca ISR 真值数据目录（None = 跳过 ISR 叠绘）----
    'jic_data_dir': r'D:\ISR\DATA\10jicamarca_is_radar(~12°S,低纬磁赤道)',

    # ---- 推理批大小（显存不足时减小）----
    'vis_batch': 2048,

    # ---- 绘图开关 ----
    'plot_global_slice':   False,   # 全球纬经度切片
    'plot_hmf2_nmf2_map':  False,   # F2 峰高 + 峰值密度分布图（按日分画布）
    'plot_edp_profile':    True,   # 垂直 EDP 廓线
}
# ─────────────────────────────────────────────


def _resolve_paths(config, mdia_cfg):
    """解析检查点路径和输出目录，填充 None 值。"""
    if config['checkpoint_path'] is None:
        config['checkpoint_path'] = os.path.join(
            _SCRIPT_DIR, 'checkpoints_fsia', 'run6', 'best_fsia_model.pth'
        )
    if config['save_dir'] is None:
        config['save_dir'] = os.path.join(
            os.path.dirname(config['checkpoint_path']), 'plots'
        )


def _build_iri_proxy(mdia_cfg, device):
    from inr_modules.data_managers.irinc_neural_proxy import IRINeuralProxy
    iri_proxy = IRINeuralProxy(layers=[4, 128, 128, 128, 128, 1]).to(device)
    proxy_state = torch.load(mdia_cfg['iri_proxy_path'], map_location=device)
    iri_proxy.load_state_dict(proxy_state)
    iri_proxy.eval()
    return iri_proxy


def _load_model(config, mdia_cfg, device):
    ckpt = config['checkpoint_path']
    if not os.path.exists(ckpt):
        raise FileNotFoundError(
            f'检查点文件不存在: {ckpt}\n'
            f'请先运行 main_fsia.py 完成训练（至少一个 epoch 保存 best_fsia_model.pth）。'
        )
    iri_proxy = _build_iri_proxy(mdia_cfg, device)
    from inr_modules.mdia.fsia_model import FSIA_INR_Model
    model = FSIA_INR_Model(iri_proxy=iri_proxy, config=mdia_cfg).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    print(f'[plot] FSIA-INR 模型已加载: {ckpt}')
    print(f'       τ_kp = {model.sw_encoder.tau_kp.item():.2f} h   '
          f'τ_solar = {model.sw_encoder.tau_solar.item():.2f} h')
    return model


def _load_sw_manager(mdia_cfg, device):
    from inr_modules.data_managers.space_weather_manager import SpaceWeatherManager
    return SpaceWeatherManager(
        txt_path=mdia_cfg['sw_path'],
        start_date_str=mdia_cfg['start_date_str'],
        total_hours=mdia_cfg['total_hours'],
        seq_len=mdia_cfg['seq_len'],
        device=device,
    )


def _build_iri_peak_manager(mdia_cfg, device):
    """加载 IRIPeakManager（FSIA v2.2）；文件不存在时返回 None（使用中性后备值）。"""
    try:
        from inr_modules.data_managers.iri_peak_manager import IRIPeakManager
        hmf2_path = mdia_cfg.get('iri_hmf2_path', '')
        nmf2_path = mdia_cfg.get('iri_nmf2_path', '')
        if hmf2_path and nmf2_path and os.path.exists(hmf2_path) and os.path.exists(nmf2_path):
            mgr = IRIPeakManager(
                hmf2_path=hmf2_path,
                nmf2_path=nmf2_path,
                total_hours=mdia_cfg.get('total_hours', 720.0),
                device=device,
            )
            print(f'[plot] IRIPeakManager 加载完成')
            return mgr
        print(f'[plot] IRIPeakManager: IRI 峰参数文件未找到，使用中性后备值')
    except Exception as e:
        print(f'[plot] IRIPeakManager 初始化失败（{e}），使用中性后备值')
    return None


def _load_jicamarca_record(data_dir, date_str='20240905'):
    """
    加载 Jicamarca ISR 数据并返回指定日期的 DayRecord。

    Args:
        data_dir:  Jicamarca HDF5 文件目录
        date_str:  目标日期字符串，如 '20240905'

    Returns:
        DayRecord dict，或 None（文件不存在 / 日期未找到）
    """
    import datetime as _dt
    if not data_dir or not os.path.isdir(data_dir):
        print(f'[plot] Jicamarca 数据目录不存在: {data_dir}，ISR 叠绘跳过')
        return None
    try:
        _isr_eval_dir = os.path.join(_SCRIPT_DIR, 'isr_evaluation')
        if _isr_eval_dir not in sys.path:
            sys.path.insert(0, _isr_eval_dir)
        from isr_loader import load_jicamarca
        start_unix = _dt.datetime(2024,  9, 1, tzinfo=_dt.timezone.utc).timestamp()
        end_unix   = _dt.datetime(2024, 10, 1, tzinfo=_dt.timezone.utc).timestamp()
        records    = load_jicamarca(data_dir, start_unix, end_unix)
        rec = next((r for r in records if r['date_str'] == date_str), None)
        if rec is None:
            print(f'[plot] 未找到 Jicamarca {date_str} 记录，ISR 叠绘跳过')
        return rec
    except Exception as e:
        print(f'[plot] Jicamarca 数据加载失败: {e}')
        return None


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[plot] 使用设备: {device}')

    from inr_modules.config_mdia import get_config_mdia
    mdia_cfg = get_config_mdia()

    _resolve_paths(CONFIG, mdia_cfg)
    os.makedirs(CONFIG['save_dir'], exist_ok=True)

    print(f'[plot] 检查点:   {CONFIG["checkpoint_path"]}')
    print(f'[plot] 输出目录: {CONFIG["save_dir"]}')

    model           = _load_model(CONFIG, mdia_cfg, device)
    sw_manager      = _load_sw_manager(mdia_cfg, device)
    iri_peak_manager = _build_iri_peak_manager(mdia_cfg, device)

    from inr_modules.mdia.visualization_mdia import (
        plot_global_slice,
        plot_altitude_profile,
        plot_hmf2_nmf2_map,
    )

    model_name  = 'FSIA-INR'
    save_dir    = CONFIG['save_dir']
    vis_days    = CONFIG['vis_days']
    vis_hours   = CONFIG['vis_hours']
    alt_levels  = CONFIG['alt_levels']

    n_slice_tasks = len(vis_days) * len(vis_hours) * CONFIG['plot_global_slice']
    n_hmf2_tasks  = len(vis_days) * CONFIG['plot_hmf2_nmf2_map']
    n_edp_tasks   = len(CONFIG['edp_hours']) * int(CONFIG['plot_edp_profile'])
    total_tasks   = n_slice_tasks + n_hmf2_tasks + n_edp_tasks
    done = 0

    for vis_day in vis_days:
        for vis_hour in vis_hours:
            if CONFIG['plot_global_slice']:
                print(f'\n[{done+1}/{total_tasks}] 全球切片 Day={vis_day} Hour={vis_hour:02d}')
                plot_global_slice(
                    model, sw_manager, device,
                    target_day=vis_day, target_hour=vis_hour,
                    save_dir=save_dir, config=mdia_cfg,
                    alt_levels=alt_levels, model_name=model_name,
                    iri_peak_manager=iri_peak_manager,
                )
                done += 1

        if CONFIG['plot_hmf2_nmf2_map']:
            print(f'\n[{done+1}/{total_tasks}] hmF2/NmF2 分布图 Day={vis_day} Hours={vis_hours}')
            plot_hmf2_nmf2_map(
                model, sw_manager, device,
                time_steps=[(vis_day, h) for h in vis_hours],
                save_dir=save_dir, config=mdia_cfg,
                label=f'day{vis_day:02d}', model_name=model_name,
                iri_peak_manager=iri_peak_manager,
            )
            done += 1

    if CONFIG['plot_edp_profile']:
        jic_record = _load_jicamarca_record(CONFIG.get('jic_data_dir'))
        for edp_h in CONFIG['edp_hours']:
            t_hour = CONFIG['edp_day'] * 24.0 + edp_h
            _lt = (edp_h + CONFIG['edp_lon'] / 15.0) % 24.0
            _lt_h = int(_lt); _lt_m = int(round((_lt - _lt_h) * 60))
            if _lt_m == 60:
                _lt_h = (_lt_h + 1) % 24; _lt_m = 0
            print(f'\n[{done+1}/{total_tasks}] EDP 廓线  '
                  f'Lat={CONFIG["edp_lat"]}°  Lon={CONFIG["edp_lon"]}°  '
                  f'Day={CONFIG["edp_day"]}  {edp_h:02d}:00 UT  '
                  f'(LT {_lt_h:02d}:{_lt_m:02d})')
            plot_altitude_profile(
                model, sw_manager, device,
                lat=CONFIG['edp_lat'], lon=CONFIG['edp_lon'],
                time_hour=t_hour,
                save_dir=save_dir, config=mdia_cfg, model_name=model_name,
                iri_peak_manager=iri_peak_manager,
                isr_record=jic_record,
            )
            done += 1

    print(f'\n[plot] 全部完成，共生成 {done} 张图。输出目录: {save_dir}')


if __name__ == '__main__':
    main()
