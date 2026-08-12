"""
FSIA-INR 评估模块

功能：
  1. evaluate_and_save_report  — 对比 Raw IRI / FNDA Background / FSIA-INR M11
  2. evaluate_parity           — 三面板 Parity 图（散点 + 密度）
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import pearsonr
from inr_modules.density_units import (
    DENSITY_UNIT_LABEL,
    log10_density_to_display,
)
from .sliding_dataset import attach_observation_background


# ======================== 内部辅助 ========================

def _collect_predictions(model, dataloader, batch_processor,
                         iri_peak_manager=None):
    """
    遍历 DataLoader，收集完整预测结果。

    Returns:
        pred   [N] — M11 Analysis
        bkg    [N] — FNDA Background (M00)
        iri    [N] — Raw IRI
        target [N] — 观测值
    """
    model.eval()
    preds, bkgs, iris, targets = [], [], [], []

    with torch.no_grad():
        for batch_data in dataloader:
            (coords, target_ne, sw_seq,
             observations_fy, observations_cosmic,
             _) = batch_processor.process_batch(batch_data)
            iri_peak = (iri_peak_manager.get_iri_peak(coords)
                        if iri_peak_manager is not None else None)
            observations_fy = attach_observation_background(
                observations_fy, model, batch_processor.sw_manager,
                iri_peak_manager)
            observations_cosmic = attach_observation_background(
                observations_cosmic, model, batch_processor.sw_manager,
                iri_peak_manager)
            Ne_fused, _, _, _, extras = model(
                coords, sw_seq,
                iri_peak=iri_peak,
                observations_fy=observations_fy,
                observations_cosmic=observations_cosmic)

            preds.append(Ne_fused.reshape(-1).cpu().numpy())
            bkgs.append(extras['ne_bkg'].reshape(-1).cpu().numpy())
            iris.append(extras['ne_iri'].reshape(-1).cpu().numpy())
            targets.append(target_ne.reshape(-1).cpu().numpy())

    return (np.concatenate(preds), np.concatenate(bkgs), np.concatenate(iris),
            np.concatenate(targets))


def _calc_metrics(y_true, y_pred):
    """计算 RMSE / R² / Pearson R / Bias"""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    r, _ = pearsonr(y_true, y_pred)
    bias = float(np.mean(y_pred - y_true))
    return rmse, r2, r, bias


# ======================== 评估报告 ========================

def evaluate_and_save_report(model, train_loader, val_loader,
                              batch_processor, save_dir,
                              iri_peak_manager=None):
    """
    计算训练集 / 验证集评估指标，保存文本报告。

    报告包含 Raw IRI、FNDA Background 和 FSIA-INR M11 三行。
    对应四列指标：RMSE / R² / Pearson R / Bias

    Args:
        model:           训练好的 FSIA_INR_Model
        train_loader:    训练集 DataLoader
        val_loader:      验证集 DataLoader
        batch_processor: SlidingWindowBatchProcessor
        save_dir:        报告保存目录

    Returns:
        dict: {'train': {...}, 'val': {...}}  各指标元组
    """
    os.makedirs(save_dir, exist_ok=True)

    print('[评估] 收集训练集预测...')
    t_pred, t_bkg, t_iri, t_true = _collect_predictions(
        model, train_loader, batch_processor, iri_peak_manager)

    print('[评估] 收集验证集预测...')
    v_pred, v_bkg, v_iri, v_true = _collect_predictions(
        model, val_loader, batch_processor, iri_peak_manager)

    t_inr = _calc_metrics(t_true, t_pred)
    t_background = _calc_metrics(t_true, t_bkg)
    t_iri_metrics = _calc_metrics(t_true, t_iri)
    v_inr = _calc_metrics(v_true, v_pred)
    v_background = _calc_metrics(v_true, v_bkg)
    v_iri_metrics = _calc_metrics(v_true, v_iri)

    tau_kp    = model.sw_encoder.tau_kp.item()
    tau_solar = model.sw_encoder.tau_solar.item()

    W = 67
    def row(label, m):
        return (f'  {label:<22}'
                f'  {m[0]:>9.5f}'
                f'  {m[1]:>9.5f}'
                f'  {m[2]:>9.5f}'
                f'  {m[3]:>+9.5f}')

    lines = [
        '=' * W,
        '   FSIA-INR 电离层重构评估报告',
        '=' * W,
        f'模型     : FSIA-INR (FY/COSMIC local-profile assimilation)',
        f'训练样本 : {len(t_true):>10,}',
        f'验证样本 : {len(v_true):>10,}',
        f'EWMA τ_kp    = {tau_kp:.2f} h',
        f'EWMA τ_solar = {tau_solar:.2f} h',
        '-' * W,
        f'  {"分量":<22}  {"RMSE":>9}  {"R²":>9}  {"R":>9}  {"Bias":>9}',
        '-' * W,
        '[训练集]',
        row('Raw IRI',          t_iri_metrics),
        row('FNDA Background',  t_background),
        row('FSIA-INR M11',     t_inr),
        '-' * W,
        '[验证集]',
        row('Raw IRI',          v_iri_metrics),
        row('FNDA Background',  v_background),
        row('FSIA-INR M11',     v_inr),
        '=' * W,
    ]

    report = '\n'.join(lines)
    print('\n' + report)

    report_path = os.path.join(save_dir, 'evaluation_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f'\n报告已保存: {report_path}')

    return {
        'train': {'analysis': t_inr, 'background': t_background,
                  'iri': t_iri_metrics},
        'val':   {'analysis': v_inr, 'background': v_background,
                  'iri': v_iri_metrics},
    }


# ======================== Parity 图 ========================

def evaluate_parity(model, val_loader, batch_processor, save_dir,
                    iri_peak_manager=None):
    """
    绘制三面板 Parity 图（验证集）。

    布局：
        上行 — 散点图  (alpha=0.05, rasterized)
        下行 — 密度图  (hist2d + LogNorm)
        三列 — Raw IRI | FNDA Background | FSIA-INR M11

    保存文件：
        parity_scatter_fsia.png
        parity_density_fsia.png

    Args:
        model:           训练好的 FSIA_INR_Model
        val_loader:      验证集 DataLoader
        batch_processor: SlidingWindowBatchProcessor
        save_dir:        保存目录
    """
    os.makedirs(save_dir, exist_ok=True)

    print('[评估] 生成 Parity 图（验证集）...')
    pred, bkg, iri, true = _collect_predictions(
        model, val_loader, batch_processor, iri_peak_manager)
    print(f'  验证样本数: {len(true):,}')

    m_iri = _calc_metrics(true, iri)
    m_bkg = _calc_metrics(true, bkg)
    m_inr = _calc_metrics(true, pred)

    true_plot = log10_density_to_display(true)
    ys_plot = [log10_density_to_display(values)
               for values in (iri, bkg, pred)]
    ax_min = min(true_plot.min(), *(values.min() for values in ys_plot))
    ax_max = max(true_plot.max(), *(values.max() for values in ys_plot))
    ax_min *= 0.95
    ax_max *= 1.05

    def _make_title(label, m):
        return (f'{label} vs 观测\n'
                f'log10 metrics: RMSE={m[0]:.4f} dex  '
                f'R²={m[1]:.4f}  R={m[2]:.4f}')

    panel_labels = ['Raw IRI', 'FNDA Background', 'FSIA-INR M11']
    colors = ['steelblue', 'gray', 'darkorange']
    titles = [
        _make_title(label, metrics)
        for label, metrics in zip(panel_labels, [m_iri, m_bkg, m_inr])
    ]

    # ---- 散点图 ----
    fig1, axes = plt.subplots(1, 3, figsize=(19, 6), dpi=150)
    for ax, y, title, color in zip(axes, ys_plot, titles, colors):
        ax.scatter(true_plot, y, alpha=0.05, s=0.5, c=color, rasterized=True)
        ax.plot([ax_min, ax_max], [ax_min, ax_max], 'r--', lw=1.5, alpha=0.8, label='1:1')
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlabel(f'观测 Ne ({DENSITY_UNIT_LABEL})', fontsize=10)
        ax.set_ylabel(f'预测 Ne ({DENSITY_UNIT_LABEL})', fontsize=10)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(ax_min, ax_max)
        ax.set_ylim(ax_min, ax_max)
        ax.set_aspect('equal', 'box')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
    plt.suptitle('FSIA-INR Parity — 散点图（验证集）', fontsize=13, fontweight='bold')
    plt.tight_layout()
    scatter_path = os.path.join(save_dir, 'parity_scatter_fsia.png')
    plt.savefig(scatter_path, dpi=150, bbox_inches='tight')
    plt.close(fig1)
    print(f'  散点图已保存: {scatter_path}')

    # ---- 密度图 ----
    density_edges = np.geomspace(ax_min, ax_max, 301)
    fig2, axes = plt.subplots(1, 3, figsize=(20, 6), dpi=150)
    for ax, y, title in zip(axes, ys_plot, titles):
        h = ax.hist2d(true_plot, y, bins=[density_edges, density_edges],
                      cmap='turbo', norm=LogNorm(), cmin=1)
        ax.plot([ax_min, ax_max], [ax_min, ax_max], 'w--', lw=1.5, alpha=0.8, label='1:1')
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlabel(f'观测 Ne ({DENSITY_UNIT_LABEL})', fontsize=10)
        ax.set_ylabel(f'预测 Ne ({DENSITY_UNIT_LABEL})', fontsize=10)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(ax_min, ax_max)
        ax.set_ylim(ax_min, ax_max)
        ax.set_aspect('equal', 'box')
        ax.grid(True, linestyle=':', alpha=0.4)
        ax.legend(fontsize=9)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='3%', pad=0.05)
        plt.colorbar(h[3], cax=cax, label='计数 (对数)')
    plt.suptitle('FSIA-INR Parity — 密度图（验证集）', fontsize=13, fontweight='bold')
    plt.tight_layout()
    density_path = os.path.join(save_dir, 'parity_density_fsia.png')
    plt.savefig(density_path, dpi=150, bbox_inches='tight')
    plt.close(fig2)
    print(f'  密度图已保存: {density_path}')
