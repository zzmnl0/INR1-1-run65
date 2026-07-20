"""
FSIA-INR 评估模块

功能：
  1. evaluate_and_save_report  — 训练集 / 验证集指标报告（RMSE / R² / Pearson R / Bias）
                                  对比两个输出：IRI Background | FSIA-INR fused
  2. evaluate_parity           — 双面板 Parity 图（散点 + 密度），IRI | FSIA-INR
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import pearsonr


# ======================== 内部辅助 ========================

def _collect_predictions(model, dataloader, batch_processor, device):
    """
    遍历 DataLoader，收集完整预测结果。

    Returns:
        pred   [N] — Ne_fused 预测值
        bkg    [N] — IRI 背景值
        target [N] — 观测值
    """
    model.eval()
    preds, bkgs, targets = [], [], []

    with torch.no_grad():
        for batch_data in dataloader:
            coords, target_ne, sw_seq, neighbors_feats, has_obs = batch_processor.process_batch(batch_data)
            Ne_fused, _, _, _, extras = model(coords, sw_seq,
                                              neighbors_feats=neighbors_feats,
                                              has_obs=has_obs)

            preds.append(Ne_fused.reshape(-1).cpu().numpy())
            bkgs.append(extras['ne_bkg'].reshape(-1).cpu().numpy())
            targets.append(target_ne.reshape(-1).cpu().numpy())

    return (np.concatenate(preds), np.concatenate(bkgs),
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
                              batch_processor, device, save_dir):
    """
    计算训练集 / 验证集评估指标，保存文本报告。

    报告包含两行对比：IRI Background | FSIA-INR fused
    对应四列指标：RMSE / R² / Pearson R / Bias

    Args:
        model:           训练好的 FSIA_INR_Model
        train_loader:    训练集 DataLoader
        val_loader:      验证集 DataLoader
        batch_processor: SlidingWindowBatchProcessor
        device:          计算设备
        save_dir:        报告保存目录

    Returns:
        dict: {'train': {...}, 'val': {...}}  各指标元组
    """
    os.makedirs(save_dir, exist_ok=True)

    print('[评估] 收集训练集预测...')
    t_pred, t_bkg, t_true = _collect_predictions(
        model, train_loader, batch_processor, device)

    print('[评估] 收集验证集预测...')
    v_pred, v_bkg, v_true = _collect_predictions(
        model, val_loader, batch_processor, device)

    t_inr = _calc_metrics(t_true, t_pred)
    t_iri = _calc_metrics(t_true, t_bkg)
    v_inr = _calc_metrics(v_true, v_pred)
    v_iri = _calc_metrics(v_true, v_bkg)

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
        f'模型     : FSIA-INR (CrossSourceAttention + PeakHead + 可学习 EWMA)',
        f'训练样本 : {len(t_true):>10,}',
        f'验证样本 : {len(v_true):>10,}',
        f'EWMA τ_kp    = {tau_kp:.2f} h',
        f'EWMA τ_solar = {tau_solar:.2f} h',
        '-' * W,
        f'  {"分量":<22}  {"RMSE":>9}  {"R²":>9}  {"R":>9}  {"Bias":>9}',
        '-' * W,
        '[训练集]',
        row('IRI Background',   t_iri),
        row('FSIA-INR fused',   t_inr),
        '-' * W,
        '[验证集]',
        row('IRI Background',   v_iri),
        row('FSIA-INR fused',   v_inr),
        '=' * W,
    ]

    report = '\n'.join(lines)
    print('\n' + report)

    report_path = os.path.join(save_dir, 'evaluation_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f'\n报告已保存: {report_path}')

    return {
        'train': {'inr': t_inr, 'iri': t_iri},
        'val':   {'inr': v_inr, 'iri': v_iri},
    }


# ======================== Parity 图 ========================

def evaluate_parity(model, val_loader, batch_processor, device, save_dir):
    """
    绘制双面板 Parity 图（验证集）。

    布局：
        上行 — 散点图  (alpha=0.05, rasterized)
        下行 — 密度图  (hist2d + LogNorm)
        两列 — IRI Background | FSIA-INR fused

    保存文件：
        parity_scatter_fsia.png
        parity_density_fsia.png

    Args:
        model:           训练好的 FSIA_INR_Model
        val_loader:      验证集 DataLoader
        batch_processor: SlidingWindowBatchProcessor
        device:          计算设备
        save_dir:        保存目录
    """
    os.makedirs(save_dir, exist_ok=True)

    print('[评估] 生成 Parity 图（验证集）...')
    pred, bkg, true = _collect_predictions(model, val_loader, batch_processor, device)
    print(f'  验证样本数: {len(true):,}')

    m_iri = _calc_metrics(true, bkg)
    m_inr = _calc_metrics(true, pred)

    ax_min = np.floor(min(true.min(), bkg.min(), pred.min()) * 10) / 10
    ax_max = np.ceil( max(true.max(), bkg.max(), pred.max()) * 10) / 10

    def _make_title(label, m):
        return (f'{label} vs 观测\n'
                f'RMSE={m[0]:.4f}  R²={m[1]:.4f}  R={m[2]:.4f}')

    panel_labels = ['IRI Background', 'FSIA-INR fused']
    ys     = [bkg,         pred]
    colors = ['steelblue', 'darkorange']
    titles = [_make_title(l, m) for l, m in zip(panel_labels, [m_iri, m_inr])]

    # ---- 散点图 ----
    fig1, axes = plt.subplots(1, 2, figsize=(13, 6), dpi=150)
    for ax, y, title, color in zip(axes, ys, titles, colors):
        ax.scatter(true, y, alpha=0.05, s=0.5, c=color, rasterized=True)
        ax.plot([ax_min, ax_max], [ax_min, ax_max], 'r--', lw=1.5, alpha=0.8, label='1:1')
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlabel('观测 Ne (log10)', fontsize=10)
        ax.set_ylabel('预测 Ne (log10)', fontsize=10)
        ax.set_xlim(ax_min, ax_max)
        ax.set_ylim(ax_min, ax_max)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
    plt.suptitle('FSIA-INR Parity — 散点图（验证集）', fontsize=13, fontweight='bold')
    plt.tight_layout()
    scatter_path = os.path.join(save_dir, 'parity_scatter_fsia.png')
    plt.savefig(scatter_path, dpi=150, bbox_inches='tight')
    plt.close(fig1)
    print(f'  散点图已保存: {scatter_path}')

    # ---- 密度图 ----
    plot_range = [[ax_min, ax_max], [ax_min, ax_max]]
    fig2, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=150)
    for ax, y, title in zip(axes, ys, titles):
        h = ax.hist2d(true, y, bins=300, range=plot_range,
                      cmap='turbo', norm=LogNorm(), cmin=1)
        ax.plot([ax_min, ax_max], [ax_min, ax_max], 'w--', lw=1.5, alpha=0.8, label='1:1')
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlabel('观测 Ne (log10)', fontsize=10)
        ax.set_ylabel('预测 Ne (log10)', fontsize=10)
        ax.set_aspect('equal')
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
