"""
FSIA-INR 训练曲线可视化工具

五视图策略（5-Panel Layout）：
  1. 精度监控  — Train/Val Pure MSE (Log Scale)
  2. 优化目标  — Total Loss & NLL (Linear Scale)
  3. 背景约束  — Background Trust & Profile Align (Log Scale)
  4. 物理约束  — Residual Smooth & Horizontal Smooth (Log Scale)
  5. EWMA 参数 — τ_kp & τ_solar 时间常数演变 (Linear Scale)
"""

import matplotlib.pyplot as plt
import numpy as np
import os


def plot_training_curves(history, save_path='fsia_training_curves.png'):
    """
    绘制 FSIA-INR 训练曲线（5 视图）

    Args:
        history:   列表，每个元素为 epoch 的 metrics dict（来自 train_fsia.py history）
        save_path: 保存路径
    """
    if len(history) == 0:
        return

    epochs = [h['epoch'] for h in history]

    def _get(key):
        return [h.get(key, np.nan) for h in history]

    train_mse       = _get('train_mse')
    val_mse         = _get('val_mse')
    total_loss      = _get('total_loss')
    train_nll       = _get('train_nll')
    bkg             = _get('bkg')
    profile_align   = _get('profile_align')
    residual_smooth = _get('residual_smooth')
    horiz_smooth    = _get('horizontal_smooth')
    tau_kp          = _get('tau_kp')
    tau_solar       = _get('tau_solar')

    has_tau = any(not np.isnan(v) for v in tau_kp)
    n_panels = 5 if has_tau else 4

    fig, axes = plt.subplots(n_panels, 1, figsize=(10, 4 * n_panels), sharex=True)
    axes = list(axes)

    # Panel 1: Accuracy
    ax = axes[0]
    ax.plot(epochs, train_mse, label='Train MSE', color='blue', lw=2, marker='o', ms=3)
    ax.plot(epochs, val_mse, label='Val MSE', color='orange', ls='--', lw=2, marker='s', ms=3)
    ax.set_yscale('log')
    ax.set_ylabel('MSE (Log)', fontsize=11)
    ax.set_title('1. Accuracy (Pure MSE)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, which='both', ls='-', alpha=0.2)

    # Panel 2: Optimization
    ax = axes[1]
    ax.plot(epochs, total_loss, label='Total Loss', color='black', lw=2, marker='o', ms=3)
    ax.plot(epochs, train_nll, label='NLL Term', color='green', ls=':', lw=2, alpha=0.7)
    ax.axhline(0, color='red', ls='-', lw=1, alpha=0.3)
    ax.set_ylabel('Loss (Linear)', fontsize=11)
    ax.set_title('2. Optimization Objective (May Be Negative)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Panel 3: Background trust & profile-peak alignment
    ax = axes[2]
    bkg_nz  = [v if v > 0 else np.nan for v in bkg]
    pa_nz   = [v if v > 0 else np.nan for v in profile_align]
    ax.plot(epochs, bkg_nz, label='Background Trust', color='navy', lw=2, marker='o', ms=3)
    ax.plot(epochs, pa_nz,  label='Profile Align (∂Ne/∂h=0)', color='teal',
            ls='--', lw=2, marker='s', ms=3)
    ax.set_yscale('log')
    ax.set_ylabel('Loss (Log)', fontsize=11)
    ax.set_title('3. Assimilation Constraints', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, which='both', ls='-', alpha=0.2)

    # Panel 4: Residual smooth & horizontal smooth
    ax = axes[3]
    smooth_nz = [v if v > 0 else np.nan for v in residual_smooth]
    horiz_nz  = [v if v > 0 else np.nan for v in horiz_smooth]
    ax.plot(epochs, smooth_nz, label='Residual Smooth (∂²Ne/∂h²)', color='purple',
            lw=2, marker='o', ms=3)
    ax.plot(epochs, horiz_nz,  label='Horizontal Smooth', color='brown',
            ls='--', lw=2, marker='s', ms=3)
    ax.set_yscale('log')
    ax.set_ylabel('Loss (Log)', fontsize=11)
    ax.set_title('4. Physical Regularization', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, which='both', ls='-', alpha=0.2)

    # Panel 5: EWMA time constants (optional)
    if has_tau:
        ax = axes[4]
        ax.plot(epochs, tau_kp, label='τ_kp (hours)', color='red', lw=2, marker='o', ms=3)
        ax.plot(epochs, tau_solar, label='τ_solar (hours)', color='orange', ls='--',
                lw=2, marker='s', ms=3, alpha=0.8)
        ax.axhline(8.0, color='red', ls=':', lw=1, alpha=0.4, label='Init: 8h')
        ax.axhline(72.0, color='orange', ls=':', lw=1, alpha=0.4, label='Init: 72h')
        ax.set_ylabel('Time Constant (h)', fontsize=11)
        ax.set_title('5. Learnable EWMA Time Constants', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9, ncol=2)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('Epochs', fontsize=11)
    else:
        axes[3].set_xlabel('Epochs', fontsize=11)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Training curves saved: {save_path}')
