"""Training-curve visualization for the run66 two-stage objective."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os


def plot_training_curves(history, save_path='fsia_training_curves.png'):
    """Plot profile validation metrics and active run66 loss terms."""
    if not history:
        return

    epochs = [h['epoch'] for h in history]

    def _get(key):
        return [h.get(key, np.nan) for h in history]

    fig, axes = plt.subplots(4, 1, figsize=(10, 14), sharex=True)

    ax = axes[0]
    ax.plot(epochs, _get('fy_profile_rmse'), label='FY profile RMSE')
    ax.plot(epochs, _get('cosmic_profile_rmse'), label='COSMIC profile RMSE')
    ax.plot(epochs, _get('val_score'), label='Selection score', ls='--')
    ax.set_ylabel('log10Ne')
    ax.set_title('Profile-balanced validation')
    ax.legend()
    ax.grid(True, alpha=0.25)

    ax = axes[1]
    ax.plot(epochs, _get('fy_obs'), label='FY Huber')
    ax.plot(epochs, _get('cosmic_obs'), label='COSMIC Huber')
    ax.plot(epochs, _get('total_loss'), label='Total', color='black')
    ax.set_ylabel('Loss')
    ax.set_title('Observation and total losses')
    ax.legend()
    ax.grid(True, alpha=0.25)

    ax = axes[2]
    ax.plot(epochs, _get('weighted_iri'), label='Weighted IRI anchor')
    ax.plot(epochs, _get('weighted_increment'), label='Weighted increment')
    ax.plot(epochs, _get('weighted_vertical'), label='Weighted vertical')
    ax.plot(epochs, _get('weighted_time'), label='Weighted time')
    ax.set_ylabel('Weighted loss')
    ax.set_title('Active auxiliary terms')
    ax.legend()
    ax.grid(True, alpha=0.25)

    ax = axes[3]
    ax.plot(epochs, _get('gradient_ratio'), label='Auxiliary / observation')
    ax.axhline(0.30, color='red', ls='--', label='30% limit')
    ax.set_ylabel('Gradient norm ratio')
    ax.set_title('Decoder gradient audit')
    ax.set_xlabel('Epoch')
    ax.legend()
    ax.grid(True, alpha=0.25)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Training curves saved: {save_path}')
