"""FSIA-INR model with frozen IRI background and local FY/COSMIC assimilation.

The active path is: IRI hidden state + two profile encoders -> NeuralETKFLayer ->
channel-wise fusion -> bounded density increment. IRI hmF2/NmF2 are passed through
as structural references; no trainable peak head is present.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .ewma_sw_encoder import DualScaleSWEncoder
    from .mdia_model import _compute_dip_features
    from .m2u_anchor_etkf import (
        anchor_mixing_weights,
        calibrate_temperature,
        flat_top_cover,
        state_distance_squared,
    )
except ImportError:
    import os as _os, sys as _sys
    _here = _os.path.dirname(_os.path.abspath(__file__))
    if _here not in _sys.path:
        _sys.path.insert(0, _here)
    from ewma_sw_encoder import DualScaleSWEncoder
    from mdia_model import _compute_dip_features
    from m2u_anchor_etkf import (
        anchor_mixing_weights,
        calibrate_temperature, flat_top_cover, state_distance_squared)


def _query_ensemble_anomalies(extras):
    query_anomalies = extras.get('query_anomalies')
    if query_anomalies is None:
        query_anomalies = torch.einsum(
            'bd,bnd->bn',
            extras['query_basis'],
            extras['latent_anomalies'],
        )
    return query_anomalies


def _source_ensemble_terms(extras, source):
    covariance = extras.get(f'ensemble_covariance_{source}')
    source_rhs = extras.get(f'ensemble_rhs_{source}')
    if covariance is None or source_rhs is None:
        obs_anomalies = extras[f'obs_anomalies_{source}']
        precision = extras[f'precision_{source}']
        innovation = extras[f'innov_{source}']
        covariance = torch.einsum(
            'bmn,bm,bmk->bnk',
            obs_anomalies,
            precision,
            obs_anomalies,
        )
        source_rhs = torch.einsum(
            'bmn,bm,bm->bn',
            obs_anomalies,
            precision,
            innovation,
        )
    return covariance, source_rhs


def solve_density_mode(extras, sources, observation_variance=None):
    """Solve one physical-density ETKF mode from a joint forward pass."""
    sources = tuple(sources)
    if len(set(sources)) != len(sources) or any(
            source not in ('FY', 'COSMIC') for source in sources):
        raise ValueError(f'invalid ETKF sources: {sources}')
    query_anomalies = _query_ensemble_anomalies(extras)
    batch, members = query_anomalies.shape
    system = (
        max(members - 1, 1)
        * torch.eye(
            members,
            device=query_anomalies.device,
            dtype=query_anomalies.dtype,
        ).expand(batch, -1, -1)
    )
    rhs = query_anomalies.new_zeros(batch, members)
    for source in sources:
        covariance, source_rhs = _source_ensemble_terms(extras, source)
        system = system + covariance
        rhs = rhs + source_rhs
    chol = torch.linalg.cholesky(system)
    weights = torch.cholesky_solve(
        rhs.unsqueeze(-1), chol).squeeze(-1)
    increment = torch.einsum('bn,bn->b', query_anomalies, weights)
    if observation_variance is None:
        return increment
    solved_query = torch.cholesky_solve(
        query_anomalies.unsqueeze(-1), chol).squeeze(-1)
    variance = (
        torch.einsum('bn,bn->b', query_anomalies, solved_query)
        + observation_variance)
    return increment, variance


def solve_density_modes(extras):
    """Solve M10/M01 together and reuse the forward-pass M11 solution."""
    shared = extras.get('mode_increments')
    if shared is not None:
        return {key: value for key, value in shared.items()}
    query_anomalies = _query_ensemble_anomalies(extras)
    batch, members = query_anomalies.shape
    base = (
        max(members - 1, 1)
        * torch.eye(
            members,
            device=query_anomalies.device,
            dtype=query_anomalies.dtype,
        ).expand(batch, -1, -1)
    )
    fy_covariance, fy_rhs = _source_ensemble_terms(extras, 'FY')
    cosmic_covariance, cosmic_rhs = _source_ensemble_terms(extras, 'COSMIC')
    systems = torch.stack(
        (base + fy_covariance, base + cosmic_covariance), dim=1)
    rhs = torch.stack((fy_rhs, cosmic_rhs), dim=1)
    weights = torch.cholesky_solve(
        rhs.unsqueeze(-1), torch.linalg.cholesky(systems)).squeeze(-1)
    increments = torch.einsum('bn,bmn->bm', query_anomalies, weights)
    result = {
        'M10': increments[:, 0],
        'M01': increments[:, 1],
    }
    joint = extras.get('ne_residual')
    result['M11'] = (
        joint.squeeze(-1) if joint is not None
        else solve_density_mode(extras, ('FY', 'COSMIC')))
    return result


# ======================== Solar Zenith Angle 计算（run25：替代 LT）========================

# Sep 1 day-of-year（非闰年/2024 闰年差 1，可忽略）— 与训练数据集起始日对齐
_DOY_SEP1_FLOAT = 245.0

def _compute_solar_features(lat_deg, lon_deg, rel_hour):
    """
    计算太阳天顶角余弦 + day-of-year 编码（run25：物理上替代 LT 当地时编码）

    使用 Spencer 7阶傅里叶公式计算太阳赤纬，精度 < 0.01°。
    所有运算可微（floor 仅用于离散化 day index，不影响梯度路径）。

    Args:
        lat_deg:  [B] 地理纬度（度）
        lon_deg:  [B] 地理经度（度，∈ [-180, 180]）
        rel_hour: [B] 自数据集起始的相对小时（用作 day-of-year 累加）
    Returns:
        cos_SZA:  [B] 太阳天顶角余弦 ∈ [-1, 1]
                       cos_SZA > 0 → 日照面（白天）
                       cos_SZA < 0 → 阴影面（夜间）
        sin_doy:  [B] sin(2π·doy/365)
        cos_doy:  [B] cos(2π·doy/365)
    """
    # ---- day-of-year + 年角 ----
    doy     = (rel_hour / 24.0).floor() + _DOY_SEP1_FLOAT
    doy_mod = doy % 365.0
    gamma   = 2.0 * math.pi * doy_mod / 365.0                                # [B] 年角

    # ---- 太阳赤纬（Spencer 1971 7阶傅里叶级数）----
    delta = (0.006918
             - 0.399912 * torch.cos(gamma)
             + 0.070257 * torch.sin(gamma)
             - 0.006758 * torch.cos(2.0 * gamma)
             + 0.000907 * torch.sin(2.0 * gamma)
             - 0.002697 * torch.cos(3.0 * gamma)
             + 0.001480 * torch.sin(3.0 * gamma))                            # [B] rad

    # ---- 当地真太阳时 + cos(SZA) ----
    LST        = (rel_hour + lon_deg / 15.0) % 24.0                          # [B]
    hour_angle = (LST - 12.0) * (math.pi / 12.0)
    lat_rad    = lat_deg * (math.pi / 180.0)
    cos_SZA = (torch.sin(delta) * torch.sin(lat_rad)
               + torch.cos(delta) * torch.cos(lat_rad) * torch.cos(hour_angle))

    # ---- 季节相位（doy_phase 与 gamma 完全同义，直接复用）----
    sin_doy = torch.sin(gamma)
    cos_doy = torch.cos(gamma)

    return cos_SZA, sin_doy, cos_doy


def _compute_local_time_features(lon_deg, rel_hour):
    """Continuous local-solar-time phase shared by every endpoint type."""
    phase = (rel_hour + lon_deg / 15.0) * (math.pi / 12.0)
    return torch.sin(phase), torch.cos(phase)


# ======================== NeuralETKFLayer 辅助函数 ========================

def _build_kalman_b_input(h_sw, lat_n, cos_SZA, sin_doy, cos_doy, sin_I):
    """B_net / PerturbationNet 输入构造（h_sw + 太阳/地磁 regime 特征）"""
    return torch.cat([
        h_sw,
        lat_n.unsqueeze(-1),
        cos_SZA.unsqueeze(-1),
        sin_doy.unsqueeze(-1),
        cos_doy.unsqueeze(-1),
        sin_I.unsqueeze(-1),
    ], dim=-1)                                                                  # [B, b_net_in=69]


# ======================== NeuralETKFLayer (run28) ========================

class NeuralETKFLayer(nn.Module):
    """Low-dimensional ETKF with a physical log10Ne observation operator."""

    def __init__(self, d_model: int = 64, b_net_in: int = 69,
                 n_members: int = 8, pert_hidden: int = 64,
                 r_fy: float = 0.04, r_cosmic: float = 0.04,
                 anomaly_parameterization: str = 'legacy_independent',
                 scale_init: float = 1.1,
                 scale_condition_max: float = 3.0,
                 coordinate_local_symmetric: bool = False,
                 coefficient_space: bool = False):
        super().__init__()
        if d_model < 1 or n_members < 2 or pert_hidden < 1:
            raise ValueError(
                'd_model and pert_hidden must be positive; n_members must be >= 2')
        if anomaly_parameterization not in (
                'legacy_independent', 'orthogonal_factor'):
            raise ValueError(
                'anomaly_parameterization must be legacy_independent or '
                'orthogonal_factor')
        if (anomaly_parameterization == 'orthogonal_factor'
                and n_members - 1 > d_model):
            raise ValueError('enkf_n_members - 1 must not exceed d_model')
        if (coordinate_local_symmetric
                and anomaly_parameterization != 'orthogonal_factor'):
            raise ValueError(
                'coordinate-local symmetric basis requires orthogonal_factor')
        if coefficient_space and d_model != n_members - 1:
            raise ValueError(
                'coefficient-space anomalies require d_model == n_members - 1')
        if scale_init <= 0 or scale_condition_max <= 1:
            raise ValueError(
                'scale_init must be positive and scale_condition_max must exceed 1')
        self.d_model     = d_model
        self.n_members   = n_members
        self.b_net_in    = b_net_in
        self.pert_hidden = pert_hidden
        self.anomaly_parameterization = anomaly_parameterization
        self.scale_init = float(scale_init)
        self.scale_condition_max = float(scale_condition_max)
        self.coordinate_local_symmetric = bool(coordinate_local_symmetric)
        self.coefficient_space = bool(coefficient_space)

        if anomaly_parameterization == 'legacy_independent':
            # Preserve v7 state-dict keys for strict read-only compatibility.
            self.P_w1 = nn.Parameter(
                torch.empty(n_members, b_net_in, pert_hidden))
            self.P_b1 = nn.Parameter(torch.empty(n_members, pert_hidden))
            self.P_w2 = nn.Parameter(
                torch.empty(n_members, pert_hidden, d_model))
            self.P_b2 = nn.Parameter(torch.empty(n_members, d_model))
            self.log_inflation = nn.Parameter(torch.zeros(()))
        else:
            rank = n_members - 1
            self.register_buffer(
                'ensemble_coefficients', self._helmert_coefficients(n_members))
            self.register_buffer(
                'state_basis',
                (torch.eye(rank, dtype=torch.float64)
                 if coefficient_space else self._dct_basis(d_model, rank)))
            self.covariance_scale_net = nn.Sequential(
                nn.Linear(b_net_in, pert_hidden),
                nn.SiLU(),
                nn.Linear(pert_hidden, rank),
            )
            nn.init.zeros_(self.covariance_scale_net[-1].weight)
            nn.init.zeros_(self.covariance_scale_net[-1].bias)

        self.register_buffer('r_fy', torch.tensor(float(r_fy)))
        self.register_buffer('r_cosmic', torch.tensor(float(r_cosmic)))

        self.last_member_weights  = None
        self.last_inflation_scale = None

        self._reset_kalman_params()

    @staticmethod
    def _helmert_coefficients(n_members):
        rank = n_members - 1
        coefficients = torch.zeros(rank, n_members, dtype=torch.float64)
        for row in range(rank):
            denominator = math.sqrt((row + 1) * (row + 2))
            coefficients[row, :row + 1] = 1.0 / denominator
            coefficients[row, row + 1] = -(row + 1) / denominator
        return coefficients * math.sqrt(rank)

    @staticmethod
    def _dct_basis(d_model, rank):
        positions = torch.arange(d_model, dtype=torch.float64).unsqueeze(1)
        modes = torch.arange(rank, dtype=torch.float64).unsqueeze(0)
        basis = torch.cos(math.pi * (positions + 0.5) * modes / d_model)
        basis[:, 0] *= 1.0 / math.sqrt(d_model)
        if rank > 1:
            basis[:, 1:] *= math.sqrt(2.0 / d_model)
        return basis

    def _reset_kalman_params(self):
        """PerturbationNet 用 nn.Linear 默认风格 uniform 初始化（per-member fan_in）。
        """
        if self.anomaly_parameterization != 'legacy_independent':
            return
        for w, b, fan_in in [
            (self.P_w1, self.P_b1, self.b_net_in),
            (self.P_w2, self.P_b2, self.pert_hidden),
        ]:
            bound = 1.0 / math.sqrt(fan_in)
            nn.init.uniform_(w, -bound, bound)
            nn.init.uniform_(b, -bound, bound)

    # ---------- 向量化扰动生成 ----------
    def _eval_perturbations(self, b_input):
        """N 个并行扰动 → ensemble 扰动 [B, N, d]"""
        if self.anomaly_parameterization == 'orthogonal_factor':
            if self.coordinate_local_symmetric:
                scales = b_input.new_full(
                    (b_input.shape[0], self.n_members - 1), self.scale_init)
            else:
                logits = self.covariance_scale_net(b_input)
                half_log_condition = 0.5 * math.log(self.scale_condition_max)
                scales = self.scale_init * torch.exp(
                    half_log_condition * torch.tanh(logits))
            state_basis = self.state_basis.to(
                device=b_input.device, dtype=b_input.dtype)
            coefficients = self.ensemble_coefficients.to(
                device=b_input.device, dtype=b_input.dtype)
            factors = state_basis.unsqueeze(0) * scales.unsqueeze(1)
            return (
                torch.einsum(
                    'bdr,rn->bnd', factors, coefficients),
                scales,
            )
        h = torch.einsum('bi,nio->bno', b_input, self.P_w1) + self.P_b1.unsqueeze(0)
        h = F.silu(h)
        delta = torch.einsum('bni,nio->bno', h, self.P_w2) + self.P_b2.unsqueeze(0)
        return delta, None                                                       # [B, N, d]

    def observation_factor_coordinates(self, phi_obs, factor_scales=None):
        """Project density-basis rows into the independent ETKF factors.

        For the orthogonal-factor parameterization this returns ``F`` such that
        ``phi_obs @ X == F @ ensemble_coefficients``.  Keeping this relation in
        one method prevents the training-only Gram diagnostic from silently
        using a different state geometry than the production ETKF.
        """
        if self.anomaly_parameterization != 'orthogonal_factor':
            raise ValueError(
                'observation factor coordinates require orthogonal_factor')
        if phi_obs.ndim != 3 or phi_obs.shape[-1] != self.d_model:
            raise ValueError(
                f'phi_obs must have shape [B, M, {self.d_model}]')
        rank = self.n_members - 1
        basis = self.state_basis.to(device=phi_obs.device, dtype=phi_obs.dtype)
        factors = torch.einsum('bmd,dr->bmr', phi_obs, basis)
        if factor_scales is None:
            scales = phi_obs.new_full((phi_obs.shape[0], rank), self.scale_init)
        else:
            if factor_scales.shape != (phi_obs.shape[0], rank):
                raise ValueError('factor_scales shape does not match phi_obs')
            scales = factor_scales.to(device=phi_obs.device, dtype=phi_obs.dtype)
        return factors * scales.unsqueeze(1)

    def set_observation_variances(
            self, r_fy, r_cosmic, r_fy_table=None, r_cosmic_table=None):
        del r_fy_table, r_cosmic_table
        values = torch.as_tensor([r_fy, r_cosmic], dtype=self.r_fy.dtype)
        if not torch.isfinite(values).all() or torch.any(values <= 0):
            raise ValueError('observation variances must be finite and positive')
        self.r_fy.fill_(float(r_fy))
        self.r_cosmic.fill_(float(r_cosmic))

    @staticmethod
    def _localization_precision(rho_squared):
        radius = rho_squared.clamp(0.0, 1.0).sqrt()
        return (1.0 - 3.0 * radius.square() + 2.0 * radius.pow(3)).clamp_min(0.0)

    @staticmethod
    def _empty_observations(reference, batch):
        return {
            'value': reference.new_zeros(batch, 0),
            'background': reference.new_zeros(batch, 0),
            'valid_mask': torch.zeros(
                batch, 0, device=reference.device, dtype=torch.bool),
            'rho_squared': reference.new_zeros(batch, 0),
        }

    def _source_terms(self, X, phi_obs, observations, variance):
        innovation = observations['value'] - observations['background']
        valid = observations['valid_mask'].to(dtype=X.dtype)
        localization = self._localization_precision(
            observations['rho_squared'])
        representativeness = observations.get(
            'representativeness_weight',
            torch.ones_like(observations['rho_squared']))
        precision = (
            valid * localization * representativeness
            / variance.clamp_min(1e-12))
        obs_anomalies = torch.einsum('bmd,bnd->bmn', phi_obs, X)
        covariance = torch.einsum(
            'bmn,bm,bmk->bnk', obs_anomalies, precision, obs_anomalies)
        rhs = torch.einsum(
            'bmn,bm,bm->bn', obs_anomalies, precision, innovation)
        return covariance, rhs, innovation, obs_anomalies, precision

    def forward(self, z_background, h_sw, phi_query, sources,
                lat_n, cos_SZA, sin_doy, cos_doy, sin_I):
        b_in = _build_kalman_b_input(h_sw, lat_n, cos_SZA, sin_doy, cos_doy, sin_I)
        anomalies, factor_scales = self._eval_perturbations(b_in)
        if self.anomaly_parameterization == 'legacy_independent':
            anomalies = anomalies - anomalies.mean(dim=1, keepdim=True)
            inflation = torch.exp(self.log_inflation).clamp(0.8, 1.5)
        else:
            inflation = anomalies.new_ones(())
        X = anomalies * torch.sqrt(inflation)
        batch, members, _ = X.shape
        empty = self._empty_observations(X, batch)
        fy_obs, fy_phi = sources.get('FY', (empty, X.new_zeros(batch, 0, self.d_model)))
        cosmic_obs, cosmic_phi = sources.get(
            'COSMIC', (empty, X.new_zeros(batch, 0, self.d_model)))
        fy_terms = self._source_terms(X, fy_phi, fy_obs, self.r_fy)
        cosmic_terms = self._source_terms(
            X, cosmic_phi, cosmic_obs, self.r_cosmic)
        eye = torch.eye(members, device=X.device, dtype=X.dtype).expand(
            batch, members, members)
        system = (
            max(members - 1, 1) * eye + fy_terms[0] + cosmic_terms[0])
        chol = torch.linalg.cholesky(system)
        weights_fy = torch.cholesky_solve(
            fy_terms[1].unsqueeze(-1), chol).squeeze(-1)
        weights_cosmic = torch.cholesky_solve(
            cosmic_terms[1].unsqueeze(-1), chol).squeeze(-1)
        weights = weights_fy + weights_cosmic
        latent_increment = torch.einsum('bn,bnd->bd', weights, X)
        z_analysis = z_background + latent_increment
        query_anomalies = torch.einsum('bd,bnd->bn', phi_query, X)
        delta_fy = torch.einsum('bn,bn->b', query_anomalies, weights_fy)
        delta_cosmic = torch.einsum(
            'bn,bn->b', query_anomalies, weights_cosmic)

        eigenvalues, eigenvectors = torch.linalg.eigh(system)
        transform = torch.einsum(
            'bnk,bk,bmk->bnm',
            eigenvectors,
            torch.sqrt(
                X.new_tensor(float(max(members - 1, 1)))
                / eigenvalues.clamp_min(1e-12)),
            eigenvectors,
        )
        analysis_anomalies = torch.einsum('bnm,bmd->bnd', transform, X)

        gain = []
        cross_covariance = []
        for terms in (fy_terms, cosmic_terms):
            weighted_y = terms[3].transpose(1, 2) * terms[4].unsqueeze(1)
            solved = torch.cholesky_solve(weighted_y, chol)
            gain.append(torch.einsum('bn,bnm->bm', query_anomalies, solved))
            cross_covariance.append(torch.einsum(
                'bn,bmn->bm', query_anomalies, terms[3])
                / max(members - 1, 1))

        w_abs = weights.abs()
        member_weights = w_abs / (w_abs.sum(dim=-1, keepdim=True) + 1e-6)
        self.last_member_weights  = member_weights
        self.last_inflation_scale = inflation.detach()
        if factor_scales is None:
            singular_values = torch.linalg.svdvals(X)
        else:
            singular_values = torch.cat([
                math.sqrt(max(members - 1, 1)) * torch.sort(
                    factor_scales, dim=-1, descending=True).values,
                X.new_zeros(batch, 1),
            ], dim=-1)
        singular_energy = singular_values.square()
        singular_probability = singular_energy / singular_energy.sum(
            dim=-1, keepdim=True).clamp_min(1e-12)
        effective_rank = torch.exp(-torch.sum(
            singular_probability * torch.log(
                singular_probability.clamp_min(1e-12)), dim=-1))
        positive_singular = singular_values[:, :max(members - 1, 1)]
        anomaly_condition = (
            positive_singular[:, 0]
            / positive_singular[:, -1].clamp_min(1e-12))
        if factor_scales is None:
            scale_saturation = X.new_zeros(batch)
        else:
            lower = self.scale_init / math.sqrt(self.scale_condition_max)
            upper = self.scale_init * math.sqrt(self.scale_condition_max)
            scale_saturation = (
                (factor_scales <= lower * 1.01)
                | (factor_scales >= upper * 0.99)
            ).to(X.dtype).mean(dim=-1)
        return {
            'z_analysis': z_analysis,
            'latent_increment': latent_increment,
            'latent_anomalies': X,
            'analysis_anomalies': analysis_anomalies,
            'transform': transform,
            'weights_FY': weights_fy,
            'weights_COSMIC': weights_cosmic,
            'innovation_FY': fy_terms[2],
            'innovation_COSMIC': cosmic_terms[2],
            'obs_anomalies_FY': fy_terms[3],
            'obs_anomalies_COSMIC': cosmic_terms[3],
            'precision_FY': fy_terms[4],
            'precision_COSMIC': cosmic_terms[4],
            'representativeness_FY': fy_obs.get(
                'representativeness_weight',
                torch.ones_like(fy_obs['rho_squared'])),
            'representativeness_COSMIC': cosmic_obs.get(
                'representativeness_weight',
                torch.ones_like(cosmic_obs['rho_squared'])),
            'ensemble_covariance_FY': fy_terms[0],
            'ensemble_covariance_COSMIC': cosmic_terms[0],
            'ensemble_rhs_FY': fy_terms[1],
            'ensemble_rhs_COSMIC': cosmic_terms[1],
            'gain_FY': gain[0],
            'gain_COSMIC': gain[1],
            'cross_covariance_FY': cross_covariance[0],
            'cross_covariance_COSMIC': cross_covariance[1],
            'delta_FY': delta_fy,
            'delta_COSMIC': delta_cosmic,
            'query_anomalies': query_anomalies,
            'system': system,
            'inflation': inflation,
            'factor_scales': factor_scales,
            'anomaly_singular_values': singular_values,
            'anomaly_effective_rank': effective_rank,
            'anomaly_condition': anomaly_condition,
            'scale_boundary_saturation': scale_saturation,
        }


# ======================== SpectralSWBranch ========================

class SpectralSWBranch(nn.Module):
    """Encode rFFT magnitudes with attention-weighted frequency pooling."""

    def __init__(self, d_model: int = 64):
        super().__init__()
        self.freq_proj = nn.Linear(2, d_model)
        self.attn_pool = nn.Sequential(
            nn.Linear(d_model, 32), nn.SiLU(), nn.Linear(32, 1),
        )

    def forward(self, sw_seq):
        """sw_seq: [B, L, 2]; returns h_sw_freq [B, d_model]"""
        F_mag   = torch.fft.rfft(sw_seq, dim=1).abs()         # [B, n_freq, 2]
        Z_f     = F.silu(self.freq_proj(F_mag))               # [B, n_freq, d]
        weights = F.softmax(self.attn_pool(Z_f), dim=1)       # [B, n_freq, 1]
        return (weights * Z_f).sum(dim=1)                     # [B, d]


# ======================== 主模型 ========================

class FSIA_INR_Model(nn.Module):
    """Feature-space FY/COSMIC assimilation over a frozen IRI background."""

    def __init__(self, iri_proxy, config):
        super().__init__()

        self.alt_min, self.alt_max = config['alt_range']
        self.seq_len = config['seq_len']

        basis_dim       = config.get('basis_dim', 64)
        sw_out_dim      = config.get('sw_out_dim', 64)
        self.sw_out_dim = int(sw_out_dim)
        sw_hidden_dim   = config.get('sw_hidden_dim', 32)
        sw_lstm_layers  = config.get('sw_lstm_layers', 2)
        tau_kp_init     = config.get('tau_kp_init', 8.0)
        tau_solar_init  = config.get('tau_solar_init', 72.0)
        if int(basis_dim) < 1:
            raise ValueError('basis_dim must be positive')

        # IRI proxy 末层隐层维度（与代理网络架构匹配：[4,128,128,128,128,1]）
        iri_hidden_dim = 128

        # ==================== [A] IRI 代理场（完全冻结）====================
        self.iri_proxy = iri_proxy
        self.iri_proxy.freeze()

        # ---- IRI 峰对齐特征网络 ----
        # 输入: cat(h_iri[128], delta_alt_iri[1], NmF2_IRI_n[1]) = 130D
        self.iri_align_net = nn.Sequential(
            nn.Linear(iri_hidden_dim + 2, 128),
            nn.SiLU(),
            nn.Linear(128, basis_dim),
        )
        # ==================== [B] 双尺度 SW 编码器（含多窗口统计）====================
        self.sw_encoder = DualScaleSWEncoder(
            seq_len=self.seq_len,
            sw_hidden_dim=sw_hidden_dim,
            sw_lstm_layers=sw_lstm_layers,
            sw_out_dim=sw_out_dim,
            tau_kp_init=tau_kp_init,
            tau_solar_init=tau_solar_init,
        )

        # ==================== [B+] SW 频域分支（run40，复活自 run28-A 简化版）====================
        self.use_sw_freq    = bool(config.get('use_sw_freq', True))
        sw_gate_bias_init   = float(config.get('sw_gate_bias_init', -2.0))
        if self.use_sw_freq:
            self.sw_freq_branch = SpectralSWBranch(
                d_model=sw_out_dim,
            )
            self.sw_gate = nn.Sequential(
                nn.Linear(sw_out_dim + 5, 32),
                nn.SiLU(),
                nn.Linear(32, sw_out_dim),
            )
            nn.init.zeros_(self.sw_gate[-1].weight)
            nn.init.constant_(self.sw_gate[-1].bias, sw_gate_bias_init)

        enkf_n_members   = int(config.get('enkf_n_members', 8))
        enkf_pert_hidden = int(config.get('enkf_pert_hidden', 64))
        self.analysis_state_semantics = config.get(
            'analysis_state_semantics', 'legacy_feature_increment')
        if self.analysis_state_semantics not in (
                'legacy_feature_increment', 'query_local_increment_coefficients',
                'shared_anchor_response'):
            raise ValueError(
                'analysis_state_semantics must be legacy_feature_increment, '
                'query_local_increment_coefficients, or shared_anchor_response')
        self.context_semantics = config.get(
            'context_semantics', 'query_conditioning')
        self.mode_basis_semantics = config.get(
            'mode_basis_semantics', 'learned_density_basis')
        self.physical_mode_dictionary = config.get(
            'physical_mode_dictionary', 'm2r_residual_legendre')
        self.allow_failed_query_local_shadow = bool(
            config.get('allow_failed_query_local_shadow', False))
        self.uses_physical_modes = (
            self.analysis_state_semantics == 'query_local_increment_coefficients')
        self.uses_shared_anchor_response = (
            self.analysis_state_semantics == 'shared_anchor_response')
        if (self.uses_shared_anchor_response
                and self.context_semantics != 'shared_error_state'):
            raise ValueError(
                'M2-U shared_anchor_response requires shared_error_state context')
        if self.uses_physical_modes:
            if enkf_n_members != 8:
                raise ValueError('M2-R physical coefficient state requires N8')
            if self.context_semantics != 'endpoint_conditioning_only':
                raise ValueError(
                    'M2-R context must use endpoint_conditioning_only semantics')
            if self.mode_basis_semantics != 'reference_whitened_physical_modes':
                raise ValueError(
                    'M2-R requires reference_whitened_physical_modes')
            if self.physical_mode_dictionary not in (
                    'm2r_residual_legendre', 'query_hmf2_legendre',
                    'endpoint_hmf2_legendre', 'background_adaptive_fixed'):
                raise ValueError('unknown physical_mode_dictionary')
            if (self.physical_mode_dictionary != 'm2r_residual_legendre'
                    and not self.allow_failed_query_local_shadow):
                raise ValueError(
                    'V0/V1/V2 query-local dictionaries failed RSR-3 and are '
                    'available only to the independent audit shadow')
        self.density_basis_semantics = config.get(
            'density_basis_semantics', 'query_conditioned')
        if self.density_basis_semantics not in (
                'query_conditioned', 'coordinate_local_symmetric',
                'endpoint_context_symmetric'):
            raise ValueError(
                'density_basis_semantics must be query_conditioned, '
                'coordinate_local_symmetric, or endpoint_context_symmetric')
        self.kalman_layer = NeuralETKFLayer(
            d_model=(enkf_n_members - 1 if self.uses_physical_modes else basis_dim),
            b_net_in=(basis_dim + sw_out_dim + 5
                      if self.uses_physical_modes else sw_out_dim + 5),
            n_members=enkf_n_members,
            pert_hidden=enkf_pert_hidden,
            r_fy=config.get('r_fy_init', 0.04),
            r_cosmic=config.get('r_cosmic_init', 0.04),
            anomaly_parameterization=config.get(
                'enkf_anomaly_parameterization', 'legacy_independent'),
            scale_init=config.get('enkf_scale_init', 1.1),
            scale_condition_max=config.get(
                'enkf_scale_condition_max', 3.0),
            coordinate_local_symmetric=(self.density_basis_semantics in (
                'coordinate_local_symmetric',
                'endpoint_context_symmetric')) and not self.uses_physical_modes,
            coefficient_space=self.uses_physical_modes,
        )
        self.enkf_n_members = enkf_n_members
        self.background_state_dim = basis_dim

        # M2-U temperatures are train-only calibrated scalars.  They are stored
        # as buffers so checkpoint restore reproduces the exact anchor partition.
        if self.uses_shared_anchor_response:
            tau_values = torch.as_tensor([
                float(config.get('m2u_tau_M10', 1.0)),
                float(config.get('m2u_tau_M01', 1.0)),
                float(config.get('m2u_tau_M11', 1.0)),
            ], dtype=torch.float32)
            if not torch.isfinite(tau_values).all() or torch.any(tau_values <= 0):
                raise ValueError('M2-U anchor temperatures must be finite and positive')
            self.register_buffer('m2u_temperatures', tau_values)
            self.m2u_space_support_km = float(
                config.get('m2u_space_support_km', 1800.0))
            self.m2u_time_support_h = float(
                config.get('m2u_time_support_h', 1.5))
            self.m2u_state_floor = float(config.get('m2u_state_floor', 0.05))
            self.m2u_state_direction_scale = float(
                config.get('m2u_state_direction_scale', 1.0))
            self.m2u_state_amplitude_scale = float(
                config.get('m2u_state_amplitude_scale', 1.0))
            self.m2u_anchor_chunk_size = int(
                config.get('m2u_anchor_chunk_size', 256))
            if self.m2u_anchor_chunk_size < 1:
                raise ValueError('m2u_anchor_chunk_size must be positive')
            if not 0.0 < self.m2u_state_floor <= 1.0:
                raise ValueError('m2u_state_floor must be in (0,1]')

        self.background_residual_cap = float(
            config.get('background_residual_cap', 0.5))
        self.fy_dlon_window = float(config.get('fy_nb_dlon', 15.0))
        self.cosmic_dlon_window = float(
            config.get('cosmic_nb_dlon', 15.0))
        self.fy_dlat_window = float(config.get('fy_nb_dlat', 5.0))
        self.fy_dt_window = float(config.get('fy_nb_dt', 1.5))
        self.cosmic_dlat_window = float(config.get('cosmic_nb_dlat', 5.0))
        self.cosmic_dt_window = float(config.get('cosmic_nb_dt', 1.5))
        self.background_decoder = nn.Sequential(
            nn.Linear(basis_dim + sw_out_dim + 2, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )

        # Shared physical observation operator.  It is nonlinear in continuous
        # coordinates/context, but affine in the low-dimensional ETKF state.
        basis_in_dim = basis_dim + sw_out_dim + 12
        if not self.uses_physical_modes:
            self.density_basis_decoder = nn.Sequential(
                nn.Linear(basis_in_dim, 64),
                nn.SiLU(),
                nn.Linear(64, basis_dim),
            )
        else:
            self.mode_residual_cap = float(config.get('mode_residual_cap', 0.25))
            mode_in_dim = 2 * (basis_dim + sw_out_dim) + 12
            self.mode_residual = nn.Sequential(
                nn.Linear(mode_in_dim, 64), nn.SiLU(), nn.Linear(64, 7))

        self._initialize_weights()

    # ------------------------------------------------------------------ #

    def get_background_with_features(self, lat, lon, alt, time):
        """
        查询冻结的 IRI proxy，返回背景场和末层隐状态。
        IRI proxy 完全冻结，使用 torch.no_grad() 避免构建无效计算图。
        """
        coords_iri = torch.stack([lat, lon, alt, time], dim=-1)
        with torch.no_grad():
            ne_bkg, h_iri = self.iri_proxy(coords_iri, return_features=True)
        return ne_bkg, h_iri   # [B,1], [B,128]

    def _density_basis(self, query_coords, target_coords, target_background,
                       z_background, h_sw, target_z_background=None,
                       target_h_sw=None):
        """Evaluate the shared continuous basis at physical target coordinates."""
        batch, count, _ = target_coords.shape
        query = query_coords[:, None, :4]
        lat = target_coords[..., 0]
        lon = target_coords[..., 1]
        alt = target_coords[..., 2]
        time = target_coords[..., 3]
        cos_sza, sin_doy, cos_doy = _compute_solar_features(
            lat.reshape(-1), lon.reshape(-1), time.reshape(-1))
        cos_sza = cos_sza.reshape(batch, count)
        sin_doy = sin_doy.reshape(batch, count)
        cos_doy = cos_doy.reshape(batch, count)
        local_descriptors = [
            (target_background - 10.5) / 1.5,
            lat / 90.0,
            torch.sin(torch.deg2rad(lon)),
            torch.cos(torch.deg2rad(lon)),
            2.0 * (alt - self.alt_min) / (self.alt_max - self.alt_min) - 1.0,
            cos_sza,
            sin_doy,
            cos_doy,
        ]
        if self.density_basis_semantics in (
                'coordinate_local_symmetric',
                'endpoint_context_symmetric'):
            descriptors = torch.stack([
                *local_descriptors,
                *[torch.zeros_like(lat) for _ in range(4)],
            ], dim=-1)
            if self.density_basis_semantics == 'endpoint_context_symmetric':
                if target_z_background is None or target_h_sw is None:
                    raise ValueError(
                        'endpoint-context basis requires target endpoint context')
                context = torch.cat(
                    [target_z_background, target_h_sw], dim=-1)
            else:
                context = target_coords.new_zeros(
                    batch, count, z_background.shape[-1] + h_sw.shape[-1])
        else:
            dlon = torch.remainder(
                lon - query[..., 1] + 180.0, 360.0) - 180.0
            descriptors = torch.stack([
                *local_descriptors,
                (lat - query[..., 0]) / 5.0,
                dlon / 15.0,
                (time - query[..., 3]) / 1.5,
                (alt - query[..., 2]) / 190.0,
            ], dim=-1)
            context = torch.cat([z_background, h_sw], dim=-1)
            context = context[:, None, :].expand(-1, count, -1)
        return self.density_basis_decoder(
            torch.cat([context, descriptors], dim=-1)) / math.sqrt(
                self.kalman_layer.d_model)

    def encode_background(self, coords, sw_seq, precomputed_h_sw=None,
                          iri_peak=None):
        """Single source of truth for the frozen/trained Background field."""
        lat, lon, alt, time = (coords[:, index] for index in range(4))
        batch = coords.shape[0]
        lat_n = lat / 90.0
        alt_n = 2.0 * (alt - self.alt_min) / (self.alt_max - self.alt_min) - 1.0
        cos_sza, sin_doy, cos_doy = _compute_solar_features(lat, lon, time)
        sin_i, _ = _compute_dip_features(lat, lon)
        if precomputed_h_sw is None:
            h_sw_time, kp_eff, f107_eff = self.sw_encoder(sw_seq)
        else:
            h_sw_time = precomputed_h_sw
            kp_eff, f107_eff = sw_seq[:, -1, 0], sw_seq[:, -1, 1]
        if self.use_sw_freq:
            h_sw_freq = self.sw_freq_branch(sw_seq)
            sw_gate_in = torch.cat([
                h_sw_time, alt_n.unsqueeze(-1), sin_i.abs().unsqueeze(-1),
                cos_sza.unsqueeze(-1), sin_doy.unsqueeze(-1),
                cos_doy.unsqueeze(-1),
            ], dim=-1)
            sw_gate = torch.sigmoid(self.sw_gate(sw_gate_in))
            h_sw = sw_gate * h_sw_freq + (1.0 - sw_gate) * h_sw_time
        else:
            h_sw_freq, sw_gate, h_sw = None, None, h_sw_time
        if iri_peak is None:
            iri_peak = torch.stack([
                torch.full((batch,), 300.0, device=coords.device),
                torch.full((batch,), 11.5, device=coords.device),
            ], dim=-1)
        hmf2, nmf2 = iri_peak[:, 0], iri_peak[:, 1]
        delta_alt = (alt - hmf2.detach()) / 190.0
        ne_iri, h_iri = self.get_background_with_features(lat, lon, alt, time)
        z_background = self.iri_align_net(torch.cat([
            h_iri, delta_alt.unsqueeze(-1),
            ((nmf2 - 11.0) / 2.0).unsqueeze(-1),
        ], dim=-1))
        background_residual = self.background_residual_cap * torch.tanh(
            self.background_decoder(torch.cat([
                z_background, h_sw, alt_n.unsqueeze(-1),
                delta_alt.unsqueeze(-1),
            ], dim=-1)))
        return {
            'ne_bkg': ne_iri + background_residual,
            'ne_iri': ne_iri,
            'background_residual': background_residual,
            'z_background': z_background,
            'h_sw': h_sw,
            'h_sw_freq': h_sw_freq,
            'sw_gate': sw_gate,
            'lat_n': lat_n,
            'alt_n': alt_n,
            'cos_sza': cos_sza,
            'sin_doy': sin_doy,
            'cos_doy': cos_doy,
            'sin_i': sin_i,
            'kp_eff': kp_eff,
            'f107_eff': f107_eff,
            'iri_peak': iri_peak,
            'delta_alt': delta_alt,
        }

    def encode_endpoint_context(self, coords, sw_seq, iri_peak=None):
        """Build one source-neutral physical context for query/obs/reference."""
        center = self.encode_background(coords, sw_seq, iri_peak=iri_peak)
        step = 10.0

        def at_offset(offset):
            shifted = coords.clone()
            shifted[:, 2] = (coords[:, 2] + offset).clamp(
                self.alt_min, self.alt_max)
            return self.encode_background(
                shifted, sw_seq, iri_peak=iri_peak)['ne_bkg'].flatten()

        minus, plus = at_offset(-step), at_offset(step)
        minus2, plus2 = at_offset(-2 * step), at_offset(2 * step)
        value = center['ne_bkg'].flatten()
        lower = coords[:, 2] < self.alt_min + step
        upper = coords[:, 2] > self.alt_max - step
        derivative = (plus - minus) / (2 * step)
        derivative = torch.where(lower, (plus - value) / step, derivative)
        derivative = torch.where(upper, (value - minus) / step, derivative)
        second = (plus - 2 * value + minus) / step ** 2
        second = torch.where(
            lower, (value - 2 * plus + plus2) / step ** 2, second)
        second = torch.where(
            upper, (value - 2 * minus + minus2) / step ** 2, second)
        sin_lst, cos_lst = _compute_local_time_features(
            coords[:, 1], coords[:, 3])
        return {
            **center,
            'basis_hmf2': center['iri_peak'][:, 0],
            'basis_nmf2': center['iri_peak'][:, 1],
            'basis_delta_alt': center['delta_alt'],
            'basis_cos_sza': center['cos_sza'],
            'basis_sin_lst': sin_lst,
            'basis_cos_lst': cos_lst,
            'basis_background_dh': derivative.detach(),
            'basis_background_d2h': second.detach(),
        }

    def _physical_mode_raw(self, unit_center, unit_hmf2, target_coords,
                           query_z=None, query_h_sw=None, target_z=None,
                           target_h_sw=None, target_background=None,
                           target_endpoint=None):
        """Seven fixed continuous log-density increment modes for M2-R R1."""
        lat, lon, alt, time = (target_coords[..., index] for index in range(4))
        dlat = (lat - unit_center[:, None, 0]) / self.fy_dlat_window
        dlon = torch.remainder(
            lon - unit_center[:, None, 1] + 180.0, 360.0) - 180.0
        dlon = dlon / self.fy_dlon_window
        dt = (time - unit_center[:, None, 3]) / self.fy_dt_window
        height = (alt - unit_hmf2[:, None]) / 190.0
        if self.physical_mode_dictionary in (
                'endpoint_hmf2_legendre', 'background_adaptive_fixed'):
            if target_endpoint is None:
                raise ValueError(
                    'endpoint physical modes require explicit endpoint context')
            height = target_endpoint['basis_delta_alt']
        cos_sza, _, _ = _compute_solar_features(
            lat.reshape(-1), lon.reshape(-1), time.reshape(-1))
        cos_sza = cos_sza.reshape_as(lat)
        center_sza, _, _ = _compute_solar_features(
            unit_center[:, 0], unit_center[:, 1], unit_center[:, 3])
        center_sin_lst, center_cos_lst = _compute_local_time_features(
            unit_center[:, 1], unit_center[:, 3])
        time_mode = (torch.sin(0.5 * math.pi * dt)
                     + 0.25 * (cos_sza - center_sza[:, None]))
        if target_endpoint is not None:
            lst_delta = (
                target_endpoint['basis_sin_lst'] * center_cos_lst[:, None]
                - target_endpoint['basis_cos_lst'] * center_sin_lst[:, None])
            time_mode = time_mode + 0.25 * lst_delta
        if self.physical_mode_dictionary == 'background_adaptive_fixed':
            derivative = 190.0 * target_endpoint['basis_background_dh']
            u = ((alt - 180.0) / 120.0).clamp(0.0, 1.0)
            low = 1.0 - 3.0 * u.square() + 2.0 * u.pow(3)
            fixed = torch.stack([
                torch.ones_like(height), -derivative, height * derivative,
                low, torch.sin(0.5 * math.pi * dlat),
                torch.sin(0.5 * math.pi * dlon),
                time_mode,
            ], dim=-1)
        else:
            fixed = torch.stack([
                torch.ones_like(height),
                height,
                0.5 * (3.0 * height.square() - 1.0),
                0.5 * (5.0 * height.pow(3) - 3.0 * height),
                torch.sin(0.5 * math.pi * dlat),
                torch.sin(0.5 * math.pi * dlon),
                time_mode,
            ], dim=-1)
        if (query_z is None or
                self.physical_mode_dictionary != 'm2r_residual_legendre'):
            return fixed
        count = target_coords.shape[1]
        if target_z is None:
            target_z = query_z[:, None, :].expand(-1, count, -1)
        if target_h_sw is None:
            target_h_sw = query_h_sw[:, None, :].expand(-1, count, -1)
        if target_background is None:
            target_background = target_coords.new_zeros(target_coords.shape[:2])
        query_context = torch.cat([query_z, query_h_sw], dim=-1)
        query_context = query_context[:, None, :].expand(-1, count, -1)
        descriptors = torch.stack([
            (target_background - 10.5) / 1.5,
            lat / 90.0,
            torch.sin(torch.deg2rad(lon)),
            torch.cos(torch.deg2rad(lon)),
            2.0 * (alt - self.alt_min) / (self.alt_max - self.alt_min) - 1.0,
            cos_sza,
            torch.sin(torch.deg2rad(15.0 * torch.remainder(time, 24.0))),
            torch.cos(torch.deg2rad(15.0 * torch.remainder(time, 24.0))),
            dlat, dlon, dt, height,
        ], dim=-1)
        residual = self.mode_residual(torch.cat([
            query_context, target_z, target_h_sw, descriptors], dim=-1))
        return fixed + self.mode_residual_cap * torch.tanh(residual)

    def _physical_reference_coords(self, unit_center):
        """Deterministic quadrature coordinates for each query-local problem."""
        dtype, device = unit_center.dtype, unit_center.device
        axis = torch.tensor([-1.0, 0.0, 1.0], device=device, dtype=dtype)
        alt = torch.linspace(
            self.alt_min, self.alt_max, 5, device=device, dtype=dtype)
        h, y, x, t = torch.meshgrid(alt, axis, axis, axis, indexing='ij')
        offsets = torch.stack([y, x, h, t], dim=-1).reshape(-1, 4)
        reference = unit_center[:, None, :4] + offsets[None] * unit_center.new_tensor([
            self.fy_dlat_window, self.fy_dlon_window, 0.0,
            self.fy_dt_window])
        reference[..., 0].clamp_(-90.0, 90.0)
        reference[..., 1] = torch.remainder(reference[..., 1] + 180.0, 360.0) - 180.0
        reference[..., 2] = offsets[None, ..., 2]
        return reference

    def _physical_mode_transform(self, unit_center, unit_hmf2,
                                 query_z=None, query_h_sw=None,
                                 query_background=None, reference_z=None,
                                 reference_h_sw=None,
                                 reference_background=None,
                                 reference_endpoint=None):
        """Reference-grid Cholesky gauge; independent of observations."""
        dtype, device = unit_center.dtype, unit_center.device
        reference = self._physical_reference_coords(unit_center)
        raw = self._physical_mode_raw(
            unit_center, unit_hmf2, reference, query_z, query_h_sw,
            reference_z, reference_h_sw,
            (reference_background if reference_background is not None else
             (query_background[:, None].expand(-1, reference.shape[1])
              if query_background is not None else None)),
            reference_endpoint)
        gram = torch.einsum('uri,urj->uij', raw, raw) / raw.shape[1]
        eigenvalues = torch.linalg.eigvalsh(gram)
        chol = torch.linalg.cholesky(gram)
        modes = torch.linalg.solve_triangular(
            chol, raw.transpose(1, 2), upper=False).transpose(1, 2)
        normalized_gram = torch.einsum(
            'uri,urj->uij', modes, modes) / modes.shape[1]
        correction = torch.linalg.cholesky(normalized_gram)
        modes = torch.linalg.solve_triangular(
            correction, modes.transpose(1, 2), upper=False).transpose(1, 2)
        chol = torch.bmm(chol, correction)
        normalized_gram = torch.einsum(
            'uri,urj->uij', modes, modes) / modes.shape[1]
        return modes, chol, {
            'mode_reference_gram_error': (
                normalized_gram - torch.eye(
                    7, device=device, dtype=dtype)).abs().amax(dim=(1, 2)),
            'raw_mode_gram_min_eigenvalue': eigenvalues[:, 0],
            'raw_mode_gram_condition': (
                eigenvalues[:, -1] / eigenvalues[:, 0].clamp_min(1e-12)),
        }

    @staticmethod
    def _apply_mode_transform(raw, chol):
        return torch.linalg.solve_triangular(
            chol, raw.transpose(1, 2), upper=False).transpose(1, 2)

    @staticmethod
    def _spectrum(matrix):
        singular = torch.linalg.svdvals(matrix)
        if singular.shape[-1] == 0:
            empty = matrix.new_zeros(matrix.shape[0])
            return singular, empty, empty
        energy = singular.square()
        probability = energy / energy.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        effective_rank = torch.exp(-torch.sum(
            probability * torch.log(probability.clamp_min(1e-12)), dim=-1))
        first_energy = probability[..., 0]
        return singular, effective_rank, first_energy

    def _physical_mode_forward(self, coords, background, observations_fy,
                               observations_cosmic, reference_context=None):
        """M2-R query-local coefficient ETKF over the existing M2-O neighbors."""
        batch = len(coords)
        center = coords[:, :4]
        hmf2 = background['iri_peak'][:, 0]
        endpoint_modes = self.physical_mode_dictionary in (
            'endpoint_hmf2_legendre', 'background_adaptive_fixed')
        if endpoint_modes and reference_context is None:
            raise ValueError('physical reference endpoint context is required')
        query_endpoint = {
            key: value[:, None] for key, value in background.items()
            if key.startswith('basis_') and value.ndim == 1}
        reference_modes, chol, mode_diagnostics = self._physical_mode_transform(
            center, hmf2, background['z_background'], background['h_sw'],
            background['ne_bkg'].flatten(),
            reference_z=(None if reference_context is None else
                         reference_context['basis_z_background']),
            reference_h_sw=(None if reference_context is None else
                            reference_context['basis_h_sw']),
            reference_background=(None if reference_context is None else
                                  reference_context['background']),
            reference_endpoint=reference_context)
        query_modes = self._apply_mode_transform(
            self._physical_mode_raw(
                center, hmf2, coords[:, None, :4],
                background['z_background'], background['h_sw'],
                background['z_background'][:, None, :],
                background['h_sw'][:, None, :], background['ne_bkg'],
                query_endpoint if endpoint_modes else None),
            chol).squeeze(1)

        def prepare(observations):
            if observations is None:
                return ({
                    'coords': coords.new_zeros(batch, 0, 4),
                    'value': coords.new_zeros(batch, 0),
                    'background': coords.new_zeros(batch, 0),
                    'valid_mask': torch.zeros(
                        batch, 0, device=coords.device, dtype=torch.bool),
                    'rho_squared': coords.new_zeros(batch, 0),
                }, coords.new_zeros(batch, 0, 7))
            required = {
                'coords', 'value', 'background', 'valid_mask', 'rho_squared',
                'basis_z_background', 'basis_h_sw'}
            if endpoint_modes:
                required.update((
                    'basis_hmf2', 'basis_nmf2', 'basis_delta_alt',
                    'basis_cos_sza', 'basis_sin_lst', 'basis_cos_lst',
                    'basis_background_dh', 'basis_background_d2h'))
            missing = required.difference(observations)
            if missing:
                raise ValueError(
                    f'M2-R observation payload missing fields: {sorted(missing)}')
            valid = observations['valid_mask']
            count = valid.shape[1]
            prepared = dict(observations)
            prepared['coords'] = torch.where(
                valid.unsqueeze(-1), observations['coords'],
                coords[:, None, :4].expand(-1, count, -1))
            prepared['background'] = torch.where(
                valid, observations['background'],
                background['ne_bkg'].expand(-1, count))
            prepared['value'] = torch.where(
                valid, observations['value'], prepared['background'])
            prepared['rho_squared'] = torch.where(
                valid, observations['rho_squared'],
                torch.ones_like(observations['rho_squared']))
            if 'representativeness_weight' in observations:
                prepared['representativeness_weight'] = torch.where(
                    valid, observations['representativeness_weight'],
                    torch.ones_like(observations['representativeness_weight']))
            modes = self._apply_mode_transform(
                self._physical_mode_raw(
                    center, hmf2, prepared['coords'],
                    background['z_background'], background['h_sw'],
                    prepared['basis_z_background'], prepared['basis_h_sw'],
                    prepared['background'],
                    prepared if endpoint_modes else None), chol)
            return prepared, modes.masked_fill(~valid.unsqueeze(-1), 0.0)

        fy_obs, fy_modes = prepare(observations_fy)
        cosmic_obs, cosmic_modes = prepare(observations_cosmic)
        coefficient_background = coords.new_zeros(batch, 7)
        etkf = self.kalman_layer(
            coefficient_background,
            torch.cat([background['z_background'], background['h_sw']], dim=-1),
            query_modes,
            {'FY': (fy_obs, fy_modes), 'COSMIC': (cosmic_obs, cosmic_modes)},
            background['lat_n'], background['cos_sza'], background['sin_doy'],
            background['cos_doy'], background['sin_i'])
        ne_delta = (etkf['delta_FY'] + etkf['delta_COSMIC']).unsqueeze(-1)
        ne_fused = background['ne_bkg'] + ne_delta
        reference_anomalies = torch.einsum(
            'brc,bnc->brn', reference_modes, etkf['latent_anomalies'])
        spectra = {}
        for name, matrix in (
                ('mode_reference', reference_modes),
                ('reference_anomalies', reference_anomalies),
                ('FY_observation_anomalies', etkf['obs_anomalies_FY'] *
                 etkf['precision_FY'].sqrt().unsqueeze(-1)),
                ('COSMIC_observation_anomalies', etkf['obs_anomalies_COSMIC'] *
                 etkf['precision_COSMIC'].sqrt().unsqueeze(-1)),
                ('joint_observation_anomalies', torch.cat([
                    etkf['obs_anomalies_FY'] *
                    etkf['precision_FY'].sqrt().unsqueeze(-1),
                    etkf['obs_anomalies_COSMIC'] *
                    etkf['precision_COSMIC'].sqrt().unsqueeze(-1),
                ], dim=1))):
            if not torch.isfinite(matrix).all():
                raise FloatingPointError(f'non-finite M2-R spectrum input: {name}')
            singular, effective_rank, first_energy = self._spectrum(matrix)
            spectra[f'{name}_singular_values'] = singular
            spectra[f'{name}_effective_rank'] = effective_rank
            spectra[f'{name}_first_energy_fraction'] = first_energy

        extras = {
            'ne_bkg': background['ne_bkg'], 'ne_iri': background['ne_iri'],
            'background_residual': background['background_residual'],
            'ne_residual': ne_delta,
            'h_iri_aligned': background['z_background'],
            'hmF2_det': hmf2.detach(),
            'peak_params': {'hmF2': hmf2, 'NmF2': background['iri_peak'][:, 1]},
            'K_FY': etkf['gain_FY'], 'K_COSMIC': etkf['gain_COSMIC'],
            'r_fy': self.kalman_layer.r_fy,
            'r_cosmic': self.kalman_layer.r_cosmic,
            'innov_FY': etkf['innovation_FY'],
            'innov_COSMIC': etkf['innovation_COSMIC'],
            'update_FY': etkf['delta_FY'].unsqueeze(-1),
            'update_COSMIC': etkf['delta_COSMIC'].unsqueeze(-1),
            'weights_FY': etkf['weights_FY'],
            'weights_COSMIC': etkf['weights_COSMIC'],
            'member_weights': self.kalman_layer.last_member_weights,
            'inflation_scale': etkf['inflation'],
            'r_ref_FY': self.kalman_layer.r_fy,
            'r_ref_COSMIC': self.kalman_layer.r_cosmic,
            'h_prior': coefficient_background,
            'h_analysis': etkf['z_analysis'],
            'basis_z_background': background['z_background'],
            'basis_h_sw': background['h_sw'],
            'latent_increment': etkf['latent_increment'],
            'latent_anomalies': etkf['latent_anomalies'],
            'analysis_anomalies': etkf['analysis_anomalies'],
            'etkf_transform': etkf['transform'], 'system': etkf['system'],
            'query_anomalies': etkf['query_anomalies'],
            'query_basis': query_modes, 'basis_FY': fy_modes,
            'basis_COSMIC': cosmic_modes,
            'factor_scales': etkf['factor_scales'],
            'anomaly_singular_values': etkf['anomaly_singular_values'],
            'anomaly_effective_rank': etkf['anomaly_effective_rank'],
            'anomaly_condition': etkf['anomaly_condition'],
            'scale_boundary_saturation': etkf['scale_boundary_saturation'],
            'coefficient_background_query': coefficient_background,
            'coefficient_analysis_query': etkf['z_analysis'],
            **mode_diagnostics, **spectra,
        }
        for source, payload in (('FY', fy_obs), ('COSMIC', cosmic_obs)):
            extras.update({
                f'obs_anomalies_{source}': etkf[f'obs_anomalies_{source}'],
                f'precision_{source}': etkf[f'precision_{source}'],
                f'representativeness_{source}': etkf[
                    f'representativeness_{source}'],
                f'ensemble_covariance_{source}': etkf[
                    f'ensemble_covariance_{source}'],
                f'ensemble_rhs_{source}': etkf[f'ensemble_rhs_{source}'],
                f'cross_covariance_{source}': etkf[f'cross_covariance_{source}'],
                f'observation_coords_{source}': payload['coords'],
                f'observation_rho_squared_{source}': payload['rho_squared'],
            })
        return (ne_fused, torch.zeros_like(ne_fused),
                torch.zeros_like(background['ne_bkg']), ne_delta, extras)

    def _m2u_state_eta(self, z_background, h_sw):
        """Continuous direction-plus-amplitude error-state embedding."""
        rank = min(self.enkf_n_members - 1, z_background.shape[-1])
        direction = F.normalize(z_background[..., :rank], dim=-1, eps=1e-6)
        amplitude = torch.linalg.vector_norm(h_sw, dim=-1, keepdim=True)
        return torch.cat([direction, amplitude], dim=-1)

    def set_m2u_temperatures(self, score_gaps):
        """Freeze train-only top-gap temperatures into the checkpoint buffer."""
        if not self.uses_shared_anchor_response:
            raise ValueError('M2-U temperatures require shared_anchor_response')
        names = ('M10', 'M01', 'M11')
        values = [calibrate_temperature(score_gaps[name]) for name in names]
        self.m2u_temperatures.copy_(self.m2u_temperatures.new_tensor(values))
        return dict(zip(names, values))

    def _shared_anchor_forward(self, coords, background, query_phi,
                               anchor_observations_fy,
                               anchor_observations_cosmic):
        """M2-U shared-anchor ETKF followed by continuous query interpolation."""
        if self.density_basis_semantics != 'endpoint_context_symmetric':
            raise ValueError('M2-U requires endpoint_context_symmetric density basis')

        def valid_catalog(catalog):
            return catalog is not None and catalog['coords'].shape[1] > 0

        catalogs = [
            ('FY', anchor_observations_fy),
            ('COSMIC', anchor_observations_cosmic),
        ]
        active = [(name, catalog) for name, catalog in catalogs
                  if valid_catalog(catalog)]
        batch = coords.shape[0]
        members = self.enkf_n_members
        state_dim = self.kalman_layer.d_model
        base = (members - 1) * torch.eye(
            members, device=coords.device, dtype=coords.dtype)

        def zero_terms(anchor_count):
            return (coords.new_zeros(anchor_count, members, members),
                    coords.new_zeros(anchor_count, members),
                    coords.new_zeros(anchor_count),
                    coords.new_zeros(anchor_count, 0, members),
                    coords.new_zeros(anchor_count, 0))

        if active:
            anchor_coords = torch.cat([
                catalog['coords'].squeeze(0) for _, catalog in active], dim=0)
            anchor_z = torch.cat([
                catalog['basis_z_background'].squeeze(0) for _, catalog in active], dim=0)
            anchor_h = torch.cat([
                catalog['basis_h_sw'].squeeze(0) for _, catalog in active], dim=0)
            anchor_background = torch.cat([
                catalog['background'].squeeze(0) for _, catalog in active], dim=0)
            anchor_source = torch.cat([
                torch.full((catalog['coords'].shape[1],), name == 'FY',
                           device=coords.device, dtype=torch.bool)
                for name, catalog in active], dim=0)
            anchor_count = anchor_coords.shape[0]
        else:
            anchor_coords = coords.new_zeros(0, 4)
            anchor_z = coords.new_zeros(0, self.background_state_dim)
            anchor_h = coords.new_zeros(0, self.sw_out_dim)
            anchor_background = coords.new_zeros(0)
            anchor_source = torch.zeros(0, device=coords.device, dtype=torch.bool)
            anchor_count = 0

        def make_anomalies(z, h, c):
            if z.shape[0] == 0:
                return c.new_zeros(0, members, state_dim), None
            lat_n = c[:, 0] / 90.0
            cos_sza, sin_doy, cos_doy = _compute_solar_features(
                c[:, 0], c[:, 1], c[:, 3])
            sin_i, _ = _compute_dip_features(c[:, 0], c[:, 1])
            b_input = _build_kalman_b_input(
                h, lat_n, cos_sza, sin_doy, cos_doy, sin_i)
            anomalies, scales = self.kalman_layer._eval_perturbations(b_input)
            if self.kalman_layer.anomaly_parameterization == 'legacy_independent':
                anomalies = anomalies - anomalies.mean(dim=1, keepdim=True)
                inflation = torch.exp(self.kalman_layer.log_inflation).clamp(0.8, 1.5)
            else:
                inflation = anomalies.new_ones(())
            return anomalies * torch.sqrt(inflation), scales

        anchor_X, _ = make_anomalies(anchor_z, anchor_h, anchor_coords)
        query_X, factor_scales = make_anomalies(
            background['z_background'], background['h_sw'], coords)
        query_anomalies = torch.einsum('bd,bnd->bn', query_phi, query_X)
        if anchor_count:
            anchor_self_phi = self._density_basis(
                anchor_coords, anchor_coords[:, None, :],
                anchor_background[:, None], anchor_z, anchor_h,
                anchor_z[:, None, :], anchor_h[:, None, :]).squeeze(1)
            anchor_self_anomalies = torch.einsum(
                'ad,and->an', anchor_self_phi, anchor_X)
        else:
            anchor_self_anomalies = coords.new_zeros(0, members)

        def source_terms(source, catalog):
            if not valid_catalog(catalog):
                return (*zero_terms(anchor_count), None)
            obs_coords = catalog['coords'].squeeze(0)
            obs_value = catalog['value'].squeeze(0)
            obs_background = catalog['background'].squeeze(0)
            obs_z = catalog['basis_z_background'].squeeze(0)
            obs_h = catalog['basis_h_sw'].squeeze(0)
            obs_phi = self._density_basis(
                anchor_coords[:1], obs_coords.unsqueeze(0),
                obs_background.unsqueeze(0), anchor_z[:1].unsqueeze(0),
                anchor_h[:1].unsqueeze(0), obs_z.unsqueeze(0),
                obs_h.unsqueeze(0)).squeeze(0)
            cover = flat_top_cover(
                anchor_coords, obs_coords,
                self.m2u_space_support_km, self.m2u_time_support_h)
            anchor_eta = self._m2u_state_eta(anchor_z, anchor_h)
            obs_eta = self._m2u_state_eta(obs_z, obs_h)
            distance = state_distance_squared(
                anchor_eta, obs_eta,
                self.m2u_state_direction_scale,
                self.m2u_state_amplitude_scale)
            state_weight = self.m2u_state_floor + (
                1.0 - self.m2u_state_floor) * torch.exp(-0.5 * distance)
            rep = catalog.get(
                'representativeness_weight',
                torch.ones_like(catalog['value'])).squeeze(0)
            valid = catalog['valid_mask'].squeeze(0).to(coords.dtype)
            precision = (cover * state_weight * rep.unsqueeze(0)
                         * valid.unsqueeze(0))
            precision = precision / (
                self.kalman_layer.r_fy if source == 'FY'
                else self.kalman_layer.r_cosmic).clamp_min(1e-12)
            innovation = obs_value - obs_background
            covariance = coords.new_zeros(anchor_count, members, members)
            rhs = coords.new_zeros(anchor_count, members)
            chunk = self.m2u_anchor_chunk_size
            for start in range(0, anchor_count, chunk):
                stop = min(start + chunk, anchor_count)
                anomalies = torch.einsum(
                    'jd,cnd->cjn', obs_phi, anchor_X[start:stop])
                covariance[start:stop] = torch.einsum(
                    'cjn,cj,cjk->cnk', anomalies, precision[start:stop],
                    anomalies)
                rhs[start:stop] = torch.einsum(
                    'cjn,cj,j->cn', anomalies, precision[start:stop],
                    innovation)
            return covariance, rhs, innovation, None, None

        fy_terms = source_terms('FY', anchor_observations_fy)
        cosmic_terms = source_terms('COSMIC', anchor_observations_cosmic)
        fy_cov, fy_rhs = fy_terms[0], fy_terms[1]
        cosmic_cov, cosmic_rhs = cosmic_terms[0], cosmic_terms[1]
        if anchor_count:
            eye = base.unsqueeze(0).expand(anchor_count, -1, -1)
            system_fy = eye + fy_cov
            system_cosmic = eye + cosmic_cov
            system_joint = eye + fy_cov + cosmic_cov
            chol_fy = torch.linalg.cholesky(system_fy)
            chol_cosmic = torch.linalg.cholesky(system_cosmic)
            chol_joint = torch.linalg.cholesky(system_joint)
            w_fy = torch.cholesky_solve(fy_rhs.unsqueeze(-1), chol_fy).squeeze(-1)
            w_cosmic = torch.cholesky_solve(
                cosmic_rhs.unsqueeze(-1), chol_cosmic).squeeze(-1)
            joint_rhs = torch.stack([fy_rhs, cosmic_rhs], dim=-1)
            w_joint_pair = torch.cholesky_solve(
                joint_rhs, chol_joint)
            w_joint = w_joint_pair.sum(dim=-1)
        else:
            system_fy = system_cosmic = system_joint = base.unsqueeze(0)[:0]
            w_fy = w_cosmic = w_joint = coords.new_zeros(0, members)

        query_eta = self._m2u_state_eta(
            background['z_background'], background['h_sw'])
        anchor_eta = self._m2u_state_eta(anchor_z, anchor_h)

        def mix(weights, tau, mask):
            if anchor_count == 0:
                return (coords.new_zeros(batch), coords.new_zeros(batch, 0),
                        torch.ones(batch, device=coords.device, dtype=coords.dtype),
                        coords.new_zeros(batch, 0))
            result = anchor_mixing_weights(
                coords[:, :4], anchor_coords, query_eta, anchor_eta,
                self.m2u_state_direction_scale,
                self.m2u_state_amplitude_scale,
                float(tau), self.m2u_space_support_km,
                self.m2u_time_support_h, anchor_mask=mask)
            # Strict core sharing uses each anchor's own decoded response;
            # query HX remains available above for diagnostics.  This makes a
            # no-observation query with identical eta/core membership receive
            # the same ETKF response as its observed anchor.
            anchor_response = torch.einsum(
                'an,an->a', anchor_self_anomalies, weights)
            increment = torch.einsum('ba,a->b', result['beta'], anchor_response)
            return (increment, result['beta'], result['background_weight'],
                    result['scores'])

        delta_fy, beta_fy, bg_fy, scores_fy = mix(
            w_fy, self.m2u_temperatures[0], anchor_source)
        delta_cosmic, beta_cosmic, bg_cosmic, scores_cosmic = mix(
            w_cosmic, self.m2u_temperatures[1], ~anchor_source)
        delta_joint, beta_joint, bg_joint, scores_joint = mix(
            w_joint, self.m2u_temperatures[2],
            torch.ones(anchor_count, device=coords.device, dtype=torch.bool))
        weights_joint_query = (torch.einsum('ba,an->bn', beta_joint, w_joint)
                               if anchor_count else coords.new_zeros(batch, members))
        self.kalman_layer.last_member_weights = (
            weights_joint_query.abs()
            / weights_joint_query.abs().sum(dim=-1, keepdim=True).clamp_min(1e-6))
        self.kalman_layer.last_inflation_scale = coords.new_ones(())
        latent_increment = torch.einsum(
            'bn,bnd->bd', weights_joint_query, query_X)
        ne_delta = delta_joint.unsqueeze(-1)
        ne_fused = background['ne_bkg'] + ne_delta
        interpolated_system = base.unsqueeze(0).expand(batch, -1, -1).clone()
        if anchor_count:
            interpolated_system = interpolated_system + torch.einsum(
                'ba,ank->bnk', beta_joint, system_joint - base)
        identity_transform = torch.eye(
            members, device=coords.device, dtype=coords.dtype).expand(
                batch, -1, -1)
        anchor_coords_diag = anchor_coords.unsqueeze(0).expand(batch, -1, -1)
        anchor_precision_fy = beta_fy
        anchor_precision_cosmic = beta_cosmic
        zero_obs = coords.new_zeros(batch, anchor_count, members)
        def diag_source(name, beta, terms, covariance, rhs):
            innovation = (terms[2].mean().expand(batch, anchor_count)
                          if terms[2] is not None and terms[2].numel()
                          else coords.new_zeros(batch, anchor_count))
            return {
                f'precision_{name}': beta,
                f'innovation_{name}': innovation,
                f'innov_{name}': innovation,
                f'obs_anomalies_{name}': query_anomalies[:, None, :].expand(
                    -1, anchor_count, -1),
                f'observation_factors_{name}': query_phi[:, None, :].expand(
                    -1, anchor_count, -1),
                f'basis_{name}': query_phi[:, None, :].expand(
                    -1, anchor_count, -1),
                f'ensemble_covariance_{name}': (
                    torch.einsum('ba,ank->bnk', beta, covariance)
                    if anchor_count else coords.new_zeros(batch, members, members)),
                f'ensemble_rhs_{name}': (
                    torch.einsum('ba,an->bn', beta, rhs)
                    if anchor_count else coords.new_zeros(batch, members)),
                f'cross_covariance_{name}': query_anomalies.mean(
                    dim=-1, keepdim=True).expand(-1, anchor_count),
                f'observation_coords_{name}': anchor_coords_diag,
                f'observation_rho_squared_{name}': (
                    (1.0 - beta).clamp_min(0.0).square()),
                f'representativeness_{name}': torch.ones_like(beta),
            }
        extras = {
            'z_analysis': background['z_background'] + latent_increment,
            'latent_increment': latent_increment,
            'latent_anomalies': query_X,
            'analysis_anomalies': query_X,
            'transform': identity_transform,
            'weights_FY': (torch.einsum('ba,an->bn', beta_fy, w_fy)
                          if anchor_count else coords.new_zeros(batch, members)),
            'weights_COSMIC': (torch.einsum('ba,an->bn', beta_cosmic, w_cosmic)
                               if anchor_count else coords.new_zeros(batch, members)),
            'weights_M11': weights_joint_query,
            'query_anomalies': query_anomalies,
            'query_basis': query_phi,
            'system': interpolated_system,
            'inflation': coords.new_ones(()),
            'factor_scales': factor_scales,
            'mode_increments': {
                'M10': delta_fy, 'M01': delta_cosmic, 'M11': delta_joint},
            'delta_FY': delta_fy, 'delta_COSMIC': delta_cosmic,
            'delta_M11': delta_joint,
            'gain_FY': beta_fy, 'gain_COSMIC': beta_cosmic,
            'shared_anchor_coords': anchor_coords_diag,
            'shared_anchor_beta_FY': beta_fy,
            'shared_anchor_beta_COSMIC': beta_cosmic,
            'shared_anchor_beta_M11': beta_joint,
            'shared_anchor_scores_M10': scores_fy,
            'shared_anchor_scores_M01': scores_cosmic,
            'shared_anchor_scores_M11': scores_joint,
            'shared_anchor_background_weight': bg_joint,
            'shared_anchor_count': anchor_count,
        }
        extras.update(diag_source('FY', beta_fy, fy_terms, fy_cov, fy_rhs))
        extras.update(diag_source(
            'COSMIC', beta_cosmic, cosmic_terms, cosmic_cov, cosmic_rhs))
        singular_values = torch.linalg.svdvals(query_X)
        energy = singular_values.square()
        probability = energy / energy.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        extras['anomaly_singular_values'] = singular_values
        extras['anomaly_effective_rank'] = torch.exp(-torch.sum(
            probability * torch.log(probability.clamp_min(1e-12)), dim=-1))
        extras['anomaly_condition'] = (
            singular_values[:, 0] / singular_values[:, -1].clamp_min(1e-12))
        extras['scale_boundary_saturation'] = coords.new_zeros(batch)
        return ne_fused, ne_delta, extras

    def forward(self, coords, sw_seq, precomputed_h_sw=None,
                iri_peak=None, observations_fy=None,
                observations_cosmic=None, analysis_unit_ids=None,
                analysis_unit_context=None, physical_reference_context=None,
                anchor_observations_fy=None,
                anchor_observations_cosmic=None):
        """Decode a joint low-dimensional ETKF analysis into physical log10Ne."""
        B = coords.shape[0]
        endpoint_modes = self.uses_physical_modes and self.physical_mode_dictionary in (
            'endpoint_hmf2_legendre', 'background_adaptive_fixed')
        background = (self.encode_endpoint_context(coords, sw_seq, iri_peak)
                      if endpoint_modes else
                      self.encode_background(
                          coords, sw_seq, precomputed_h_sw, iri_peak))
        ne_bkg = background['ne_bkg']
        ne_iri = background['ne_iri']
        background_residual = background['background_residual']
        f_iri = background['z_background']
        h_sw = background['h_sw']
        lat_n = background['lat_n']
        cos_SZA = background['cos_sza']
        sin_doy = background['sin_doy']
        cos_doy = background['cos_doy']
        sin_I = background['sin_i']
        _iri_peak = background['iri_peak']
        hmF2_det = _iri_peak[:, 0].detach()
        peak_params = {'hmF2': _iri_peak[:, 0], 'NmF2': _iri_peak[:, 1]}

        if self.uses_physical_modes:
            if analysis_unit_ids is not None or analysis_unit_context is not None:
                raise ValueError(
                    'M2-R query-local semantics do not accept analysis-unit inputs')
            return self._physical_mode_forward(
                coords, background, observations_fy, observations_cosmic,
                physical_reference_context)

        if self.uses_shared_anchor_response:
            query_phi = self._density_basis(
                coords, coords[:, None, :4], ne_bkg, f_iri, h_sw,
                f_iri[:, None, :], h_sw[:, None, :]).squeeze(1)
            ne_fused, ne_delta, etkf = self._shared_anchor_forward(
                coords, background, query_phi,
                anchor_observations_fy, anchor_observations_cosmic)
            extras = {
                'ne_bkg': ne_bkg,
                'ne_iri': ne_iri,
                'background_residual': background_residual,
                'ne_residual': ne_delta,
                'h_iri_aligned': f_iri,
                'hmF2_det': hmF2_det,
                'peak_params': peak_params,
                'K_FY': etkf['gain_FY'],
                'K_COSMIC': etkf['gain_COSMIC'],
                'r_fy': self.kalman_layer.r_fy,
                'r_cosmic': self.kalman_layer.r_cosmic,
                'innov_FY': etkf['innov_FY'],
                'innov_COSMIC': etkf['innov_COSMIC'],
                'update_FY': etkf['delta_FY'].unsqueeze(-1),
                'update_COSMIC': etkf['delta_COSMIC'].unsqueeze(-1),
                'member_weights': self.kalman_layer.last_member_weights,
                'inflation_scale': etkf['inflation'],
                'r_ref_FY': self.kalman_layer.r_fy,
                'r_ref_COSMIC': self.kalman_layer.r_cosmic,
                'h_prior': f_iri,
                'h_analysis': etkf['z_analysis'],
                'basis_z_background': f_iri,
                'basis_h_sw': h_sw,
                'query_coords': coords,
            }
            extras.update(etkf)
            extras['ne_bkg'] = ne_bkg
            extras['ne_iri'] = ne_iri
            extras['background_residual'] = background_residual
            return (ne_fused, torch.zeros_like(ne_fused),
                    torch.zeros_like(ne_bkg), ne_delta, extras)

        def prepare(observations):
            if observations is None:
                empty = {
                    'coords': coords.new_zeros(B, 0, 4),
                    'value': coords.new_zeros(B, 0),
                    'background': coords.new_zeros(B, 0),
                    'valid_mask': torch.zeros(
                        B, 0, device=coords.device, dtype=torch.bool),
                    'rho_squared': coords.new_zeros(B, 0),
                }
                return empty, coords.new_zeros(
                    B, 0, self.kalman_layer.d_model)
            required = {
                'coords', 'value', 'background', 'valid_mask',
                'rho_squared'}
            if self.density_basis_semantics == 'endpoint_context_symmetric':
                required.update(('basis_z_background', 'basis_h_sw'))
            missing = required.difference(observations)
            if missing:
                raise ValueError(
                    f'observation payload missing fields: {sorted(missing)}')
            valid = observations['valid_mask']
            count = valid.shape[1]
            query_coords = coords[:, None, :4].expand(-1, count, -1)
            query_background = ne_bkg.expand(-1, count)
            prepared = dict(observations)
            prepared['coords'] = torch.where(
                valid.unsqueeze(-1), observations['coords'], query_coords)
            prepared['background'] = torch.where(
                valid, observations['background'], query_background)
            prepared['value'] = torch.where(
                valid, observations['value'], query_background)
            prepared['rho_squared'] = torch.where(
                valid, observations['rho_squared'],
                torch.ones_like(observations['rho_squared']))
            if 'representativeness_weight' in observations:
                prepared['representativeness_weight'] = torch.where(
                    valid, observations['representativeness_weight'],
                    torch.ones_like(observations[
                        'representativeness_weight']))
            target_z_background = target_h_sw = None
            if self.density_basis_semantics == 'endpoint_context_symmetric':
                target_z_background = torch.where(
                    valid.unsqueeze(-1), observations['basis_z_background'],
                    f_iri[:, None, :].expand(-1, count, -1))
                target_h_sw = torch.where(
                    valid.unsqueeze(-1), observations['basis_h_sw'],
                    h_sw[:, None, :].expand(-1, count, -1))
            phi = self._density_basis(
                coords, prepared['coords'], prepared['background'],
                f_iri, h_sw, target_z_background, target_h_sw)
            phi = phi.masked_fill(~valid.unsqueeze(-1), 0.0)
            return prepared, phi

        query_phi = self._density_basis(
            coords, coords[:, None, :4], ne_bkg, f_iri, h_sw,
            f_iri[:, None, :], h_sw[:, None, :]).squeeze(1)
        fy_obs, phi_fy = prepare(observations_fy)
        cosmic_obs, phi_cosmic = prepare(observations_cosmic)
        etkf = self.kalman_layer(
            f_iri, h_sw, query_phi,
            {'FY': (fy_obs, phi_fy), 'COSMIC': (cosmic_obs, phi_cosmic)},
            lat_n, cos_SZA, sin_doy, cos_doy, sin_I)
        Ne_delta = (
            etkf['delta_FY'] + etkf['delta_COSMIC']).unsqueeze(-1)
        Ne_fused = ne_bkg + Ne_delta
        log_var = torch.zeros_like(Ne_fused)
        ne_placeholder = torch.zeros_like(ne_bkg)

        extras = {
            'ne_bkg':         ne_bkg,
            'ne_iri':         ne_iri,
            'background_residual': background_residual,
            'ne_residual':    Ne_delta,
            'h_iri_aligned':  f_iri,
            'hmF2_det':       hmF2_det,
            'peak_params':    peak_params,
            'K_FY':           etkf['gain_FY'],
            'K_COSMIC':       etkf['gain_COSMIC'],
            'r_fy':           self.kalman_layer.r_fy,
            'r_cosmic':       self.kalman_layer.r_cosmic,
            'innov_FY':       etkf['innovation_FY'],
            'innov_COSMIC':   etkf['innovation_COSMIC'],
            'update_FY':      etkf['delta_FY'].unsqueeze(-1),
            'update_COSMIC':  etkf['delta_COSMIC'].unsqueeze(-1),
            'weights_FY':     etkf['weights_FY'],
            'weights_COSMIC': etkf['weights_COSMIC'],
            'member_weights':   getattr(self.kalman_layer, 'last_member_weights', None),
            'inflation_scale':  getattr(self.kalman_layer, 'last_inflation_scale', None),
            'r_ref_FY':       self.kalman_layer.r_fy,
            'r_ref_COSMIC':   self.kalman_layer.r_cosmic,
            'h_prior':        f_iri,
            'h_analysis':     etkf['z_analysis'],
            'basis_z_background': f_iri,
            'basis_h_sw': h_sw,
            'latent_increment': etkf['latent_increment'],
            'latent_anomalies': etkf['latent_anomalies'],
            'analysis_anomalies': etkf['analysis_anomalies'],
            'etkf_transform': etkf['transform'],
            'obs_anomalies_FY': etkf['obs_anomalies_FY'],
            'obs_anomalies_COSMIC': etkf['obs_anomalies_COSMIC'],
            'precision_FY': etkf['precision_FY'],
            'precision_COSMIC': etkf['precision_COSMIC'],
            'representativeness_FY': etkf['representativeness_FY'],
            'representativeness_COSMIC': etkf[
                'representativeness_COSMIC'],
            'ensemble_covariance_FY': etkf['ensemble_covariance_FY'],
            'ensemble_covariance_COSMIC': etkf['ensemble_covariance_COSMIC'],
            'ensemble_rhs_FY': etkf['ensemble_rhs_FY'],
            'ensemble_rhs_COSMIC': etkf['ensemble_rhs_COSMIC'],
            'cross_covariance_FY': etkf['cross_covariance_FY'],
            'cross_covariance_COSMIC': etkf['cross_covariance_COSMIC'],
            'observation_coords_FY': fy_obs['coords'],
            'observation_coords_COSMIC': cosmic_obs['coords'],
            'observation_rho_squared_FY': fy_obs['rho_squared'],
            'observation_rho_squared_COSMIC': cosmic_obs['rho_squared'],
            'query_coords': coords,
            'query_basis': query_phi,
            'query_anomalies': etkf['query_anomalies'],
            'basis_FY': phi_fy,
            'basis_COSMIC': phi_cosmic,
            'factor_scales': etkf['factor_scales'],
            'anomaly_singular_values': etkf['anomaly_singular_values'],
            'anomaly_effective_rank': etkf['anomaly_effective_rank'],
            'anomaly_condition': etkf['anomaly_condition'],
            'scale_boundary_saturation': etkf['scale_boundary_saturation'],
        }

        return Ne_fused, log_var, ne_placeholder, Ne_delta, extras

    def _initialize_weights(self):
        """Initialize Background conservatively and keep ETKF gradients alive."""
        if not self.uses_physical_modes:
            nn.init.xavier_uniform_(
                self.density_basis_decoder[-1].weight, gain=0.1)
            nn.init.zeros_(self.density_basis_decoder[-1].bias)
        else:
            nn.init.zeros_(self.mode_residual[-1].weight)
            nn.init.zeros_(self.mode_residual[-1].bias)
        nn.init.zeros_(self.background_decoder[-1].weight)
        nn.init.zeros_(self.background_decoder[-1].bias)
        nn.init.zeros_(self.iri_align_net[-1].weight)
        nn.init.zeros_(self.iri_align_net[-1].bias)


if __name__ == '__main__':
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from data_managers.irinc_neural_proxy import IRINeuralProxy

    config = {
        'alt_range': (120.0, 500.0), 'seq_len': 36, 'basis_dim': 64,
        'sw_hidden_dim': 32, 'sw_lstm_layers': 2, 'sw_out_dim': 64,
        'tau_kp_init': 8.0, 'tau_solar_init': 72.0,
        'enkf_n_members': 8, 'enkf_pert_hidden': 64, 'use_sw_freq': True,
    }
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    iri_proxy = IRINeuralProxy(layers=[4, 128, 128, 128, 128, 1]).to(device)
    model = FSIA_INR_Model(iri_proxy, config).to(device)
    B = 8
    coords = torch.tensor(
        [[-12.0, -76.8, 250.0 + i, 48.0] for i in range(B)],
        device=device)
    sw_seq = torch.randn(B, 36, 2, device=device)
    iri_peak = torch.stack([
        torch.full((B,), 300.0, device=device),
        torch.full((B,), 11.5, device=device),
    ], dim=-1)
    m00, _, _, delta00, background = model(coords, sw_seq, iri_peak=iri_peak)
    observation = {
        'coords': coords[:, None, :],
        'value': background['ne_bkg'].detach() + 0.1,
        'background': background['ne_bkg'].detach(),
        'valid_mask': torch.ones(B, 1, dtype=torch.bool, device=device),
        'rho_squared': torch.zeros(B, 1, device=device),
    }
    m10, _, _, delta10, _ = model(
        coords, sw_seq, iri_peak=iri_peak, observations_fy=observation)
    model.zero_grad()
    m10.sum().backward()
    assert torch.equal(m00, background['ne_bkg'])
    assert torch.equal(delta00, torch.zeros_like(delta00))
    assert torch.isfinite(m10).all() and torch.isfinite(delta10).all()
    assert model.density_basis_decoder[-1].weight.grad.abs().sum() > 0
    print('FSIA density-observation ETKF self-test passed')
