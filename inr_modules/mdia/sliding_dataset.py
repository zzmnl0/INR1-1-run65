"""Assemble FY targets, space-weather history, and local FY/COSMIC profiles."""

import numpy as np
import torch


REPRESENTATIVENESS_SHAPE = (4, 3, 3, 3, 4)
REPRESENTATIVENESS_SOURCE_PAIRS = {
    ('FY', 'FY'): 0,
    ('FY', 'COSMIC'): 1,
    ('COSMIC', 'FY'): 2,
    ('COSMIC', 'COSMIC'): 3,
}


def load_representativeness_kernel(path):
    if not path:
        return None
    with np.load(path, allow_pickle=False) as cells:
        cell_id = cells['cell_id']
        stable = cells['stable']
    expected = np.arange(np.prod(REPRESENTATIVENESS_SHAPE))
    if (cell_id.shape != expected.shape
            or not np.array_equal(cell_id, expected)
            or stable.shape != expected.shape):
        raise ValueError('representativeness cells have an incompatible schema')
    return torch.from_numpy(
        stable.reshape(REPRESENTATIVENESS_SHAPE).astype(np.float32))


def load_empirical_covariance_targets(path):
    """Load frozen train-only covariance targets used only by Analysis loss."""
    if not path:
        return None
    with np.load(path, allow_pickle=False) as cells:
        cell_id = cells['cell_id']
        stable = cells['stable']
        covariance = cells['covariance']
    expected = np.arange(np.prod(REPRESENTATIVENESS_SHAPE))
    if (cell_id.shape != expected.shape
            or not np.array_equal(cell_id, expected)
            or stable.shape != expected.shape
            or covariance.shape != expected.shape
            or not np.isfinite(covariance).all()):
        raise ValueError('empirical covariance cells have an incompatible schema')
    return {
        'stable': torch.from_numpy(
            stable.reshape(REPRESENTATIVENESS_SHAPE).astype(bool)),
        'covariance': torch.from_numpy(
            covariance.reshape(REPRESENTATIVENESS_SHAPE).astype(np.float32)),
    }


def empirical_covariance_token_targets(
        query_coords, observation_coords, rho_squared, valid_mask,
        target_source, observation_source, targets):
    """Map physical query/observation pairs to frozen profile-blocked cells."""
    pair = REPRESENTATIVENESS_SOURCE_PAIRS[
        (target_source, observation_source)]
    device = query_coords.device
    dtype = query_coords.dtype
    boundaries = query_coords.new_tensor([200.0, 300.0])
    target_altitude = torch.bucketize(
        query_coords[:, 2].contiguous(), boundaries, right=True)[:, None]
    observation_altitude = torch.bucketize(
        observation_coords[..., 2].contiguous(), boundaries, right=True)
    target_lt = torch.remainder(
        query_coords[:, 3] + query_coords[:, 1] / 15.0, 24.0)[:, None]
    observation_lt = torch.remainder(
        observation_coords[..., 3] + observation_coords[..., 1] / 15.0,
        24.0)
    target_day = (target_lt >= 6.0) & (target_lt < 18.0)
    observation_day = (
        (observation_lt >= 6.0) & (observation_lt < 18.0))
    local_time_class = torch.where(
        target_day & observation_day,
        torch.ones_like(observation_altitude),
        torch.where(
            ~target_day & ~observation_day,
            torch.zeros_like(observation_altitude),
            torch.full_like(observation_altitude, 2)))
    rho_bin = torch.clamp(
        (torch.sqrt(torch.clamp(rho_squared, min=0.0)) * 4.0).long(),
        max=3)
    stable = targets['stable'].to(device=device)[
        pair, target_altitude, observation_altitude, local_time_class, rho_bin]
    covariance = targets['covariance'].to(device=device, dtype=dtype)[
        pair, target_altitude, observation_altitude, local_time_class, rho_bin]
    return covariance, valid_mask.bool() & stable


def _smooth_day_probability(local_time):
    def smoothstep(value):
        value = value.clamp(0.0, 1.0)
        return value.square() * (3.0 - 2.0 * value)

    rise = smoothstep((local_time - 5.0) / 2.0)
    fall = 1.0 - smoothstep((local_time - 17.0) / 2.0)
    return rise * fall


def _interpolation_brackets(values, centers):
    upper = torch.bucketize(values.contiguous(), centers).clamp(
        1, len(centers) - 1)
    lower = upper - 1
    fraction = ((values - centers[lower])
                / (centers[upper] - centers[lower])).clamp(0.0, 1.0)
    return lower, upper, 1.0 - fraction, fraction


def attach_representativeness_weight(
        payload, query_coords, target_source, observation_source, stable_grid,
        floor=0.25):
    """Attach a continuous frozen precision multiplier without changing masks."""
    if payload is None or stable_grid is None:
        return payload
    if not 0.0 < floor <= 1.0:
        raise ValueError('representativeness floor must be in (0, 1]')
    pair = REPRESENTATIVENESS_SOURCE_PAIRS[
        (target_source, observation_source)]
    valid = payload['valid_mask'].bool()
    query = query_coords[:, :4]
    observation = torch.where(
        valid.unsqueeze(-1), payload['coords'][..., :4], query[:, None, :])
    dtype = payload['value'].dtype
    device = payload['value'].device
    grid = stable_grid.to(device=device, dtype=dtype)
    grid = floor + (1.0 - floor) * grid[pair]

    altitude_centers = query.new_tensor([160.0, 250.0, 400.0])
    rho_centers = query.new_tensor([0.125, 0.375, 0.625, 0.875])
    target_brackets = _interpolation_brackets(
        query[:, 2:3], altitude_centers)
    observation_brackets = _interpolation_brackets(
        observation[..., 2], altitude_centers)
    rho = torch.sqrt(torch.clamp(torch.where(
        valid, payload['rho_squared'], 0.0), min=0.0))
    rho_brackets = _interpolation_brackets(rho, rho_centers)

    target_lt = torch.remainder(query[:, 3] + query[:, 1] / 15.0, 24.0)
    observation_lt = torch.remainder(
        observation[..., 3] + observation[..., 1] / 15.0, 24.0)
    target_day = _smooth_day_probability(target_lt)[:, None]
    observation_day = _smooth_day_probability(observation_lt)
    local_time_weights = torch.stack([
        (1.0 - target_day) * (1.0 - observation_day),
        target_day * observation_day,
        1.0 - (
            (1.0 - target_day) * (1.0 - observation_day)
            + target_day * observation_day),
    ], dim=-1)

    weight = torch.zeros_like(payload['rho_squared'])
    for target_index, target_weight in zip(
            target_brackets[:2], target_brackets[2:]):
        for observation_index, observation_weight in zip(
                observation_brackets[:2], observation_brackets[2:]):
            for rho_index, rho_weight in zip(
                    rho_brackets[:2], rho_brackets[2:]):
                cell = grid[
                    target_index, observation_index, :, rho_index]
                weight = weight + (
                    target_weight * observation_weight * rho_weight
                    * (cell * local_time_weights).sum(dim=-1))
    result = dict(payload)
    result['representativeness_weight'] = torch.where(
        valid, weight, torch.ones_like(weight))
    return result


def query_observation_payload(index, coords, device, exclude_profile_ids=None,
                              allowed_profile_ids=None):
    if index is None:
        return None
    kwargs = {'exclude_profile_ids': exclude_profile_ids}
    if allowed_profile_ids is not None:
        kwargs['allowed_profile_ids'] = allowed_profile_ids
    payload = index.query_observation_batch(
        coords.detach().cpu().numpy(), **kwargs)
    return {
        key: torch.from_numpy(value).to(device, non_blocking=True)
        for key, value in payload.items()
    }


def attach_observation_background(
        payload, model, sw_manager, iri_peak_manager=None, chunk_size=4096):
    """Evaluate shared Background and optional endpoint context at valid tokens."""
    if payload is None:
        return None
    valid = payload['valid_mask']
    background = payload['value'].new_zeros(payload['value'].shape)
    endpoint_context = getattr(
        model, 'density_basis_semantics', None) == 'endpoint_context_symmetric'
    if endpoint_context:
        background_state_dim = getattr(
            model, 'background_state_dim', model.kalman_layer.d_model)
        basis_z_background = payload['value'].new_zeros(
            *payload['value'].shape, background_state_dim)
        basis_h_sw = payload['value'].new_zeros(
            *payload['value'].shape, model.sw_out_dim)
    flat_coords = payload['coords'][valid]
    valid_indices = valid.flatten().nonzero(as_tuple=True)[0]
    flat_background = background.flatten()
    if len(flat_coords):
        unique_coords, inverse = torch.unique(
            flat_coords, dim=0, sorted=False, return_inverse=True)
    else:
        unique_coords, inverse = flat_coords, torch.empty(
            0, device=flat_coords.device, dtype=torch.long)
    unique_background = background.new_empty(len(unique_coords))
    if endpoint_context:
        unique_z_background = basis_z_background.new_empty(
            len(unique_coords), background_state_dim)
        unique_h_sw = basis_h_sw.new_empty(
            len(unique_coords), model.sw_out_dim)
    with torch.no_grad():
        for start in range(0, len(unique_coords), chunk_size):
            coords = unique_coords[start:start + chunk_size]
            sw_seq = sw_manager.get_drivers_sequence(coords[:, 3])
            peak = (iri_peak_manager.get_iri_peak(coords)
                    if iri_peak_manager is not None else None)
            encoded = model.encode_background(coords, sw_seq, iri_peak=peak)
            unique_background[start:start + len(coords)] = (
                encoded['ne_bkg'].flatten())
            if endpoint_context:
                unique_z_background[start:start + len(coords)] = (
                    encoded['z_background'])
                unique_h_sw[start:start + len(coords)] = encoded['h_sw']
    flat_background[valid_indices] = unique_background[inverse]
    result = dict(payload)
    result['background'] = background
    if endpoint_context:
        basis_z_background.reshape(
            -1, model.kalman_layer.d_model)[valid_indices] = (
                unique_z_background[inverse])
        basis_h_sw.reshape(-1, model.sw_out_dim)[valid_indices] = (
            unique_h_sw[inverse])
        result['basis_z_background'] = basis_z_background
        result['basis_h_sw'] = basis_h_sw
    return result


class SlidingWindowBatchProcessor:
    """
    滑动窗口批次处理器

    Receives one FY batch and returns the tensors consumed by FSIA training
    and evaluation.
    """

    def __init__(self, sw_manager, device='cuda',
                 fy_nb_index=None, cosmic_nb_index=None,
                 representativeness_kernel=None,
                 representativeness_floor=0.25,
                 empirical_covariance_targets=None):
        """
        Args:
            sw_manager:       SpaceWeatherManager 实例
            device:           计算设备
            fy_nb_index:      FYNeighborhoodIndex 实例（run61，可选）
            cosmic_nb_index:  COSMICNeighborhoodIndex 实例（run64，可选）
        """
        self.sw_manager = sw_manager
        self.device = device
        self.fy_nb_index = fy_nb_index          # run61: FY 邻域索引
        self.cosmic_nb_index = cosmic_nb_index  # run64: COSMIC 邻域索引
        self.representativeness_kernel = representativeness_kernel
        self.representativeness_floor = float(representativeness_floor)
        self.empirical_covariance_targets = empirical_covariance_targets

    def process_batch(self, batch_item, query_neighbors=True):
        """
        Convert one FY batch and query the local FY/COSMIC profile indexes.

        支持两种输入形式（run65 P2）：
            - 单张量：[Batch, 5 or 7]（backward compat）
            - 元组：  (batch_data [Batch, 5 or 7], batch_ds_idx [Batch])
                      batch_ds_idx 用于 O(1) 预计算邻域查询

        支持两种 FY 数据格式：
            5 列：[Lat_geo, Lon_geo, Alt, Time, Ne_Log]
                  → coords [Batch, 4]（无 AACGM）
            7 列：[Lat_geo, Lon_geo, Alt, Time, Ne_Log, Lat_aacgm, Lon_aacgm]
                  → coords [Batch, 5]（含 AACGM 磁纬，由 add_aacgm_to_fy.py 生成）

        Args:
            batch_item: [Batch, 5/7] Tensor，或 (batch_data, batch_ds_idx) 元组

        Returns:
            coords:    [Batch, 4] 或 [Batch, 5] — (Lat_geo, Lon_geo, Alt, Time[, Lat_aacgm])
            target_ne: [Batch, 1] 真值 Ne（对数）
            sw_seq:    [Batch, Seq, 2] 空间天气序列
        """
        # 解包元组（run65 P2：DataLoader 现返回 (data, ds_idx)）
        if isinstance(batch_item, (tuple, list)):
            if len(batch_item) == 3:
                batch_data, _, profile_ids = batch_item
            else:
                batch_data, _ = batch_item
                profile_ids = torch.arange(len(batch_data), dtype=torch.long)
            profile_ids_np = profile_ids.cpu().numpy()
        else:
            batch_data = batch_item
            profile_ids = torch.arange(len(batch_data), dtype=torch.long)
            profile_ids_np = None

        batch_data = batch_data.to(self.device, non_blocking=True)

        ncols = batch_data.shape[1]

        if ncols >= 7:
            # 7 列：cols [0,1,2,3] = geo coords; col 4 = Ne_Log; cols [5,6] = AACGM
            # 将 AACGM lat (col 5) 追加为 coords 第 5 列
            coords    = torch.cat([batch_data[:, :4], batch_data[:, 5:6]], dim=1)  # [B, 5]
            target_ne = batch_data[:, 4:5]  # [Batch, 1]
        else:
            # 5 列（原始格式）
            coords    = batch_data[:, :4]   # [Batch, 4]
            target_ne = batch_data[:, 4:5]  # [Batch, 1]

        # 查询空间天气序列
        sw_seq = self.sw_manager.get_drivers_sequence(coords[:, 3])  # [Batch, Seq, 2]

        # run61/run65: FY 邻域观测查询（P2：优先用预计算数据）
        if query_neighbors and self.fy_nb_index is not None:
            observations_fy = query_observation_payload(
                self.fy_nb_index, coords, self.device,
                exclude_profile_ids=profile_ids_np)
        else:
            observations_fy = None

        # run64/run65: COSMIC-2 邻域观测查询（P2：优先用预计算数据）
        if query_neighbors and self.cosmic_nb_index is not None:
            observations_cosmic = query_observation_payload(
                self.cosmic_nb_index, coords, self.device)
        else:
            observations_cosmic = None

        return (coords, target_ne, sw_seq,
                observations_fy, observations_cosmic,
                profile_ids.to(self.device, non_blocking=True))
