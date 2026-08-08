"""Paired regression metrics and deterministic grouped bootstrap."""

import numpy as np


def regression_metrics(observation, prediction):
    observation = np.asarray(observation, dtype=np.float64).reshape(-1)
    prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)
    finite = np.isfinite(observation) & np.isfinite(prediction)
    observation, prediction = observation[finite], prediction[finite]
    if observation.size < 2:
        return {'n': int(observation.size), 'rmse': np.nan,
                'pearson_r': np.nan, 'ccc': np.nan}
    error = prediction - observation
    obs_mean, pred_mean = observation.mean(), prediction.mean()
    obs_var, pred_var = observation.var(), prediction.var()
    covariance = np.mean(
        (observation - obs_mean) * (prediction - pred_mean))
    denom_r = np.sqrt(obs_var * pred_var)
    return {
        'n': int(observation.size),
        'rmse': float(np.sqrt(np.mean(error * error))),
        'pearson_r': float(covariance / denom_r) if denom_r > 1e-30 else np.nan,
        'ccc': float(2.0 * covariance / (
            obs_var + pred_var + (obs_mean - pred_mean) ** 2 + 1e-30)),
    }


def paired_group_sufficient_statistics(observation, prediction, baseline,
                                       unit_ids):
    observation = np.asarray(observation, dtype=np.float64).reshape(-1)
    prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)
    baseline = np.asarray(baseline, dtype=np.float64).reshape(-1)
    unit_ids = np.asarray(unit_ids).reshape(-1)
    if not (len(observation) == len(prediction) == len(baseline) == len(unit_ids)):
        raise ValueError('paired bootstrap arrays must have equal lengths')
    finite = (np.isfinite(observation) & np.isfinite(prediction)
              & np.isfinite(baseline))
    observation, prediction, baseline, unit_ids = (
        values[finite] for values in
        (observation, prediction, baseline, unit_ids))
    _, inverse = np.unique(unit_ids, return_inverse=True)
    count = np.bincount(inverse).astype(np.float64)

    def sums(values):
        return np.bincount(inverse, weights=values, minlength=len(count))

    return np.stack([
        count, sums(observation), sums(observation ** 2),
        sums(prediction), sums(prediction ** 2), sums(observation * prediction),
        sums((prediction - observation) ** 2),
        sums(baseline), sums(baseline ** 2), sums(observation * baseline),
        sums((baseline - observation) ** 2),
    ], axis=1)


def _metrics_from_sums(values, candidate=True):
    n, sum_o, sum_o2 = values[:3]
    offset = 3 if candidate else 7
    sum_p, sum_p2, sum_op = values[offset:offset + 3]
    sse = values[6 if candidate else 10]
    mean_o, mean_p = sum_o / n, sum_p / n
    var_o = max(sum_o2 / n - mean_o ** 2, 0.0)
    var_p = max(sum_p2 / n - mean_p ** 2, 0.0)
    covariance = sum_op / n - mean_o * mean_p
    denom_r = np.sqrt(var_o * var_p)
    return np.asarray([
        2.0 * covariance / (var_o + var_p + (mean_o - mean_p) ** 2 + 1e-30),
        np.sqrt(sse / n),
        covariance / denom_r if denom_r > 1e-30 else np.nan,
    ])


def paired_group_bootstrap_from_statistics(stats, replicates=2000, seed=42):
    """Compare paired models from one sufficient-statistics row per unit."""
    stats = np.asarray(stats, dtype=np.float64)
    if stats.ndim != 2 or stats.shape[1] != 11:
        raise ValueError('paired bootstrap statistics must have shape [units, 11]')
    if len(stats) < 2:
        raise ValueError('paired bootstrap requires at least two sampling units')
    rng = np.random.default_rng(seed)
    deltas = np.empty((replicates, 3), dtype=np.float64)
    for index in range(replicates):
        sampled = rng.integers(0, len(stats), size=len(stats))
        total = stats[sampled].sum(axis=0)
        candidate_metrics = _metrics_from_sums(total, True)
        baseline_metrics = _metrics_from_sums(total, False)
        deltas[index] = (
            candidate_metrics[0] - baseline_metrics[0],
            baseline_metrics[1] - candidate_metrics[1],
            candidate_metrics[2] - baseline_metrics[2],
        )
    names = ('delta_ccc', 'delta_rmse_gain', 'delta_pearson_r')
    candidate = _metrics_from_sums(stats.sum(axis=0), True)
    baseline = _metrics_from_sums(stats.sum(axis=0), False)
    estimates = np.asarray([
        candidate[0] - baseline[0],
        baseline[1] - candidate[1],
        candidate[2] - baseline[2],
    ])
    intervals = {
        name: {
            'estimate': float(value),
            'ci95': [float(bound) for bound in np.nanquantile(
                deltas[:, column], [0.025, 0.975])],
        }
        for column, (name, value) in enumerate(zip(names, estimates))
    }
    decision = 'inconclusive'
    decisive_metric = None
    for name in names:
        lower, upper = intervals[name]['ci95']
        if lower > 0.0:
            decision, decisive_metric = 'pass', name
            break
        if upper < 0.0:
            decision, decisive_metric = 'fail', name
            break
    return {
        'sampling_units': int(len(stats)),
        'replicates': int(replicates),
        'seed': int(seed),
        **intervals,
        'decision': decision,
        'decisive_metric': decisive_metric,
    }


def paired_group_bootstrap(observation, prediction, baseline, unit_ids,
                           replicates=2000, seed=42):
    """Compare prediction with baseline using grouped paired resampling."""
    return paired_group_bootstrap_from_statistics(
        paired_group_sufficient_statistics(
            observation, prediction, baseline, unit_ids),
        replicates=replicates, seed=seed)
