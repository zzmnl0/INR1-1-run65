"""Select V1/V2 only after paired train-only query-local representation gates."""

import argparse
import json
from pathlib import Path

import numpy as np


def _ci(values, seed=42, draws=2000):
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return None
    rng = np.random.default_rng(seed)
    boot = values[rng.integers(0, len(values), (draws, len(values)))].mean(1)
    return {
        'profiles': int(len(values)),
        'mean_difference': float(values.mean()),
        'bootstrap_95_ci': np.quantile(boot, [0.025, 0.975]).tolist(),
    }


def _paired_rank(candidate, control):
    result = {}
    for target in ('FY', 'COSMIC'):
        result[target] = {}
        for source in ('FY', 'COSMIC', 'joint'):
            candidate_rows = candidate['targets'][target][
                'paired_effective_rank'][source]
            control_rows = control['targets'][target][
                'paired_effective_rank'][source]
            if candidate_rows is None or control_rows is None:
                result[target][source] = None
                continue
            common = sorted(set(candidate_rows['profile_means']).intersection(
                control_rows['profile_means']))
            result[target][source] = _ci([
                candidate_rows['profile_means'][key]
                - control_rows['profile_means'][key] for key in common])
    return result


def _candidate_gate(candidate, control):
    paired = _paired_rank(candidate, control)
    rank = all(
        row is not None and row['bootstrap_95_ci'][0] > 0
        for rows in paired.values() for row in rows.values())
    first_energy = all(
        candidate['targets'][target]['first_mode_proxy_q50'][source]['M2-R']
        <= control['targets'][target]['first_mode_proxy_q50'][source]['M2-R']
        for target in ('FY', 'COSMIC') for source in ('FY', 'COSMIC', 'joint')
        if candidate['targets'][target]['first_mode_proxy_q50'][source]['M2-R']
        is not None)
    strata = True
    for target in ('FY', 'COSMIC'):
        for source in ('FY', 'COSMIC', 'joint'):
            candidate_rows = candidate['targets'][target]['strata'][source]
            control_rows = control['targets'][target]['strata'][source]
            for cell in set(candidate_rows).intersection(control_rows):
                if min(candidate_rows[cell]['queries'], control_rows[cell]['queries']) >= 10:
                    ratio = (candidate_rows[cell]['M2-R_median']
                             / control_rows[cell]['M2-R_median'] - 1.0)
                    strata &= ratio >= -0.05
    required = (
        'reference_geometry_passed', 'posterior_innovation_all_sources_decreased',
        'direction_accuracy_improved', 'covariance_sign_accuracy_improved',
        'squared_error_not_above_M2O', 'profile_paired_better_than_rank1')
    internal = all(candidate['gates'].get(key, False) for key in required)
    return {
        'paired_effective_rank_vs_V0': paired,
        'all_rank_CI_lower_bounds_positive_vs_V0': rank,
        'first_mode_energy_not_above_V0': first_energy,
        'major_strata_not_degraded_over_5pct_vs_V0': strata,
        'all_candidate_gates_passed': rank and first_energy and strata and internal,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--v0', type=Path, required=True)
    parser.add_argument('--v1', type=Path, required=True)
    parser.add_argument('--v2', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    reports = {
        name: json.loads(path.read_text(encoding='utf-8'))
        for name, path in (('V0', args.v0), ('V1', args.v1), ('V2', args.v2))}
    expected = {'V0': 'query_hmf2_legendre',
                'V1': 'endpoint_hmf2_legendre',
                'V2': 'background_adaptive_fixed'}
    for name, dictionary in expected.items():
        if reports[name]['semantics']['dictionary'] != dictionary:
            raise ValueError(f'{name} report has wrong dictionary')
    if len({json.dumps(report['selected_profiles'], sort_keys=True)
            for report in reports.values()}) != 1:
        raise ValueError('V0/V1/V2 reports do not use identical target profiles')
    gates = {name: _candidate_gate(reports[name], reports['V0'])
             for name in ('V1', 'V2')}
    selected = ('V1' if gates['V1']['all_candidate_gates_passed'] else
                'V2_REQUIRES_LOW_NIGHT_ONLY_FAILURE_REVIEW'
                if gates['V2']['all_candidate_gates_passed'] else None)
    output = {
        'schema_version': 1,
        'partition': 'train',
        'locked_test_accessed': False,
        'isr_accessed': False,
        'gates': gates,
        'selected_dictionary': selected,
        'training_authorized': selected == 'V1',
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + '.tmp')
    temporary.write_text(
        json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False) + '\n',
        encoding='utf-8')
    temporary.replace(args.output)


if __name__ == '__main__':
    main()
