import unittest

import numpy as np
import pytest

from run_m00_condition_diagnostic import (
    _harmful_strata,
    harmful_correction_metrics,
    paired_date_bootstrap_harmful,
)


class HarmfulCorrectionTests(unittest.TestCase):
    def test_classification_uses_observation_residual_and_h_threshold(self):
        result = harmful_correction_metrics(
            [.1, .1, 0.0, np.nan], [0.0] * 4,
            [.15, .25, .1, .2], [1, 1, 1, 1])
        # d=.1,c=.15 is useful; d=.1,c=.25 is same-direction harmful;
        # d=0 is near-zero observation residual, but remains H-harmful.
        self.assertEqual(result['useful_point_count'], 1)
        self.assertEqual(result['same_direction_over_point_count'], 1)
        self.assertEqual(result['near_zero_observation_residual_point_count'], 1)
        self.assertEqual(result['harmful_point_count'], 2)
        self.assertEqual(result['invalid_point_count'], 1)

    def test_reverse_harmful_and_useful_classes(self):
        # IRI=0, observed requires +1; M00 moves -1 (harmful/reverse),
        # then +0.5 (useful), and zero is reported separately.
        result = harmful_correction_metrics(
            [1.0, 1.0, 1.0], [0.0] * 3, [-1.0, .5, 0.0], [1, 1, 2])
        self.assertEqual(result['reverse_point_count'], 1)
        self.assertEqual(result['harmful_point_count'], 1)
        self.assertEqual(result['useful_point_count'], 1)
        self.assertEqual(result['near_zero_c_count'], 1)
        self.assertAlmostEqual(result['profile_metrics'][0]['m00_rmse'],
                               np.sqrt(4.25 / 2.0))

    def test_date_bootstrap_weights_profiles_not_dates(self):
        common = {
            'coords': np.asarray([[0, 0, 250, 0], [0, 0, 250, 1],
                                  [0, 0, 250, 24]], dtype=float),
            'observation': np.asarray([.1, .1, .1]),
            'iri': np.zeros(3),
            'profile_id': np.asarray([10, 11, 12]),
        }
        candidate = {**common, 'm00': np.asarray([.25, .15, .25])}
        control = {**common, 'm00': np.asarray([.15, .15, .15])}
        result = paired_date_bootstrap_harmful(
            candidate, control, np.ones(3, dtype=bool), replicates=20, seed=42)
        assert result['estimable']
        # Day 0 has two profiles (harmful fraction .5), day 1 has one (1.0).
        assert result['delta']['harmful_fraction'] == pytest.approx(2 / 3)

    def test_profile_equal_weight_does_not_average_signs_first(self):
        result = harmful_correction_metrics(
            [1, -1, 1, -1], [0, 0, 0, 0], [-1, 1, .5, -.5],
            [10, 10, 20, 20])
        self.assertEqual(result['profile_count'], 2)
        self.assertAlmostEqual(result['reverse_fraction'], 0.5)
        self.assertEqual(result['direction_point_count'], 4)

    def test_mismatched_arrays_fail(self):
        with self.assertRaises(ValueError):
            harmful_correction_metrics([1], [0], [0], [1, 2])

    def test_h_formula_boundaries_and_near_zero_observation_error(self):
        result = harmful_correction_metrics(
            [1.0, 1.0, .02, 0.0], [0.0] * 4,
            [-1.0, 2.1, 1e-6, 1e-4], [1, 2, 3, 4])
        # H is the direct squared-error difference: c^2 - 2*c*d.
        self.assertAlmostEqual(result['mean_H'], np.mean(
            [3.0, 2.1 ** 2 - 2 * 2.1, 1e-12 - 4e-8, 1e-8]))
        self.assertEqual(result['same_direction_over_point_count'], 1)
        self.assertEqual(result['near_zero_observation_error_point_count'], 1)
        self.assertEqual(result['direction_point_count'], 2)

    def test_plan_counterexamples_keep_classes_disjoint(self):
        result = harmful_correction_metrics(
            [.1, .1, 0.0, .1, np.nan, .1],
            [0.0] * 6,
            [.15, .25, .1, 0.0, .25, np.inf],
            [1, 2, 3, 4, 5, 6])
        self.assertEqual(result['useful_point_count'], 1)
        self.assertEqual(result['harmful_point_count'], 2)
        self.assertEqual(result['near_zero_observation_error_point_count'], 1)
        self.assertEqual(result['near_zero_c_count'], 1)
        self.assertEqual(result['invalid_point_count'], 2)

    def test_height_and_local_time_boundaries_are_fixed(self):
        coords = np.asarray([
            [0.0, 0.0, 120.0, 5.999],
            [0.0, 0.0, 200.0, 6.0],
            [0.0, 0.0, 300.0, 17.999],
            [0.0, 0.0, 500.0, 18.0],
        ])
        lt, day, hour, _, height = _harmful_strata(coords)
        self.assertEqual(day.tolist(), ['night', 'day', 'day', 'night'])
        self.assertEqual(hour.tolist(), ['3', '6', '15', '18'])
        self.assertEqual(height.tolist(), ['120-200', '200-300', '300-500', '300-500'])

    def test_date_bootstrap_requires_two_dates_and_is_reproducible(self):
        base = {
            'observation': np.asarray([1., 1., 1., 1.]),
            'iri': np.zeros(4), 'm00': np.asarray([.5, .5, .2, .2]),
            'profile_id': np.asarray([1, 1, 2, 2]),
            'coords': np.asarray([[0., 0., 250., 0.], [0., 0., 250., 1.],
                                  [0., 0., 250., 24.], [0., 0., 250., 25.]])}
        control = dict(base, m00=np.asarray([.4, .4, .3, .3]))
        mask = np.ones(4, dtype=bool)
        result = paired_date_bootstrap_harmful(base, control, mask)
        self.assertTrue(result['estimable'])
        self.assertEqual(result['replicates'], 2000)
        self.assertEqual(result['seed'], 42)
        one_day = dict(base, coords=base['coords'].copy())
        one_day['coords'][:, 3] = 0
        self.assertFalse(paired_date_bootstrap_harmful(
            one_day, dict(control, coords=one_day['coords']), mask)['estimable'])

    def test_log_domain_direction_matches_linear_domain_sign(self):
        log_result = harmful_correction_metrics([1.0], [0.0], [-.5], [1])
        linear_result = harmful_correction_metrics(
            [10.0], [1.0], [10 ** -.5], [1])
        self.assertEqual(log_result['reverse_point_count'],
                         linear_result['reverse_point_count'])


if __name__ == '__main__':
    unittest.main()
