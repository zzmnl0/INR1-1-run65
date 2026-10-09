import tempfile
import unittest
import json
import tempfile
from pathlib import Path
from unittest import mock

import run_m00_low_budget as runner
from isr_evaluation.background_m00_eval import (
    _component_diagnostic_rows,
    _reporting_masks,
    night_v2_acceptance,
)
from inr_modules.mdia.train_fsia import _checkpoint_sha256


class NightV2RunnerTest(unittest.TestCase):
    def test_no_selected_checkpoint_has_no_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'checkpoint.pth'
            path.write_bytes(b'checkpoint')
            self.assertIsNone(_checkpoint_sha256(None))
            self.assertIsNone(_checkpoint_sha256(path, skip=True))
            self.assertIsNotNone(_checkpoint_sha256(path))

    def test_fixed_isr_gate(self):
        rows = []
        for station in ('Jicamarca', 'PokerFlat'):
            for stratum in ('main_200_500_night', 'main_200_500_day',
                            'main_200_300_night', 'main_300_500_night'):
                rows.append({'station': station, 'stratum': stratum,
                             'model': 'M00_A1S', 'rmse': 1.0,
                             'ccc': 0.5, 'bias': 0.1})
                rows.append({'station': station, 'stratum': stratum,
                             'model': 'M00_NIGHT_V2', 'rmse': 0.99,
                             'ccc': 0.51, 'bias': 0.08})
        self.assertTrue(night_v2_acceptance(rows)['passed'])
        rows[1]['ccc'] = 0.504
        self.assertFalse(night_v2_acceptance(rows)['passed'])

    def test_isr_gate_rejects_nonfinite_metric(self):
        rows = []
        for station in ('Jicamarca', 'PokerFlat'):
            for stratum in ('main_200_500_night', 'main_200_500_day',
                            'main_200_300_night', 'main_300_500_night'):
                for model in ('M00_A1S', 'M00_NIGHT_V2'):
                    rows.append({'station': station, 'stratum': stratum,
                                 'model': model, 'rmse': 1.0, 'ccc': 0.5,
                                 'bias': 0.1})
        rows[1]['rmse'] = float('nan')
        self.assertFalse(night_v2_acceptance(rows)['passed'])

    def test_isr_fixed_lt_and_sza_partitions(self):
        masks = dict(_reporting_masks(
            [250, 250], [1, 7], cos_sza=[-0.1, 0.1]))
        self.assertEqual(masks['main_200_300_lt_00_03'].tolist(), [True, False])
        self.assertEqual(masks['main_200_300_sza_night'].tolist(), [True, False])
        self.assertEqual(masks['main_200_300_sza_day'].tolist(), [False, True])

    def test_component_cap_uses_final_residual(self):
        row = next(row for row in _component_diagnostic_rows(
            'Jicamarca', [250], [0], [0.6], [0.6], [0.2], 1.0, 1.0)
                   if row['stratum'] == 'main_200_300_all')
        self.assertEqual(row['total_residual_near_cap_rate'], 0.0)

    def test_budget_counts_events_not_assumptions(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'budget.json'
            with mock.patch.object(runner, 'NIGHT_V2_BUDGET_PATH', path):
                self.assertEqual(runner._night_v2_budget()['training_launches'], 0)
                runner._night_v2_budget(('training_launched', {'run': 'x'}))
                payload = runner._night_v2_budget()
                self.assertEqual(payload['training_launches'], 1)
                self.assertEqual(payload['events'][0]['event'], 'training_launched')

    def test_isr_reuse_is_identity_checked_and_not_rewritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir, output = root / 'run', root / 'isr'
            run_dir.mkdir(); output.mkdir()
            checkpoint = run_dir / 'epoch_01_background_model.pth'
            best = run_dir / 'best_background_model.pth'
            checkpoint.touch(); best.touch()
            canonical, comparison = root / 'canonical.npz', root / 'comparison.npz'
            canonical.touch(); comparison.touch()
            rows = []
            for station in ('Jicamarca', 'PokerFlat'):
                for stratum in ('main_200_500_night', 'main_200_500_day',
                                'main_200_300_night', 'main_300_500_night'):
                    rows += [
                        {'station': station, 'stratum': stratum,
                         'model': 'M00_A1S', 'rmse': 1., 'ccc': .5, 'bias': .1},
                        {'station': station, 'stratum': stratum,
                         'model': 'M00_NIGHT_V2', 'rmse': .99, 'ccc': .51, 'bias': .08},
                    ]
            report = {'checkpoint': str(best.resolve()),
                      'canonical_cache': str(canonical.resolve()),
                      'comparison_cache': str(comparison.resolve()),
                      'candidate_name': 'M00_NIGHT_V2',
                      'reference_name': 'M00_A1S', 'diagnostic_only': False,
                      'row_count': 206230, 'stratified_metrics': rows}
            report_path = output / 'isr_m00_report.json'
            report_path.write_text(json.dumps(report), encoding='utf-8')
            before = report_path.read_bytes()
            development = {'passed': True, 'selected_epoch': 1,
                           'weight_identity': {'valid': True}}
            replay_path = root / 'replay.json'
            replay_path.write_text(json.dumps({'metrics': {}}), encoding='utf-8')
            with (mock.patch.object(runner, 'NIGHT_V2_ISR_OUTPUT', output),
                  mock.patch.object(runner, 'ISR_CACHE', canonical),
                  mock.patch.object(runner, 'ISR_COMPARISON_CACHE', comparison),
                  mock.patch.object(runner, '_manifest_config', return_value={
                      'background_selection_baseline': str(replay_path),
                      'background_night_v2_supported_cells': []}),
                  mock.patch.object(runner, '_load_history', return_value=[]),
                  mock.patch.object(runner, '_night_v2_weight_identity', return_value={'valid': True}),
                  mock.patch('inr_modules.mdia.train_fsia.select_night_generalization_v2',
                             return_value={'eligible': True, 'selected_epoch': 1}),
                  mock.patch('run_m00_condition_diagnostic.cached_night_v2_isr_diagnostic',
                             return_value={'common_valid_count': 206230}),
                  mock.patch.object(runner.torch, 'load',
                                    return_value={'x': runner.torch.tensor(1)})):
                result = runner.run_night_v2_isr(run_dir, development)
            self.assertTrue(result['passed'])
            self.assertEqual(report_path.read_bytes(), before)

    def test_v2_flags_rejected_for_legacy_experiment(self):
        with mock.patch('sys.argv', ['run_m00_low_budget.py',
                                     '--experiment', 'baseline',
                                     '--evaluate-only']):
            with self.assertRaises(SystemExit):
                runner.main()


if __name__ == '__main__':
    unittest.main()


def test_diagnostic_failure_never_launches_training(tmp_path, monkeypatch):
    import run_m00_condition_diagnostic as diagnostic
    monkeypatch.setattr(diagnostic, 'prepare_night_v2_diagnostic',
                        lambda *args, **kwargs: {'can_train': False, 'status': 'diagnostic_not_supported'})
    monkeypatch.setattr(diagnostic, 'cached_night_v2_isr_diagnostic', lambda *args: {})
    monkeypatch.setattr(runner, 'NIGHT_V2_DIAGNOSTIC_DIR', tmp_path)
    monkeypatch.setattr(runner, '_night_v2_budget', lambda *args: {'training_launches': 0})
    monkeypatch.setattr(runner, '_write_night_v2_plan', lambda value: value)
    monkeypatch.setattr(runner, 'run_config', lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError('trained')))
    assert runner.run_seed42_magnetic_daynight_bias_v2()['status'] == 'diagnostic_not_supported'


def test_completed_reuse_dev_failure_never_launches_isr(tmp_path, monkeypatch):
    import json
    import run_m00_condition_diagnostic as diagnostic
    control = tmp_path / 'control'; control.mkdir()
    (control / 'run_manifest.json').write_text(json.dumps({'resolved_training': {
        'resolved_background_stratum_bias_weight': runner.NIGHT_V2_OLD_WEIGHT}}))
    monkeypatch.setattr(runner, 'EXPLICIT_CONDITION_RUN', control)
    monkeypatch.setattr(runner, 'NIGHT_V2_RUN', tmp_path)
    monkeypatch.setattr(runner, 'NIGHT_V2_DIAGNOSTIC_DIR', tmp_path)
    monkeypatch.setattr(diagnostic, 'prepare_night_v2_diagnostic',
                        lambda *args, **kwargs: {'can_train': True})
    monkeypatch.setattr(diagnostic, 'cached_night_v2_isr_diagnostic', lambda *args: {})
    monkeypatch.setattr(runner, '_night_v2_complete', lambda *args: True)
    monkeypatch.setattr(runner, '_manifest_config', lambda *args: {'skip_hash_verification': False})
    monkeypatch.setattr(runner, '_config_mismatches', lambda *args: {})
    monkeypatch.setattr(runner, '_night_v2_budget', lambda *args: {'training_launches': 1})
    monkeypatch.setattr(runner, '_write_night_v2_plan', lambda value: value)
    monkeypatch.setattr(runner, 'evaluate_night_v2', lambda *args: {'passed': False})
    monkeypatch.setattr(runner, 'run_config', lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError('retrained')))
    monkeypatch.setattr(runner, 'run_night_v2_isr', lambda *args: (_ for _ in ()).throw(AssertionError('ISR launched')))
    assert runner.run_seed42_magnetic_daynight_bias_v2()['status'] == 'dev_not_passed'
