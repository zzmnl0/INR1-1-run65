"""Focused P0-A regression tests for the common ISR/GIRO peak QA contract."""

import numpy as np

from isr_evaluation.peak_qa import search_peak_profile


def _parabola(altitude, peak=300.0, curvature=0.001):
    return 12.0 - curvature * (np.asarray(altitude) - peak) ** 2


def test_regular_jicamarca_like_profile_has_valid_peak():
    altitude = np.arange(200.0, 501.0, 15.0)
    result = search_peak_profile(altitude, _parabola(altitude, 305.0))
    assert result.status == 'valid'
    assert result.hmf2_valid and result.nmf2_valid
    assert abs(result.hmf2_km - 305.0) <= 1.0


def test_peak_contract_rejects_insufficient_levels_and_peak_gap():
    short_altitude = np.array([200.0, 250.0, 300.0, 350.0])
    assert search_peak_profile(short_altitude, _parabola(short_altitude)).status == 'insufficient_levels'

    altitude = np.array([
        200., 210., 220., 230., 240., 250., 260., 270., 280., 290.,
        320., 330., 340., 350., 360., 370., 380., 390., 400., 410.,
        420., 430., 440., 450., 460., 470., 480., 490., 500.,
    ])
    assert search_peak_profile(altitude, _parabola(altitude)).status == 'gap_at_peak'


def test_boundary_flat_and_multipeak_statuses_are_explicit():
    altitude = np.arange(200.0, 501.0, 10.0)
    assert search_peak_profile(altitude, _parabola(altitude, 200.0)).status == 'lower_censored'

    flat = search_peak_profile(altitude, _parabola(altitude, 300.0, 1e-5))
    assert flat.status == 'flat_peak'
    assert flat.nmf2_valid and not flat.hmf2_valid

    first = 12.0 - 0.003 * (altitude - 260.0) ** 2
    second = 11.98 - 0.003 * (altitude - 340.0) ** 2
    double = search_peak_profile(altitude, np.maximum(first, second))
    assert double.status == 'ambiguous_multipeak'
    assert double.nmf2_valid and not double.hmf2_valid


def test_no_extrapolation_rejects_profile_without_two_sided_support():
    altitude = np.arange(260.0, 501.0, 10.0)
    result = search_peak_profile(altitude, _parabola(altitude, 275.0))
    assert result.status in {'lower_censored', 'insufficient_bracketing'}
