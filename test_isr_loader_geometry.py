"""P0-A WGS84 line-of-sight geometry regressions without production ISR data."""

import numpy as np
import pytest

from isr_evaluation.isr_loader import (
    _validate_beam_geometry,
    _wgs84_los_to_geodetic,
)


def test_vertical_poker_beam_adds_station_height_without_horizontal_shift():
    lat, lon, altitude = _wgs84_los_to_geodetic(
        65.13, -147.471, 0.215, np.array([105178.203125]), 14.04, 90.0)
    np.testing.assert_allclose(lat, [65.13], atol=1e-7)
    np.testing.assert_allclose(lon, [-147.471], atol=1e-7)
    np.testing.assert_allclose(altitude, [105.393203125], atol=2e-5)


def test_slant_beam_produces_finite_wgs84_coordinates():
    lat, lon, altitude = _wgs84_los_to_geodetic(
        65.13, -147.471, 0.215, np.array([200000.0]), 90.0, 45.0)
    assert np.isfinite(lat[0]) and np.isfinite(lon[0]) and np.isfinite(altitude[0])
    assert not np.isclose(lat[0], 65.13)
    assert lon[0] > -147.471
    assert altitude[0] > 0.215


class _Beam:
    def __init__(self, azimuth, elevation, beam_id, ranges):
        self.values = {
            '1D Parameters/azm': np.asarray(azimuth, dtype=float),
            '1D Parameters/elm': np.asarray(elevation, dtype=float),
            '1D Parameters/beamid': np.asarray(beam_id, dtype=float),
            'range': np.asarray(ranges, dtype=float),
        }

    def __getitem__(self, key):
        class _Dataset:
            def __init__(self, values):
                self.values = values

            def __getitem__(self, item):
                return self.values[item]

        return _Dataset(self.values[key])


def test_time_varying_beam_or_unknown_range_scale_is_rejected():
    station = {'latitude_deg': 65.13, 'longitude_deg': -147.471, 'altitude_km': 0.215}
    moving = _Beam([0.0, 1.0], [90.0, 90.0], [1, 1], [100000.0, 120000.0])
    with pytest.raises(ValueError, match='varies'):
        _validate_beam_geometry(moving, station)
    implausible = _Beam([0.0], [90.0], [1], [100.0])
    with pytest.raises(ValueError, match='meter-to-kilometre'):
        _validate_beam_geometry(implausible, station)
