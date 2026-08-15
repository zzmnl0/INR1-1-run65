"""P0-A WGS84 line-of-sight geometry regressions without production ISR data."""

from pathlib import Path

import numpy as np
import pytest

import isr_evaluation.isr_loader as loader_module
from isr_evaluation.isr_loader import (
    _select_hdf_files,
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


@pytest.mark.parametrize(
    "loader_name,suffix",
    [("load_jicamarca", ".hdf5"), ("load_poker_flat", ".h5")],
)
def test_explicit_isr_file_list_never_scans_or_opens_unlisted_sibling(
        tmp_path, monkeypatch, loader_name, suffix):
    allowed = tmp_path / f"allowed{suffix}"
    unlisted_broken = tmp_path / f"unlisted_broken{suffix}"
    allowed.write_bytes(b"synthetic allowed placeholder")
    unlisted_broken.write_bytes(b"not an HDF file")
    opened = []
    hashed = []
    sized = []

    def forbidden_glob(_pattern):
        raise AssertionError("explicit file_paths must not scan the directory")

    def failing_hdf_open(path, *_args, **_kwargs):
        opened.append(Path(path).resolve())
        raise OSError("synthetic stop before HDF reads")

    def tracked_hash(path):
        hashed.append(Path(path).resolve())
        raise AssertionError("hash must follow a successful allowed HDF open")

    def tracked_size(path):
        sized.append(Path(path).resolve())
        raise AssertionError("size must follow a successful allowed HDF open")

    monkeypatch.setattr(loader_module.glob, "glob", forbidden_glob)
    monkeypatch.setattr(loader_module.h5py, "File", failing_hdf_open)
    monkeypatch.setattr(loader_module, "_sha256", tracked_hash)
    monkeypatch.setattr(loader_module.os.path, "getsize", tracked_size)
    monkeypatch.setattr(loader_module.traceback, "print_exc", lambda: None)

    loader = getattr(loader_module, loader_name)
    assert loader(
        str(tmp_path), 0.0, 1.0, file_paths=[allowed]) == []
    assert opened == [allowed.resolve()]
    assert unlisted_broken.resolve() not in opened
    assert hashed == []
    assert sized == []


@pytest.mark.parametrize(
    "loader_name,suffix",
    [("load_jicamarca", ".hdf5"), ("load_poker_flat", ".h5")],
)
def test_explicit_empty_isr_file_list_performs_no_discovery_or_io(
        tmp_path, monkeypatch, loader_name, suffix):
    (tmp_path / f"unlisted_broken{suffix}").write_bytes(b"not HDF")
    monkeypatch.setattr(
        loader_module.glob, "glob",
        lambda _pattern: (_ for _ in ()).throw(
            AssertionError("explicit empty list must not glob")))
    monkeypatch.setattr(
        loader_module.h5py, "File",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("explicit empty list must not open HDF")))
    monkeypatch.setattr(
        loader_module, "_sha256",
        lambda _path: (_ for _ in ()).throw(
            AssertionError("explicit empty list must not hash")))
    monkeypatch.setattr(
        loader_module.os.path, "getsize",
        lambda _path: (_ for _ in ()).throw(
            AssertionError("explicit empty list must not stat")))
    loader = getattr(loader_module, loader_name)
    assert loader(str(tmp_path), 0.0, 1.0, file_paths=[]) == []


@pytest.mark.parametrize(
    "loader_name,suffix",
    [("load_jicamarca", ".hdf5"), ("load_poker_flat", ".h5")],
)
def test_strict_explicit_isr_file_errors_are_not_silently_skipped(
        tmp_path, monkeypatch, loader_name, suffix):
    allowed = tmp_path / f"allowed{suffix}"
    allowed.write_bytes(b"synthetic placeholder")
    monkeypatch.setattr(
        loader_module.h5py, "File",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            OSError("synthetic allowed-file failure")))
    loader = getattr(loader_module, loader_name)
    with pytest.raises(OSError, match="allowed-file failure"):
        loader(
            str(tmp_path), 0.0, 1.0, file_paths=[allowed],
            fail_on_file_error=True)


def test_explicit_file_list_is_lexical_and_rejects_ambiguous_containers(
        tmp_path, monkeypatch):
    monkeypatch.setattr(
        loader_module.glob, "glob",
        lambda _pattern: (_ for _ in ()).throw(
            AssertionError("explicit file selection must not glob")))
    selected = _select_hdf_files(
        str(tmp_path), "*.h5", [Path("a.h5"), Path("b.h5")])
    assert selected == sorted([
        str((tmp_path / "a.h5").resolve()),
        str((tmp_path / "b.h5").resolve()),
    ], key=loader_module.os.path.normcase)
    with pytest.raises(TypeError, match="iterable"):
        _select_hdf_files(str(tmp_path), "*.h5", "a.h5")
    with pytest.raises(ValueError, match="duplicate lexical"):
        _select_hdf_files(str(tmp_path), "*.h5", ["a.h5", ".\\a.h5"])
