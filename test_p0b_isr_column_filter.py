"""P0-B ISR UTC-date column filtering without production density access."""

import datetime as dt

import numpy as np
import pytest

import isr_evaluation.isr_loader as loader_module
import isr_evaluation.audit_m2w2_error_chain as runner


def _unix(value):
    return dt.datetime.fromisoformat(value).replace(
        tzinfo=dt.timezone.utc).timestamp()


def _declared_identity(path, sha256, size_bytes):
    return {
        'path': str(path.resolve()),
        'sha256': sha256,
        'size_bytes': size_bytes,
    }


class _Dataset:
    def __init__(self, values, *, allowed_2d_columns=None):
        self.values = np.asarray(values)
        self.shape = self.values.shape
        self.allowed_2d_columns = allowed_2d_columns
        self.reads = []

    def __getitem__(self, item):
        self.reads.append(item)
        if self.allowed_2d_columns is not None:
            if not isinstance(item, tuple) or len(item) != 2:
                raise AssertionError('restricted 2-D data must be sliced by column')
            rows, columns = item
            if rows != slice(None):
                raise AssertionError('restricted 2-D data must retain its row axis')
            actual = np.asarray(columns, dtype=np.int64)
            np.testing.assert_array_equal(actual, self.allowed_2d_columns)
        return self.values[item]


class _Group:
    def __init__(self, values):
        self.values = values

    def __getitem__(self, key):
        return self.values[key]

    def keys(self):
        return self.values.keys()


class _File(_Group):
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False


def test_cross_midnight_jicamarca_reads_only_allowed_density_column(
        tmp_path, monkeypatch):
    timestamps = np.asarray([
        _unix('2024-09-05T23:55:00'),
        _unix('2024-09-06T00:05:00'),
        _unix('2024-09-07T00:05:00'),
    ])
    allowed_columns = np.asarray([1], dtype=np.int64)
    ne = _Dataset(
        [[1.1e11, 2.2e11, 9.9e99]], allowed_2d_columns=allowed_columns)
    dne = _Dataset(
        [[1.0e10, 1.0e10, 9.9e98]], allowed_2d_columns=allowed_columns)
    layout = _Group({
        'timestamps': _Dataset(timestamps),
        'gdalt': _Dataset([250.0]),
        '2D Parameters/ne': ne,
        '2D Parameters/dne': dne,
        '1D Parameters/gdlatr': _Dataset([-11.95]),
        '1D Parameters/gdlonr': _Dataset([-76.87]),
    })
    fake_hdf = _File({'Data/Array Layout': layout})
    path = tmp_path / 'jro20240905_cross_midnight.hdf5'

    monkeypatch.setattr(loader_module.h5py, 'File', lambda *_a, **_k: fake_hdf)
    monkeypatch.setattr(
        loader_module, '_sha256',
        lambda _path: (_ for _ in ()).throw(
            AssertionError('restricted mixed-date HDF must not be fully hashed')))
    monkeypatch.setattr(loader_module.os.path, 'getsize', lambda _path: 123)
    declared_identity = _declared_identity(path, 'a' * 64, 123)

    records, audit = loader_module.load_jicamarca(
        str(tmp_path), _unix('2024-09-06T00:00:00'),
        _unix('2024-09-06T23:59:59'),
        allowed_dates_utc={'20240906'}, file_paths=[path],
        source_file_identities=[declared_identity],
        fail_on_file_error=True, return_access_audit=True)

    assert len(records) == 1
    np.testing.assert_array_equal(records[0]['ts_1d'], timestamps[[1]])
    np.testing.assert_array_equal(records[0]['ne_2d'], [[2.2e11]])
    assert len(ne.reads) == 1 and len(dne.reads) == 1
    segment = audit['files'][0]['segments'][0]
    assert segment['total_time_columns'] == 3
    assert segment['allowed_time_columns'] == 1
    assert segment['excluded_time_columns'] == 2
    assert segment['allowed_column_spans_inclusive'] == [[1, 1]]
    assert segment['allowed_dates_utc'] == ['20240906']
    assert {row['dataset'] for row in segment['dataset_reads']} == {'ne', 'dne'}
    assert all(row['materialized_dates_utc'] == ['20240906']
               for row in segment['dataset_reads'])
    file_row = audit['files'][0]
    assert file_row['materialized_source_identity'] == declared_identity
    allowed_identity = file_row['materialized_allowed_content_identity']
    assert set(allowed_identity) == {
        'schema', 'path', 'sha256', 'framed_array_count',
        'allowed_time_column_count'}
    assert allowed_identity['schema'] == 'isr_allowed_materialized_content_v1'
    assert allowed_identity['path'] == str(path.resolve())
    assert len(allowed_identity['sha256']) == 64
    assert allowed_identity['framed_array_count'] == 6
    assert allowed_identity['allowed_time_column_count'] == 1
    assert audit['totals'] == {
        'time_columns': 3,
        'allowed_time_columns': 1,
        'excluded_time_columns': 2,
        'density_dataset_column_reads': 2,
        'excluded_density_columns_materialized': 0,
    }


def test_file_without_allowed_dates_never_reads_density(tmp_path, monkeypatch):
    timestamps = np.asarray([
        _unix('2024-09-08T01:00:00'),
        _unix('2024-09-08T02:00:00'),
    ])
    ne = _Dataset([[9.9e99, 9.9e99]], allowed_2d_columns=np.asarray([], dtype=int))
    dne = _Dataset([[9.9e98, 9.9e98]], allowed_2d_columns=np.asarray([], dtype=int))
    layout = _Group({
        'timestamps': _Dataset(timestamps),
        # These must also remain unread because the metadata decision is empty.
        'gdalt': _Dataset([250.0]),
        '2D Parameters/ne': ne,
        '2D Parameters/dne': dne,
        '1D Parameters/gdlatr': _Dataset([-11.95]),
        '1D Parameters/gdlonr': _Dataset([-76.87]),
    })
    fake_hdf = _File({'Data/Array Layout': layout})
    path = tmp_path / 'jro20240908_locked_only.hdf5'
    monkeypatch.setattr(loader_module.h5py, 'File', lambda *_a, **_k: fake_hdf)
    hash_calls = []
    monkeypatch.setattr(
        loader_module, '_sha256',
        lambda value: hash_calls.append(value) or (_ for _ in ()).throw(
            AssertionError('zero-allowed-column file must not be hashed')))
    monkeypatch.setattr(
        loader_module.os.path, 'getsize',
        lambda _path: (_ for _ in ()).throw(
            AssertionError('zero-allowed-column file must not be sized')))
    declared_identity = _declared_identity(path, 'b' * 64, 123)

    records, audit = loader_module.load_jicamarca(
        str(tmp_path), _unix('2024-09-06T00:00:00'),
        _unix('2024-09-06T23:59:59'),
        allowed_dates_utc={'20240906'}, file_paths=[path],
        source_file_identities=[declared_identity],
        fail_on_file_error=True, return_access_audit=True)

    assert records == []
    assert ne.reads == [] and dne.reads == []
    file_row = audit['files'][0]
    assert 'materialized_source_identity' not in file_row
    assert 'materialized_allowed_content_identity' not in file_row
    assert hash_calls == []
    segment = file_row['segments'][0]
    assert segment['allowed_time_columns'] == 0
    assert segment['excluded_time_columns'] == 2
    assert segment['dataset_reads'] == []
    assert audit['totals']['excluded_density_columns_materialized'] == 0


def test_allowed_density_file_keeps_identity_without_final_record(
        tmp_path, monkeypatch):
    timestamps = np.asarray([_unix('2024-09-06T00:05:00')])
    allowed_columns = np.asarray([0], dtype=np.int64)
    layout = _Group({
        'timestamps': _Dataset(timestamps),
        # Outside the requested 120--500 km domain, so no DayRecord survives.
        'gdalt': _Dataset([100.0]),
        '2D Parameters/ne': _Dataset(
            [[2.2e11]], allowed_2d_columns=allowed_columns),
        '2D Parameters/dne': _Dataset(
            [[1.0e10]], allowed_2d_columns=allowed_columns),
        '1D Parameters/gdlatr': _Dataset([-11.95]),
        '1D Parameters/gdlonr': _Dataset([-76.87]),
    })
    fake_hdf = _File({'Data/Array Layout': layout})
    path = tmp_path / 'jro20240906_no_final_record.hdf5'
    monkeypatch.setattr(loader_module.h5py, 'File', lambda *_a, **_k: fake_hdf)
    monkeypatch.setattr(
        loader_module, '_sha256',
        lambda _path: (_ for _ in ()).throw(
            AssertionError('restricted HDF must not be fully hashed')))
    monkeypatch.setattr(loader_module.os.path, 'getsize', lambda _path: 789)
    declared_identity = _declared_identity(path, 'c' * 64, 789)

    records, audit = loader_module.load_jicamarca(
        str(tmp_path), _unix('2024-09-06T00:00:00'),
        _unix('2024-09-06T23:59:59'),
        allowed_dates_utc={'20240906'}, file_paths=[path],
        source_file_identities=[declared_identity],
        fail_on_file_error=True, return_access_audit=True)

    assert records == []
    assert audit['files'][0]['materialized_source_identity'] == declared_identity
    content_identity = audit['files'][0][
        'materialized_allowed_content_identity']
    assert content_identity['framed_array_count'] == 6
    assert content_identity['allowed_time_column_count'] == 1
    assert audit['totals']['allowed_time_columns'] == 1
    assert audit['totals']['density_dataset_column_reads'] == 2


def test_multiday_poker_beam_slices_density_and_coordinate_columns(
        tmp_path, monkeypatch):
    timestamps = np.asarray([
        _unix('2024-09-05T23:55:00'),
        _unix('2024-09-06T00:05:00'),
        _unix('2024-09-07T00:05:00'),
    ])
    allowed_columns = np.asarray([1], dtype=np.int64)
    restricted = {
        'ne': _Dataset(
            [[1.1e11, 2.2e11, 9.9e99]],
            allowed_2d_columns=allowed_columns),
        'dne': _Dataset(
            [[1.0e10, 1.0e10, 9.9e98]],
            allowed_2d_columns=allowed_columns),
        'cgm_lat': _Dataset(
            [[64.0, 65.0, 66.0]], allowed_2d_columns=allowed_columns),
        'cgm_lon': _Dataset(
            [[210.0, 211.0, 212.0]], allowed_2d_columns=allowed_columns),
    }
    beam = _Group({
        'timestamps': _Dataset(timestamps),
        'range': _Dataset([250000.0]),
        '1D Parameters/azm': _Dataset([0.0, 0.0, 0.0]),
        '1D Parameters/elm': _Dataset([90.0, 90.0, 90.0]),
        '1D Parameters/beamid': _Dataset([1.0, 1.0, 1.0]),
        '2D Parameters/ne': restricted['ne'],
        '2D Parameters/dne': restricted['dne'],
        '2D Parameters/cgm_lat': restricted['cgm_lat'],
        '2D Parameters/cgm_long': restricted['cgm_lon'],
    })
    metadata_dtype = np.dtype([('name', 'S64'), ('value', 'S64')])
    metadata = np.asarray([
        (b'Instrument latitude', b'65.13'),
        (b'Instrument longitude', b'-147.471'),
        (b'Instrument altitude', b'0.215'),
    ], dtype=metadata_dtype)
    fake_hdf = _File({
        'Metadata/Experiment Parameters': _Dataset(metadata),
        'Data/Array Layout': _Group({'beam_1': beam}),
    })
    path = tmp_path / 'pfa20240905_cross_midnight.h5'
    monkeypatch.setattr(loader_module.h5py, 'File', lambda *_a, **_k: fake_hdf)
    monkeypatch.setattr(
        loader_module, '_sha256',
        lambda _path: (_ for _ in ()).throw(
            AssertionError('restricted mixed-date HDF must not be fully hashed')))
    monkeypatch.setattr(loader_module.os.path, 'getsize', lambda _path: 456)
    declared_identity = _declared_identity(path, 'b' * 64, 456)

    records, audit = loader_module.load_poker_flat(
        str(tmp_path), _unix('2024-09-06T00:00:00'),
        _unix('2024-09-06T23:59:59'),
        allowed_dates_utc={'20240906'}, file_paths=[path],
        source_file_identities=[declared_identity],
        fail_on_file_error=True, return_access_audit=True)

    assert len(records) == 1
    np.testing.assert_array_equal(records[0]['ts_1d'], timestamps[[1]])
    np.testing.assert_array_equal(records[0]['ne_2d'], [[2.2e11]])
    assert all(len(dataset.reads) == 1 for dataset in restricted.values())
    segment = audit['files'][0]['segments'][0]
    assert segment['allowed_column_spans_inclusive'] == [[1, 1]]
    assert {row['dataset'] for row in segment['dataset_reads']} == {
        'ne', 'dne', 'cgm_lat', 'cgm_lon'}
    assert audit['totals']['density_dataset_column_reads'] == 2
    assert audit['totals']['excluded_density_columns_materialized'] == 0
    assert audit['files'][0]['materialized_source_identity'] == declared_identity
    assert audit['files'][0]['source_identity_semantics'] == (
        'p0a_contract_attested_no_whole_hdf_reread_v1')
    allowed_identity = audit['files'][0][
        'materialized_allowed_content_identity']
    assert set(allowed_identity) == {
        'schema', 'path', 'sha256', 'framed_array_count',
        'allowed_time_column_count'}
    assert allowed_identity['schema'] == 'isr_allowed_materialized_content_v1'
    assert allowed_identity['path'] == str(path.resolve())
    assert len(allowed_identity['sha256']) == 64
    assert allowed_identity['framed_array_count'] == 12
    assert allowed_identity['allowed_time_column_count'] == 1
    assert all(len(row['materialized_payload_sha256']) == 64
               for row in segment['dataset_reads'])

    # Feed the real four-dataset Poker loader ledger to the runner validator.
    jicamarca_path = (tmp_path / 'jro20240908_metadata_only.hdf5').resolve()
    jicamarca_identity = {
        'path': str(jicamarca_path), 'sha256': 'a' * 64, 'size_bytes': 1}
    jicamarca_audit = {
        'isr_column_access_audit_schema_version': 1,
        'station': 'Jicamarca',
        'filter_semantics': (
            'timestamps_metadata_first_then_explicit_2d_column_slice_v1'),
        'allowed_dates_utc': ['20240906'],
        'excluded_date_values_persisted': False,
        'files': [{
            'path': str(jicamarca_path),
            'segments': [{
                'segment_id': 'array_layout',
                'total_time_columns': 1,
                'allowed_time_columns': 0,
                'excluded_time_columns': 1,
                'allowed_column_spans_inclusive': [],
                'allowed_dates_utc': [],
                'dataset_reads': [],
            }],
        }],
        'totals': {
            'time_columns': 1,
            'allowed_time_columns': 0,
            'excluded_time_columns': 1,
            'density_dataset_column_reads': 0,
            'excluded_density_columns_materialized': 0,
        },
    }
    attestation = runner._validate_isr_column_access_audits(
        {'Jicamarca': jicamarca_audit, 'PokerFlat': audit},
        {'Jicamarca': [jicamarca_identity],
         'PokerFlat': [declared_identity]},
        {'20240906'})
    assert attestation['stations']['PokerFlat'][
        'density_dataset_column_reads'] == 2


@pytest.mark.parametrize(
    'loader_name,filename', [
        ('load_jicamarca', 'jro20240906_missing_identity.hdf5'),
        ('load_poker_flat', 'pfa20240906_missing_identity.h5'),
    ])
def test_restricted_loader_rejects_missing_p0a_identity_before_hdf_open(
        tmp_path, monkeypatch, loader_name, filename):
    path = tmp_path / filename
    monkeypatch.setattr(
        loader_module.h5py, 'File',
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError('identity gate must precede HDF open')))
    monkeypatch.setattr(
        loader_module, '_sha256',
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError('identity gate must precede whole-file SHA256')))
    with pytest.raises(ValueError, match='requires P0-A source_file_identities'):
        getattr(loader_module, loader_name)(
            str(tmp_path), _unix('2024-09-06T00:00:00'),
            _unix('2024-09-06T23:59:59'),
            allowed_dates_utc={'20240906'}, file_paths=[path],
            fail_on_file_error=True)


def test_restricted_loader_rejects_declared_size_drift_without_whole_hash(
        tmp_path, monkeypatch):
    timestamps = np.asarray([_unix('2024-09-06T00:05:00')])
    allowed_columns = np.asarray([0], dtype=np.int64)
    layout = _Group({
        'timestamps': _Dataset(timestamps),
        'gdalt': _Dataset([250.0]),
        '2D Parameters/ne': _Dataset(
            [[2.2e11]], allowed_2d_columns=allowed_columns),
        '2D Parameters/dne': _Dataset(
            [[1.0e10]], allowed_2d_columns=allowed_columns),
        '1D Parameters/gdlatr': _Dataset([-11.95]),
        '1D Parameters/gdlonr': _Dataset([-76.87]),
    })
    path = tmp_path / 'jro20240906_size_drift.hdf5'
    monkeypatch.setattr(
        loader_module.h5py, 'File',
        lambda *_args, **_kwargs: _File({'Data/Array Layout': layout}))
    monkeypatch.setattr(
        loader_module, '_sha256',
        lambda _path: (_ for _ in ()).throw(
            AssertionError('restricted HDF must not be fully hashed')))
    monkeypatch.setattr(loader_module.os.path, 'getsize', lambda _path: 124)
    with pytest.raises(ValueError, match='size differs'):
        loader_module.load_jicamarca(
            str(tmp_path), _unix('2024-09-06T00:00:00'),
            _unix('2024-09-06T23:59:59'),
            allowed_dates_utc={'20240906'}, file_paths=[path],
            source_file_identities=[_declared_identity(path, 'd' * 64, 123)],
            fail_on_file_error=True)


def test_guarded_column_reader_rejects_time_axis_mismatch():
    dataset = _Dataset(np.ones((2, 3)))
    segment = {
        'total_time_columns': 4,
        'allowed_dates_utc': ['20240906'],
        'allowed_column_spans_inclusive': [[1, 1]],
        'dataset_reads': [],
    }
    try:
        loader_module._read_allowed_columns(
            dataset, np.asarray([1]), segment, 'ne')
    except ValueError as exc:
        assert 'time axis differs' in str(exc)
    else:
        raise AssertionError('time-axis mismatch must fail before materialization')
    assert dataset.reads == []
