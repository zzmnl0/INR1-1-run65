"""Pure/synthetic regressions for the read-only M2-W2 P0-B runner."""

from __future__ import annotations

from copy import deepcopy
import datetime as dt
import hashlib
import inspect
import json
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

import isr_evaluation.audit_m2w2_error_chain as runner
import isr_evaluation.isr_loader as isr_loader
from inr_modules.mdia.p0b_audit import load_p0b_contract


ROOT = Path(__file__).resolve().parent
CONTRACT_PATH = ROOT / "m2w2_contracts" / "p0b_audit_contract_v2.json"


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _selection(
        count: int, *, station: str = "Jicamarca",
        query_split: str = "train", date_utc: str = "20240902",
        altitudes: np.ndarray | None = None) -> runner.RawSelection:
    integers = np.arange(count, dtype=np.int64)
    floats = np.arange(count, dtype=np.float32)
    altitude_values = (120.0 + floats if altitudes is None
                       else np.asarray(altitudes, dtype=np.float32))
    if altitude_values.shape != (count,):
        raise ValueError("synthetic altitudes must match selection count")
    return runner.RawSelection(
        station=station,
        date_utc=date_utc,
        query_split=query_split,
        keys=np.asarray([f"key-{value}" for value in integers], dtype="U64"),
        query_ids=integers,
        timestamps=integers,
        latitudes=floats,
        longitudes=floats,
        altitudes=altitude_values,
        relative_hours=floats,
        aacgm_latitudes=floats,
        aacgm_mlt_hours=np.mod(floats, 24.0),
        observations_log10=9.0 + floats,
    )


def test_runner_schema_is_exactly_the_frozen_cache_schema():
    contract, _ = load_p0b_contract(CONTRACT_PATH)
    runner._validate_runner_schemas(contract)


def test_aacgm_coordinate_contract_and_query_conversion_are_explicit():
    contract, _ = load_p0b_contract(CONTRACT_PATH)
    status = runner._coordinate_runtime_status(contract)
    assert status["aacgm"] == {
        "status": "computed",
        "package": "aacgmv2",
        "version": "2.7.0",
        "method": "ALLOWTRACE",
        "fields": ["aacgm_latitude_deg", "aacgm_mlt_hour"],
    }
    assert status["qd"]["status"] == "unavailable"
    assert status["qd"]["proxy_substitution_used"] is False
    timestamp = int(dt.datetime(
        2024, 9, 3, 3, tzinfo=dt.timezone.utc).timestamp())
    magnetic_latitude, magnetic_mlt = runner._compute_aacgm_query_coordinates(
        np.asarray([-12.0, 65.13]), np.asarray([-77.0, -147.471]),
        np.asarray([300.0, 300.0]), np.asarray([timestamp, timestamp]))
    assert magnetic_latitude.dtype == np.float32
    assert magnetic_mlt.dtype == np.float32
    assert np.isfinite(magnetic_latitude).all()
    assert np.all((magnetic_mlt >= 0.0) & (magnetic_mlt < 24.0))


def test_runner_reads_frozen_counts_from_data_scope():
    contract, _ = load_p0b_contract(CONTRACT_PATH)
    assert runner._contract_expected_profile_counts(contract) == {
        "train": {"FY": 44_625, "COSMIC": 56_099},
        "development": {"FY": 11_591, "COSMIC": 13_924},
    }
    assert runner._contract_expected_query_counts(contract) == {
        "train": 141_223,
        "development": 34_438,
    }


def test_p0a_json_identity_is_checked_before_and_after_parse(tmp_path, monkeypatch):
    checkpoint_sha = "a" * 64
    date_split_sha = "b" * 64
    contract_path = tmp_path / "isr_evaluation_contract.json"
    contract_path.write_text(json.dumps({
        "evaluation_schema_version": 2,
        "token_partitions": ["train", "development"],
        "candidate_checkpoint": {
            "sha256": checkpoint_sha,
            "checkpoint_format_version": 14,
            "model_domain_semantics": runner.EXPECTED_DOMAIN,
            "model_alt_range_km": list(runner.EXPECTED_MODEL_RANGE),
            "observation_alt_range_km": list(runner.EXPECTED_OBSERVATION_RANGE),
            "peak_search_alt_range_km": list(runner.EXPECTED_OBSERVATION_RANGE),
            "date_split": {"sha256": date_split_sha},
        },
        "isr_input_files": [],
    }), encoding="utf-8")
    expected = {
        "sha256": _digest(contract_path),
        "size_bytes": contract_path.stat().st_size,
    }
    real_identity = runner.artifact_identity
    stages = []

    def recording_identity(path):
        stages.append(Path(path))
        return real_identity(path)

    monkeypatch.setattr(runner, "artifact_identity", recording_identity)
    identity = runner._load_p0a_identity(
        contract_path, checkpoint_sha, date_split_sha, expected)
    assert len(stages) == 2
    assert identity.sha256 == expected["sha256"]
    assert identity.size_bytes == expected["size_bytes"]

    with pytest.raises(ValueError, match="before parse"):
        runner._load_p0a_identity(
            contract_path, checkpoint_sha, date_split_sha,
            {**expected, "sha256": "c" * 64})


def test_npz_shard_publication_never_overwrites_existing_path(tmp_path):
    path = tmp_path / "batch_000000_query.npz"
    runner._atomic_savez(path, {
        "query_id": np.asarray([1], dtype=np.int64),
        "value": np.asarray([2.0], dtype=np.float32),
    })
    digest_before = _digest(path)
    with pytest.raises(FileExistsError, match="already exists"):
        runner._atomic_savez(path, {
            "query_id": np.asarray([9], dtype=np.int64),
            "value": np.asarray([8.0], dtype=np.float32),
        })
    assert _digest(path) == digest_before
    with np.load(path, allow_pickle=False) as payload:
        assert payload["query_id"].tolist() == [1]


def test_runtime_indexes_force_strict_exact_token_preload(monkeypatch):
    import inr_modules.data_managers.FY_dataloader as fy_module
    import inr_modules.data_managers.iri_peak_manager as iri_module
    import inr_modules.data_managers.space_weather_manager as sw_module

    captured = {}

    class FakeIndex:
        def __init__(self, path, config):
            source = "FY" if "fy" in path.lower() else "COSMIC"
            captured[source] = dict(config)

    class FakeManager:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(fy_module, "FYNeighborhoodIndex", FakeIndex)
    monkeypatch.setattr(fy_module, "COSMICNeighborhoodIndex", FakeIndex)
    monkeypatch.setattr(iri_module, "IRIPeakManager", FakeManager)
    monkeypatch.setattr(sw_module, "SpaceWeatherManager", FakeManager)
    allowlists = {
        "FY": np.asarray([1, 2], dtype=np.int64),
        "COSMIC": np.asarray([3, 4], dtype=np.int64),
    }
    runner._load_runtime_managers({
        "sw_path": "sw.txt",
        "start_date_str": "2024-09-01 00:00:00",
        "total_hours": 24,
        "seq_len": 3,
        "iri_hmf2_path": "hmf2.npy",
        "iri_nmf2_path": "nmf2.npy",
        "fy_path": "fy.npy",
        "cosmic_path": "cosmic.npy",
    }, torch.device("cpu"), allowlists)
    for source in ("FY", "COSMIC"):
        config = captured[source]
        assert config["strict_preload_token_only"] is True
        assert config["neighbor_directory_semantics"] == (
            "token_exact_positive_support_v1")
        np.testing.assert_array_equal(
            config["strict_preload_allowed_profile_ids"], allowlists[source])


def test_preflight_constructs_and_validates_runtime_indexes_before_return_gate():
    source = inspect.getsource(runner._run)
    preflight_gate = source.index("if args.preflight_only:")
    assert source.index("_load_runtime_managers(") < preflight_gate
    assert source.index("_validate_loaded_train_only_indexes(") < preflight_gate
    assert source.index('manifest["satellite_index_preflight"]') < preflight_gate
    preflight_body = source[preflight_gate:]
    assert preflight_body.index("_run_preflight_inference_suite(") < (
        preflight_body.index("checkpoint_after = sha256_file(checkpoint)"))
    assert '"read_only_model_inference_executed": True' in preflight_body
    assert '"stratified_probe_count": preflight_summary["probe_count"]' in (
        preflight_body)


def test_preflight_inference_runs_one_capped_batch_without_persistence(monkeypatch):
    selection = _selection(300)
    captured = {}

    def fake_infer(
            model, sw_manager, iri_peak_manager, indexes, allowlists,
            frozen_contract, actual_selection, sl, batch_id, device,
            next_token_row_id):
        captured.update({
            "selection": actual_selection,
            "slice": sl,
            "batch_id": batch_id,
            "next_token_row_id": next_token_row_id,
        })
        return (
            {
                "query_id": np.arange(sl.stop - sl.start, dtype=np.int64),
                "raw_coverage_code": np.full(
                    sl.stop - sl.start, "joint", dtype="U12"),
                "coverage_code": np.full(
                    sl.stop - sl.start, "joint", dtype="U12"),
                "FY_dominant_profile_valid": np.ones(
                    sl.stop - sl.start, dtype=bool),
                "COSMIC_dominant_profile_valid": np.ones(
                    sl.stop - sl.start, dtype=bool),
            },
            {
                "token_row_id": np.arange(5, dtype=np.int64),
                "source": np.asarray(
                    ["FY", "FY", "COSMIC", "COSMIC", "FY"], dtype="U6"),
                "altitude_km": np.asarray(
                    [225.0, 275.0, 350.0, 450.0, 300.0], dtype=np.float32),
            },
            {
                "query_id": np.arange(9, dtype=np.int64),
                "token_row_id": np.asarray(
                    [0, 1, 2, 3, 4, 0, 1, 2, 3], dtype=np.int64),
                "source": np.asarray(
                    ["FY", "FY", "COSMIC", "COSMIC", "FY",
                     "FY", "FY", "COSMIC", "COSMIC"], dtype="U6"),
                "localized_precision": np.ones(9, dtype=np.float32),
            },
            5,
            {
                "query_closure_max_abs_error": 1e-7,
                "edge_closure_max_abs_error": 2e-7,
                "recomputed_summary_max_abs_error": 3e-7,
            },
        )

    monkeypatch.setattr(runner, "_infer_batch", fake_infer)
    contract, _ = load_p0b_contract(CONTRACT_PATH)
    summary = runner._run_preflight_inference_batch(
        object(), object(), object(), {}, {}, contract, selection, 256,
        torch.device("cpu"))
    assert captured["selection"] is selection
    assert captured["slice"] == slice(0, 256)
    assert captured["batch_id"] == 0
    assert captured["next_token_row_id"] == 0
    assert summary["query_rows"] == 256
    assert summary["token_rows"] == 5
    assert summary["edge_rows"] == 9
    assert summary["npz_written"] is False
    assert summary["numeric_predictions_persisted"] is False
    assert summary["recomputed_summary_max_abs_error"] == pytest.approx(3e-7)
    assert summary["positive_localized_precision_edges_by_source"] == {
        "FY": 5, "COSMIC": 4}
    assert summary["effective_joint_queries"] == 256
    assert summary[
        "positive_localized_precision_edges_by_height_deletion_band"] == {
            "drop_200_250": 2,
            "drop_250_300": 2,
            "drop_300_400": 3,
            "drop_400_500": 2,
        }


def _stratified_preflight_selections():
    altitudes = np.asarray([120.0, 180.0, 200.0, 260.0, 300.0, 500.0])
    return [
        _selection(
            len(altitudes), station=station, query_split=query_split,
            date_utc=("20240902" if query_split == "train" else "20240904"),
            altitudes=altitudes)
        for station in ("Jicamarca", "PokerFlat")
        for query_split in ("train", "development")
    ]


def test_preflight_suite_covers_station_split_and_three_altitude_bands(monkeypatch):
    contract, _ = load_p0b_contract(CONTRACT_PATH)
    calls = []

    def fake_batch(
            model, sw_manager, iri_peak_manager, indexes, allowlists,
            frozen_contract, selection, batch_size, device):
        calls.append((
            selection.station, selection.query_split,
            selection.altitudes.copy(), batch_size))
        return {
            "status": "full_chain_batch_pass_no_numeric_persistence",
            "station": selection.station,
            "date_utc": selection.date_utc,
            "query_split": selection.query_split,
            "query_rows": selection.count,
            "positive_localized_precision_edges_by_source": {
                "FY": 1, "COSMIC": 1},
            "effective_joint_queries": 1,
            "valid_dominant_profile_queries_by_source": {
                "FY": 1, "COSMIC": 1},
            "positive_localized_precision_edges_by_height_deletion_band": {
                "drop_200_250": 1,
                "drop_250_300": 1,
                "drop_300_400": 1,
                "drop_400_500": 1,
            },
            "npz_written": False,
            "numeric_predictions_persisted": False,
        }

    monkeypatch.setattr(runner, "_run_preflight_inference_batch", fake_batch)
    result = runner._run_preflight_inference_suite(
        object(), object(), object(), {}, {}, contract,
        _stratified_preflight_selections(), torch.device("cpu"))

    assert result["probe_count"] == 12
    assert len(calls) == 12
    assert {(station, split) for station, split, _, _ in calls} == {
        (station, split)
        for station in ("Jicamarca", "PokerFlat")
        for split in ("train", "development")
    }
    observed_cells = {
        (probe["station"], probe["query_split"],
         tuple(probe["altitude_band_km"]))
        for probe in result["probes"]
    }
    assert len(observed_cells) == 12
    assert all(batch_size == 32 for _, _, _, batch_size in calls)
    for offset in range(0, len(calls), 3):
        assert calls[offset][2].tolist() == [120.0, 180.0]
        assert calls[offset + 1][2].tolist() == [200.0, 260.0]
        assert calls[offset + 2][2].tolist() == [300.0, 500.0]
    assert result["numeric_predictions_persisted"] is False
    assert result["npz_written"] is False
    assert result["activation"]["status"] == "pass"
    assert result["activation"]["observed"][
        "effective_joint_queries"] == 12


def test_preflight_suite_stops_immediately_on_probe_failure(monkeypatch):
    contract, _ = load_p0b_contract(CONTRACT_PATH)
    calls = []

    def fail_on_fifth_probe(
            model, sw_manager, iri_peak_manager, indexes, allowlists,
            frozen_contract, selection, batch_size, device):
        calls.append((selection.station, selection.query_split))
        if len(calls) == 5:
            raise RuntimeError("synthetic preflight inference failure")
        return {
            "status": "full_chain_batch_pass_no_numeric_persistence",
            "station": selection.station,
            "date_utc": selection.date_utc,
            "query_split": selection.query_split,
            "query_rows": selection.count,
            "npz_written": False,
            "numeric_predictions_persisted": False,
        }

    monkeypatch.setattr(
        runner, "_run_preflight_inference_batch", fail_on_fifth_probe)
    with pytest.raises(RuntimeError, match="synthetic preflight inference failure"):
        runner._run_preflight_inference_suite(
            object(), object(), object(), {}, {}, contract,
            _stratified_preflight_selections(), torch.device("cpu"))
    assert len(calls) == 5


def test_preflight_suite_rejects_unactivated_source_and_counterfactual_paths(
        monkeypatch):
    contract, _ = load_p0b_contract(CONTRACT_PATH)

    def incomplete_batch(
            model, sw_manager, iri_peak_manager, indexes, allowlists,
            frozen_contract, selection, batch_size, device):
        return {
            "status": "full_chain_batch_pass_no_numeric_persistence",
            "station": selection.station,
            "date_utc": selection.date_utc,
            "query_split": selection.query_split,
            "query_rows": selection.count,
            "positive_localized_precision_edges_by_source": {
                "FY": 1, "COSMIC": 0},
            "effective_joint_queries": 0,
            "valid_dominant_profile_queries_by_source": {
                "FY": 1, "COSMIC": 0},
            "positive_localized_precision_edges_by_height_deletion_band": {
                "drop_200_250": 1,
                "drop_250_300": 1,
                "drop_300_400": 0,
                "drop_400_500": 1,
            },
            "npz_written": False,
            "numeric_predictions_persisted": False,
        }

    monkeypatch.setattr(
        runner, "_run_preflight_inference_batch", incomplete_batch)
    with pytest.raises(ValueError, match="did not activate every frozen"):
        runner._run_preflight_inference_suite(
            object(), object(), object(), {}, {}, contract,
            _stratified_preflight_selections(), torch.device("cpu"))


def test_preflight_cli_help_declares_read_only_model_inference():
    help_text = " ".join(runner.build_parser().format_help().lower().split())
    assert "read-only model-inference preflight" in help_text
    assert "without model inference" not in help_text


@pytest.mark.parametrize("field", [
    "untracked_generated_artifacts_allowed",
    "explicit_path_staging_only",
    "git_add_all_forbidden",
    "accepted_history_rewrite_forbidden",
])
def test_runner_rejects_version_control_policy_drift(field):
    contract, _ = load_p0b_contract(CONTRACT_PATH)
    changed = deepcopy(contract)
    changed["version_control"][field] = False
    with pytest.raises(ValueError):
        runner._validate_version_control_contract(
            changed, runner.EXPECTED_IMPLEMENTATION_TAG)


def test_runtime_rejects_tag_object_identity_change(monkeypatch):
    initial = {
        "head": "a" * 40,
        "branch": runner.EXPECTED_IMPLEMENTATION_BRANCH,
        "implementation_tag_object_sha": "b" * 40,
        "implementation_tag_commit": "a" * 40,
    }
    changed = dict(initial, implementation_tag_object_sha="c" * 40)
    monkeypatch.setattr(runner, "_git_provenance", lambda **kwargs: changed)
    with pytest.raises(ValueError, match="tag_object"):
        runner._assert_git_provenance_stable(
            initial, runner.EXPECTED_IMPLEMENTATION_TAG)


def test_runner_git_provenance_uses_shared_machine_status(monkeypatch):
    calls = []
    head = "a" * 40
    tag_object = "b" * 40

    def machine_status(root):
        calls.append(root)
        return []

    def fake_run(arguments, **kwargs):
        command = tuple(arguments[2:])
        outputs = {
            ("rev-parse", "HEAD"): head,
            ("branch", "--show-current"): runner.EXPECTED_IMPLEMENTATION_BRANCH,
            ("rev-parse", f"{runner.EXPECTED_IMPLEMENTATION_TAG}^{{commit}}"): head,
            ("rev-parse", f"{runner.EXPECTED_IMPLEMENTATION_TAG}^{{tag}}"): tag_object,
            ("cat-file", "-t", runner.EXPECTED_IMPLEMENTATION_TAG): "tag",
        }
        return type("Completed", (), {
            "stdout": outputs.get(command, ""), "stderr": "", "returncode": 0})()

    monkeypatch.setattr(runner, "tracked_git_status", machine_status)
    monkeypatch.setattr(runner.shutil, "which", lambda name: "rtk")
    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    provenance = runner._git_provenance(enforce=True)
    assert provenance["tracked_status"] == []
    assert calls == [runner.ROOT]


def test_batch_ids_remain_unique_across_multiple_records_for_one_station_day():
    first = _selection(3)
    second = _selection(2)
    identities = [
        (item.station, item.date_utc, batch_id, sl.start, sl.stop)
        for item, batch_id, sl in runner._iter_scope_batches(
            [first, second], batch_size=2)
    ]
    assert [value[2] for value in identities] == [0, 1, 2]
    assert len({value[:3] for value in identities}) == len(identities)


def test_isr_contract_candidates_are_not_excluded_by_filename_date(
        tmp_path, monkeypatch):
    jicamarca = tmp_path / "jicamarca"
    poker = tmp_path / "poker"
    jicamarca.mkdir()
    poker.mkdir()
    allowed_jro = jicamarca / "jro20240902_000000.hdf5"
    allowed_pfa = poker / "pfa20240904.001_lp_fit_05min.001.h5"
    allowed_jro.write_bytes(b"allowed Jicamarca")
    allowed_pfa.write_bytes(b"allowed Poker Flat")

    # These files deliberately do not exist.  Selection may validate only the
    # contract/path syntax; timestamp metadata in the loader decides columns.
    locked_jro = jicamarca / "jro20240905_locked.hdf5"
    out_of_range_pfa = poker / "pfa20240831.out_of_range.h5"
    contract = {
        "isr_input_files": [
            {
                "path": str(allowed_jro),
                "sha256": _digest(allowed_jro),
                "size_bytes": allowed_jro.stat().st_size,
            },
            {
                "path": str(allowed_pfa),
                "sha256": _digest(allowed_pfa),
                "size_bytes": allowed_pfa.stat().st_size,
            },
            {"path": str(locked_jro), "sha256": "a" * 64, "size_bytes": 1},
            {"path": str(out_of_range_pfa), "sha256": "b" * 64,
             "size_bytes": 1},
        ]
    }
    monkeypatch.setattr(
        runner, "sha256_file",
        lambda path: (_ for _ in ()).throw(
            AssertionError("candidate selection must not hash HDF bytes")))
    selected = runner._select_allowed_isr_files(
        contract,
        dt.datetime(2024, 9, 1, tzinfo=dt.timezone.utc),
        {"train": [1], "development": [3], "locked_test": [4]},
        jicamarca,
        poker,
    )
    assert [Path(row["path"]) for row in selected["Jicamarca"]] == sorted(
        [allowed_jro.resolve(), locked_jro.resolve()])
    assert [Path(row["path"]) for row in selected["PokerFlat"]] == sorted(
        [out_of_range_pfa.resolve(), allowed_pfa.resolve()])


def test_full_cli_requires_explicit_preflight_acceptance(monkeypatch):
    called = []
    monkeypatch.setattr(runner, "_run", lambda args: called.append(args) or 0)
    with pytest.raises(ValueError, match="requires --preflight-acceptance"):
        runner.main([
            "--checkpoint", "candidate.pth",
            "--expected-checkpoint-sha256", "a" * 64,
            "--output-dir", "new-output",
        ])
    assert called == []


def test_runner_accepts_only_zero_excluded_density_column_access(tmp_path):
    allowed_dates = {"20240902"}
    selected = {}
    audits = {}
    for station, suffix in (("Jicamarca", ".hdf5"), ("PokerFlat", ".h5")):
        path = (tmp_path / f"{station}{suffix}").resolve()
        identity = {"path": str(path), "sha256": "a" * 64, "size_bytes": 10}
        selected[station] = [identity]
        dataset_names = ["ne", "dne"] if station == "Jicamarca" else [
            "ne", "dne", "cgm_lat", "cgm_lon"]
        reads = [{
            "dataset": name,
            "materialized_column_count": 1,
            "materialized_column_spans_inclusive": [[0, 0]],
            "materialized_dates_utc": ["20240902"],
            "excluded_columns_materialized": 0,
            "materialized_payload_sha256": hashlib.sha256(
                name.encode("utf-8")).hexdigest(),
        } for name in dataset_names]
        audits[station] = {
            "isr_column_access_audit_schema_version": 1,
            "station": station,
            "filter_semantics": (
                "timestamps_metadata_first_then_explicit_2d_column_slice_v1"),
            "allowed_dates_utc": ["20240902"],
            "excluded_date_values_persisted": False,
            "files": [{
                "path": str(path),
                "materialized_source_identity": identity,
                "source_identity_semantics": (
                    "p0a_contract_attested_no_whole_hdf_reread_v1"),
                "materialized_allowed_content_identity": {
                    "schema": "isr_allowed_materialized_content_v1",
                    "path": str(path),
                    "sha256": hashlib.sha256(
                        f"allowed:{station}".encode("utf-8")).hexdigest(),
                    "framed_array_count": len(reads) + 1,
                    "allowed_time_column_count": 1,
                },
                "segments": [{
                    "segment_id": "segment",
                    "total_time_columns": 2,
                    "allowed_time_columns": 1,
                    "excluded_time_columns": 1,
                    "allowed_column_spans_inclusive": [[0, 0]],
                    "allowed_dates_utc": ["20240902"],
                    "dataset_reads": reads,
                }],
            }],
            "totals": {
                "time_columns": 2,
                "allowed_time_columns": 1,
                "excluded_time_columns": 1,
                "density_dataset_column_reads": 2,
                "excluded_density_columns_materialized": 0,
            },
        }
    attestation = runner._validate_isr_column_access_audits(
        audits, selected, allowed_dates)
    assert attestation["status"] == "pass"
    assert attestation["excluded_density_columns_materialized"] == 0
    assert attestation[
        "locked_or_out_of_scope_timestamp_metadata_persisted"] is False
    assert attestation["stations"]["PokerFlat"][
        "density_dataset_column_reads"] == 2
    assert attestation["stations"]["PokerFlat"][
        "coordinate_dataset_column_reads"] == 2
    assert attestation["stations"]["PokerFlat"][
        "coordinate_dataset_fields"] == ["cgm_lat", "cgm_lon"]
    assert attestation["density_dataset_column_reads"] == 4
    assert attestation["coordinate_dataset_column_reads"] == 2
    assert [row["path"] for row in attestation[
        "materialized_allowed_content_identities"]] == sorted(
            [str((tmp_path / "Jicamarca.hdf5").resolve()),
             str((tmp_path / "PokerFlat.h5").resolve())],
            key=lambda value: value.lower())
    assert len(attestation["stations"]["PokerFlat"][
        "allowed_payload_sha256"]) == 64
    assert "excluded_time_columns" not in json.dumps(attestation)

    incorrect_density_total = deepcopy(audits)
    incorrect_density_total["PokerFlat"]["totals"][
        "density_dataset_column_reads"] = 4
    with pytest.raises(ValueError, match="aggregate access ledger"):
        runner._validate_isr_column_access_audits(
            incorrect_density_total, selected, allowed_dates)

    audits["PokerFlat"]["files"][0]["segments"][0]["dataset_reads"][0][
        "excluded_columns_materialized"] = 1
    with pytest.raises(ValueError, match="excluded column"):
        runner._validate_isr_column_access_audits(
            audits, selected, allowed_dates)


def test_real_poker_loader_audit_separates_density_and_coordinate_columns(
        tmp_path, monkeypatch):
    def unix(value: str) -> float:
        return dt.datetime.fromisoformat(value).replace(
            tzinfo=dt.timezone.utc).timestamp()

    timestamps = np.asarray([
        unix("2024-09-02T00:05:00"),
        unix("2024-09-05T00:05:00"),
    ], dtype=np.float64)
    jicamarca_path = tmp_path / "jro20240902_mixed.hdf5"
    with h5py.File(jicamarca_path, "w") as hdf:
        layout = hdf.require_group("Data/Array Layout")
        layout.create_dataset("timestamps", data=timestamps)
        layout.create_dataset("gdalt", data=np.asarray([250.0]))
        density = layout.create_group("2D Parameters")
        density.create_dataset("ne", data=np.asarray([[2.2e11, 9.9e19]]))
        density.create_dataset("dne", data=np.asarray([[1.0e10, 9.9e18]]))
        station = layout.create_group("1D Parameters")
        station.create_dataset("gdlatr", data=np.asarray([-11.95]))
        station.create_dataset("gdlonr", data=np.asarray([-76.87]))

    poker_path = tmp_path / "pfa20240902_mixed.h5"
    with h5py.File(poker_path, "w") as hdf:
        metadata = hdf.require_group("Metadata")
        metadata_dtype = np.dtype([("name", "S64"), ("value", "S64")])
        metadata.create_dataset("Experiment Parameters", data=np.asarray([
            (b"Instrument latitude", b"65.13"),
            (b"Instrument longitude", b"-147.471"),
            (b"Instrument altitude", b"0.215"),
        ], dtype=metadata_dtype))
        beam = hdf.require_group("Data/Array Layout").create_group("beam_1")
        beam.create_dataset("timestamps", data=timestamps)
        beam.create_dataset("range", data=np.asarray([250000.0]))
        one_d = beam.create_group("1D Parameters")
        one_d.create_dataset("azm", data=np.asarray([0.0, 0.0]))
        one_d.create_dataset("elm", data=np.asarray([90.0, 90.0]))
        one_d.create_dataset("beamid", data=np.asarray([1, 1]))
        two_d = beam.create_group("2D Parameters")
        two_d.create_dataset("ne", data=np.asarray([[2.5e11, 8.8e19]]))
        two_d.create_dataset("dne", data=np.asarray([[1.0e10, 8.8e18]]))
        two_d.create_dataset("cgm_lat", data=np.asarray([[65.0, -88.0]]))
        two_d.create_dataset("cgm_long", data=np.asarray([[211.0, -77.0]]))

    def identity(path: Path) -> dict[str, object]:
        return {
            "path": str(path.resolve()),
            "sha256": _digest(path),
            "size_bytes": path.stat().st_size,
        }

    jicamarca_identity = identity(jicamarca_path)
    poker_identity_before = identity(poker_path)
    monkeypatch.setattr(
        isr_loader, "_sha256",
        lambda _path: (_ for _ in ()).throw(
            AssertionError("P0-B must reuse the P0-A whole-file SHA")))
    start = unix("2024-09-02T00:00:00")
    end = unix("2024-09-02T23:59:59")
    jicamarca_records, jicamarca_audit = isr_loader.load_jicamarca(
        "", start, end, allowed_dates_utc={"20240902"},
        file_paths=[jicamarca_path],
        source_file_identities=[jicamarca_identity],
        fail_on_file_error=True, return_access_audit=True)
    poker_records, poker_audit_before = isr_loader.load_poker_flat(
        "", start, end, allowed_dates_utc={"20240902"},
        file_paths=[poker_path], source_file_identities=[poker_identity_before],
        fail_on_file_error=True, return_access_audit=True)
    assert len(jicamarca_records) == len(poker_records) == 1

    # Mutating only excluded-date columns changes the declared whole-file SHA,
    # but must not change the allowed-content identity or its column ledger.
    with h5py.File(poker_path, "r+") as hdf:
        beam = hdf["Data/Array Layout/beam_1/2D Parameters"]
        beam["ne"][:, 1] = 7.7e19
        beam["dne"][:, 1] = 7.7e18
        beam["cgm_lat"][:, 1] = -66.0
        beam["cgm_long"][:, 1] = -55.0
    poker_identity_after = identity(poker_path)
    assert poker_identity_after["sha256"] != poker_identity_before["sha256"]
    _, poker_audit_after = isr_loader.load_poker_flat(
        "", start, end, allowed_dates_utc={"20240902"},
        file_paths=[poker_path], source_file_identities=[poker_identity_after],
        fail_on_file_error=True, return_access_audit=True)
    content_before = poker_audit_before["files"][0][
        "materialized_allowed_content_identity"]
    content_after = poker_audit_after["files"][0][
        "materialized_allowed_content_identity"]
    assert content_after["sha256"] == content_before["sha256"]
    assert content_after["allowed_time_column_count"] == 1

    selected = {
        "Jicamarca": [jicamarca_identity],
        "PokerFlat": [poker_identity_after],
    }
    attestation = runner._validate_isr_column_access_audits(
        {"Jicamarca": jicamarca_audit, "PokerFlat": poker_audit_after},
        selected, {"20240902"})
    assert poker_audit_after["totals"]["density_dataset_column_reads"] == 2
    assert attestation["stations"]["PokerFlat"][
        "density_dataset_column_reads"] == 2
    assert attestation["stations"]["PokerFlat"][
        "coordinate_dataset_column_reads"] == 2
    assert attestation["stations"]["PokerFlat"][
        "density_dataset_fields"] == ["ne", "dne"]
    assert attestation["stations"]["PokerFlat"][
        "coordinate_dataset_fields"] == ["cgm_lat", "cgm_lon"]


def test_full_missing_preflight_fails_before_output_directory_creation(
        tmp_path, monkeypatch):
    output = tmp_path / "full-output"
    checkpoint = tmp_path / "candidate.pth"
    checkpoint.write_bytes(b"synthetic")
    git = {
        "head": "a" * 40,
        "branch": runner.EXPECTED_IMPLEMENTATION_BRANCH,
        "implementation_tag_object_sha": "b" * 40,
        "implementation_tag_commit": "a" * 40,
    }
    monkeypatch.setattr(runner, "_git_provenance", lambda **kwargs: git)
    monkeypatch.setattr(runner, "_python_environment", lambda: {"test": True})
    monkeypatch.setattr(runner, "_validate_identity_contract", lambda *args: None)
    args = runner.build_parser().parse_args([
        "--checkpoint", str(checkpoint),
        "--expected-checkpoint-sha256", "a" * 64,
        "--output-dir", str(output),
        "--preflight-acceptance", str(tmp_path / "missing"
                                      / "preflight_acceptance.json"),
    ])
    with pytest.raises(FileNotFoundError, match="requires preflight_acceptance"):
        runner._run(args)
    assert not output.exists()


def _write_synthetic_preflight_acceptance(tmp_path, monkeypatch):
    preflight_dir = tmp_path / "preflight"
    preflight_dir.mkdir()
    checkpoint = tmp_path / "candidate.pth"
    checkpoint.write_bytes(b"synthetic checkpoint")
    checkpoint_sha = _digest(checkpoint)
    p0a_contract = tmp_path / "isr_evaluation_contract.json"
    p0a_contract.write_text(
        json.dumps({"evaluation_schema_version": 2}), encoding="utf-8")
    p0a_identity = runner.artifact_identity(p0a_contract)
    date_split = tmp_path / "date_split.json"
    date_split.write_text(json.dumps({"train": [0]}), encoding="utf-8")
    date_split_sha = _digest(date_split)
    frozen_contract_path = tmp_path / "p0b_contract.json"
    frozen_contract_path.write_text(json.dumps({"synthetic": True}), encoding="utf-8")
    frozen_contract_sha = _digest(frozen_contract_path)

    query_counts = {"train": 2, "development": 1}
    profile_counts = {
        "train": {"FY": 3, "COSMIC": 4},
        "development": {"FY": 1, "COSMIC": 2},
    }
    frozen_contract = {
        "identity": {"date_split_sha256": date_split_sha},
        "output_lifecycle": {"preflight_probe": {
            "stations": ["Jicamarca", "PokerFlat"],
            "query_splits": ["train", "development"],
            "altitude_bands_km": [
                [120.0, 200.0], [200.0, 300.0], [300.0, 500.0]],
        }},
        "data_scope": {
            "P0A_ISR_contract_identity": p0a_identity,
            "ISR_queries": {"expected_finite_query_counts": query_counts},
            "satellite_tokens": {
                "expected_train_profile_counts": profile_counts["train"]},
            "satellite_profile_index_metadata": {
                "expected_development_profile_counts": profile_counts[
                    "development"]},
        },
    }
    git = {
        "head": "a" * 40,
        "branch": runner.EXPECTED_IMPLEMENTATION_BRANCH,
        "implementation_tag_object_sha": "b" * 40,
        "implementation_tag_commit": "a" * 40,
    }
    python_environment = {"python": "synthetic", "numpy": "synthetic"}
    coordinate_status = {
        "aacgm": {"status": "computed"},
        "qd": {"status": "unavailable"},
    }
    column_access = {
        "status": "pass",
        "excluded_density_columns_materialized": 0,
    }
    isr_source_files = [{
        "path": str(tmp_path / "isr.h5"),
        "sha256": "c" * 64,
        "size_bytes": 10,
    }]
    profile_metadata = {
        source: {
            "train_unique_profiles": profile_counts["train"][source],
            "development_unique_profiles": profile_counts["development"][source],
            "train_profile_id_sha256": (
                "1" * 64 if source == "FY" else "2" * 64),
            "profile_index_sha256": (
                "3" * 64 if source == "FY" else "4" * 64),
        }
        for source in runner.SOURCES
    }
    allowlist_identity = runner._train_allowlist_identity_from_metadata(
        profile_metadata)
    token_directory_identity = {
        source: {
            "semantics": "exact_train_only_compact_token_arrays_v1",
            "sha256": ("7" * 64 if source == "FY" else "8" * 64),
            "token_rows": 2,
            "unique_profiles": 1,
            "arrays": {
                "token_coords": {"dtype": "<f4", "shape": [2, 4]},
                "token_values": {"dtype": "<f4", "shape": [2]},
                "token_profile_ids": {"dtype": "<i8", "shape": [2]},
                "token_ids": {"dtype": "<i8", "shape": [2]},
            },
        }
        for source in runner.SOURCES
    }
    input_data_identity = {
        "date_split_manifest": date_split_sha,
        "FY": "5" * 64,
        "COSMIC": "6" * 64,
    }
    source_provenance = {"synthetic": {"sha256": "d" * 64, "size_bytes": 1}}
    monkeypatch.setattr(
        runner, "_source_provenance", lambda *args: deepcopy(source_provenance))

    runtime = {
        "audit_schema_version": runner.P0B_AUDIT_SCHEMA_VERSION,
        "status": "runtime_contract_bound",
        "frozen_p0b_contract": {
            "path": str(frozen_contract_path.resolve()),
            "sha256": frozen_contract_sha,
        },
        "candidate_checkpoint": {
            "path": str(checkpoint.resolve()),
            "sha256": checkpoint_sha,
        },
        "p0a_dependency": {
            "contract_path": str(p0a_contract.resolve()),
            "contract_sha256": p0a_identity["sha256"],
            "contract_size_bytes": p0a_identity["size_bytes"],
        },
        "date_split": {
            "path": str(date_split.resolve()),
            "sha256": date_split_sha,
        },
        "satellite_profile_metadata": profile_metadata,
        "source_provenance": source_provenance,
        "git": git,
        "python_environment": python_environment,
        "coordinate_enrichment": coordinate_status,
        "ISR_column_access_audit": column_access,
        "profile_cap_status": "not_applicable_in_v14",
        "train_allowlist_identity": allowlist_identity,
        "train_token_directory_identity": token_directory_identity,
        "runtime_identities": {
            "checkpoint_path": {
                "path": str(checkpoint.resolve()),
                "sha256": checkpoint_sha,
            },
            "date_split_path": {
                "path": str(date_split.resolve()),
                "sha256": date_split_sha,
            },
            "python_environment": python_environment,
            "input_data_sha256": input_data_identity,
            "train_allowlist_identity": allowlist_identity,
            "train_token_directory_identity": token_directory_identity,
            "coordinate_enrichment": coordinate_status,
            "ISR_column_access_audit": column_access,
            "ISR_source_file_sha256": isr_source_files,
            "profile_cap_status": "not_applicable_in_v14",
        },
    }
    manifest = {
        "audit_schema_version": runner.P0B_AUDIT_SCHEMA_VERSION,
        "status": "preflight_pass",
        "preflight_only": True,
        "checkpoint_sha256_before": checkpoint_sha,
        "checkpoint_sha256_after": checkpoint_sha,
        "date_split_sha256": date_split_sha,
        "raw_registry": {
            "date_partition_counts": query_counts,
            "allowed_query_count": sum(query_counts.values()),
        },
        "satellite_profile_metadata": profile_metadata,
        "git": git,
        "git_after": git,
        "python_environment": python_environment,
        "input_data_sha256": input_data_identity,
        "train_allowlist_identity": allowlist_identity,
        "train_token_directory_identity": token_directory_identity,
        "coordinate_enrichment": coordinate_status,
        "ISR_column_access_audit": column_access,
        "profile_cap_status": "not_applicable_in_v14",
        "isr_source_files": isr_source_files,
        "inference_preflight": {
            "status": "stratified_full_chain_pass_no_numeric_persistence",
            "probe_count": 12,
            "activation": {"status": "pass"},
            "numeric_predictions_persisted": False,
            "npz_written": False,
        },
    }
    failure = {
        "audit_schema_version": runner.P0B_AUDIT_SCHEMA_VERSION,
        "status": "no_failures",
        "entries": [],
    }
    runtime_path = runner.atomic_write_json(
        preflight_dir / "audit_contract.json", runtime)
    manifest_path = runner.atomic_write_json(
        preflight_dir / "manifest.json", manifest)
    failure_path = runner.atomic_write_json(
        preflight_dir / "failure_ledger.json", failure)
    marker_path = runner.write_completion_marker_atomically(
        preflight_dir / runner.PREFLIGHT_ACCEPTANCE_FILENAME,
        {
            "status": "preflight_pass",
            "full_audit_complete": False,
            "read_only_model_inference_executed": True,
            "stratified_probe_count": 12,
            "numeric_predictions_persisted": False,
            "npz_written": False,
            "locked_test_query_rows_written": 0,
            "satellite_token_partition": "train",
            "profile_cap_status": "not_applicable_in_v14",
            "query_partition_counts": query_counts,
            "satellite_profile_counts": profile_counts,
            "candidate_checkpoint_sha256": checkpoint_sha,
            "p0b_contract_sha256": frozen_contract_sha,
            "p0a_contract_sha256": p0a_identity["sha256"],
            "date_split_sha256": date_split_sha,
            "git_head": git["head"],
            "implementation_tag_object_sha": git[
                "implementation_tag_object_sha"],
            "python_environment": python_environment,
            "input_data_sha256": input_data_identity,
            "train_allowlist_identity": allowlist_identity,
            "train_token_directory_identity": token_directory_identity,
            "coordinate_enrichment": coordinate_status,
            "ISR_column_access_audit": column_access,
        },
        [runtime_path, manifest_path, failure_path],
        artifact_root=preflight_dir,
    )
    kwargs = {
        "acceptance_path": marker_path,
        "output_dir": tmp_path / "full",
        "checkpoint": checkpoint,
        "expected_checkpoint_sha256": checkpoint_sha,
        "frozen_contract": frozen_contract,
        "frozen_contract_path": frozen_contract_path,
        "frozen_contract_sha256": frozen_contract_sha,
        "p0a_contract_path": p0a_contract,
        "git_provenance": git,
        "python_environment": python_environment,
        "coordinate_status": coordinate_status,
    }
    return kwargs, {
        "marker": marker_path,
        "runtime": runtime_path,
        "manifest": manifest_path,
    }


def test_completed_preflight_acceptance_binds_to_matching_full_identity(
        tmp_path, monkeypatch):
    kwargs, _ = _write_synthetic_preflight_acceptance(tmp_path, monkeypatch)
    dependency = runner._validate_preflight_acceptance(**kwargs)
    assert dependency["sha256"] == _digest(kwargs["acceptance_path"])
    assert dependency["checkpoint_sha256"] == kwargs[
        "expected_checkpoint_sha256"]
    assert dependency["python_environment"] == kwargs["python_environment"]
    assert not kwargs["output_dir"].exists()


@pytest.mark.parametrize("drift_side", [
    "preflight_marker",
    "preflight_artifact",
    "full_runtime",
])
def test_preflight_or_full_identity_drift_is_rejected_before_full_directory(
        tmp_path, monkeypatch, drift_side):
    kwargs, paths = _write_synthetic_preflight_acceptance(tmp_path, monkeypatch)
    if drift_side == "preflight_marker":
        marker = json.loads(paths["marker"].read_text(encoding="utf-8"))
        marker["candidate_checkpoint_sha256"] = "e" * 64
        runner.atomic_write_json(paths["marker"], marker)
    elif drift_side == "preflight_artifact":
        manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
        manifest["status"] = "mutated_after_acceptance"
        runner.atomic_write_json(paths["manifest"], manifest)
    else:
        kwargs["python_environment"] = {
            **kwargs["python_environment"], "torch": "drifted"}

    with pytest.raises((ValueError, FileNotFoundError)):
        runner._validate_preflight_acceptance(**kwargs)
    assert not kwargs["output_dir"].exists()


def _refresh_synthetic_preflight_artifact_inventory(paths):
    marker = json.loads(paths["marker"].read_text(encoding="utf-8"))
    root = paths["marker"].parent
    marker["artifacts"] = [
        runner.artifact_identity(root / record["path"], root=root)
        for record in marker["artifacts"]
    ]
    runner.atomic_write_json(paths["marker"], marker)


@pytest.mark.parametrize("ledger", [
    "runtime_contract",
    "runtime_identities",
    "manifest",
    "completion_marker",
])
def test_preflight_rejects_each_token_directory_identity_ledger_drift(
        tmp_path, monkeypatch, ledger):
    kwargs, paths = _write_synthetic_preflight_acceptance(tmp_path, monkeypatch)
    marker = json.loads(paths["marker"].read_text(encoding="utf-8"))
    changed_identity = deepcopy(marker["train_token_directory_identity"])
    changed_identity["FY"]["sha256"] = "9" * 64
    if ledger in {"runtime_contract", "runtime_identities"}:
        runtime = json.loads(paths["runtime"].read_text(encoding="utf-8"))
        if ledger == "runtime_contract":
            runtime["train_token_directory_identity"] = changed_identity
        else:
            runtime["runtime_identities"][
                "train_token_directory_identity"] = changed_identity
        runner.atomic_write_json(paths["runtime"], runtime)
        _refresh_synthetic_preflight_artifact_inventory(paths)
    elif ledger == "manifest":
        manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
        manifest["train_token_directory_identity"] = changed_identity
        runner.atomic_write_json(paths["manifest"], manifest)
        _refresh_synthetic_preflight_artifact_inventory(paths)
    else:
        marker["train_token_directory_identity"] = changed_identity
        runner.atomic_write_json(paths["marker"], marker)

    with pytest.raises(ValueError, match="train token directory identity"):
        runner._validate_preflight_acceptance(**kwargs)
    assert not kwargs["output_dir"].exists()


@pytest.mark.parametrize("target", ["marker", "manifest"])
def test_preflight_rejects_nonfinite_json_in_marker_or_artifact(
        tmp_path, monkeypatch, target):
    kwargs, paths = _write_synthetic_preflight_acceptance(tmp_path, monkeypatch)
    path = paths[target]
    text = path.read_text(encoding="utf-8")
    text = text.replace("{", '{"forbidden_nonfinite": 1e999,', 1)
    path.write_text(text, encoding="utf-8")
    if target == "manifest":
        _refresh_synthetic_preflight_artifact_inventory(paths)
    with pytest.raises(ValueError, match="non-finite"):
        runner._validate_preflight_acceptance(**kwargs)
    assert not kwargs["output_dir"].exists()


def test_runner_publishes_completion_markers_as_terminal_success_writes():
    source = inspect.getsource(runner._run)
    first_preflight_branch = source.index("if not args.preflight_only:")
    preflight_branch = source.index("if args.preflight_only:", first_preflight_branch)
    full_branch = source.index("next_token_row_id = 0", preflight_branch)
    failure_branch = source.index("except BaseException as exc:", full_branch)

    preflight_source = source[preflight_branch:full_branch]
    preflight_marker = preflight_source.index("write_completion_marker_atomically(")
    assert preflight_source.index(
        "token_directory_after = build_train_token_directory_identity(indexes)"
    ) < preflight_marker
    assert preflight_source.rfind("atomic_write_json(", 0, preflight_marker) >= 0
    assert "atomic_write_json(" not in preflight_source[preflight_marker:]
    assert preflight_marker < preflight_source.index("return 0", preflight_marker)

    full_source = source[full_branch:failure_branch]
    full_marker = full_source.index("write_completion_marker_atomically(")
    assert full_source.index(
        "token_directory_after = build_train_token_directory_identity(indexes)"
    ) < full_marker
    assert full_source.rfind("atomic_write_json(", 0, full_marker) >= 0
    assert "atomic_write_json(" not in full_source[full_marker:]
    assert full_marker < full_source.index("return 0", full_marker)

    failure_source = source[failure_branch:]
    assert "write_completion_marker_atomically(" not in failure_source
    assert "_write_failure_ledger(" in failure_source


def _write_profile_products(root: Path, source: str):
    physical = root / f"{source}.npy"
    np.save(physical, np.zeros((3, 5), dtype=np.float32))
    index = root / f"{source}_index.npz"
    np.savez(
        index,
        profile_id=np.asarray([10, 20, 30], dtype=np.int64),
        pass_profile=np.asarray([True, True, True]),
        representative_time=np.asarray([1.0, 25.0, 49.0], dtype=np.float64),
        date_code=np.asarray([20240901, 20240902, 20240903], dtype=np.int64),
        kept_points=np.ones(3, dtype=np.int64),
        output_start=np.asarray([0, 1, 2], dtype=np.int64),
        output_end=np.asarray([1, 2, 3], dtype=np.int64),
        h_cut_km=np.asarray([210.0, 220.0, 230.0], dtype=np.float64),
    )
    return physical, index


def test_train_allowlist_cross_checks_dates_boundaries_and_omits_h_cut(tmp_path):
    fy_path, fy_index = _write_profile_products(tmp_path, "FY")
    cosmic_path, cosmic_index = _write_profile_products(tmp_path, "COSMIC")
    expected = {
        "train": {"FY": 1, "COSMIC": 1},
        "development": {"FY": 1, "COSMIC": 1},
    }
    config = {
        "fy_path": str(fy_path),
        "cosmic_path": str(cosmic_path),
        "fy_profile_index_path": str(fy_index),
        "cosmic_profile_index_path": str(cosmic_index),
        "observation_alt_range": [200.0, 500.0],
    }
    allowlists, metadata = runner._build_train_only_allowlists(
        config,
        {"train": [0], "development": [1], "locked_test": [2]},
        {"allowed_profile_summary": expected},
        dt.datetime(2024, 9, 1, tzinfo=dt.timezone.utc),
        expected,
    )
    assert allowlists["FY"].tolist() == [10]
    assert allowlists["COSMIC"].tolist() == [10]
    assert all("h_cut" not in json.dumps(value) for value in metadata.values())
    assert all(value["classification_only_transient"] for value in metadata.values())
    assert all(not value["locked_test_metadata_persisted"]
               for value in metadata.values())

    with np.load(fy_index, allow_pickle=False) as product:
        arrays = {name: product[name] for name in product.files}
    arrays["date_code"] = arrays["date_code"].copy()
    arrays["date_code"][0] = 20240909
    np.savez(fy_index, **arrays)
    with pytest.raises(ValueError, match="date_code"):
        runner._build_train_only_allowlists(
            config,
            {"train": [0], "development": [1], "locked_test": [2]},
            {"allowed_profile_summary": expected},
            dt.datetime(2024, 9, 1, tzinfo=dt.timezone.utc),
            expected,
        )
