# FSIA-INR Maintenance Notes

Use `AGENTS.md` for repository workflow, commands, style, and data-safety
requirements. This file records only architecture details that are easy to
misinterpret when changing the model.

## Active Data Flow

`main_fsia.py` builds real-profile FY and COSMIC neighborhood indexes and calls
`train_fsia`. Each batch follows:

1. A frozen IRI neural proxy supplies background `log10(Ne)` and hidden features.
2. `FYObsEncoder` and `COSMICObsEncoder` encode local profile neighborhoods.
3. `NeuralETKFLayer` applies independently masked FY and COSMIC updates.
4. Channel-wise residual fusion and the gated decoder predict a bounded
   correction to the IRI background.

FY uses `fy_202409_clean1.npy` for physical values and the matching
`fy_202409_clean3.npy` profile IDs. COSMIC profile IDs come from column 6 of
`cosmic_september_2024.npy`. Both indexes group by profile ID before sorting by
representative profile time.

## Active Objectives

Training uses supervised density loss (Huber warmup, then uncertainty NLL),
height-adaptive IRI background regularization, IRI hidden-state reconstruction,
and profile peak alignment. GIRO peak training, TEC, voxel pooling, global
analysis state, shape/uplift/depletion losses, and vertical smoothness are not
part of the active path.

## Checkpoint Compatibility

Run65 checkpoints must load with `strict=True`. The following zero-valued state
keys remain only for compatibility and are frozen:

- `proj_frame_offset.weight`
- `kalman_layer.H_FY_u`, `H_FY_v`, and `H_FY_a.*`
- `kalman_layer.H_COSMIC_u`, `H_COSMIC_v`, and `H_COSMIC_a.*`

The two `R_*_net` modules are also frozen because their outputs are detached in
the run65 inference path. Changing that behavior is a run66 model change, not a
cleanup.

Training-state format v2 records optimizer parameter names. Older full states
are migrated by dropping optimizer slots for parameters frozen by this cleanup.

Never overwrite historical checkpoints or generated P2/P3 artifacts.
