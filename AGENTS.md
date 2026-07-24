# Repository Guidelines

## Project Structure & Module Organization

This repository implements FSIA-INR, a feature-space data-assimilation model for reconstructing three-dimensional ionospheric electron density (`Ne`). Top-level scripts are the main workflows: `main_fsia.py` trains and evaluates the model, `plot_fsia.py` generates figures, and `evaluate_giro_peak.py` validates GIRO peak parameters. Core code lives under `inr_modules/`:

- `data_managers/`: FY-3, COSMIC-2, IRI peak, and space-weather loading/indexing.
- `mdia/fsia_model.py`: model components and forward data flow.
- `mdia/train_fsia.py`: training, validation, checkpointing, and losses.
- `mdia/physics_losses_mdia.py`: physics-informed objectives.
- `isr_evaluation/`: independent ISR comparison tools.

Treat `inr_modules/config_mdia.py` as the configuration source of truth. Generated checkpoints and figures belong in configured output directories, not source folders.

## Build, Test, and Development Commands

Run commands from the repository root in the configured `pytorch_cpu` environment:

```powershell
python main_fsia.py
python plot_fsia.py
python evaluate_giro_peak.py
python -m inr_modules.mdia.fsia_model
```

The final command is the lightweight model self-test and should pass before submission. This is a Python project with no separate build step.

## Coding Style & Naming Conventions

Use four-space indentation and PEP 8 naming: `snake_case` for functions and variables, `PascalCase` for classes, and uppercase names for constants. Preserve tensor-shape comments (for example, `[B, K, 10]`) around model interfaces. Prefer existing managers, encoders, and configuration keys over new abstractions or dependencies. Keep changes localized and update comments that name obsolete run versions.

## Testing Guidelines

No dedicated test framework or coverage threshold is currently configured. For non-trivial logic, add one focused `test_*.py` regression test or extend the relevant module self-test. Test safe fallback behavior, tensor shapes, missing-neighbor masks, and IRI-baseline degradation. Do not require production datasets for unit-level checks.

## Commit & Pull Request Guidelines

Existing commits use short imperative subjects such as `Fix true-profile assimilation grouping`. Keep each commit single-purpose. Pull requests should describe the data flow affected, configuration changes, validation command and result, and any checkpoint compatibility impact. Include representative plots only when numerical or visualization behavior changes.

## Configuration & Data Safety

Dataset paths are machine-specific absolute paths. Do not commit datasets, checkpoints, credentials, or generated reports. Validate required paths before long training runs, and preserve the no-observation fallback to the frozen IRI background.
