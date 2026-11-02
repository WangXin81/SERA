# Setup, Experiments, and Verification

Run these commands from the repository root in the documented environment.

| Script | Purpose |
| --- | --- |
| `link_resources.sh` | Link a dataset root and copy a model resource tree into the checkout |
| `prepare_model_resources.py` | Download verified CLIP checkpoints and restore the 39 reference text-feature caches |
| `prepare_reference_images.py` | Materialize the server image subset from an existing dataset tree |
| `verify_reference_data.py` | Verify images and required loader annotations against the reference inventory |
| `run_experiment.sh` | Train and evaluate one dataset-backbone-shot-protocol cell |
| `reproduce_grid.sh` | Run all 165 cells for one formal protocol |
| `reproduce_uec_internal_grid.sh` | Run the paired UEC evidence-group ablation grid |
| `summarize_uec_internal.py` | Consolidate fresh paired UEC JSON outputs |
| `rebuild_paper_tables.py` | Reconstruct and compare the frozen paper tables |
| `verify_package.py` | Check package structure, result coverage, protocol metadata, and leakage declarations |
| `check_data.py` | Check the full-bundle dataset inventory and image paths |
| `check_environment.py` | Check the reference numerical software versions |

## One Experiment

```bash
bash scripts/run_experiment.sh caltech101 vit_b16 1 full_val 2024
```

The positional arguments are `DATASET BACKBONE SHOTS [PROTOCOL] [SEED]`.
Supported protocols are listed in the [configuration guide](../configs/README.md).
The launcher supplies the explicit switches for REA, MEA, and VPER required by
the selected protocol. Calling `train.py` with parser defaults alone does not
enable the complete method.

`PYTHON_BIN`, `DATA_ROOT`, `MODEL_ROOT`, and `OUTPUT_ROOT` select the Python
executable and local resource/output roots. `BATCH_SIZE` and `NUM_WORKERS`
are launcher settings; retain batch size 32 for the formal protocol.
The default output directory is `outputs/<protocol>/`.

## Reference Checks

```bash
python scripts/verify_package.py --source-only
python scripts/rebuild_paper_tables.py
```

These checks do not launch training. After attaching data and model resources,
run the additional checks in [Package Verification](../docs/VERIFICATION.md).
The complete experiment command table is in the [repository README](../README.md).
