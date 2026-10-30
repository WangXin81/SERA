# Dataset Loaders and Split Metadata

This directory contains the loaders for the eleven classification datasets,
shared dataset utilities, and frozen split metadata under `jsons/`.

The loaders sample the K-shot support set from training images and expose
separate validation and query sets. When a split JSON is absent from a dataset
directory, the shared utilities fall back to the bundled file with the same
name. ImageNet uses its loader's deterministic per-class validation split.

See [Dataset Preparation](../docs/DATASETS.md) for dataset identifiers, expected
image directories, source pages, and the ImageNet protocol. After preparing
the image tree, run from the repository root:

```bash
python scripts/check_data.py
```

This checks the frozen full-bundle inventory and verifies that every referenced
image exists. Dataset images are prepared separately from the source checkout.

[Repository README](../README.md)
