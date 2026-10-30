# Package Verification

The source distribution and the complete server bundle share the same
numerical implementation and reference records. Their verification steps
differ because the source checkout excludes dataset images and model weights.

## Source Checkout

```bash
python scripts/verify_package.py --source-only
python scripts/rebuild_paper_tables.py
```

The first command checks required source files, result
coverage, protocol metadata, and leakage declarations. `--source-only` skips
the requirement for local pretrained checkpoints. It does not run inference
or check dataset images.

The second command reconstructs the frozen paper tables and compares their
values with the archived CSV files. It does not train a model or replace the
reference archive.

## Prepared Data and Model Resources

After following the [data guide](DATASETS.md) and [model guide](RESOURCES.md):

```bash
python scripts/verify_package.py
python scripts/verify_reference_data.py
python scripts/check_environment.py
```

These commands additionally check local checkpoint presence, image paths and
reference integrity metadata, and the reference numerical environment. The separate
`python scripts/check_data.py` command checks the original full-bundle
inventories and loader splits, including cached metadata. That strict bundle
inventory differs from image-only verification of a prepared source checkout.

[Repository README](../README.md) · [Reproducibility](REPRODUCIBILITY.md)
