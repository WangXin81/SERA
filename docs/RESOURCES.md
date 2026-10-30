# Model Resources

SERA uses the pretrained CLIP ViT-B/32, ViT-B/16, and ViT-L/14 backbones from
[OpenAI CLIP](https://github.com/openai/CLIP). The model assets have the following
layout after preparation:

```text
pretrained/
`-- clip/
    |-- ViT-B-32.pt
    |-- ViT-B-16.pt
    |-- ViT-L-14.pt
    `-- text_feats/
```

## Preparation

From the repository root:

```bash
python scripts/prepare_model_resources.py
```

The command downloads the three original CLIP checkpoints and restores the
39 archived text-feature caches from `resources/SERA_text_features.tar.gz`.
All assets are validated automatically against the reference resource metadata
embedded in the preparation script.

The checkpoint files are downloaded separately from the source repository.
An existing model resource tree with a `clip/` subdirectory is attached with:

```bash
bash scripts/link_resources.sh /path/to/dataset-root /path/to/model-resource-root
```

When the preparation command has already created `pretrained/`, attach only
the dataset root:

```bash
bash scripts/link_resources.sh /path/to/dataset-root
python scripts/verify_package.py
```

## Text Features

Target-class and source-vocabulary text features are stored in
`pretrained/clip/text_feats/`. The included cache files preserve the class
names, templates, and backbone settings of the reference experiments.
`build_class_text_features` and `build_vocab_free_features` in `sera/runner.py`
also generate these features when a cache is absent.

SERA trains task-specific adaptation parameters from the support set. The
released resources are the pretrained backbones and reference text features;
the archived experiments do not contain one universal SERA checkpoint.

[Repository README](../README.md) · [Dataset Preparation](DATASETS.md)
