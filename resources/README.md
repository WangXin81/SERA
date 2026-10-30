# Model Resources

The reference model resources comprise three pretrained CLIP checkpoints and
39 archived text-feature caches.

| Resource | Distribution |
| --- | --- |
| CLIP ViT-B/32, ViT-B/16, ViT-L/14 checkpoints | Downloaded from the original OpenAI URLs by `scripts/prepare_model_resources.py` |
| Target-class and source-vocabulary text features | Included in `SERA_text_features.tar.gz` |

From the repository root:

```bash
python scripts/prepare_model_resources.py
```

The command prepares and validates all 42 assets under `pretrained/clip/`.
Existing files with different contents are rejected. To restore only the
archived text features:

```bash
python scripts/prepare_model_resources.py --text-features-only
```

SERA performs task-specific adaptation using the support set. The reference
bundle contains pretrained CLIP checkpoints and text features. A single
universal trained SERA checkpoint is not part of the archived experiment output.

[Repository README](../README.md) · [Resource Details](../docs/RESOURCES.md)
