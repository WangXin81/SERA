# Model Resources

The reference model resources comprise three pretrained CLIP checkpoints and
39 archived text-feature caches.

| Resource | Distribution |
| --- | --- |
| CLIP ViT-B/32, ViT-B/16, ViT-L/14 checkpoints | Downloaded from the original OpenAI URLs by `scripts/prepare_model_resources.py` |
| Target-class and source-vocabulary text features | Included in `SERA_text_features.tar.gz` |

## Pretrained Checkpoints

| Backbone | Filename | Source |
| --- | --- | --- |
| ViT-B/32 | `ViT-B-32.pt` | [OpenAI checkpoint](https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt) |
| ViT-B/16 | `ViT-B-16.pt` | [OpenAI checkpoint](https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt) |
| ViT-L/14 | `ViT-L-14.pt` | [OpenAI checkpoint](https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt) |

## Preparation

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
