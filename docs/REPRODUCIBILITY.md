# Reproducibility Protocol

## Verified Environment

- Python 3.10
- PyTorch 2.5.1 with CUDA 12.1 (`2.5.1+cu121`)
- torchvision 0.20.1 with CUDA 12.1 (`0.20.1+cu121`)
- NumPy 2.2.6
- NVIDIA RTX 4090 for the release-validation runs

The numerical stack is part of the verified experiment definition. Before any
experiment, `scripts/run_experiment.sh` invokes `scripts/check_environment.py`
and stops on a version mismatch. An environment-specific Python executable can
be selected explicitly:

```bash
PYTHON_BIN=/path/to/verified/environment/bin/python \
  bash scripts/run_experiment.sh caltech101 vit_b16 1 full_val 2024
```

Reproduction uses the software versions listed above. Other Python environments
may contain incompatible library versions and are not covered by these
validation results.

## Formal Grid

- Seed: 2024
- Backbones: ViT-B/16, ViT-B/32, ViT-L/14
- Shots: 1, 2, 4, 8, 16
- Datasets: Caltech101, DTD, EuroSAT, FGVC Aircraft, Food101, ImageNet,
  Oxford Flowers, Oxford Pets, Stanford Cars, SUN397, and UCF101
- Base-training epochs: 50
- Batch size: 32
- Residual adaptation iterations: 500 per shot
- Multi-view evaluation: deterministic FiveCrop inputs
- Residual-adapter placement: `all` in every public launcher. The argument is
  active only when REA or residual-reference warmup is enabled; the historical
  `top3` value stored in frozen no-REA metadata is inactive because both paths
  are disabled.

The full grid contains 11 datasets x 3 backbones x 5 shot settings = 165
cells. It can be launched with:

```bash
bash scripts/reproduce_grid.sh full_val
```

The paired UEC internal ablation uses the same trained model and collected
logits for both controlled variants. It removes only one UEC evidence family
from the routing pool at a time while keeping REA, MEA, VPER, seed, epochs,
batch size, validation protocol, and balanced profile unchanged:

```bash
bash scripts/reproduce_uec_internal_grid.sh
```

`base_alignment_only` corresponds to SERA w/o VFR; it retains only
`clip/mlp/ada/total` UEC evidence. `vocabulary_free_relation_only` corresponds
to SERA w/o BA; it retains only `sim_*` UEC evidence. The command writes one
paired JSON per dataset/backbone/shot cell and summarizes all 330 variant
results after completing the 165-cell grid.

Reference JSON outputs are read-only under `results/reference/`; fresh outputs
are written under `outputs/`. The consolidated reference table is
`results/tables/sera_common11_results.csv`.

## Reference Archive Integrity

The package contains 990 archived JSON files representing 1,155 reported
results: 165 full SERA results plus 990 ablation results across six variants.
The archive includes 165 no-REA, 165 no-MEA, 330 routing-protocol, and 330
UEC internal-evidence results; the UEC results are stored in 165 paired JSON
files, each containing the same-run full baseline and both controlled variants.
Every packaged record was checked against the frozen experiment archive. Final
accuracy, dataset, backbone, shot, seed, protocol, route (after publication-name
translation), and leakage declarations agree with the authoritative records.

Every archived record declares `query_leakage=false` and
`test_label_used=false`. `scripts/verify_package.py` checks counts, complete
grid axes, protocol membership, leakage declarations, and result structure.

## Representative Rerun Validation

Twelve cells spanning all 11 datasets, all three backbones, all five shot
values, and four protocols were rerun through the publication entry point in
the verified environment. The table compares each fresh run with its archived
paper result.

| Dataset | Backbone | Shots | Protocol | Fresh final accuracy | Archive final accuracy | Absolute difference |
|---|---|---:|---|---:|---:|---:|
| UCF101 | ViT-B/16 | 2 | `no_rea` | 0.7718741894 | 0.7718741894 | 0 |
| Caltech101 | ViT-B/16 | 1 | `full_val` | 0.9488843679 | 0.9488843679 | 0 |
| Oxford Pets | ViT-L/14 | 1 | `no_mea` | 0.9468520284 | 0.9460343122 | 0.0008177161 |
| DTD | ViT-B/16 | 1 | `no_rea` | 0.5821512938 | 0.5821512938 | 0 |
| EuroSAT | ViT-B/32 | 1 | `full_val` | 0.8307407498 | 0.8307407498 | 0 |
| FGVC Aircraft | ViT-L/14 | 4 | `no_rea` | 0.5091509223 | 0.5076507926 | 0.0015001297 |
| Food101 | ViT-B/32 | 8 | `no_rea` | 0.8209900856 | 0.8210890889 | 0.0000990033 |
| Oxford Flowers | ViT-B/16 | 16 | `no_rea` | 0.9890377522 | 0.9890377522 | 0 |
| Stanford Cars | ViT-L/14 | 2 | `no_rea` | 0.8286283016 | 0.8293744326 | 0.0007461310 |
| SUN397 | ViT-B/32 | 4 | `no_rea` | 0.7205541730 | 0.7205541730 | 0 |
| ImageNet | ViT-B/16 | 8 | `no_rea` | 0.7282000184 | 0.7282000184 | 0 |
| DTD | ViT-B/32 | 1 | `support_only` | 0.5366430283 | 0.5366430283 | 0 |

Eight of the twelve cells reproduced the archived final accuracy exactly, and
Food101 differed by less than 0.0001. All fresh records retained false leakage
flags. The remaining differences were investigated by rerunning the preserved
frozen source in the same verified environment:

- Food101 reproduced the publication package result exactly rather than the
  archived value.
- Oxford Pets and FGVC Aircraft also differed when rerun directly from the
  frozen source.
- Repeated Stanford Cars runs selected different validation routes and final
  scores in both the publication and frozen layouts.

These checks distinguish packaging equivalence from fresh-run variability.
The publication namespace and directory reorganization do not alter the
numerical pipeline, but the training and validation-route search is not bitwise
deterministic for every cell. The archived full, component, routing, and paired
UEC records are the reference snapshots. The released configuration retains the
recorded seed, algorithms, route-search procedure, and kernel settings.

## Distribution and External Resources

The full reproducibility bundle includes the SERA implementation, dataset
loaders, split metadata, protocol scripts, documentation, archived result
records, dataset images under `data/`, pretrained CLIP weights under
`pretrained/`, and the required licensed reference implementation under
`third_party/`. The bundle contains no symbolic links and does not resolve
code, data, model, or result dependencies through another SERA experiment
directory.

Frozen split metadata is bundled under `datasets/jsons/`, and the loaders use
it as a fallback when a dataset directory lacks the corresponding split file.
For a source-only distribution that intentionally omits large resources,
`scripts/link_resources.sh` can attach a dataset root and physically copy a
model cache into `pretrained/`; this step is not required by the full bundle.

## Verification

For the source checkout, first run:

```bash
python scripts/verify_package.py --source-only
python scripts/rebuild_paper_tables.py
```

After preparing the resources described in [Dataset Preparation](DATASETS.md)
and [Model Resources](RESOURCES.md), run the checks below.


Run the following after every copy, archive extraction, or server deployment:

```bash
python scripts/verify_package.py
python scripts/check_data.py
python scripts/check_environment.py
```

The commands validate package structure and hashes, archived-result coverage
and leakage flags, all eleven dataset splits and image paths, and the numerical
environment.

[Repository README](../README.md) · [Documentation](README.md)
