# Dataset Preparation

Prepare the dataset image tree separately from the source checkout. Use the
frozen metadata under `datasets/jsons/` for the paper partitions; a dataset's
original release split is not a substitute for the archived SERA split.

## Dataset Sources and Image Layout

The source pages below identify the original datasets. The last column gives
the image location expected by the released loaders, relative to `data/`.
Preserve the paths referenced in the bundled split JSON files.

| Dataset | Command-line Identifier | Source | Image Location |
| --- | --- | --- | --- |
| Caltech101 | `caltech101` | [CaltechDATA](https://data.caltech.edu/records/mzrjq-6wc02) | `Caltech101/101_ObjectCategories/` |
| DTD | `dtd` | [Oxford VGG](https://www.robots.ox.ac.uk/~vgg/data/dtd/) | `dtd/images/` |
| EuroSAT | `eurosat` | [Dataset repository](https://github.com/phelber/EuroSAT) | `eurosat/2750/` |
| FGVC Aircraft | `fgvc` or `fgvc_aircraft` | [Oxford VGG](https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/) | `fgvc_aircraft/images/` |
| Food101 | `food101` | [ETH Zurich](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/) | `Food101/images/` |
| Oxford Flowers 102 | `oxford_flowers` | [Oxford VGG](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/) | `Flower102/jpg/` |
| Oxford Pets | `oxford_pets` | [Oxford VGG](https://www.robots.ox.ac.uk/~vgg/data/pets/) | `OxfordPets/images/` |
| Stanford Cars | `stanford_cars` | [Dataset project](https://ai.stanford.edu/~jkrause/cars/car_dataset.html) | `StanfordCars/cars_train/`, `StanfordCars/cars_test/` |
| SUN397 | `sun397` | [Dataset project](https://vision.princeton.edu/projects/2010/SUN/) | `sun397/SUN397/` |
| UCF101 | `ucf101` | [UCF CRCV](https://www.crcv.ucf.edu/data/UCF101.php) | `UCF101/UCF-101-midframes/` |
| ImageNet-1K | `imagenet` | [ImageNet](https://www.image-net.org/) | `imagenet/images/train/`, `imagenet/images/val/` |

UCF101 uses the prepared middle-frame image classification tree. The video
archive alone does not have the layout required by the loader. Use the frame
names and class directories recorded in `split_zhou_UCF101.json`.
The [CoOp dataset guide](https://github.com/KaiyangZhou/CoOp/blob/main/DATASETS.md#ucf101)
provides the extracted `UCF-101-midframes.zip` used by this classification layout.

## Reference Bundle

The full reproducibility bundle stores the verified image tree directly under
`data/`. It contains no links to parent projects or historical code trees.
Dataset split metadata is also bundled under `datasets/jsons/` so that the
experimental partitions remain explicit and versioned.

Operating-system resource-fork metadata is excluded from the image bundle.
Dataset images, annotation files, and experimental split metadata are retained.

## ImageNet Routing and Query Split

For ImageNet-1K, support images come from the official training archive. The
official validation images are split deterministically with seed 20260614 into
25 routing-validation and 25 held-out query images per class. The two subsets
are disjoint. Query labels are used only for final accuracy computation.

Keep the server reference training image pool when reproducing the frozen
support selections. Adding other training images changes the sampling pool.
The reference pool contains exactly 16 training images per class (16,000 total)
and all 50,000 official validation images. The validation images are divided
into 25 routing-validation and 25 query images per class by the released loader.
The full-bundle inventory is recorded in `scripts/check_data.py`; it describes
the prepared reference tree rather than the size of every original dataset.

## Reference Image Inventory

`datasets/manifests/reference_data_files.csv.gz` contains the paths, file sizes,
and reference integrity metadata exported from the server reference tree.
`reference_data_summary.json` records the file counts and byte totals for all
eleven dataset directories. The inventory contains metadata only; image bytes
are not distributed in this repository.

The manifest also fixes the SUN397 image subset and the UCF101 middle-frame
files. A prepared source tree with the documented dataset directory names is
converted into the reference image subset with:

```bash
python scripts/prepare_reference_images.py \
  --source-root /path/to/source-dataset-root \
  --output-root /path/to/reference-dataset-root --dataset imagenet
```

The command selects the exact image filenames in the server manifest, verifies
their contents, and copies them into a separate output root. `--dataset sun397`
and `--dataset UCF101` select the corresponding reference files.
Existing files with different contents are rejected. Dataset annotation files
required by the loaders remain part of the dataset preparation described above.

## Setup and Validation

Verify the prepared image files against the reference checksums:

```bash
python scripts/verify_reference_data.py --root /path/to/dataset-root
```

The ImageNet check also rejects additional image files and class directories,
because they change the support sampling pool or validation/query partition.
`--dataset imagenet` restricts the check to one dataset. `--sizes-only` performs
a faster path-and-size check. The default check verifies image files; adding
`--include-metadata` also checks the original annotation, preparation-record,
and cached-metadata files present in the complete server snapshot.

The stricter original full-bundle inventory check is:

```bash
python scripts/check_data.py
```

For a source-only copy, create the local resource link with:

```bash
bash scripts/link_resources.sh /path/to/DATA /path/to/model
```

The model resource root contains a `clip/` subdirectory with the required
weights and text-feature caches.

The GitHub source distribution excludes image files and model weights. Split
metadata and the numerical implementation are included. Follow
[Model Resources](RESOURCES.md) for checkpoint preparation and
[Package Verification](VERIFICATION.md) for source-only checks.

[Repository README](../README.md) · [Documentation](README.md)
