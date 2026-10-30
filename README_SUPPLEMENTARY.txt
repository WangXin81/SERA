SERA: SUPPORT EVIDENCE ROUTING AND ADAPTATION
SUPPLEMENTARY EXECUTION GUIDE
===========================================

1. Overview
-----------
This source distribution contains the SERA implementation for few-shot image
classification, including Unified Evidence Construction (UEC), Residual Evidence
Adaptation (REA), Multi-view Evidence Aggregation (MEA), and Validation-Protected
Evidence Routing (VPER). It includes experiment launchers, frozen result records,
split metadata, environment specifications, and verification scripts.

The repository README provides the main workflow. Detailed guides are indexed
in docs/README.md. Dataset images and pretrained weights are prepared separately
as described in docs/DATASETS.md and docs/RESOURCES.md.

2. Reference environment
------------------------
Linux x86_64; NVIDIA GPU; Python 3.10; PyTorch 2.5.1+cu121;
torchvision 0.20.1+cu121; NumPy 2.2.6. The archived release-validation runs used
an NVIDIA RTX 4090. Full dependency specifications are in environment.yml and
requirements.txt.

3. Installation and source verification
---------------------------------------
Run from the repository root:

  conda env create -f environment.yml
  conda activate sera
  python scripts/verify_package.py --source-only
  python scripts/rebuild_paper_tables.py

These checks validate the distributed files and archived result tables without
starting an experiment.

4. External resources
----------------------
Prepare the eleven datasets according to docs/DATASETS.md. The model resource
root must contain clip/ViT-B-32.pt, clip/ViT-B-16.pt, and clip/ViT-L-14.pt.
The 39 server reference text-feature caches are included in
resources/SERA_text_features.tar.gz. The preparation command downloads the
three CLIP checkpoints, restores those caches, and verifies all 42 assets.

  python scripts/prepare_model_resources.py
  bash scripts/link_resources.sh /path/to/dataset-root
  python scripts/verify_package.py
  python scripts/verify_reference_data.py
  python scripts/check_environment.py

The linking script creates the data link. The optional model root argument
copies an existing model resource tree into pretrained/. A full server bundle
with local resources does not require linking. scripts/check_data.py checks
the stricter original full-bundle inventory, including cached metadata.

5. Experiment execution
------------------------
Run one complete SERA cell:

  bash scripts/run_experiment.sh caltech101 vit_b16 1 full_val 2024

The paper uses SERA-V, SERA-SO, and SERA-SF for the full_val, support_only, and
support_fixed launcher protocols, respectively. The archived record names are
SERA_val, SERA_support_only, and SERA_support_fixed.

The arguments are DATASET BACKBONE SHOTS PROTOCOL SEED. Supported protocols are
full_val, no_rea, no_mea, support_only, and support_fixed. The formal settings
use seed 2024, 50 base-training epochs, batch size 32, and 500 REA iterations per
shot. The launcher checks the numerical environment before training.

Run the full 165-cell grid:

  bash scripts/reproduce_grid.sh full_val

Run the paired UEC evidence-group ablation:

  bash scripts/reproduce_uec_internal_grid.sh

The paired experiment evaluates both controlled variants using the same trained
model and collected logits. See configs/README.md and docs/REPRODUCIBILITY.md.

6. Outputs and reference results
--------------------------------
New records are written under outputs/. The canonical results/reference/ tree
contains 990 JSON files: 825 standalone core records and 165 paired UEC files.
These represent 1,155 reported results. The paper archive includes copies of
original records and external-method comparison values; copies are not extra
experiments. results/README.md describes the complete archive structure.

  python scripts/rebuild_paper_tables.py

This command reconstructs 825 core rows, 330 UEC ablation rows, and the UEC
summaries in a temporary directory and compares them with the frozen CSV tables.
Use --output-dir outputs/rebuilt_tables to retain a regenerated copy.

7. Evaluation boundaries and rerun variability
----------------------------------------------
REA uses support images and labels only. MEA wraps the base evaluation path.
VPER uses the labels allowed by the selected routing protocol. Query/test labels
are used only for final accuracy computation. Archived records retain
query_leakage=false and test_label_used=false.

Fresh stochastic training or validation-route selection can differ from frozen
results. The archived representative rerun checks and their observed differences
are documented in docs/REPRODUCIBILITY.md.
