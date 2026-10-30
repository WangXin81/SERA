# SERA method-to-code mapping

This document maps the paper terminology to the verified implementation in
`sera/runner.py`. It describes the formal `balanced` evidence profile used by
the reported experiments. Optional exploratory evidence families are disabled
by this profile and do not enter the reported SERA results.

## End-to-end execution order

1. Load the pretrained CLIP/VLM backbone and construct target-class text
   prototypes.
2. Train the lightweight base-alignment components on the K-shot support set.
3. Train REA independently from support images and support labels only.
4. Evaluate the base path on support, routing-validation, and query sets. When
   MEA is enabled, each image is evaluated through FiveCrop and the per-view
   class logits are averaged. Image features are normalized per view, averaged,
   and normalized again.
5. Construct UEC evidence from the aggregated base outputs and support labels.
6. Form a baseline UEC pool and a candidate pool that additionally contains the
   independent REA evidence branch.
7. Use VPER to select a branch or probability-fusion rule under the requested
   routing protocol, then apply that fixed rule to query logits.

MEA is an evaluation wrapper around the base path; it is not a serial trainable
module between the image encoder and UEC. REA is evaluated separately and does
not pass through MEA.

## Problem setting

The implementation receives a K-shot support set `S`, an optional official
routing-validation set `V`, and a held-out query/test set `Q`. Query labels are
used only to compute the final reported accuracy. They are never used for model
training, residual adaptation, route selection, or fusion-weight selection.

## UEC: Unified Evidence Construction

UEC operates in a common C-class logit space. Under
`--evidence_pool_profile balanced`, it retains exactly two evidence families:

### Base alignment evidence

- `clip`: semantic CLIP alignment evidence.
- `mlp`: visual adaptation evidence.
- `ada`: residual logit evidence.
- `total`: aligned fusion evidence produced by the base model.

`forward_logits` computes these four class-logit vectors.
`train_base_alignment` optimizes the trainable base-alignment components;
`collect_outputs` collects the resulting evaluation outputs.

### Vocabulary-free relation evidence

`build_sim_logits` constructs relation evidence from support image features,
support labels, target text prototypes, and source vocabulary prototypes:

- `sim_raw`: direct relation evidence.
- `sim_z`: standardized relation evidence.
- `sim_teacher`: teacher-guided relation evidence.
- `sim_select`: selected relational mapping evidence.

`filter_evidence_pool(..., pool_profile="balanced")` excludes composite,
subspace, and auxiliary support branches from the formal pool.

## REA: Residual Evidence Adaptation

`train_residual_evidence_branch` trains an independent low-rank residual branch
using only support images and support labels. It returns support, validation,
and query class-logit vectors under the SERA key `residual_evidence`.
This class-logit vector is termed Residual Adaptation Evidence.

REA is added only to the VPER candidate pool. It is not included in the balanced
UEC baseline pool and is not automatically adopted.

## MEA: Multi-view Evidence Aggregation

`FiveCropEvalTransform` creates five deterministic spatial crops. With
`--enable_multi_view_evidence_aggregation`, the support-evaluation,
routing-validation, and query loaders use this transform. `collect_outputs`
runs every crop through the same trained base model and computes:

- the mean of per-view base class logits; and
- the normalized mean of per-view normalized image features.

The aggregated features feed relation-evidence construction. REA remains a
separate candidate path and is evaluated without this MEA wrapper.

## VPER: Validation-Protected Evidence Routing

The baseline pool contains UEC branches. The candidate pool contains the same
UEC branches plus REA. `select_fusion_by_mode` searches the permitted route or
probability-fusion rules. For the official-validation protocol, VPER adopts the
candidate only when its routing-validation accuracy is at least the baseline
accuracy plus `--evidence_routing_guard_margin`; otherwise it falls back to the
UEC route.

The formal protocols are:

- `SERA_val`: official validation labels select the route; query labels are not
  used.
- `SERA_support_only`: support labels select the route; no validation or query
  labels are used.
- `SERA_support_fixed`: fixed global probability fusion; no label-based route
  search is performed.

The implementation ranks all single branches, retains the top eight for
single-branch candidates, and uses the top six as the ordered pool for adjacent
two- and three-branch probability-fusion windows. The formal fusion grid step
is 0.05.

## Formal ablations

- `SERA_no_REA`: removes the independent residual candidate while retaining
  UEC, MEA, and VPER.
- `SERA_no_MEA`: disables FiveCrop evaluation while retaining UEC, REA, and
  VPER.
- `SERA_support_only` and `SERA_support_fixed`: change only the VPER protocol.
- `SERA w/o VFR` (`base_alignment_only`): removes vocabulary-free relation
  evidence from the UEC pool while retaining base alignment evidence.
- `SERA w/o BA` (`vocabulary_free_relation_only`): removes base alignment
  evidence from the UEC pool while retaining vocabulary-free relation evidence.

The paired UEC ablations retain REA, MEA, and VPER and reuse the same trained
model and collected logits. They are exported through
`--save_uec_ablation_json` or `scripts/reproduce_uec_internal_grid.sh`.

The formal grid contains 3 backbones x 5 shot settings x 11 datasets = 165
cells for each full or module-ablation configuration.
