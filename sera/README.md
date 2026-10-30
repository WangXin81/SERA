# SERA Implementation

The numerical implementation is organized around `runner.py`, which contains
training, evidence construction, evaluation, and routing.

| Path | Role |
| --- | --- |
| [`runner.py`](runner.py) | Argument parsing, base alignment, UEC, REA, MEA, VPER, and result export |
| `clip_encoder/` | CLIP-compatible encoder and tokenizer assets |
| `low_rank/` | Low-rank adaptation layers used by REA |
| [`../train.py`](../train.py) | Repository command-line entry point |
| `../third_party/reference_clip/` | Reference CLIP dependency used by REA |

UEC constructs the baseline evidence pool. REA is trained independently on the
support set and enters the candidate pool. MEA aggregates base-path outputs at
evaluation time. VPER selects the final prediction rule under the requested
protocol.

The [Method-to-Code Map](../docs/METHOD_TO_CODE.md) documents the corresponding
functions and data boundaries. Use the [experiment launchers](../scripts/README.md)
for the formal settings, including the switches required to enable the complete
method.

[Repository README](../README.md)
