# Third-party components

This package bundles the SERA CLIP-compatible encoder under
`sera/clip_encoder/` and the low-rank adaptation layers used by REA under
`sera/low_rank/`. Dataset loader code is bundled under `datasets/`. These
components provide package-local dependencies and retain the existing source
notices in the affected files.

REA also requires a reference CLIP implementation. The required source is
vendored under `third_party/reference_clip/clip/`. Its license and a concise
provenance notice are preserved in `third_party/reference_clip/`. The SERA
experiment entry points and module descriptions use the terminology documented
in `docs/METHOD_TO_CODE.md`.

Bundled components retain their existing copyright and license notices.
These notices accompany the corresponding source files and remain part of the
distribution.
