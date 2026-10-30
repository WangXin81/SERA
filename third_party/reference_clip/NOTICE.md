# Reference CLIP implementation

The source files under `clip/` are derived from the official implementation
associated with *Low-Rank Few-Shot Adaptation of Vision-Language Models*
(CVPRW 2024) by Maxime Zanella and Ismail Ben Ayed:

https://github.com/MaxZanella/CLIP-LoRA

The bundled files are used only as the reference CLIP implementation required
by SERA's Residual Evidence Adaptation module. Packaging changes are limited
to repository placement and import namespaces; SERA's public modules,
command-line options, configurations, and result fields use SERA terminology.

See `LICENSE` in this directory for the applicable AGPL-3.0 license text.
