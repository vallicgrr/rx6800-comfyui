# RX 6800 / Radeon 780M workflow evidence (2026-10-01)

These are ComfyUI API prompt graphs, not frontend editor workflows. Submit one as
the `prompt` field of a POST /prompt request to your own ComfyUI server.
RX graphs live under rx6800/; the 780M copies are under 780m/.
The JSON files retain the base prompts, seeds and output prefixes. For the two
final Music RX rows, the measured seeds differ from these base graphs:
20 s / 178.2 s used 527121; 120 s / 1157.9 s used 527124. Both 780M rows
and the archived Music graphs use 527001. Apply the RX override to all seed
inputs in the corresponding Music graph. measured-runs.json records these overrides.

This is a workflow archive, not a complete installation kit. It contains no model
weights, custom-node implementations, or RX runtime patches. The article describes
the tested ComfyUI/PyTorch versions. The RX timing depends on the local helper node,
comfy_kitchen changes, and H3 GGUF loader changes described in the lab report.
GGUF, LTXVideo and Hunyuan3DWrapper nodes must be available for their graphs.

The Hunyuan3D source image is included under input/. Place it in your ComfyUI input
folder, respecting the image path in the selected graph (780M uses rx6800_bench/).
LTX uses saved conditioning files identified in the graphs. Those tensors are NOT
included, so the LTX graphs require the original conditioning files to run.
manifest.json lists registered node types and common file inputs for each graph.

The accompanying lab report is chronological. Its first comparison table is current;
later sections preserve abandoned approaches and earlier measurements.
