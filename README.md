# ComfyUI on the Radeon RX 6800 (gfx1030, Windows ROCm)

Runtime fixes, workflow graphs and benchmark notes behind the post
*ComfyUI on the RX 6800: fixing INT8, GPU VAE decode, and VRAM spill*
(shino.dev, October 2026).

This is a snapshot of one tested setup, not a maintained project. The fixes target the
exact versions listed below and may stop working, or become unnecessary, with other
ComfyUI, comfy-kitchen or ROCm builds.

## Results

Same API graph on both cards, one card generating at a time. Single runs; repeated RX runs
varied by up to about 10%. See `workflows/README.md` and the post for conditions.

| Workflow | RX 6800 | Radeon 780M | Speedup |
| --- | ---: | ---: | ---: |
| MiniMax Music 3, 20 s song | 178.2 s | 470.5 s | 2.6× |
| MiniMax Music 3, full 120 s song | 1157.9 s | 2886.4 s | 2.5× |
| MiniMax H3 Q3_K_XL, 640×384, 56 frames, 6 steps | 233.0 s | 636.0 s | 2.7× |
| MiniMax H3 Q2_K, 768×448, 56 frames, 6 steps | 353.2 s | 825.1 s | 2.3× |
| Hunyuan3D-2mini full-body mesh | 102.3 s | 171.7 s | 1.7× |
| Qwen-Image-2512 Q4_K_M, 1216×640, 20 steps | 345.7 s | 1093.2 s | 3.2× |
| Qwen-Image 2.1 INT8, 1216×640, 25 steps | 104.7 s | 208.8 s | 2.0× |
| LTX 2.3 Q2_K, 512×288, 97 frames + audio, 8 steps | 119.5 s | 288.9 s | 2.4× |
| LTX 2.5 Q3_K_S, 512×288, 97 frames + audio, 25 steps | 290.2 s | 784.5 s | 2.7× |

The two MiniMax Music RX rows used seeds 527121 (20 s) and 527124 (120 s); everything else,
including both 780M Music rows, used the seeds in the graphs. `workflows/measured-runs.json`
records the overrides.

## Tested setup

| | RX 6800 | Radeon 780M |
| --- | --- | --- |
| OS | Windows 11 | Windows 11 (same PC) |
| ComfyUI | 0.37.2 | 0.37.2 |
| Python | 3.12.10 | 3.12.10 |
| PyTorch | `2.12.0+rocm10.2.0a20260929` (TheRock nightly) | `2.12.0+rocm7.15.0a20260728` |
| comfy-kitchen / comfy-aimdo | 0.2.35 / 0.5.5 | 0.2.35 / 0.5.5 |
| Launch flags | `--cuda-device 1 --use-pytorch-cross-attention --reserve-vram 2` | `--use-pytorch-cross-attention --reserve-vram 6` |
| Custom nodes | ComfyUI-GGUF (`6ea2651` + both patches), ComfyUI-LTXVideo, ComfyUI-Hunyuan3DWrapper, **ComfyUI-RX6800-Fixes** | same, without ComfyUI-RX6800-Fixes |

## What is in this repository

- `ComfyUI-RX6800-Fixes/` — a custom node with no nodes of its own. On import it overrides
  a few ComfyUI and comfy-kitchen functions; every override checks for `gfx1030` and otherwise
  calls the original. Copy the folder into `ComfyUI/custom_nodes/`.
- `patches/` — changes to [ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF) at commit
  `6ea2651`, applied in order with `git apply` from the ComfyUI-GGUF folder:
  - `01-ComfyUI-GGUF-local-ltxv-gemma4.patch`: local changes that predate this work and
    were in place for both cards (Gemma 4 text arch, BF16 LTX-AV raw parameters).
  - `02-ComfyUI-GGUF-minimax-h3.patch`: loads the MiniMax H3 GGUF files used here (a Qwen3-VL
    text encoder in ComfyUI key layout without llama.cpp metadata, and an H3 diffusion
    model arch signature). Needed on both cards.
- `workflows/` — ComfyUI API prompt graphs for both cards, `manifest.json` (node types and
  file inputs per graph), `measured-runs.json`, and the Hunyuan3D input image.
- `docs/lab-report.md` — the chronological lab report, including abandoned approaches and
  earlier measurements. Its first table is current.

Not included: model weights, the LTX conditioning tensors the LTX graphs load, or the
original FP32/FP16 fallbacks that the report describes as intermediate steps.

## The fixes (all gfx1030-only)

| Fix | Problem on this setup | Effect measured |
| --- | --- | --- |
| INT8 linear through rocBLAS | `torch._int_mm` uses hipBLASLt, which has no gfx1030 kernels (`HIPBLAS_STATUS_INVALID_VALUE`). rocBLAS does have INT8 GEMM kernels for gfx1030 (generic "fallback" builds), called here through ctypes with a fixed 64 MB workspace so it is safe under HIP graph capture. | MiniMax Music 3 runs at all; vs an FP16 dequantize path: 2-row decode 2.1 → 1.0 ms, Qwen-Image 2.1 137 → 105 s |
| Conv3d as per-tap Conv2d | MIOpen has no CK kernels for gfx1030; 3D convs run slowly and long kernels hit `unspecified launch failure` | LTX 2.3 video VAE decode 44 s (CPU) → 1.9 s (GPU) |
| Conv1d as per-tap matmul | Dilated audio-VAE convs only get MIOpen's naive kernel (the GEMM solver needs a workspace PyTorch does not pass) | 436 → 9 ms per layer; MiniMax 20 s decode 125 → 2.5 s |
| Chunked attention | No memory-efficient SDPA kernel for gfx1030; large score matrices spill into system memory | H3 ~6 min → ~34 s/step; Hunyuan3D refinement no longer stalls |
| MiniMax Music 3 AR depth decoder in FP16 | The 4-layer depth decoder re-dequantizes 570M INT8 weights 7× per token | ~1.1 GB extra VRAM during generation; AR 2:52 → 2:07 per 20 s song |
| MiniMax DiT window batching | Long songs denoise ~690-frame windows one call at a time | ~9% faster diffusion |
| MiniMax `aligned_condition` weighted sum | The original `einsum` becomes a rocBLAS GEMM with an invalid launch grid at ~3000 frames | 2-minute songs complete |
| Unload idle models at stage boundaries | ComfyUI sizes VRAM from `mem_get_info`, which reports ~9 GB free while the previous stage's model stays resident; activations then page into system memory | No slowdown after long songs; LTX 2.5 decode 35 → 8 s |

## Install on an equivalent setup

1. Install the custom nodes listed above, then apply the two patches to ComfyUI-GGUF:
   ```bash
   cd ComfyUI/custom_nodes/ComfyUI-GGUF
   git checkout 6ea2651
   git apply path/to/patches/01-ComfyUI-GGUF-local-ltxv-gemma4.patch
   git apply path/to/patches/02-ComfyUI-GGUF-minimax-h3.patch
   ```
2. Copy `ComfyUI-RX6800-Fixes/` into `ComfyUI/custom_nodes/`. ComfyUI imports custom nodes in
   directory order, so ComfyUI-Hunyuan3DWrapper must load first for its attention patch; the
   node logs a warning if it did not.
3. Launch with the RX 6800 flags above. `--cpu-vae` is not needed.
4. Submit a graph from `workflows/rx6800/` as the `prompt` field of `POST /prompt`.

The rocBLAS binding loads `rocblas.dll` from the TheRock wheel layout
(`site-packages/_rocm_sdk_libraries/bin`); other ROCm packagings need that path adjusted.

## License

GPL-3.0, matching ComfyUI, which the custom node builds on. The ComfyUI-GGUF patches modify
Apache-2.0 code. Workflow graphs and the input image are provided for reproduction.
