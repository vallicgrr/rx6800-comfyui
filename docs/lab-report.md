## Corrections to the retained lab notes

The chronological notes below retain earlier summaries. For the final comparison,
the RX Music 20 s and 120 s rows used seeds 527121 and 527124 respectively;
both 780M rows used 527001. The archived Music graphs retain 527001.
Repeated RX runs varied by up to about 10% (LTX 2.3: 119.5 and 107.1 s;
H3 Q3: 233.0 and 221.7 s). Short-lyrics 120 s attempts stopped early between
about 15 and 65 s; only the first stopped at about 30 s (749 tokens).
Both installations depend on the shared ComfyUI-GGUF loader changes for H3.
The intended INT8 calculation was restored on RX; the 780M kernel path was
not verified. The rocBLAS gfx1030 INT8 kernels are generic fallback builds.

---

# RX 6800 ComfyUI benchmark and troubleshooting report

**Run date:** 2026-09-30  
**Machine:** Windows PC with AMD Radeon RX 6800 (gfx1030, 16 GB VRAM) and Radeon 780M iGPU  
**Scope:** Run MiniMax, LTX video, and Hunyuan3D workflows informed by shino.dev articles on the RX 6800; keep results/settings for a later 780M comparison. The 780M comparisons have not been run.

## Final RX 6800 vs Radeon 780M comparison (2026-10-01)

Every row below was measured in this session with the **same API workflow file** on both cards (models, prompts, seeds, steps, resolution; only the output folder differs). Workflows: `benchmarks\*.json` (RX) and `benchmarks\780m\*.json` (780M). Times are ComfyUI execution time from queue start to the last node, measured through the websocket.

| Workflow | RX 6800 | Radeon 780M | RX speedup |
|---|---:|---:|---:|
| MiniMax Music 3, 20 s song (`minimax_music3_20s_api`) | **178.2 s** | 470.5 s | 2.6× |
| MiniMax Music 3, full 2-minute song (`minimax_music3_120s_full_api`) | **1157.9 s** | 2886.4 s | 2.5× |
| MiniMax H3 Q3_K_XL t2va, 640×384, 56 frames, 6 turbo steps | **233.0 s** | 636.0 s | 2.7× |
| MiniMax H3 Q2_K t2va, 768×448, 56 frames, 6 turbo steps | **353.2 s** | 825.1 s | 2.3× |
| Hunyuan3D-2mini full-body mesh, 30 steps, octree 384 | **102.3 s** | 171.7 s | 1.7× |
| Qwen-Image-2512 Q4_K_M OG image, 1216×640, 20 steps, CFG 2.5 | **345.7 s** | 1093.2 s | 3.2× |
| Qwen-Image 2.1 int8 OG image, 1216×640, 25 steps | **104.7 s** | 208.8 s | 2.0× |
| LTX 2.3 Q2_K, 512×288, 97 frames + audio, 8 steps | **119.5 s** | 288.9 s | 2.4× |
| LTX 2.5 Q3_K_S, 512×288, 97 frames + audio, 25 steps | **290.2 s** | 784.5 s | 2.7× |

**Conditions.**
- One card at a time: the RX server was stopped during every 780M run; during the RX runs the 780M server was idle, and before the RX H3 runs its models were released with ComfyUI's `/free` (unload models, free memory). Without that, the RX H3 runs paged system RAM (28.8 GB machine) and took 726–1127 s.
- 780M: ComfyUI-TheRock as installed (`--use-pytorch-cross-attention --reserve-vram 6`, GPU VAE, torch 2.12.0+rocm7.15), plus the shared ComfyUI-GGUF loader edits that let the MiniMax H3 GGUF files load at all.
- RX 6800: this tree with the fixes described below (`ComfyUI-RX6800-Fixes`, the RX venv's comfy_kitchen INT8/na3d changes), `--cuda-device 1 --use-pytorch-cross-attention --reserve-vram 2`, GPU VAE.
- Run order: on both cards H3 Q3 was the first job after a server start and H3 Q2 the second (each reloaded the 13 GB H3 text encoder). Each row is a single run; repeated RX runs during this work varied by about ±5%.
- Same seed, different numerics (RX FP16 vs 780M BF16), so outputs are not bit-identical across cards. Essentially the same image on both cards: Qwen-Image-2512 and Qwen-Image 2.1 (mean pixel difference 3.0 and 1.1 / 255), LTX 2.3 (7.6 / 255 at the middle frame), and the Hunyuan3D meshes. Different but equally coherent results from the same prompt: MiniMax H3, and LTX 2.5 (its `euler_ancestral` sampler injects new noise every step). MiniMax Music audio differs, and the 780M's is quieter (RMS 0.06 vs 0.16 at 20 s).

## Follow-up fixes (2026-09-30, later the same day)

The failing workflows now complete, MiniMax H3 runs for the first time, and the RX instance no longer forces CPU VAE. The original sections below are kept for the record; where they conflict, this section is current.

| Workflow | Before | After fixes | Output |
|---|---|---|---|
| **MiniMax Music 3**, 20 s, same API workflow (`minimax_music3_20s_api.json`) | No output; ~7.2 s/token (≈1 h projected for 501 tokens) | **457 s (7:37) end to end**: AR 501 tokens in 3:46 (2.2 tokens/s), diffusion 30 steps in 1:17 | [`minimax_music3_20s_00002.mp3`](../output/rx6800/minimax_music3_20s_00002.mp3) |
| **Hunyuan3D-2mini** full body, 30 steps, guidance 5.5, seed 680003, octree 384, 50k faces | Refinement stalled for 10+ min; no GLB | **103 s end to end** (sampling 33 s; decode + mesh ~20 s) | [`hunyuan3d2mini_knight_fullbody_masked_final_00001_.glb`](../output/rx6800/hunyuan3d2mini_knight_fullbody_masked_final_00001_.glb) |
| **LTX 2.3 video VAE** decode, 97 frames 512×288 (standalone test, random latent) | 44 s on CPU (`--cpu-vae`) | **1.9 s on GPU** (3.3 s cold process); matches CPU to 3e-6 | — |
| **LTX 2.5 diffusion video VAE** (`ltx-2.5-video-vae-bf16`) decode, same shape | >1 h on CPU (stopped); GPU OOM | **42.5 s on GPU** | — |

| **MiniMax H3** text-to-video+audio, 768×448, 56 frames (2.3 s) at 24 fps, Q2_K video model + turbo LoRA, 6 steps `res_multistep`, seed 680033 (`minimax_h3_t2va_768x448_turbo_api.json`) | Not run (780M article: OOM). First RX attempt: text-encoder GGUF rejected, then ~6 min/step with 14.5 GB spilled to shared memory | **299 s end to end**, ~34 s/step; prompt encoding ~30 s | [`minimax_h3_t2va_768x448_56f_turbo6_00001_.mp4`](../output/rx6800/minimax_h3_t2va_768x448_56f_turbo6_00001_.mp4). Coherent motion and audio; soft detail as expected at Q2/6 steps/low res. |
| **MiniMax H3 Q3_K_XL** t2va, 640×384, 56 frames, turbo LoRA 6 steps, same prompt/seed (`minimax_h3_q3kxl_t2va_640x384_turbo_api.json`) | 300 s/step: 9.3 GB model fully in VRAM but activations spilled 1.2 GB to shared memory | **221.7 s end to end**, 23.5 s/step, 98 MB shared | [`minimax_h3_q3kxl_t2va_640x384_56f_turbo6_00001_.mp4`](../output/rx6800/minimax_h3_q3kxl_t2va_640x384_56f_turbo6_00001_.mp4). Clearly sharper than Q2. Same size as the 780M's Aug 26 Q3 runs (`H3_Q3K_XL_56f_*`), whose steps/timing were not recorded. |

The LTX workflows themselves were not saved, so they were not rerun end to end; only their VAE decode was measured.

Root causes and changes:

- **MiniMax int8 GEMM.** The ROCm nightly's hipBLASLt has no gfx1030 kernels (`TensileLibrary_lazy_gfx1030.dat` missing), so `torch._int_mm` always fails. `comfy_kitchen\backends\eager\quantization.py` now probes `_int_mm` once per device and, on failure, runs the int8 GEMM as an fp16 rocBLAS GEMM: exact fp32-output for K > 32768, otherwise a 2^-7 pre-scaled fp16 HGEMM (~2e-4 relative rounding, well below int8 activation-quantization error). Also avoids a per-call weight transpose copy. About 16× faster per token than the FP32 fallback. Backup: `quantization.py.before-rx6800-fp16-fallback`.
- **Video VAE on GPU.** MIOpen has no CK conv kernels for gfx1030, so `Conv3d` runs 3–6× slower than the same work as `Conv2d`; long kernels plausibly caused the earlier HIP `unspecified launch failure` (Windows TDR). New RX-local node `custom_nodes\ComfyUI-RX6800-Fixes` runs stride-1 `comfy.ops` `Conv3d` as a sum of per-time-tap `Conv2d` calls on gfx1030. `--cpu-vae` removed from `start_rx6800.ps1`. fp16 LTX VAE decode is numerically wrong on this card; the VAE stays fp32 (ComfyUI's default here).
- **LTX 2.5 diffusion VAE OOM.** comfy_kitchen's eager `na3d` stacks many masked-SDPA tiles, bounded only on CPU; ROCm gfx1030 SDPA has no memory-efficient kernel, so it materialized >12 GB of scores. `comfy_kitchen\backends\eager\na.py` now applies the CPU score budget on HIP too. Backup: `na.py.before-rx6800-hip-score-budget`.
- **Hunyuan3D stall.** It was not mesh extraction: FlashVDM refinement runs cross attention with up to ~230k queries per call, and ROCm SDPA's math path built ~7.5 GB score tensors that spilled into shared memory (GPU near idle, 2–9 s per call, hundreds of calls). `ComfyUI-RX6800-Fixes` swaps the wrapper's attention for fixed 32k-query chunks of fp16 matmul + softmax on gfx1030, without editing the shared (symlinked) Hunyuan3D wrapper. ComfyUI's `attention_split` was also tried: fast in isolation, but under FlashVDM's memory pressure it chose oversized chunks (~6 min).
- **Large attention in general.** The same missing kernel made MiniMax H3's ~23k-token video self-attention spill (VRAM full, 14.5 GB shared, GPU idle). `ComfyUI-RX6800-Fixes` also wraps `comfy.ops.scaled_dot_product_attention` on gfx1030: unmasked, non-causal attention whose score matrix would exceed 256M elements runs in fixed query chunks (fp16 matmul + softmax, max error ~2e-4 vs SDPA), capped at 64M score elements per chunk (a 256M cap pushed H3 Q3_K_XL into shared memory; the smaller cap costs Hunyuan3D decode ~4 s, 20 → 24 s). Smaller or masked attention is untouched. ~364 s/step → ~34 s/step for H3.
- **MiniMax H3 GGUF loading** (edits to the shared ComfyUI-GGUF node, symlinked from ComfyUI-TheRock, so they also apply to the 780M instance). `qwen3vl_32b_minimax_h3-Q2_K_M.gguf` and `minimax_h3_fl2va_pruned-*.gguf` are ComfyUI-layout state dicts without llama.cpp/arch metadata. `loader.py` now loads arch-less text encoders with `model.layers.*` keys as-is and restores the Qwen-VL patch-embed conv weight to 5-D (GGUF stores at most 4 dims); `tools/convert.py` gets a MiniMax H3 arch signature. Backups: `loader.py.before-rx6800-comfy-layout-te`, `tools/convert.py.before-rx6800-minimax-h3`.
- **Hunyuan3D input.** `input\rx6800_knight_front_518.png` has gray side bars and a white card behind the character, which Hunyuan3D reconstructs as a slab (8.4M raw faces). The new `input\rx6800_knight_front_518_masked.png` crops the knight from the original on a flood-filled white background and pads it square at ~90% fill.

## Setup and changes

- Installed and used a separate ComfyUI tree at `ComfyUI-RX6800`, running on `http://127.0.0.1:8190`. The existing ComfyUI/780M instance on port 8188 was left alone.
- RX instance uses Python 3.12.10, ComfyUI 0.37.2, and `torch 2.12.0+rocm10.2.0a20260929` with ROCm target gfx1030.
- Configured model search paths to share existing weights from `ComfyUI-TheRock\models`. Model files were not copied or downloaded for this setup. Custom-node code for GGUF, LTX, and Hunyuan3D is linked into the RX tree.
- RX launch options originally included `--cuda-device 1`, `--use-pytorch-cross-attention`, `--cpu-vae`, and `--reserve-vram 2` (`--cpu-vae` since removed; see follow-up). CPU VAE decoding avoids failures seen in the GPU video VAE path; it does not automatically move Hunyuan3D's custom VAE to CPU.
- Added an RX-local `Hy3DVAEDecodeCPU` helper under `custom_nodes\ComfyUI-RX6800-Hy3DCPUDecoder` to try that custom Hunyuan3D CPU path without editing the shared Hunyuan3D node code.
- To work around RX ROCm's failing int8 GEMM for MiniMax Music 3, changed the RX venv's `comfy_kitchen\backends\eager\quantization.py` to use FP32 matrix multiplication for that operation. The prior version is backed up as `quantization.py.before-rx6800-fp32-fallback`.

## Results

| Workflow and RX settings | RX 6800 result | shino.dev comparison | Output / caveat |
|---|---:|---|---|
| **LTX 2.3**; 512×288, 97 frames, 24 fps, 8 steps, Q2_K, seed 680023; native tiled VAE decode; CPU VAE; audio enabled | **158.25 s (2:38)** | Article reports about 4:30 for a 97-frame run with audio on Radeon 780M. | [`ltx23_512x288_97f_cpuvae_00001_.mp4`](../output/rx6800/ltx23_512x288_97f_cpuvae_00001_.mp4) |
| **LTX 2.5**; 512×288, 97 frames, 24 fps, 25 steps, Q3_K_S, seed 680025; native 2×2 tiled VAE decode; CPU VAE | **351.73 s (5:52)** | Article's 512×288/97-frame/8-step 780M run is about 4:18. Its cited quality run is 1344×768/Q6_K/25 steps at about 1:27. | [`ltx25_Q3_512x288_97f_25steps_cpuvae_00001_.mp4`](../output/rx6800/ltx25_Q3_512x288_97f_25steps_cpuvae_00001_.mp4). RX settings differ in steps and quantization, so these times are not a like-for-like comparison. |
| **Hunyuan3D-2mini**, shape-only; image center-cropped to 518×518; 30 steps, guidance 5.5, seed 680003, VAE 384, maximum 50,000 faces | **173.34 s (2:53)** | The article describes a 518×518 shape-only workflow but gives no runtime. | [`hunyuan3d2mini_knight_00001_.glb`](../output/rx6800/hunyuan3d2mini_knight_00001_.glb). Gray, untextured mesh; the center crop cuts off some of the original 512×768 character. |
| **MiniMax Music 3**, article-style 20-second generation, 501 autoregressive steps | **No output.** First attempt failed at ROCm int8 GEMM. With the FP32 fallback, 13/501 tokens took about 93 s (~7.2 s/token); stopped as impractically slow. | Article reports about 8.5 minutes for 20 s on Radeon 780M: roughly 5 minutes autoregressive generation plus 2.5 minutes sampling. | No audio file. API prompt/workflow payload retained at `benchmarks\minimax_music3_20s_api.json`. |

The Hunyuan3D full-body retry used a padded square input at `input\rx6800_knight_front_518.png`, rather than cropping the character. Shape sampling completed and all 64 FlashVDM CPU VAE chunks decoded. The following mesh extraction then used several CPU cores for many minutes without progress or an exported GLB; that run was stopped. No corrected full-body output or reliable end-to-end time was obtained.

## Errors and attempted fixes

### MiniMax Music 3 int8 matrix multiplication

- **Error:** ROCm `HIPBLAS_STATUS_INVALID_VALUE` in `comfy_kitchen`'s `fast_int8_mm` path on gfx1030.
- **Tried:** RX-local fallback to FP32 matrix multiplication. It avoided the immediate HIPBLAS error but was too slow: 13/501 autoregressive tokens in ~93 seconds. The generation was interrupted, so no audio resulted.
- **Not tried:** A scaled FP16 or chunked matmul approximation, other ROCm kernels/backends, changing the quantization/model, or running MiniMax on the 780M. The FP16 approach considered could change numeric accuracy and was not validated. MiniMax H3 video was also not run; the cited article reports that H3 ran out of memory on 780M, but this is not a test of the RX 6800.

### Hunyuan3D VAE / mesh extraction

- **Error:** GPU custom VAE decode failed in scaled dot-product attention on gfx1030 and terminated the ComfyUI server. The regular LTX GPU sampling path did run; this is specific to the Hunyuan3D custom VAE path encountered here.
- **Tried:** Restarting the RX server with `--cpu-vae` (which alone did not move this custom VAE to CPU), then using the RX-local `Hy3DVAEDecodeCPU` helper. It decoded all 64 FlashVDM chunks but the mesh extraction did not finish in the observed many-minute window. That attempt was stopped and the RX ComfyUI server was restarted successfully.
- **Not tried:** Lowering the octree resolution below 384, lowering/tuning chunk sizes, disabling FlashVDM, testing a different marching-cubes backend, or profiling/optimizing the CPU extraction. The already successful cropped-input GLB remains the only completed Hunyuan3D result.

### Video VAE

- **Error:** The initial LTX 2.3 GPU tiled-VAE decode hit repeated HIP `unspecified launch failure` errors.
- **Tried:** Restarting ComfyUI with `--cpu-vae` and using the LTX-native tiled decoder. LTX 2.3 then completed. LTX 2.5 also completed with CPU VAE and native 2×2 tiling.
- **Not tried:** Further GPU VAE kernel/attention experimentation; no need was established after both requested video samples completed via CPU VAE.

## GPU fan observation

During LTX sampling, the RX 6800 showed about 84–88% activity in the Windows GPU engine view. Fans cycling off and on at light load is consistent with AMD Zero RPM behavior; it was not evidence that the card was idle during generation. See [AMD's Zero RPM FAQ](https://www.amd.com/en/resources/support-articles/faqs/DH-020.html).

## 780M comparison (2026-09-30 evening)

Same API workflows and seeds, run on the existing ComfyUI-TheRock instance (port 8188, ComfyUI 0.37.2, torch 2.12.0+rocm7.15, launch flags unchanged: `--use-pytorch-cross-attention --reserve-vram 6`, GPU VAE). The RX server was stopped during these runs so the two instances did not share RAM. No TheRock settings were changed; the knight input was uploaded to TheRock's `input\rx6800_bench\`. Workflow copies: `benchmarks\780m\`. Outputs: `ComfyUI-TheRock\output\780m_bench\`.

| Workflow | RX 6800 (fixed) | Radeon 780M (stock) | Notes |
|---|---:|---:|---|
| MiniMax Music 3, 20 s | 457 s | **470.5 s** | Nearly equal; the RX is held back by its int8 fallback (the 780M's ROCm has native int8 GEMM). Same seed, different numerics: 780M output is noticeably quieter (RMS 0.062 vs 0.156). |
| Hunyuan3D-2mini full body | 103 s | **171.7 s** | Meshes are nearly identical (same seed), cross-checking the RX attention/conv fixes. The FlashVDM stall does not occur on the 780M. |
| MiniMax H3 Q2_K t2va 768×448, 56 frames, 6 turbo steps | 299 s | **825.1 s** | Run after the 780M server was restarted (it now has the shared ComfyUI-GGUF loader edits). |
| MiniMax H3 Q3_K_XL t2va 640×384, 56 frames, 6 turbo steps | 221.7 s | **636.0 s** | Both outputs coherent and sharp; different scenes from the same seed (RX fp16 vs 780M bf16 numerics). 780M audio is very quiet (RMS 0.011 vs 0.093), the same pattern as MiniMax Music 3. |
| LTX 2.3 video VAE decode (standalone, 97 frames 512×288) | 1.9 s full frame | **13.1 s tiled** (first run); full frame aborts in bf16 `conv3d`; a second tiled run crashed with HIP `unspecified launch failure` | Tiled = 512 px / 64-frame tiles (VAEDecodeTiled defaults). 780M VAE runs bf16. |
| LTX 2.5 diffusion video VAE decode | 42.5 s | **OOM** (full frame and tiled) | Stock comfy_kitchen `na3d`; the RX venv has the score-budget patch, the 780M venv does not. |

The 780M server was running (idle) during the standalone VAE tests. Each 780M workflow was run once, on a server that had been up since 12:50.

## Qwen-Image OG backgrounds on the RX 6800

Workflows reconstructed from the prompt graphs embedded in the site's 780M OG PNGs (`og_local_ogp_qwen_image_00001_.png`, `og_qwen21_780m_s7201_00001_.png`): same prompts, negatives, seeds and samplers, 1216×640. Only the save prefix changed. Saved as `benchmarks\qwen_image_2512_og_local_ogp_1216x640_api.json` and `benchmarks\qwen_image_21_og_s7201_1216x640_api.json`.

| Model and settings | RX 6800 | Radeon 780M | Output |
|---|---:|---:|---|
| Qwen-Image-2512 Q4_K_M, 20 steps, CFG 2.5, Euler/simple, seed 1405 | **376.4 s** (15.4 s/step, 12.7 GB fully in VRAM) | 15 min 45 s average (published OGP post) | [`og_local_ogp_qwen_image_2512_00001_.png`](../output/rx6800/og_local_ogp_qwen_image_2512_00001_.png), nearly identical to the published background |
| Qwen-Image 2.1 int8 convrot, 25 steps, CFG 1, tiled VAE, seed 7201 | **137.0 s** (4.7–4.9 s/step); 211.8 s with the earlier int8 emulation | 3.3–3.4 min (published Qwen-Image 2.1 post) | [`og_qwen21_s7201_fp16deq_00001_.png`](../output/rx6800/og_qwen21_s7201_fp16deq_00001_.png) |

**int8 linear on the RX 6800 (second revision).** `int8_linear` in the RX venv's comfy_kitchen now dequantizes the int8 weight to fp16 per call and runs one fp16 GEMM (after the convrot activation rotation), instead of quantizing activations and emulating the int8 GEMM with int32 round trips. At Qwen-Image 2.1's 4096→24576 layer with 3200 tokens: 244 ms → 56 ms; relative error against an fp32 reference 4e-4 (the emulated path was ~9e-3 because of activation quantization). MiniMax Music 3 rerun: 429.9 s (diffusion 1:17 → 0:59; the 501-token AR stage stays at ~3:37, so it is not matmul-bound).

## MiniMax Music 3 AR speedup (third int8 revision)

The 501-token AR stage was bound by the int8 linears (~392 ms of ~440 ms per token: a 36-layer Qwen3-8B pass plus 7 passes of a 4-layer depth decoder, ~11B int8 weights read per token). rocBLAS FP16 GEMM with 2-row inputs streams large weights at only ~100 GB/s but does better on tiles up to ~6144×6144, and the fallback spent a full extra pass scaling the dequantized weight. For inputs of ≤ 64 rows, `int8_linear` now converts 6144×6144 weight tiles to FP16, multiplies unscaled (input pre-divided by 2^7 to keep FP16 in range), and scales the small output. Linear time per token 392 → 233 ms; error vs a float64 reference 2.9e-4 (was 4.2e-4). Larger inputs keep the dequantize-then-GEMM path.

| MiniMax Music 3, 20 s | AR 501 tokens | Diffusion 30 steps | Total |
|---|---:|---:|---:|
| Before (second revision) | 3:37 (2.3 tokens/s) | 0:59 | 429.9 s |
| After, fresh server (includes ~105 s model load) | 2:52 (2.9 tokens/s) | 1:00 | **373.0 s** |
| After, warm server (seed 527002) | 2:54 | 1:03 | **268.2 s** |
| Radeon 780M, stock (first music run on its server) | — | — | 470.5 s |

## Conv1d fix and remaining-bottleneck profiling (2026-10-01)

**MiniMax Music 3 audio decode.** Per-node timing on a freshly started server showed `VAEDecodeAudioTiled` taking 125 s (28 s warm). The DAV decoder's dilated `Conv1d` layers (768 ch × 12,288 samples, kernel 7, dilation 1/3/9) have only two MIOpen solvers on gfx1030: `ConvDirectNaiveConvFwd` and `GemmFwdRest`, which needs a 264 MB–1 GB workspace that PyTorch does not provide. MIOpen therefore ran a naive kernel and re-ran its find search for each shape in every new process ("Find-db regenerating"). `ComfyUI-RX6800-Fixes` now runs stride-1, ungrouped `Conv1d` on gfx1030 as one matmul per kernel tap: 436 → 9 ms for the dilation-9 layer, output within 6e-7 of the original.

| MiniMax Music 3, 20 s, fresh server | TextEncode (AR) | KSampler | Audio decode | Total |
|---|---:|---:|---:|---:|
| Before Conv1d fix | 180.8 s | 64.4 s | 125.4 s | 371.6 s |
| After | 180.7 s | 65.7 s | **2.5 s** | **249.5 s** |

Warm server after the fix: 249.7 s. MiniMax H3 Q3_K_XL rerun with the fix: sampling unchanged at 23.4 s/step, audio decode 1.4 s.

**Where the remaining time goes (measured, not yet changed):**

- Qwen-Image-2512 (1216×640, CFG batch): per step ~10.4 s matmul at ~24 TFLOPS (about 74% of the RX 6800's fp16 peak), ~1.4 s GGUF dequantization, ~3.6 s other. Near the rocBLAS ceiling.
- MiniMax H3 Q3_K_XL: GGUF dequantization ~1.3 s per forward; large fp16 GEMMs at 17–25 TFLOPS dominate. Near the rocBLAS ceiling.
- MiniMax Music 3 AR (~0.34 s/token): ~233 ms int8 linears (rocBLAS skinny-GEMM throughput), the rest small ops; Comfy's graph capture is active (`graph breaks: 0`). The depth decoder (4 layers, run 7× per token) re-dequantizes its 570M int8 weights each pass; keeping an FP16 copy for the duration of one generate call (~1.1 GB) is estimated to save ~20 s per 20 s song but needs a model-code change.
- MiniMax Music 3 DiT (~2.1 s/step in the server): int8 linears ~0.86 s/step on an idle GPU; the rest could not be attributed reliably (cProfile and torch.profiler both misattribute time on ROCm/Windows).

## MiniMax Music 3 depth-decoder FP16 copy (2026-10-01)

The AR loop's 4-layer depth decoder runs 7 times per token and dequantized its 570M INT8 weights on every pass. `ComfyUI-RX6800-Fixes` now wraps `MiniMaxMusic3AR.generate` on gfx1030: before generation it dequantizes the 16 quantized decoder Linear weights once to FP16 (~1.1 GB), routes those layers (including the fused SwiGLU down-projection via `comfy.ops.linear_input_act`) through plain FP16 linears for that call only, and drops the copies when the call returns. ComfyUI model code is unchanged. Per-layer error against the INT8 path ≤ 3.6e-4 on the real weights; graph capture still reports 0 breaks.

| MiniMax Music 3, 20 s | AR 501 tokens | TextEncode node | KSampler | Audio decode | Total |
|---|---:|---:|---:|---:|---:|
| Before, fresh server | 2:52 (2.9 tokens/s) | 180.7 s | 65.7 s | 2.5 s | 249.5 s |
| After, fresh server | **2:07 (3.9 tokens/s)** | 132.0 s | 66.1 s | 2.3 s | **201.4 s** |
| After, warm server (seed 527005) | — | 128.9 s | 63.4 s | 3.5 s | **196.0 s** |
| Radeon 780M, stock | | | | | 470.5 s |

Output peaks near or slightly above 1.0 appear in runs before and after this change (e.g. 1.015 and 1.021 before), so they are not caused by it.

## MiniMax Music 3 diffusion window batching and a 2-minute run (2026-10-01)

The DiT denoises songs longer than ~8 s as overlapping 689-frame windows with a 344-frame hop (5 windows for 20 s, 30 for 120 s), one transformer call each. `ComfyUI-RX6800-Fixes` now stacks same-length windows (up to 8) into one batch on gfx1030; windows are independent within a step, and the output is scattered and averaged exactly like the original loop. Final latents after 4 steps differ from the sequential loop by 0.17% relative (different GEMM kernel rounding).

Component timing showed the DiT is compute-bound: a 20 s step is ~33 TFLOP (feed-forward ~60%, attention core <10%), and at 1.42 s/step standalone that is ~23 TFLOPS, about the best rocBLAS reaches on this card. Batching therefore gives ~9%: server KSampler 63–66 s → **58.1 s** (2.1 → 1.93 s/step) for 20 s.

**2-minute song** (`benchmarks\minimax_music3_120s_api.json`: same prompt with "20-second" changed to "2-minute", `max_duration` and latent length 120 s):

| Stage | Measured | Note |
|---|---:|---|
| TextEncode (AR) | 196.3 s | The model emitted its stop token at 749/3001 tokens (~30 s of content) at 3.9 tokens/s |
| KSampler, 30 steps × 30 windows | 386.0 s | 12.6–12.9 s/step |
| Audio decode | 39.2 s | |
| **Total** | **623.5 s** | 120 s of non-silent audio (RMS 0.06–0.15 per 10 s) |

A 2-minute song whose AR runs the full 3001 tokens would take about 3001 / 3.9 ≈ 770 s for the AR stage, so roughly **20 minutes** in total.

**Audio decode slowdown after a long song (fixed below; original notes).** In a server process that has run a 2-minute song, later audio decodes slow down: 20 s songs 2.3 s → 16–22 s, and a second 2-minute run's decode took 168 s (39 s in the first). Restarting the server restores 2.3 s; `/free` (unload models and empty cache) does not. Tile size (512–1536) makes no difference. The same decoder runs 2.0 s standalone on an otherwise idle GPU but 27–32 s while the idle server holds ~9 GB, with Conv1d and Snake (memory-bound ops) ~15–20× slower, which points to VRAM oversubscription paging decode buffers into system memory. The DAV decode already requests ~7.6 GB per tile from ComfyUI, so the crowding memory is likely held outside its accounting (comfy-aimdo dynamic-VRAM state after a long generation); not confirmed. A second 2-minute run with seed 527010 emitted more AR tokens (TextEncode 426 s) and took 984.7 s in total.

## Full 2-minute MiniMax Music 3 benchmark (2026-10-01)

`benchmarks\minimax_music3_120s_full_api.json` (780M copy in `benchmarks\780m\`): the 20 s benchmark's style (96 BPM Japanese art-pop, same voice and instrumentation, seed 527001), with a caption describing a full two-minute structure and original lyrics for intro, two verses, three choruses, a bridge and an outro. `max_duration` and latent length 120 s.

The first RX attempt generated the full 3001-token AR sequence, then failed in the DiT's `aligned_condition`: its `einsum("blht,l->bht")` over the 8 text-encoder layers becomes a rocBLAS GEMM shaped [4096 × frames, 8] × [8, 1], whose launch grid is invalid on gfx1030 at 2986 frames (`hipErrorInvalidConfiguration` → `HIPBLAS_STATUS_INTERNAL_ERROR`; 501 frames works). The stock code fails the same way. `ComfyUI-RX6800-Fixes` now computes that weighted sum as eight multiply-adds on gfx1030 (max difference vs einsum at 501 frames: 0.003 in FP16).

| RX 6800, fresh server | Time |
|---|---:|
| TextEncode (AR 2986/3000 frames ≈ 119 s of content, 3.58 tokens/s) | 843.5 s |
| KSampler (30 steps × 30 windows, 12.4–12.7 s/step) | 380.2 s |
| Audio decode | 10.0 s |
| **Total** | **1235.2 s (20.6 min)** |

Output [`minimax_music3_120s_full_00001.mp3`](../output/rx6800/minimax_music3_120s_full_00001.mp3): 120.0 s; RMS per 10 s 0.07–0.21, quieter for the first 30 s and rising and falling afterwards.

**780M control** (stock TheRock code, same workflow; RX server stopped during the run):

| Stage | RX 6800 | Radeon 780M |
|---|---:|---:|
| TextEncode (AR) | 843.5 s, 2986/3000 frames | 1835.4 s, 2986/3000 frames |
| KSampler | 380.2 s | 976.5 s |
| Audio decode | 10.0 s | 68.1 s |
| **Total** | **1235.2 s (20.6 min)** | **2886.4 s (48.1 min)** |

Both cards ran the AR stage to the same length, so the early stop in the earlier 2-minute runs came from the short 20 s lyrics, not from the RX numerics. The 780M's einsum did not fail (gfx1103). 780M output: 120.0 s, overall RMS 0.152 (same as the RX), RMS per 10 s 0.07–0.21.

## Native INT8 through rocBLAS (2026-10-01)

hipBLASLt (what `torch._int_mm` uses) has no gfx1030 kernels, but plain rocBLAS ships INT8×INT8→INT32 Tensile kernels for gfx1030 (generic "fallback" builds). `comfy_kitchen\backends\eager\quantization.py` in the RX venv now calls `rocblas_gemm_ex` through ctypes (same `rocblas.dll` torch loads), with one handle per device bound to torch's current stream and a fixed 64 MB workspace so rocBLAS never allocates during HIP graph capture. `int8_linear` is back to its original algorithm on the RX (per-row INT8 activation quantization, INT8 GEMM, scale), the same math the 780M runs; `mm_int8` is bit-exact. Error against a float64 reference ~0.9% (activation quantization), versus ~3e-4 for the previous FP16-dequant path. Backup: `quantization.py.before-rx6800-rocblas-int8`.

Raw GEMM: ~30–31 TOPS INT8 vs ~25–27 TFLOPS FP16 at large M; 2-row decode 1.04 vs 2.11 ms (24576×4096).

| Workflow | Before (FP16-dequant fallback) | rocBLAS INT8 | 780M |
|---|---:|---:|---:|
| MiniMax Music 3, 20 s, fresh server | 193.3 s (AR 2:07, 3.9 tokens/s) | **180.3 s** (AR 1:50, 4.55 tokens/s) | 470.5 s |
| Qwen-Image 2.1 OG, 1216×640, 25 steps | 137.0 s (4.7 s/step) | **105.3 s** (3.55 s/step) | ~200 s |

Full 2-minute MiniMax song (same server process, after the 20 s and Qwen runs): **1199.6 s** vs 1235.2 s. TextEncode 843.5 → 745.4 s (2986 frames, ~4.0 tokens/s; long KV caches make late tokens slower than the 20 s song's 4.55/s), KSampler 380.2 → 404.0 s, audio decode 10.0 → 48.0 s. The decode and part of the KSampler increase match the open slowdown-after-long-AR issue above rather than the INT8 change.

MiniMax diffusion is unchanged (~1.9 s/step) on a fresh server. Qwen-Image 2.1 output: [`og_qwen21_s7201_rocblas_int8_00001_.png`](../output/rx6800/og_qwen21_s7201_rocblas_int8_00001_.png), visually the same image. HIP graph capture of the AR loop still reports 0 breaks.

## Fix: MiniMax slowdown after long songs (2026-10-01)

Cause: VRAM oversubscription across MiniMax stages. ComfyUI sizes VRAM from `mem_get_info`, which on this Windows/ROCm setup reports ~9 GB free while the 8.7 GB text encoder (and then the DiT) stay resident, so it never unloads them. A GPU memory trace showed a 20 s song peaking at 11.9 GB (AR) / 14.8 GB (DiT, decode) on a fresh server, while after a 2-minute job the next song sat at 15.4–16.1 GB and its decode put 2 GB in shared system memory. Large allocations that overflow VRAM run at ~51 GB/s instead of ~450 GB/s. The extra residency was invisible to torch, comfy-aimdo and the HIP memory pool counters; the exact owner was not identified.

Fix in `ComfyUI-RX6800-Fixes` (gfx1030 only): unload idle models at MiniMax stage boundaries, before sampling a MiniMax Music 3 DiT (`comfy.sample.sample` / `sample_custom`) and before decoding with the DAV audio VAE (`VAE.decode` / `decode_tiled`). The next song reloads them from RAM, as it already did between workflows.

| Same server process, in order | AR | KSampler | Decode | Total |
|---|---:|---:|---:|---:|
| 20 s song | 115.0 s | 59.7 s | 2.2 s | 178.2 s |
| Full 2-minute song | 743.5 s | 403.5 s | 9.9 s | **1157.9 s** |
| 20 s song after it | 113.3 s | 59.7 s | **2.0 s** | 175.2 s |
| 20 s song again | 113.2 s | 59.7 s | 2.1 s | 175.2 s |

Before the fix the same order gave a 1035.7 s KSampler for the 2-minute song and 16–22 s decodes afterwards.

## LTX 2.3 / 2.5 rebuilt and rerun on GPU VAE (2026-10-01)

The original LTX workflows were recovered from the prompt graphs embedded in the original outputs (`ltx23_512x288_97f_cpuvae_00001_.mp4`, `ltx25_Q3_512x288_97f_25steps_cpuvae_00001_.mp4`): same GGUF models, saved conditioning (`ltx23_ball_*`, `ltx25_ball_*` / `*_blurry_negative`), seeds, steps and tiling. The only change is `LTXVTiledVAEDecode` `working_device` `cpu` → `auto`; the server no longer runs `--cpu-vae`. Saved as `benchmarks\ltx23_512x288_97f_gpuvae_api.json` and `benchmarks\ltx25_Q3_512x288_97f_25steps_gpuvae_api.json`. (`working_device` only places the tile accumulation buffers; the VAE itself runs wherever ComfyUI loaded it.)

The first GPU run of LTX 2.5 decoded in 33–39 s (slower than the old CPU decode) with 1.6 GB in shared system memory: the same oversubscription as MiniMax, here with the ~10 GB LTX model resident during the decode. The stage-boundary unload in `ComfyUI-RX6800-Fixes` now also runs before any video VAE decode (5-D latents) on gfx1030, keeping only the VAE being used, so LTX's per-tile `vae.decode` calls unload the diffusion model once.

| Workflow | Original (`--cpu-vae`) | GPU VAE, before unload fix | **GPU VAE, final** |
|---|---:|---:|---:|
| LTX 2.3, 512×288, 97 f, 8 steps, Q2_K | 158.25 s | 134.8 s (decode 5.3 s) | **119.5 s** (sampler 107.5 s, decode 8.2 s); 107.1 s on a second run |
| LTX 2.5, 512×288, 97 f, 25 steps, Q3_K_S, 2×2 tiles | 351.73 s | 373.5 s (decode 33.0 s) | **290.2 s** (sampler 279.3 s, decode 7.7 s) |

Frames from the original CPU-VAE and new GPU-VAE videos are visually identical at the start, middle and end; mean pixel values and audio RMS match. LTX 2.5's audio track is near-silent (RMS 0.001) in both the original and the new run.

## Earlier note on a 780M comparison

For a fair rerun, match the RX resolution, frame count, FPS, steps, quantization, seed, audio setting, tiled-decoder settings, prompts/conditioning, and input image. Avoid copying weights: both instances can point to the existing shared model directory.

## Source articles

- [LTX video on Radeon 780M](https://shino.dev/posts/ltx-video-780m/)
- [LTX 2.5 split pipeline](https://shino.dev/posts/ltx25-split-pipeline/)
- [MiniMax Music 3 on Radeon 780M](https://shino.dev/posts/minimax-music3-780m/)
- [Sprites to meshes with Hunyuan3D](https://shino.dev/posts/sprites-to-meshes-780m/)
