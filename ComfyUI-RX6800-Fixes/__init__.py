"""RX 6800 (gfx1030) runtime fixes for ComfyUI on Windows ROCm.

Every override checks for gfx1030 and otherwise defers to the original code.

INT8 linear: torch._int_mm goes through hipBLASLt, which ships no gfx1030 kernels
(HIPBLAS_STATUS_INVALID_VALUE). Plain rocBLAS does have INT8 x INT8 -> INT32 kernels for
gfx1030, so comfy_kitchen's eager int8_linear is replaced on gfx1030 with the same math
(per-row activation quantization, INT8 GEMM, scales) through rocblas_gemm_ex via ctypes.

Conv3d: MIOpen has no CK kernels for gfx1030, so 3D convs run 3-6x slower than the
same work as per-time-tap Conv2d calls; long kernels also trip Windows TDR ("unspecified
launch failure" in video VAE decode).

Conv1d: the only MIOpen solvers for MiniMax Music 3's dilated audio-VAE convs are a naive
kernel and a GEMM needing a workspace PyTorch does not pass; one matmul per kernel tap is
40x+ faster, and MIOpen's per-process search for those shapes cost ~2 min per server start.

Attention: ROCm SDPA has no memory-efficient kernel for gfx1030, so large attention
(Hunyuan3D FlashVDM's ~200k-query cross attention, MiniMax H3's ~23k-token video self
attention) materializes multi-GB scores and spills to shared memory, leaving the GPU idle.
"""
import ctypes
import logging
import os
import sys

import torch
import torch.nn.functional as F

import comfy_kitchen.backends.eager as ck_eager
from comfy_kitchen.backends.eager import quantization as ck_quantization

import comfy.ldm.minimax_music.ar
import comfy.ldm.minimax_music.dav
import comfy.ldm.minimax_music.dit
import comfy.model_management
import comfy.ops
import comfy.sample
import comfy.sd
from comfy.quant_ops import QuantizedTensor

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

_gfx1030_cache: dict[int, bool] = {}


def _is_gfx1030(tensor):
    if not tensor.is_cuda or torch.version.hip is None:
        return False
    index = tensor.get_device()
    if index not in _gfx1030_cache:
        _gfx1030_cache[index] = "gfx1030" in torch.cuda.get_device_properties(index).gcnArchName
    return _gfx1030_cache[index]


# rocBLAS INT8 GEMM: ~1.2x the FP16 GEMM rate at large M and 2x at 2-row decode shapes,
# bit-exact INT32 accumulation. A fixed per-device workspace keeps rocBLAS from allocating
# during HIP graph capture (MiniMax Music 3's AR loop is graph-captured).
_ROCBLAS = None
_rocblas_handles: dict[int, tuple] = {}
_ROCBLAS_WORKSPACE_BYTES = 64 * 1024 * 1024


def _rocblas_handle(device):
    global _ROCBLAS
    if _ROCBLAS is None:
        # TheRock ROCm wheels ship rocblas.dll next to the torch package.
        path = os.path.join(os.path.dirname(torch.__file__), "..", "_rocm_sdk_libraries", "bin", "rocblas.dll")
        _ROCBLAS = ctypes.CDLL(os.path.abspath(path))
        _ROCBLAS.rocblas_gemm_ex.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int, ctypes.c_int32, ctypes.c_uint32,
        ]
        _ROCBLAS.rocblas_set_stream.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        _ROCBLAS.rocblas_set_workspace.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
    index = device.index if device.index is not None else torch.cuda.current_device()
    entry = _rocblas_handles.get(index)
    if entry is None:
        with torch.cuda.device(index):
            handle = ctypes.c_void_p()
            if _ROCBLAS.rocblas_create_handle(ctypes.byref(handle)) != 0:
                raise RuntimeError("rocblas_create_handle failed")
            workspace = torch.empty(_ROCBLAS_WORKSPACE_BYTES, dtype=torch.uint8, device=device)
            _ROCBLAS.rocblas_set_workspace(handle, workspace.data_ptr(), _ROCBLAS_WORKSPACE_BYTES)
        entry = (handle, workspace)
        _rocblas_handles[index] = entry
    return entry[0]


def _rocblas_int8_mm(x, weight):
    """x [M, K] @ weight [N, K].T with INT32 accumulation (row-major, both contiguous)."""
    m, k = x.shape
    n = weight.shape[0]
    out = torch.empty((m, n), dtype=torch.int32, device=x.device)
    handle = _rocblas_handle(x.device)
    _ROCBLAS.rocblas_set_stream(handle, torch.cuda.current_stream(x.device).cuda_stream)
    one, zero = ctypes.c_int32(1), ctypes.c_int32(0)
    # Column-major view: out.T [N, M] = weight [N, K] (stored as K x N, transposed) @ x.T [K, M].
    # 112/111 = rocblas_operation_transpose/none, 160/162 = rocblas_datatype_i8_r/i32_r.
    status = _ROCBLAS.rocblas_gemm_ex(
        handle, 112, 111, n, m, k, ctypes.byref(one),
        weight.data_ptr(), 160, k, x.data_ptr(), 160, k, ctypes.byref(zero),
        out.data_ptr(), 162, n, out.data_ptr(), 162, n, 162, 0, 0, 0,
    )
    if status != 0:
        raise RuntimeError(f"rocblas_gemm_ex int8 failed with status {status}")
    return out


_original_int8_linear = ck_eager.int8_linear


def _int8_linear(x, weight, weight_scale, bias=None, out_dtype=torch.bfloat16, convrot=False, convrot_groupsize=256,
                 input_act=None, input_act_weight=None, input_act_eps=0.0, residual=None, residual_scale=None):
    if not _is_gfx1030(x):
        return _original_int8_linear(x, weight, weight_scale, bias, out_dtype, convrot, convrot_groupsize,
                                     input_act, input_act_weight, input_act_eps, residual, residual_scale)
    x = ck_quantization._apply_input_act(x, input_act, input_act_weight, input_act_eps)
    weight = weight.to(device=x.device).contiguous()
    weight_scale = weight_scale.to(device=x.device, dtype=torch.float32).reshape(1, -1)
    if convrot:
        hadamard = ck_quantization._build_hadamard(convrot_groupsize, device=x.device, dtype=x.dtype)
        x = ck_quantization._rotate_activation(x, hadamard, convrot_groupsize)
    x_8, x_scale = ck_quantization.quantize_int8_rowwise(x.reshape(-1, x.shape[-1]))
    result = _rocblas_int8_mm(x_8.contiguous(), weight).float()
    result = result.mul_(x_scale.float().reshape(-1, 1)).mul_(weight_scale).to(out_dtype)
    if bias is not None:
        result = result + bias.to(device=result.device, dtype=result.dtype).reshape(1, -1)
    result = result.reshape(*x.shape[:-1], weight.shape[0])
    return ck_quantization._apply_residual(result, residual, residual_scale)


# comfy_kitchen's registry resolves eager implementations with getattr on this module per call.
ck_eager.int8_linear = _int8_linear


def _conv3d_as_conv2d(x, weight, bias, padding):
    pad_t, pad_h, pad_w = padding
    if pad_t:
        x = F.pad(x, (0, 0, 0, 0, pad_t, pad_t))
    batch, channels, frames, height, width = x.shape
    out_frames = frames - weight.shape[2] + 1
    # [B, T, C, H, W] makes every temporal window a contiguous [B*T', C, H, W] view.
    x = x.transpose(1, 2).contiguous()
    out = None
    for i in range(weight.shape[2]):
        window = x[:, i:i + out_frames].reshape(batch * out_frames, channels, height, width)
        y = F.conv2d(window, weight[:, :, i], None, padding=(pad_h, pad_w))
        out = y if out is None else out.add_(y)
    out = out.reshape(batch, out_frames, *out.shape[1:]).transpose(1, 2).contiguous()
    if bias is not None:
        out += bias.reshape(1, -1, 1, 1, 1)
    return out


_original_conv3d_forward = comfy.ops.disable_weight_init.Conv3d._conv_forward


def _conv3d_forward(self, input, weight, bias, autopad=None, *args, **kwargs):
    if (not args and not kwargs and _is_gfx1030(input) and self.groups == 1 and self.padding_mode == "zeros"
            and isinstance(self.padding, tuple) and tuple(self.stride) == (1, 1, 1) and tuple(self.dilation) == (1, 1, 1)):
        if autopad == "causal_zero":
            weight = weight[:, :, -input.shape[2]:, :, :]
        return _conv3d_as_conv2d(input, weight, bias, self.padding)
    return _original_conv3d_forward(self, input, weight, bias, autopad, *args, **kwargs)


comfy.ops.disable_weight_init.Conv3d._conv_forward = _conv3d_forward


def _conv1d_as_matmul(x, weight, bias, padding, dilation):
    x = F.pad(x, (padding, padding))
    out_len = x.shape[-1] - dilation * (weight.shape[-1] - 1)
    out = None
    for i in range(weight.shape[-1]):
        y = torch.matmul(weight[:, :, i], x[:, :, i * dilation:i * dilation + out_len])
        out = y if out is None else out.add_(y)
    if bias is not None:
        out += bias.reshape(1, -1, 1)
    return out


_original_conv1d_forward = comfy.ops.disable_weight_init.Conv1d._conv_forward


def _conv1d_forward(self, input, weight, bias):
    if (input.ndim == 3 and _is_gfx1030(input) and self.groups == 1 and self.padding_mode == "zeros"
            and isinstance(self.padding, tuple) and tuple(self.stride) == (1,)):
        return _conv1d_as_matmul(input, weight, bias, self.padding[0], self.dilation[0])
    return _original_conv1d_forward(self, input, weight, bias)


comfy.ops.disable_weight_init.Conv1d._conv_forward = _conv1d_forward


# Max attention-score elements per chunk (64M: 128 MB of fp16 scores; larger chunks pushed
# MiniMax H3 Q3_K_XL at 640x384 past VRAM into shared memory). Chunk sizes are fixed
# from the shapes: attention_split sizes them from reported free memory, which overcommits
# on this card (Hunyuan3D decode 20s vs ~6min).
SCORE_BUDGET = 2 ** 26


def _chunked_attention(q, k, v, scale=None):
    batch, heads, n_q, _ = q.shape
    chunk = max(1, SCORE_BUDGET // (batch * heads * k.shape[-2]))
    kt = k.transpose(-1, -2) * (q.shape[-1] ** -0.5 if scale is None else scale)
    return torch.cat([(q[:, :, i:i + chunk] @ kt).softmax(dim=-1) @ v for i in range(0, n_q, chunk)], dim=2)


_original_sdpa = comfy.ops.scaled_dot_product_attention


def _sdpa(q, k, v, *args, **kwargs):
    if (not args and q.ndim == 4 and _is_gfx1030(q) and kwargs.get("attn_mask") is None
            and not kwargs.get("is_causal", False) and not kwargs.get("enable_gqa", False)
            and kwargs.get("dropout_p", 0.0) == 0.0
            and q.shape[0] * q.shape[1] * q.shape[2] * k.shape[-2] > SCORE_BUDGET):
        return _chunked_attention(q, k, v, kwargs.get("scale"))
    return _original_sdpa(q, k, v, *args, **kwargs)


comfy.ops.scaled_dot_product_attention = _sdpa


def _hy3d_attention(q, k, v):
    if _is_gfx1030(q):
        return _chunked_attention(q, k, v)
    return F.scaled_dot_product_attention(q, k, v)


# MiniMax Music 3 AR: the 4-layer depth decoder runs 7 times per token and re-dequantizes its
# 570M INT8 weights on every pass. On gfx1030 an FP16 copy held for one generate() call
# (~1.1 GB) cuts it from ~82 to ~36 ms per token (~23 s per 20 s song).
_depth_fp16 = {}


def _depth_fp16_forward(module):
    def forward(input, *args, **kwargs):
        return F.linear(input.to(torch.float16), _depth_fp16[module]).to(input.dtype)
    return forward


_original_linear_input_act = comfy.ops.linear_input_act


def _linear_input_act(linear, x, input_act, act_weight=None, act_eps=0.0, residual=None, residual_scale=None):
    weight = _depth_fp16.get(linear)
    if weight is None:
        return _original_linear_input_act(linear, x, input_act, act_weight, act_eps, residual, residual_scale)
    out = F.linear(comfy.ops._eager_input_act(x, input_act, act_weight, act_eps).to(torch.float16), weight).to(x.dtype)
    return out if residual is None else torch.addcmul(residual, out, residual_scale)


comfy.ops.linear_input_act = _linear_input_act

_original_ar_generate = comfy.ldm.minimax_music.ar.MiniMaxMusic3AR.generate


def _ar_generate(self, input_ids, seed, max_audio_frames, device, *args, **kwargs):
    if torch.device(device).type != "cuda" or torch.version.hip is None or "gfx1030" not in torch.cuda.get_device_properties(torch.device(device)).gcnArchName:
        return _original_ar_generate(self, input_ids, seed, max_audio_frames, device, *args, **kwargs)
    linears = [m for m in self.model.audio_decoder.layers.modules() if isinstance(getattr(m, "weight", None), QuantizedTensor)]
    try:
        for m in linears:
            probe = torch.empty((1, m.in_features), dtype=torch.float16, device=device)
            weight, bias, offload_stream = comfy.ops.cast_bias_weight(m, probe, offloadable=True, compute_dtype=torch.float16, want_requant=True)
            _depth_fp16[m] = (weight.dequantize() if isinstance(weight, QuantizedTensor) else weight).to(torch.float16).contiguous()
            comfy.ops.uncast_bias_weight(m, weight, bias, offload_stream)
            m.forward = _depth_fp16_forward(m)
        return _original_ar_generate(self, input_ids, seed, max_audio_frames, device, *args, **kwargs)
    finally:
        for m in linears:
            _depth_fp16.pop(m, None)
            m.__dict__.pop("forward", None)


comfy.ldm.minimax_music.ar.MiniMaxMusic3AR.generate = _ar_generate


# MiniMax Music 3 DiT: songs longer than ~8 s are denoised as overlapping ~690-frame windows,
# one transformer call each (~1400 GEMM rows, ~10 TFLOPS on gfx1030). Windows are independent
# within a step, so same-length windows are stacked into one batch to reach the larger-GEMM rate.
DIT_WINDOW_BATCH = 8
_original_dit_forward = comfy.ldm.minimax_music.dit.MiniMaxMusic3DiT.forward


def _dit_forward(self, x, timestep, context, conditioning_scale, **kwargs):
    window = comfy.ldm.minimax_music.dit.latent_length(comfy.ldm.minimax_music.dit.MAX_CONDITION_FRAMES)
    if not _is_gfx1030(x) or x.shape[-1] <= window:
        return _original_dit_forward(self, x, timestep, context, conditioning_scale, **kwargs)
    condition = self.aligned_condition(context)
    condition = condition * conditioning_scale[:, :1, :1]
    if condition.shape[-1] < x.shape[-1]:
        condition = torch.nn.functional.pad(condition, (0, x.shape[-1] - condition.shape[-1]))
    else:
        condition = condition[..., :x.shape[-1]]

    hop = comfy.ldm.minimax_music.dit.latent_length(comfy.ldm.minimax_music.dit.CONDITION_HOP_FRAMES)
    spans = []
    start = 0
    while start < x.shape[-1]:
        end = min(start + window, x.shape[-1])
        spans.append((start, end))
        if end == x.shape[-1]:
            break
        start += hop

    output = torch.zeros_like(x)
    count = torch.zeros((1, 1, x.shape[-1]), device=x.device, dtype=x.dtype)
    batch = x.shape[0]
    by_length = {}
    for span in spans:
        by_length.setdefault(span[1] - span[0], []).append(span)
    for group in by_length.values():
        for i in range(0, len(group), DIT_WINDOW_BATCH):
            chunk = group[i:i + DIT_WINDOW_BATCH]
            out = self.diffusion_transformer(
                torch.cat([x[..., s:e] for s, e in chunk]),
                timestep.repeat(len(chunk)),
                torch.cat([condition[..., s:e] for s, e in chunk]),
            )
            for j, (s, e) in enumerate(chunk):
                output[..., s:e] -= out[j * batch:(j + 1) * batch]
                count[..., s:e] += 1
    return output / count


comfy.ldm.minimax_music.dit.MiniMaxMusic3DiT.forward = _dit_forward

_original_aligned_condition = comfy.ldm.minimax_music.dit.MiniMaxMusic3DiT.aligned_condition


def _aligned_condition(self, hidden):
    # The 8-layer einsum becomes a [4096*frames, 8] x [8, 1] rocBLAS GEMM whose launch grid is
    # invalid on gfx1030 past ~2000 frames (HIPBLAS_STATUS_INTERNAL_ERROR on 2-minute songs).
    if not _is_gfx1030(hidden):
        return _original_aligned_condition(self, hidden)
    frames = hidden.shape[1]
    hidden = hidden.transpose(1, 2).reshape(hidden.shape[0], 8, 4096, frames)
    weights = torch.softmax(comfy.ops.cast_to_input(self.cond_layer_logits, hidden), dim=0)
    mixed = hidden[:, 0] * weights[0]
    for layer in range(1, 8):
        mixed.add_(hidden[:, layer] * weights[layer])
    mixed = comfy.ops.cast_to_input(self.cond_layer_scale, mixed) * mixed
    condition = self.latent_conditioners(mixed)
    return torch.nn.functional.interpolate(condition, size=comfy.ldm.minimax_music.dit.latent_length(frames), mode="nearest")


comfy.ldm.minimax_music.dit.MiniMaxMusic3DiT.aligned_condition = _aligned_condition


# Stage boundaries (MiniMax Music 3 sampling/decode, video VAE decode): ComfyUI sizes VRAM from
# mem_get_info, which on Windows reports ~9 GB free while the previous stage's model (MiniMax's
# 8.7 GB text encoder, its DiT, or a ~10 GB LTX model) stays resident. The real headroom runs
# out and large activations page into system memory: a 2-minute song's diffusion went
# 380 s -> 1036 s, the next 20 s decode 2 s -> 16 s, LTX 2.5's video decode ~10 s -> 35 s.
# The previous stage is idle by then, so unload it first; the next run reloads from RAM.
def _free_idle_models(device, keep=None):
    device = torch.device(device)
    if device.type == "cuda" and torch.version.hip is not None and "gfx1030" in torch.cuda.get_device_properties(device).gcnArchName:
        keep_loaded = [m for m in comfy.model_management.current_loaded_models if m.model is keep]
        comfy.model_management.free_memory(1e30, device, keep_loaded=keep_loaded)
        comfy.model_management.soft_empty_cache()


_original_vae_decode = comfy.sd.VAE.decode
_original_vae_decode_tiled = comfy.sd.VAE.decode_tiled


def _decode_needs_room(vae, samples):
    # MiniMax audio and video VAEs (LTX 2.5: 35 s with the diffusion model resident, see report).
    return isinstance(vae.first_stage_model, comfy.ldm.minimax_music.dav.MiniMaxMusic3DAV) or samples.ndim == 5


def _vae_decode(self, samples_in, *args, **kwargs):
    if _decode_needs_room(self, samples_in):
        _free_idle_models(self.device, keep=self.patcher)
    return _original_vae_decode(self, samples_in, *args, **kwargs)


def _vae_decode_tiled(self, samples, *args, **kwargs):
    if _decode_needs_room(self, samples):
        _free_idle_models(self.device, keep=self.patcher)
    return _original_vae_decode_tiled(self, samples, *args, **kwargs)


comfy.sd.VAE.decode = _vae_decode
comfy.sd.VAE.decode_tiled = _vae_decode_tiled

_original_sample = comfy.sample.sample
_original_sample_custom = comfy.sample.sample_custom


def _is_minimax_music_dit(model):
    return isinstance(getattr(model.model, "diffusion_model", None), comfy.ldm.minimax_music.dit.MiniMaxMusic3DiT)


def _sample(model, *args, **kwargs):
    if _is_minimax_music_dit(model):
        _free_idle_models(model.load_device)
    return _original_sample(model, *args, **kwargs)


def _sample_custom(model, *args, **kwargs):
    if _is_minimax_music_dit(model):
        _free_idle_models(model.load_device)
    return _original_sample_custom(model, *args, **kwargs)


comfy.sample.sample = _sample
comfy.sample.sample_custom = _sample_custom


_hy3d_patched = 0
for name, module in list(sys.modules.items()):
    if "Hunyuan3DWrapper" in name and name.endswith("autoencoders.attention_processors"):
        module.scaled_dot_product_attention = _hy3d_attention
        _hy3d_patched += 1
if _hy3d_patched == 0:
    logging.warning("RX6800 fixes: Hunyuan3DWrapper not loaded yet; FlashVDM attention not patched")
