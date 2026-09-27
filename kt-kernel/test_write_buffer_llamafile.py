"""Functional test: LLAMAFILE write_weight_scale_to_buffer (issue #2108, suggestion C).

Builds a small LLAMA_MOE (TP count 1) with Q8_0 expert weights, submits the
write_weight_scale_to_buffer task, and verifies the bf16 staging buffers match
the dequantized fp32 reference. CPU-only.
"""

import ctypes

import torch

import kt_kernel.kt_kernel_ext as ext

torch.manual_seed(7)

HIDDEN = 512
INTER = 2048  # llamafile: intermediate % 32 == 0 and must span 8 NUMA TP nodes
N_EXPERTS = 4
N_EXPTOK = 8

# Q8_0 quantize via ggml python helper if available; else use torch quantize
CPUInfer = ext.CPUInfer(4)

gate_f32 = (torch.randn((N_EXPERTS, INTER, HIDDEN)) / 4.0).contiguous()
up_f32 = (torch.randn((N_EXPERTS, INTER, HIDDEN)) / 4.0).contiguous()
down_f32 = (torch.randn((N_EXPERTS, HIDDEN, INTER)) / 4.0).contiguous()


def quantize_q8_0(t: torch.Tensor) -> torch.Tensor:
    """Quantize contiguous fp32 tensor [rows, k] to Q8_0 (32-elem blocks)."""
    import ggml  # type: ignore

    rows, k = t.shape
    assert k % 32 == 0
    nblocks = rows * (k // 32)
    out = torch.empty(nblocks * 34, dtype=torch.uint8)  # 2B scale + 32B data
    for r in range(rows):
        row = t[r]
        nb = k // 32
        bufs = []
        for b in range(nb):
            block = row[b * 32 : (b + 1) * 32]
            d = block.abs().max() / 127.0
            qs = torch.clamp(torch.round(block / d), -128, 127).to(torch.int8)
            bufs.append(d.half().view(torch.uint8).unsqueeze(0))
            bufs.append(qs.view(torch.uint8))
        out[r * (k // 32) * 34 : (r + 1) * (k // 32) * 34] = torch.cat(bufs)
    return out


def quantize_q8_0_fast(t: torch.Tensor) -> torch.Tensor:
    rows, k = t.shape
    nb = k // 32
    blocks = t.reshape(rows, nb, 32)
    d = blocks.abs().amax(dim=-1) / 127.0
    qs = torch.clamp(torch.round(blocks / d.unsqueeze(-1)), -128, 127).to(torch.int8)
    scale_bytes = d.half().contiguous().view(torch.uint8).view(rows, nb, 2)
    out = torch.empty(rows * nb * 34, dtype=torch.uint8)
    out_view = out.reshape(rows, nb, 34)
    out_view[:, :, :2] = scale_bytes
    out_view[:, :, 2:] = qs.view(torch.uint8)
    return out


def quantize_q8_0_torch(t):
    return quantize_q8_0_fast(t)


gate_q = quantize_q8_0_torch(gate_f32.reshape(-1, HIDDEN)).reshape(N_EXPERTS, -1)
up_q = quantize_q8_0_torch(up_f32.reshape(-1, HIDDEN)).reshape(N_EXPERTS, -1)
down_q = quantize_q8_0_torch(down_f32.reshape(-1, INTER)).reshape(N_EXPERTS, -1)
print("quantized sizes:", gate_q.numel(), up_q.numel(), down_q.numel())

config = ext.moe.MOEConfig(N_EXPERTS, N_EXPTOK, HIDDEN, INTER, 0)
config.max_len = 256
config.group_max_len = 256
config.group_min_len = 10
config.m_block = 32
config.pool = CPUInfer.backend_
config.gate_proj = gate_q.data_ptr()
config.up_proj = up_q.data_ptr()
config.down_proj = down_q.data_ptr()
config.gate_scale = 0
config.up_scale = 0
config.down_scale = 0
config.gate_type = ext.kvcache.ggml_type.Q8_0
config.up_type = ext.kvcache.ggml_type.Q8_0
config.down_type = ext.kvcache.ggml_type.Q8_0
config.hidden_type = ext.kvcache.ggml_type.FP32

moe = ext.moe.MOE(config)
phys_map = torch.arange(N_EXPERTS, dtype=torch.int32).contiguous()
CPUInfer.submit(moe.load_weights_task(phys_map.data_ptr()))
CPUInfer.sync()
print("weights loaded")

# Staging buffers (TP count 1 -> single ptr in each list)
EXPERT_ID = 2
# Case 1: single GPU TP slot
w13_buf = torch.zeros((2 * INTER, HIDDEN), dtype=torch.bfloat16).contiguous()
w2_buf = torch.zeros((HIDDEN, INTER), dtype=torch.bfloat16).contiguous()
w13_ptrs = [w13_buf.data_ptr()]
w2_ptrs = [w2_buf.data_ptr()]
scale_ptrs = [0]
task_pair = moe.write_weight_scale_to_buffer_task(1, EXPERT_ID, w13_ptrs, scale_ptrs, w2_ptrs, scale_ptrs)
CPUInfer.submit(task_pair)
CPUInfer.sync()

# Reference: dequantize the actual Q8_0 bytes (Q8_0 is lossy, so the reference
# must come from the quantized data, not the original fp32), then -> bf16
def dequant_q8_0(q: torch.Tensor, shape: tuple) -> torch.Tensor:
    """Dequantize flattened Q8_0 bytes -> fp32 tensor of `shape` [rows, k]."""
    rows, k = shape
    nb = k // 32
    v = q.reshape(-1, 34)
    d = v[:, :2].contiguous().view(torch.float16).reshape(-1).float()  # [total_blocks]
    qs = v[:, 2:].contiguous().view(torch.int8).float()  # [total_blocks, 32]
    return (qs * d.unsqueeze(-1)).reshape(rows, k)


gate_ref = (
    dequant_q8_0(gate_q, (N_EXPERTS * INTER, HIDDEN)).reshape(N_EXPERTS, INTER, HIDDEN)[EXPERT_ID].to(torch.bfloat16)
)
up_ref = (
    dequant_q8_0(up_q, (N_EXPERTS * INTER, HIDDEN)).reshape(N_EXPERTS, INTER, HIDDEN)[EXPERT_ID].to(torch.bfloat16)
)
down_ref = (
    dequant_q8_0(down_q, (N_EXPERTS * HIDDEN, INTER)).reshape(N_EXPERTS, HIDDEN, INTER)[EXPERT_ID].to(torch.bfloat16)
)

torch.testing.assert_close(w13_buf[:INTER], gate_ref, rtol=1e-2, atol=1e-3)
torch.testing.assert_close(w13_buf[INTER:], up_ref, rtol=1e-2, atol=1e-3)
torch.testing.assert_close(w2_buf, down_ref, rtol=1e-2, atol=1e-3)
print("PASS: single gpu_tp_count=1")

# Case 2: two GPU TP slots, INTER split evenly
half = INTER // 2
w13_a = torch.zeros((2 * half, HIDDEN), dtype=torch.bfloat16).contiguous()
w13_b = torch.zeros((2 * half, HIDDEN), dtype=torch.bfloat16).contiguous()
w2_a = torch.zeros((HIDDEN, half), dtype=torch.bfloat16).contiguous()
w2_b = torch.zeros((HIDDEN, half), dtype=torch.bfloat16).contiguous()
w13_ptrs2 = [w13_a.data_ptr(), w13_b.data_ptr()]
w2_ptrs2 = [w2_a.data_ptr(), w2_b.data_ptr()]
scale_ptrs2 = [0, 0]
task_pair = moe.write_weight_scale_to_buffer_task(2, EXPERT_ID, w13_ptrs2, scale_ptrs2, w2_ptrs2, scale_ptrs2)
CPUInfer.submit(task_pair)
CPUInfer.sync()

gate_ref_a, gate_ref_b = gate_ref[:half], gate_ref[half:]
up_ref_a, up_ref_b = up_ref[:half], up_ref[half:]
torch.testing.assert_close(w13_a[:half], gate_ref_a)
torch.testing.assert_close(w13_a[half:], up_ref_a)
torch.testing.assert_close(w13_b[:half], gate_ref_b)
torch.testing.assert_close(w13_b[half:], up_ref_b)
torch.testing.assert_close(w2_a, down_ref[:, :half])
torch.testing.assert_close(w2_b, down_ref[:, half:])
print("PASS: multi gpu_tp_count=2")
print("ALL PASS: write_weight_scale_to_buffer matches fp32 reference")
