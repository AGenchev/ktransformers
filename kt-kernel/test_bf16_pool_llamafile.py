"""Functional test: LLAMAFILE write_weight_scale_to_buffer with BF16 expert pool.

Builds a small LLAMA_MOE with Q8_0 GGUF-style expert weights, creates a
SYNTHETIC BF16 pool (pool.bin + pool.json) with DIFFERENT values, points
KT_BF16_EXPERT_POOL at it, and verifies that write_weight_scale_to_buffer
copies the POOL bytes (not the dequantized GGUF) into the staging buffers,
for gpu_tp_count=1 and 2. CPU-only.

The fallback path (env unset / bad path -> GGUF dequant) is covered by
test_write_buffer_llamafile.py (no env) and the bad-path subprocess below.
"""

import os
import subprocess
import sys

import torch

torch.manual_seed(11)

HIDDEN = 512
INTER = 2048
N_EXPERTS = 4
N_EXPTOK = 8
LAYER_IDX = 0  # GeneralMOEConfig.layer_idx default; pool must cover layer 0

import kt_kernel.kt_kernel_ext as ext

CPUInfer = ext.CPUInfer(4)

# GGUF-side weights (Q8_0): deliberately different from the pool values.
gate_f32 = (torch.randn((N_EXPERTS, INTER, HIDDEN)) / 4.0).contiguous()
up_f32 = (torch.randn((N_EXPERTS, INTER, HIDDEN)) / 4.0).contiguous()
down_f32 = (torch.randn((N_EXPERTS, HIDDEN, INTER)) / 4.0).contiguous()


def quantize_q8_0_fast(t: torch.Tensor) -> torch.Tensor:
    rows, k = t.shape
    nb = k // 32
    blocks = t.reshape(rows, nb, 32)
    d = blocks.abs().amax(dim=-1) / 127.0
    qs = torch.clamp(torch.round(blocks / d.unsqueeze(-1)), -128, 127).to(torch.int8)
    scale_bytes = d.half().contiguous().view(torch.uint8).view(rows, nb, 2)
    out = torch.empty(rows * nb * 34, dtype=torch.uint8)
    ov = out.reshape(rows, nb, 34)
    ov[:, :, :2] = scale_bytes
    ov[:, :, 2:] = qs.view(torch.uint8)
    return out


gate_q = quantize_q8_0_fast(gate_f32.reshape(-1, HIDDEN)).reshape(N_EXPERTS, -1)
up_q = quantize_q8_0_fast(up_f32.reshape(-1, HIDDEN)).reshape(N_EXPERTS, -1)
down_q = quantize_q8_0_fast(down_f32.reshape(-1, INTER)).reshape(N_EXPERTS, -1)

# Synthetic POOL values: independent from gate_f32/up_f32/down_f32.
pool_gate = (torch.randn((N_EXPERTS, INTER, HIDDEN)) / 4.0).to(torch.bfloat16).contiguous()
pool_up = (torch.randn((N_EXPERTS, INTER, HIDDEN)) / 4.0).to(torch.bfloat16).contiguous()
pool_down = (torch.randn((N_EXPERTS, HIDDEN, INTER)) / 4.0).to(torch.bfloat16).contiguous()

# Write the synthetic pool: layer 0 only, experts 0..N_EXPERTS-1.
per_expert = 2 * INTER * HIDDEN * 2 + HIDDEN * INTER * 2
pool_dir = "/tmp/kt_test_bf16_pool"
os.makedirs(pool_dir, exist_ok=True)
bin_path = os.path.join(pool_dir, "pool.bin")
with open(bin_path, "wb") as f:
    for e in range(N_EXPERTS):
        f.write(pool_gate[e].contiguous().view(torch.uint8).numpy().tobytes())
        f.write(pool_up[e].contiguous().view(torch.uint8).numpy().tobytes())
        f.write(pool_down[e].contiguous().view(torch.uint8).numpy().tobytes())
import json

with open(os.path.join(pool_dir, "pool.json"), "w") as f:
    json.dump(
        {
            "dtype": "BF16",
            "moe_layer_ids": [LAYER_IDX],
            "n_experts": N_EXPERTS,
            "inter": INTER,
            "hidden": HIDDEN,
            "roles": ["gate_proj", "up_proj", "down_proj"],
            "per_expert_bytes": per_expert,
            "total_bytes": N_EXPERTS * per_expert,
        },
        f,
    )

os.environ["KT_BF16_EXPERT_POOL"] = pool_dir

config = ext.moe.MOEConfig(N_EXPERTS, N_EXPTOK, HIDDEN, INTER, 0)
config.layer_idx = LAYER_IDX
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
print("weights loaded; pool:", os.environ["KT_BF16_EXPERT_POOL"])

EXPERT_ID = 2

# Case 1: gpu_tp_count=1 — expect POOL bytes, exactly.
w13_buf = torch.zeros((2 * INTER, HIDDEN), dtype=torch.bfloat16).contiguous()
w2_buf = torch.zeros((HIDDEN, INTER), dtype=torch.bfloat16).contiguous()
task = moe.write_weight_scale_to_buffer_task(
    1, EXPERT_ID, [w13_buf.data_ptr()], [0], [w2_buf.data_ptr()], [0]
)
CPUInfer.submit(task)
CPUInfer.sync()

torch.testing.assert_close(w13_buf[:INTER], pool_gate[EXPERT_ID])
torch.testing.assert_close(w13_buf[INTER:], pool_up[EXPERT_ID])
torch.testing.assert_close(w2_buf, pool_down[EXPERT_ID])
# And prove the pool was actually used: values must DIFFER from the GGUF dequant.
assert not torch.allclose(w13_buf[:INTER].float(), gate_f32[EXPERT_ID], rtol=1e-2, atol=1e-1), (
    "w13 matches GGUF dequant -- pool source was NOT used!"
)
print("PASS: pool source, gpu_tp_count=1 (exact bf16 match, differs from GGUF dequant)")

# Case 2: gpu_tp_count=2, INTER split evenly.
half = INTER // 2
w13_a = torch.zeros((2 * half, HIDDEN), dtype=torch.bfloat16).contiguous()
w13_b = torch.zeros((2 * half, HIDDEN), dtype=torch.bfloat16).contiguous()
w2_a = torch.zeros((HIDDEN, half), dtype=torch.bfloat16).contiguous()
w2_b = torch.zeros((HIDDEN, half), dtype=torch.bfloat16).contiguous()
task = moe.write_weight_scale_to_buffer_task(
    2, EXPERT_ID, [w13_a.data_ptr(), w13_b.data_ptr()], [0, 0], [w2_a.data_ptr(), w2_b.data_ptr()], [0, 0]
)
CPUInfer.submit(task)
CPUInfer.sync()

g, u, d = pool_gate[EXPERT_ID], pool_up[EXPERT_ID], pool_down[EXPERT_ID]
torch.testing.assert_close(w13_a[:half], g[:half])
torch.testing.assert_close(w13_a[half:], u[:half])
torch.testing.assert_close(w13_b[:half], g[half:])
torch.testing.assert_close(w13_b[half:], u[half:])
torch.testing.assert_close(w2_a, d[:, :half])
torch.testing.assert_close(w2_b, d[:, half:])
print("PASS: pool source, gpu_tp_count=2 (slot mapping matches dequant path)")

# Case 3: bad pool path -> fallback to GGUF dequant (fresh process needed:
# the pool singleton latches on first resolve).
child = r"""
import os, sys, torch
os.environ["KT_BF16_EXPERT_POOL"] = "/nonexistent_pool_dir"
import kt_kernel.kt_kernel_ext as ext
HIDDEN, INTER, N_EXPERTS = 512, 2048, 4
CPUInfer = ext.CPUInfer(4)
g32 = (torch.randn((N_EXPERTS, INTER, HIDDEN)) / 4.0).contiguous()
def q8(t):
    rows, k = t.shape; nb = k // 32
    blocks = t.reshape(rows, nb, 32)
    d = blocks.abs().amax(dim=-1) / 127.0
    qs = torch.clamp(torch.round(blocks / d.unsqueeze(-1)), -128, 127).to(torch.int8)
    sb = d.half().contiguous().view(torch.uint8).view(rows, nb, 2)
    out = torch.empty(rows * nb * 34, dtype=torch.uint8); ov = out.reshape(rows, nb, 34)
    ov[:, :, :2] = sb; ov[:, :, 2:] = qs.view(torch.uint8)
    return out
gq = q8(g32.reshape(-1, HIDDEN)).reshape(N_EXPERTS, -1)
cfg = ext.moe.MOEConfig(N_EXPERTS, 8, HIDDEN, INTER, 0)
cfg.max_len = 256; cfg.group_max_len = 256; cfg.group_min_len = 10; cfg.m_block = 32
cfg.pool = CPUInfer.backend_
cfg.gate_proj = gq.data_ptr(); cfg.up_proj = gq.data_ptr(); cfg.down_proj = gq.data_ptr()
cfg.gate_scale = 0; cfg.up_scale = 0; cfg.down_scale = 0
cfg.gate_type = cfg.up_type = cfg.down_type = ext.kvcache.ggml_type.Q8_0
cfg.hidden_type = ext.kvcache.ggml_type.FP32
moe = ext.moe.MOE(cfg)
pm = torch.arange(N_EXPERTS, dtype=torch.int32).contiguous()
CPUInfer.submit(moe.load_weights_task(pm.data_ptr())); CPUInfer.sync()
w13 = torch.zeros((2 * INTER, HIDDEN), dtype=torch.bfloat16).contiguous()
w2 = torch.zeros((HIDDEN, INTER), dtype=torch.bfloat16).contiguous()
CPUInfer.submit(moe.write_weight_scale_to_buffer_task(1, 1, [w13.data_ptr()], [0], [w2.data_ptr()], [0]))
CPUInfer.sync()
def dq(q, rows, k):
    v = q.reshape(-1, 34)
    d = v[:, :2].contiguous().view(torch.float16).reshape(-1).float()
    qs = v[:, 2:].contiguous().view(torch.int8).float()
    return (qs * d.unsqueeze(-1)).reshape(rows, k)
ref = dq(gq, N_EXPERTS * INTER, HIDDEN).reshape(N_EXPERTS, INTER, HIDDEN)[1].to(torch.bfloat16)
torch.testing.assert_close(w13[:INTER], ref, rtol=1e-2, atol=1e-3)
print("PASS: bad pool path falls back to GGUF dequant")
"""
r = subprocess.run([sys.executable, "-c", child], capture_output=True, text=True, timeout=600)
if r.returncode != 0 or "PASS: bad pool path" not in r.stdout:
    print(r.stdout[-2000:])
    print(r.stderr[-2000:])
    sys.exit("FALLBACK TEST FAILED")
print(r.stdout.strip().splitlines()[-1])
print("ALL PASS: BF16 expert pool consumer")
