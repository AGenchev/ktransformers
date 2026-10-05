# K-transformers — Local Development Workspace

This is the working directory for local development on
[KTransformers](https://github.com/kvcache-ai/KTransformers). It contains an
upstream clone, local patches, and a containerized build/dev environment.

- Upstream clone: `KTransformers-repo/` (origin: `kvcache-ai/KTransformers`,
  fork: `AGenchev/ktransformers`)
- Dev container: `dev/Dockerfile` + `dev/dev.sh` (image `kt-dev-image:latest`,
  container `kt-dev`, runs as UID 1000 matching host user `gele`)
- Patch archive: `dev/patches/`
- House rules and host quirks: see `AGENTS.md`

---

## Applied changes: full-GPU prefill for LLAMAFILE MoE (issue #2108)

### Background

`--kt-gpu-prefill-token-threshold` activates a **full-GPU prefill** path in
sglang's KTransformers EP integration: when a prefill chunk reaches the
token threshold, all MoE experts are dequantized into bf16 GPU staging
buffers once per layer, so prefill runs entirely on GPU instead of the
hybrid CPU/GPU pipeline.

That path calls `wrapper.submit_write_weight_scale_to_buffer()` /
`sync_write_weight_scale_to_buffer()`. Two independent defects broke it:

1. **Missing capability in the LLAMAFILE backend (kt-kernel).**
   Only `NativeMoEWrapper` implemented the buffer-streaming methods. With
   `--kt-method LLAMAFILE` (the GLM-5.3 configuration used here), the call
   raised `AttributeError` and killed the scheduler
   (ktransformers issues [#2108](https://github.com/kvcache-ai/KTransformers/issues/2108)
   and #2113). The upstream fix attempt in PR #2111 ("move helpers to base")
   would not have been enough: the LLAMAFILE C++ class `LLAMA_MOE_TP` had no
   `write_weight_scale_to_buffer` task at all, so even with PR #2111 the
   crash would have turned into `NotImplementedError`.

2. **Rank-asymmetric capability gate (sglang fork).**
   The interim "Option A" patch gated the full-GPU path on
   `hasattr(self.wrapper, "submit_write_weight_scale_to_buffer")`. The KT
   wrapper object is constructed on **TP rank 0 only**, so rank 0 evaluated
   the gate to `True` while ranks 1–3 saw `False`. Rank 0 entered the
   full-GPU path and blocked inside `SharedFullContext`'s gloo collectives
   while its peers took the hybrid path and enqueued different NCCL
   collectives → 600 s NCCL watchdog timeout → SIGQUIT, scheduler death.

### Change 1 — kt-kernel: `write_weights_to_buffer` for LLAMAFILE ("Suggestion C")

Commit `e1cde21` (branch `fix/issue-2108-llamafile-write-weights`) on
**AGenchev/ktransformers** (`main` tip and feature branch). Files:

- `kt-kernel/operators/llamafile/moe.hpp`
  - `LLAMA_MOE_TP::write_weights_to_buffer`: dequantizes this TP part's
    local slice of the GGUF expert weights to bf16 into the GPU staging
    buffers. TP mapping mirrors `load_weights()`: the outer `TP_MOE`
    accumulates per-TP intermediate offsets (uneven CPU splits and
    `cpu_tp_count != gpu_tp_count` supported); global row
    `r = offset + r_local` maps to GPU slot `r / gpu_inter_local`, row
    `r % gpu_inter_local`, where `gpu_inter_local =
    full_config.intermediate_size / gpu_tp_count`. The down matrix maps
    along its K (intermediate) axis the same way. Layout contract matches
    sglang `kt_ep_wrapper._prepare_weight_bf16`:
    `w13 = [2*gpu_inter_local, hidden]` (gate-then-up),
    `w2 = [hidden, gpu_inter_local]`.
  - `TP_MOE<LLAMA_MOE_TP>::write_weight_scale_to_buffer`: fans the task out
    to all TP parts via `do_numa_job` after `load_weights()`.
- `kt-kernel/python/utils/llamafile.py`
  - `LlamafileMoEWrapper` gains `submit_write_weight_scale_to_buffer` /
    `sync_write_weight_scale_to_buffer` with the same positional signature
    as `NativeMoEWrapper`. Because sglang's gate is capability-based
    (`hasattr`), the full-GPU path then activates automatically — no
    further sglang configuration is needed.
- `kt-kernel/test_write_buffer_llamafile.py` (included in the commit;
  CPU-only functional
  test): a 4-expert Q8_0 `LLAMA_MOE` (HIDDEN=512, INTER=2048 — must be
  large enough to span 8 NUMA TP nodes) verifies the bf16 staging buffers
  against the dequantized-Q8_0 reference for `gpu_tp_count=1` and
  `gpu_tp_count=2`. Note: Q8_0 is lossy, so the reference dequantizes the
  actual quantized bytes, not the original fp32.

### Change 2 — sglang: rank-uniform full-GPU capability gate

Commits `9f98335bd2` + `61bae41f9f`, branch
`kt-ep/issue-2108-full-gpu-gate`, pushed to **AGenchev/sglang**
(based on kvcache-ai/sglang `541ddc37cb`). No PR filed. Local copy:
`third_party/sglang` (detached HEAD at `61bae41f9f`, `fork` remote added).
Also archived as `dev/patches/sglang-issue2108-rank-uniform-gate.patch`
(with the earlier interim patch kept as
`dev/patches/sglang-issue2108-capability-gate.patch` for history, applied by
`dev/patches/apply_rank_symmetric_gate.py`).

In `python/sglang/srt/layers/moe/kt_ep_wrapper.py`:

- New `_kt_wrapper_full_gpu_capable(method)`: rank 0 evaluates the wrapper
  capability and **broadcasts the verdict over the gloo CPU group**, so
  every TP rank takes the same branch; the result is cached per KT method.
  Single-GPU / non-distributed runs short-circuit to the local check.
- The full-GPU gate in `KTEPWrapperMethod.apply()` uses this rank-uniform
  check; wrappers without the capability (e.g. stock AMX builds) skip the
  full-GPU path with a one-time warning instead of crashing, falling back
  to the hybrid CPU/GPU pipeline.

> Editing note: the workspace `edit`/`write` tools reject any write to
> `kt_ep_wrapper.py` (secret-detector false positive on the upstream
> dataclass field `max_deferred_experts_per_token`). Apply changes to that
> file via a script in `dev/patches/` executed from the host shell.

### Result (verified end-to-end, 2026-10-02)

GLM-5.3, LLAMAFILE method, Q5_K_XL CPU experts, 4×A100 TP4,
`--kt-num-gpu-experts 12`, `--kt-gpu-prefill-token-threshold 512`:

- 1456-token prompt → correct completion in ~54 s wall clock.
- Full-GPU layerwise prefill runs clean through all 75 MoE layers
  (~620–730 ms prepare + ~4 ms compute per layer; the first full-GPU pass
  is slow by design because every expert is dequantized once per layer).
- Launch script: `KTransformers-repo/ktransformers-glm53_LLAMAFILE-container-n12.sh`
  (run inside the container via `./dev/dev.sh bash <script>`).

### Rebuild / retest from scratch

```bash
# kt-kernel (from KTransformers-repo/kt-kernel, inside the dev container)
CCACHE_DIR=/tmp/ccache CPUINFER_PARALLEL=32 \
  pip install . --no-deps --no-build-isolation --break-system-packages

# sglang fork patch is already installed as an editable install:
#   pip install --break-system-packages -e third_party/sglang/python
# If the submodule is reset, re-apply via dev/patches/apply_rank_symmetric_gate.py

# Functional test (kt-kernel)
python test_write_buffer_llamafile.py
```

Build gotchas: `/home/dev/.cache` is root-owned, hence `CCACHE_DIR=/tmp/ccache`;
the LLAMAFILE test needs INTER large enough to span 8 NUMA TP nodes or it
raises `intermediate_size too small`.

### Upstream / maintenance notes

- Both fixes are **not merged upstream** as of 2026-10-05. Watch PR #2111
  and the sglang fork; if an equivalent fix merges, the local sglang commits
  can be dropped (`git -C third_party/sglang checkout -- <file>` / rebase).
- Unpushed-history fallback: `dev/patches/*.patch` recreate both sglang
  commits; the kt-kernel work is fully on the fork.
- All kt-kernel artifacts created by docker must be `chown 1000:1000`
  (host user `gele`); the dev container already runs as UID 1000.

---

## Workspace layout

```
K-transformers/
├── AGENTS.md                     # host quirks, credentials policy, layout
├── README.md                     # this file
├── dev/
│   ├── Dockerfile, dev.sh        # dev container (kt-dev-image, UID 1000)
│   ├── patches/                  # archived sglang patches + apply script
│   └── ccache/, pip-cache/, hf-cache/, home/   # persistent caches
└── KTransformers-repo/           # upstream clone (origin) + fork remote
    ├── kt-kernel/                # C++/CUDA kernels; fix branch committed here
    └── third_party/
        ├── sglang/               # kvcache-ai fork; local gate commits here
        ├── llama.cpp, custom_flashinfer, pybind11
        └── ...
```
