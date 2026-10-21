#!/bin/bash
# GLM-5.3 LLAMAFILE launch, issue #2108 full-GPU prefill variant:
# chunked-prefill-size 12288 (vs 4096 in the -n12 baseline).
# Every prefill chunk >= kt-gpu-prefill-token-threshold (512) takes the
# full-GPU path after the rank-uniform capability gate fix; a larger chunk
# means fewer chunks and fewer per-chunk prepare passes.
# Port 1027. Run from project root: ./dev/dev.sh bash <this-script>
#
# Variant: NUMA interleave experiment (2026-10-20). Server wrapped in
# `numactl --interleave=all` so pages (GGUF mmap weights, CPU expert
# copies, threadpool working set) spread evenly over all 8 NUMA nodes
# instead of first-touch clustering; goal: decode DRAM efficiency
# (implied ~230-240 GB/s -> closer to the 460 GB/s DDR5 ceiling).
# Single-variable change vs pool-filemode.sh: all other flags identical.
# Baseline decode before this change: 13.9-15.4 tok/s.
#
# Variant: pool FILE MODE (no KT_POOL_MEMFD). The memfd replica copy
# single-threadedly first-touches all ~1.4 TB of shmem page cache onto one
# NUMA node (kernel pwrite() ignores the process interleave policy), that
# node fills at ~280 GiB and the copy collapses from ~1.4 GB/s to ~0.1 GB/s.
# File mode maps pool.bin directly: no full-copy step at startup, pages are
# read per-layer by the prefetch thread (mlock is now OFF by default —
# KT_POOL_LOCK_MAX_MB opts in — full-pool mlock OOM-killed a rank on the
# 2.2 TiB host, see kt-kernel "default pool mlock off"), direct DMA stays
# dormant (staged path serves requests) — the previously verified configuration.

export CUDA_HOME=$CONDA_PREFIX
export TORCH_CUDA_ARCH_LIST="8.0+PTX"
export FLASHINFER_CUDA_ARCH_LIST=8.0

rm -rf ~/.cache/flashinfer ~/.cache/torch_extensions ~/.cache/tvm-ffi ~/.cache/sglang

export PYTORCH_ALLOC_CONF=expandable_segments:True
export SGLANG_ENABLE_JIT_DEEPGEMM=0 # can't fuse CPU+GPU at the same time
export SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0 # downgrade TP mem imbalance to warning
export KT_BF16_EXPERT_POOL=/work/models/GLM-5.3-bf16-expert-pool
export KT_POOL_PREFETCH=1
# unset: KT_POOL_DIRECT_DMA (needs a registerable mapping; file mapping is not)
# unset: KT_POOL_MEMFD (single-node first-touch collapse, see header)

numactl --interleave=all python -m sglang.launch_server \
  --model /work/weights/GLM-5.3-BF16 \
  --model-path '/work/weights/GLM-5.3-BF16' \
  --kt-weight-path '/work/weights/GLM-5.3-GGUF/UD-Q5_K_XL' \
  --kt-cpuinfer 48 \
  --kt-threadpool-count 8 \
  --sleep-on-idle \
  --kt-num-gpu-experts 12 \
  --kt-method LLAMAFILE \
  --kt-enable-dynamic-expert-update \
  --kt-expert-placement-strategy uniform \
  --kt-gpu-prefill-token-threshold 512 \
  --kt-max-deferred-experts-per-token 0 \
  --disable-shared-experts-fusion \
  --enable-p2p-check \
  --enable-mixed-chunk \
  --chunked-prefill-size 12288 \
  --max-prefill-tokens 12288 \
  --watchdog-timeout 3000 \
  --tensor-parallel-size 4 \
  --trust-remote-code \
  --mem-fraction-static 0.95 \
  --kv-cache-dtype auto \
  --max-total-tokens 256000 \
  --context-length 256000 \
  --max-running-requests 4 \
  --cuda-graph-max-bs 4 \
  --attention-backend flashinfer \
  --fp8-gemm-backend auto \
  --tool-call-parser glm47 \
  --reasoning-parser glm45 \
  --served-model-name IO_model \
  --host 0.0.0.0 \
  --port 1027
