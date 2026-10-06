#!/bin/bash
# GLM-5.3 LLAMAFILE launch, issue #2108 full-GPU prefill variant:
# chunked-prefill-size 12288 (vs 4096 in the -n12 baseline).
# Every prefill chunk >= kt-gpu-prefill-token-threshold (512) takes the
# full-GPU path after the rank-uniform capability gate fix; a larger chunk
# means fewer chunks and fewer per-chunk prepare passes.
# Port 1027. Run from project root: ./dev/dev.sh bash <this-script>

export CUDA_HOME=$CONDA_PREFIX
export TORCH_CUDA_ARCH_LIST="8.0+PTX"
export FLASHINFER_CUDA_ARCH_LIST=8.0

rm -rf ~/.cache/flashinfer ~/.cache/torch_extensions ~/.cache/tvm-ffi ~/.cache/sglang

export PYTORCH_ALLOC_CONF=expandable_segments:True
export SGLANG_ENABLE_JIT_DEEPGEMM=0 # can't fuse CPU+GPU at the same time
export SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0 # downgrade TP mem imbalance to warning

python -m sglang.launch_server \
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
