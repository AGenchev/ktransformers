#!/bin/bash

export CUDA_HOME=$CONDA_PREFIX
export TORCH_CUDA_ARCH_LIST="8.0+PTX"
export FLASHINFER_CUDA_ARCH_LIST=8.0


rm -rf ~/.cache/flashinfer # clear cached JIT compiled things
rm -rf ~/.cache/torch_extensions
rm -rf ~/.cache/tvm-ffi
rm -rf ~/.cache/sglang

# TODO:Check w GLM FP8 Precision
export PYTORCH_ALLOC_CONF=expandable_segments:True
export SGLANG_ENABLE_JIT_DEEPGEMM=0 # can't fuse CPU+GPU at the same time

# export TRANSFORMERS_VERBOSITY=info

# Tunning: python tuning_fused_moe_triton.py     --model /work/weights/GLM-5.2-FP16     --tp-size 4     --disable-shared-experts-fusion     --tune

# pip install sgl-kernel --index-url https://docs.sglang.ai/whl/cu130/

# requires hopper: --fp8-gemm-backend cutlass

#python3 -m sglang.launch_server --host 0.0.0.0 --port 1027 --model /work/weights/GLM-5.2-FP16 --kt-weight-path /work/weights/GLM-5.2-FP16-AMXINT8-NUMA8 --kt-cpuinfer 48
#--kt-threadpool-count 8 --kt-num-gpu-experts 8 --kt-method AMXINT8 --kt-gpu-prefill-token-threshold 500 --kt-enable-dynamic-expert-update --attention-backend flashinfer --trust-remote-code --mem-fraction-static 0.9 --chunked-prefill-size 32768
#--max-running-requests 32 --max-total-tokens 32768 --watchdog-timeout 3000 --enable-mixed-chunk --tensor-parallel-size 4 --enable-p2p-check --disable-shared-experts-fusion
# --attention-backend flashinfer is preferred for GLM
# --attention-backend triton - works
# --kt-gpu-prefill-token-threshold might lead it to crash on Ampere, we want to fix it
# match to max prefill tokens --kt-gpu-prefill-token-threshold 4096 \
# --max-prefill-tokens 4096 -
# from 18 experts, no much change in speed


# https://ktransformers.net/en/docs/inference/long-context-deployment#step-3-wait-for-ready-and-verify Prefill time grows linearly with context length. A 600K token request spends roughly 20–25 minutes in prefill
# HTTP client timeout must be ≥ 1800 s, timeout=3600 recommended.
# export SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR='/work/weights/GLM-5.3-GGUF/expert_stats_BF16-vs-UD-Q5_num2'
# mkdir -p "$SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR"
#  --record-kt-gpu-expert-distribution \
#  --expert-distribution-recorder-mode stat \



python -m sglang.launch_server \
  --model /work/weights/GLM-5.3-BF16 \
  --model-path '/work/weights/GLM-5.3-BF16' \
  --kt-weight-path '/work/weights/GLM-5.3-GGUF/UD-Q5_K_XL' \
  --kt-cpuinfer 48 \
  --kt-threadpool-count 8 \
  --sleep-on-idle \
  --kt-num-gpu-experts 2 \
  --kt-method LLAMAFILE \
  --kt-enable-dynamic-expert-update \
  --kt-expert-placement-strategy uniform \
  --kt-gpu-prefill-token-threshold 512 \
  --kt-max-deferred-experts-per-token 0 \
  --disable-shared-experts-fusion \
  --enable-p2p-check \
  --enable-mixed-chunk \
  --chunked-prefill-size 4096 \
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
  # --dtype bfloat16
