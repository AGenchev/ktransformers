#!/bin/bash
# kt-dev контейнер на еднократна употреба (ephemeral, --rm).
#
# Стекът е BAKED в image kt-dev-baked:20261022 (kt-kernel + sglang-kt +
# sgl_kernel sm100 .so в dist-packages) — provision.sh НЕ е нужен.
# Всичко stateful е на bind mounts (repo, models, weights, caches, home).
#
# Употреба:
#   ./dev/dev-once.sh                                  → сървър на порт 1027 (file-mode pool)
#   ./dev/dev-once.sh bash                             → интерактивна обвивка в /work/KTransformers-repo
#   ./dev/dev-once.sh bash <launcher.sh>               → друг лънчър по избор
#
# Проверено 2026-10-26:
#   - /work/weights  ← /home/gele/LLMs/weights (282 BF16 shards + GGUF, read-only)
#   - /work/models   ← K-transformers/models   (BF16 expert pool 1.4 TB, rw)
#   - пътищата в лънчърите сочат тези mount точки; работещите стартове ги доказват.
#
# Бележки:
#   - --rm: контейнерът се чисти при изход; volumes остават.
#   - Изход при kill -9 може да остави ghost GPU памет → просто стартирай наново
#     (или рестартирай контейнера веднъж, познат driver артефакт).
#   - Ако image-ът липсва (нова машина): построй го по dev/Dockerfile + provision.sh,
#     после docker commit / dev/update-image.sh (виж историята в AGENTS.md).

IMAGE=kt-dev-baked:20261022

exec docker run --rm -it \
  --gpus all --ipc=host --network host --privileged \
  --ulimit memlock=-1 \
  -u 1000:1000 \
  -v /home/gele/projects/K-transformers/KTransformers-repo:/work/KTransformers-repo \
  -v /home/gele/projects/K-transformers/models:/work/models \
  -v /home/gele/projects/K-transformers/dev/provision.sh:/work/provision.sh:ro \
  -v /home/gele/LLMs/weights:/work/weights:ro \
  -v /home/gele/projects/K-transformers/dev/home:/home/dev \
  -v /home/gele/projects/K-transformers/dev/ccache:/home/dev/.ccache \
  -v /home/gele/projects/K-transformers/dev/hf-cache:/home/dev/.cache/huggingface \
  -v /home/gele/projects/K-transformers/dev/pip-cache:/home/dev/.cache/pip \
  -w /work/KTransformers-repo \
  "$IMAGE" \
  "${@:-bash ktransformers-glm53_LLAMAFILE-container-n12-c12288-pool-filemode.sh}"
