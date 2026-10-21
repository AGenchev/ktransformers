# KT-NUMA memfd инцидент & решение (2026-10-20/21)

## Изводът в едно изречение
memfd + zram + свършена RAM = невъзстановим D-state deadlock на scheduler-а;
kt-kernel вече се самозащитава (MemAvailable guard), а prod конфигурацията
остава file-mode pool.

## Хронология на теста (2026-10-20)
1. **Паралелен memfd copy** (kt-kernel `fd5c460`): 8 CPU-pinned writer нишки
   (по една на NUMA възел от `KT_POOL_NUMA_NODES`), shmem страниците алокират
   node-local под политиката на пишещата задача.
   - KT_POOL_NUMA_NODES=4-7: 2.4 GB/s старт, но **пад на ~0.3 GB/s на ~380 GiB**
     — възли 4–7 не могат да поберат 1368 GiB (reclaim-stall).
   - KT_POOL_NUMA_NODES=0-7: 2.4 GB/s до ~380 GiB, после пак колапс — защото
     общият RAM бюджет: 1368 (shmem) + 524 (GGUF anon) + ~560 (container)
     **> 2.21 TiB физическа RAM**. Kernel-ът изтласква shmem страници в zram
     (zram стигна 1.46 TB).
2. **Deadlock**: page fault върху zram-backed shmem страница → scheduler-ът
   замръзна в D-state (`lock_mm_and_find_vma`), host load = 324, контейнерът
   стана unkillable (`docker rm -f` → "did not receive an exit event", защото
   init-ът стои в невъзобновим disk sleep). Оставен изкисьнал за нощта.

## Защо това е фундаментално (не бъг в нашия код)
- shmem/memfd страниците **са swappable** (за разлика от anon + mlock).
- zram е с по-висок приоритет от page cache при anonymous pressure — ядрото
  компресира shmem страници в zram вместо да ги evict-не (те са "dirty shmem").
- Page fault върху страница, която се decompress-ва от zram при изчерпан
 _MEM_, не може да се обходи от userspace — Process не може да бъде kill-нат
  докато е в невъзобновим D-state.

## Корекции (2026-10-21, kt-kernel main)
1. **`cc6f051` — memfd safety guard**: преди memfd copy-я чете
   `/proc/meminfo MemAvailable`; изисква `MemAvailable >= pool + 300 GiB`,
   иначе автоматичен fallback към file mapping с ясно съобщение.
2. **`pool.sh` лънчър → file mode** (по подразбиране), с пълния incident
   запис в коментара. memfd се пробва ръчно с:
   `KT_POOL_MEMFD=1 KT_POOL_DIRECT_DMA=1 KT_POOL_NUMA_NODES=0-7`.

## Какво ОСТАВА валидно (печалбите от цикъла)
- **Паралелният copy работи**: 2.4 GB/s × 8 writers (10×+ спрямо 0.1 GB/s
  единичен) — кодът е в `fd5c460` и чака хост с достатъчно RAM.
- **NUMA placement** (`1e6a598`): `numa_alloc_onnode` за GGUF resident копия +
  CPU pin на worker пуловете + prefetch pin към GPU-сокета — активирано,
  prep 287 ms/слой, decode ~14.8 tok/s.
- **Mincore skip probe**: при топъл рестарт prefetch-ът е 0 s вместо 20 мин.
- **mlock guard** (`c65dd3f`): default off, opt-in cap (OOM урока от 19-ти).

## Текущо състояние на системата
- kt-dev контейнер: **зомби/D-state от 20-ти вечерта** — изисква пресъздаване:
  ```bash
  docker rm -f kt-dev   # ако пак откаже — чак или reboot
  # пресъздаване по dev/dev.sh рецептата (ръчно docker run или ./dev/dev.sh)
  docker cp /tmp/sgl_sm100.so kt-dev:/usr/local/lib/python3.12/dist-packages/sgl_kernel/sm100/common_ops.abi3.so
  docker exec kt-dev bash /work/provision.sh
  ```
  (provision.sh е idempotent — sglang-kt + kt-kernel са в persistent user-site)
- След пресъздаването: `pool-filemode.sh` лънчърът (известната добра
  конфигурация).

## Unpushed kt-kernel commits (main)
`645d1a8` (memfd pool) → `f8d3d6a` (copy logging) → `c65dd3f` (mlock guard)
→ `1e6a598` (NUMA placement) → `fd5c460` (parallel copy) → `cc6f051` (guard)

## Какво би отключило memfd/direct-DMA в бъдеще
- **+300 GiB RAM** на хоста (т.е. ~2.5 TiB), или
- **int4 pool** (~414 GB вместо 1368 GiB — част от 8-картовия GPU-only план),
  при който memfd изобщо не е нужен за direct-DMA: целият pool се събира на
  GPU-тата.
