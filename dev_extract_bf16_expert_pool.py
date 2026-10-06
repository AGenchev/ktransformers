#!/usr/bin/env python3
"""Extract routed-expert BF16 weights from safetensors shards into a flat pool.

Pool layout (for the LLAMAFILE full-GPU staging fast path, issue #2108):
  pool.bin  : concatenation over (moe_layer_index L, expert_id E) of
              [gate_proj (inter,hidden) | up_proj (inter,hidden) |
               down_proj (hidden,inter)] in BF16, row-major, no padding.
  pool.json : metadata {layers: {L: n_experts}, dims, offsets}
Dense layers (no mlp.experts.*) and shared_experts are skipped.

Reads are streamed shard-by-shard with bounded memory; writes are
sequential per (L, E) thanks to an intermediate staging of the current
layer in tmp files, so pool.bin stays append-ordered by layer.
"""
import glob
import json
import mmap
import os
import shutil
import struct
import sys
import tempfile

SRC_GLOB = "/work/weights/GLM-5.3-BF16/*.safetensors"
OUT_DIR = "/work/models/GLM-5.3-bf16-expert-pool"
TMP = os.path.join(OUT_DIR, "tmp")

ROLE_ORDER = ("gate_proj", "up_proj", "down_proj")


def shard_headers():
    for path in sorted(glob.glob(SRC_GLOB)):
        with open(path, "rb") as fh:
            (n,) = struct.unpack("<Q", fh.read(8))
            hdr = json.loads(fh.read(n))
        yield path, hdr, 8 + n  # data start offset


def main():
    os.makedirs(TMP, exist_ok=True)
    role_dims = {}  # role -> shape (gate/up [inter,hidden], down [hidden,inter])
    layers = {}  # L -> {expert_id: {role: (shard_path, data_off, size)}}
    for path, hdr, data_start in shard_headers():
        with open(path, "rb") as fh:
            for name, info in hdr.items():
                if name == "__metadata__" or info["dtype"] != "BF16":
                    continue
                parts = name.split(".")
                # model.layers.N.mlp.experts.E.<role>.weight
                if (
                    len(parts) == 8
                    and parts[0] == "model"
                    and parts[1] == "layers"
                    and parts[3] == "mlp"
                    and parts[4] == "experts"
                    and parts[7] == "weight"
                ):
                    L, E, role = int(parts[2]), int(parts[5]), parts[6]  # noqa: E741
                    dims = info["shape"]
                    if role in role_dims:
                        assert dims == role_dims[role], (
                            f"{name} shape {dims} != {role_dims[role]}"
                        )
                    else:
                        role_dims[role] = dims
                    layers.setdefault(L, {}).setdefault(E, {})[role] = (
                        path,
                        data_start + info["data_offsets"][0],
                        info["data_offsets"][1] - info["data_offsets"][0],
                    )

    if not layers:
        sys.exit("no expert tensors found")
    layer_ids = sorted(layers)
    # Fail fast on incomplete (L, E, role) coverage.
    for L in layer_ids:
        for E, roles in layers[L].items():
            missing = set(ROLE_ORDER) - set(roles)
            if missing:
                sys.exit(f"layer {L} expert {E} missing roles: {sorted(missing)}")
    n_experts = {L: len(layers[L]) for L in layer_ids}
    expert_counts = set(n_experts.values())
    assert len(expert_counts) == 1, f"uneven experts per layer: {expert_counts}"
    E_total = expert_counts.pop()
    inter, hidden = role_dims["gate_proj"]
    assert role_dims["up_proj"] == [inter, hidden], role_dims
    assert role_dims["down_proj"] == [hidden, inter], role_dims
    per_expert_bytes = 3 * inter * hidden * 2
    total = len(layer_ids) * E_total * per_expert_bytes
    print(
        f"MoE layers: {len(layer_ids)} (ids {layer_ids[0]}..{layer_ids[-1]}), "
        f"experts/layer: {E_total}, dims inter={inter} hidden={hidden}, "
        f"pool size: {total / 2**40:.2f} TiB"
    )

    pool_path = os.path.join(OUT_DIR, "pool.bin")
    out = open(pool_path, "wb")
    REC = 1 << 20  # 1 MiB copy chunks
    done = 0
    for L in layer_ids:
        # Stage the whole layer into a tmp file so out-of-order shards
        # still yield a strictly sequential pool write.
        tmp_path = os.path.join(TMP, f"layer_{L:03d}.bin")
        with open(tmp_path, "wb") as tmp:
            for E in range(E_total):
                for role in ROLE_ORDER:
                    src, off, size = layers[L][E][role]
                    assert size == inter * hidden * 2, (role, size)
                    with open(src, "rb") as fh:
                        fh.seek(off)
                        mm = mmap.mmap(fh.fileno(), off + size, access=mmap.ACCESS_READ)
                        left = size
                        pos = off
                        while left:
                            chunk = mm[pos : pos + min(REC, left)]
                            tmp.write(chunk)
                            pos += len(chunk)
                            left -= len(chunk)
                        mm.close()
        with open(tmp_path, "rb") as tmp:
            shutil.copyfileobj(tmp, out, REC)
        os.remove(tmp_path)
        done += E_total * per_expert_bytes
        print(f"layer {L} done ({done / 2**40:.2f}/{total / 2**40:.2f} TiB)", flush=True)

    out.close()
    with open(os.path.join(OUT_DIR, "pool.json"), "w") as fh:
        json.dump(
            {
                "dtype": "BF16",
                "moe_layer_ids": layer_ids,
                "n_experts": E_total,
                "inter": inter,
                "hidden": hidden,
                "roles": list(ROLE_ORDER),
                "per_expert_bytes": per_expert_bytes,
                "total_bytes": total,
            },
            fh,
            indent=2,
        )
    print("pool complete:", pool_path)


if __name__ == "__main__":
    main()
