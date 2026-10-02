"""Atomic local multi-shard checkpoint format for trusted course artifacts."""

import hashlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_checkpoint(root, *, step, shards, metadata):
    import torch
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    destination = root / f"step-{step:08d}"
    if step < 0 or not shards:
        raise ValueError("nonnegative new step and nonempty shards required")
    if destination.exists():
        # Committed versions are never overwritten: tell the caller how to proceed.
        raise ValueError(f"checkpoint {destination} already exists; save under a new step, "
                         "or move or delete that directory first (existing versions are never overwritten)")
    required = {"model", "optimizer", "rng", "cursor"}
    if any(not required <= shard.keys() for shard in shards):
        raise ValueError("each shard needs model, optimizer, RNG, and data cursor")
    if not {"model_version", "data_version", "tokenizer_version", "mesh"} <= metadata.keys():
        raise ValueError("checkpoint needs model/data/tokenizer/mesh provenance")
    with TemporaryDirectory(prefix=".writing-", dir=root) as temporary:
        directory = Path(temporary)
        records = []
        for rank, shard in enumerate(shards):
            path = directory / f"rank-{rank:04d}.pt"
            with path.open("wb") as stream:
                torch.save(shard, stream)
                stream.flush()
                os.fsync(stream.fileno())
            records.append({"rank": rank, "file": path.name, "sha256": _digest(path)})
        manifest = {"format": 1, "step": step, "world_size": len(shards),
                    "metadata": metadata, "shards": records}
        marker = directory / "COMMITTED.json"
        with marker.open("w") as stream:
            json.dump(manifest, stream, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        # Same filesystem rename publishes the complete directory in one operation.
        os.replace(directory, destination)
    return destination


def load_checkpoint(directory):
    import torch
    directory = Path(directory)
    marker = directory / "COMMITTED.json"
    if not marker.is_file():
        raise ValueError("checkpoint has no completion marker")
    manifest = json.loads(marker.read_text())
    records = manifest.get("shards", [])
    if manifest.get("format") != 1 or len(records) != manifest.get("world_size"):
        raise ValueError("invalid checkpoint format or shard count")
    if [record.get("rank") for record in records] != list(range(len(records))):
        raise ValueError("missing or repeated shard owner")
    shards = []
    for rank, record in enumerate(records):
        name = f"rank-{rank:04d}.pt"
        path = directory / name
        if record.get("file") != name or not path.is_file() or path.is_symlink():
            raise ValueError("missing or invalid shard path")
        if _digest(path) != record.get("sha256"):
            raise ValueError("shard checksum mismatch")
        shards.append(torch.load(path, map_location="cpu", weights_only=True))
    return manifest, shards


def load_latest(root):
    failures = []
    for directory in sorted(Path(root).glob("step-*"), reverse=True):
        try:
            return load_checkpoint(directory)
        except (ValueError, OSError, KeyError, json.JSONDecodeError) as error:
            failures.append(f"{directory.name}: {error}")
    raise ValueError("no complete checkpoint: " + "; ".join(failures))
