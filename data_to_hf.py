#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path

import datasets as ds
import pyarrow as pa
import pyarrow.ipc as ipc
from huggingface_hub import HfApi


def load_graph_bytes(path: Path) -> bytes:
    if path.suffix == ".arrow":
        with pa.memory_map(str(path), "rb") as source:
            table = ipc.open_file(source).read_all()
        return table.column("graph")[0].as_py()
    if path.suffix == ".gpickle":
        with open(path, "rb") as f:
            return f.read()
    raise ValueError(f"Unsupported file type: {path}")


def main():
    parser = argparse.ArgumentParser(description="Convert gpickles to arrow and upload to HuggingFace")
    parser.add_argument("--path", default="my_dataset/processed", help="Path to processed data folder")
    parser.add_argument("--repo", required=True, help="HuggingFace dataset repo id, e.g. username/dataset")
    parser.add_argument("--token", default=None, help="HuggingFace token with write permission")
    parser.add_argument("--commit_message", default="upload processed data", help="Commit message")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Use only the first N files from the dataset folder",
    )
    args = parser.parse_args()

    data_dir = Path(args.path)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Processed dataset folder not found: {args.path}")

    zones_map_path = data_dir / "_zones_map.json"
    zones_map = None
    if zones_map_path.is_file():
        with zones_map_path.open("r", encoding="utf-8") as zf:
            zones_map = json.load(zf)

    files = sorted([p for p in data_dir.iterdir() if p.suffix in {".gpickle", ".arrow"}])
    if args.limit is not None:
        files = files[: args.limit]
    if not files:
        raise RuntimeError("No .gpickle or .arrow files found")

    def gen():
        for p in files:
            yield {"graph": load_graph_bytes(p)}

    features = ds.Features({"graph": ds.Value("binary")})
    dataset = ds.Dataset.from_generator(gen, features=features)
    # The datasets library >=4.0 removed the `with_metadata` helper.
    # Instead of attaching arbitrary metadata to the Dataset object,
    # we upload the optional `_zones_map.json` file separately below.

    dataset.push_to_hub(args.repo, token=args.token, commit_message=args.commit_message)

    if zones_map is not None:
        api = HfApi()
        api.upload_file(
            path_or_fileobj=str(zones_map_path),
            path_in_repo="_zones_map.json",
            repo_id=args.repo,
            repo_type="dataset",
            token=args.token,
            commit_message=args.commit_message,
        )


if __name__ == "__main__":
    main()
