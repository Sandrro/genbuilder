#!/usr/bin/env python3
import argparse
import json
import os
import pickle
from pathlib import Path
import io

import datasets as ds
import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc
from huggingface_hub import HfApi
import networkx as nx  # noqa: F401 - needed for pickle loading


def load_graph_record(path: Path, K: int):
    """Load a graph file and return serialized bytes plus zoning info.

    The graph is unpickled so that any stored ``zone_onehot`` attribute can be
    reshaped to a flat vector of length ``K`` before re-serializing. This keeps
    functional zone information intact when uploaded to HuggingFace datasets.
    """

    if path.suffix == ".arrow":
        with pa.memory_map(str(path), "rb") as source:
            table = ipc.open_file(source).read_all()
        graph_bytes = table.column("graph")[0].as_py()
    elif path.suffix == ".gpickle":
        with open(path, "rb") as f:
            graph_bytes = f.read()
    else:
        raise ValueError(f"Unsupported file type: {path}")

    g = pickle.loads(graph_bytes)
    zoh = g.graph.get("zone_onehot")
    if zoh is None:
        zone_onehot = [0.0] * K
        zone_id = -1
    else:
        zoh_arr = np.asarray(zoh, dtype=np.float32).reshape(-1)
        if K and zoh_arr.size != K:
            raise ValueError(
                f"zone_onehot length {zoh_arr.size} != expected {K} for {path}"
            )
        zone_onehot = zoh_arr.tolist()
        zone_id = int(np.argmax(zoh_arr)) if zoh_arr.size > 0 else -1
        g.graph["zone_onehot"] = zoh_arr
        g.graph["zone_id"] = zone_id

    return pickle.dumps(g), zone_id, zone_onehot


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

    K = len(zones_map["map"]) if zones_map and "map" in zones_map else None

    def gen():
        for p in files:
            try:
                g_bytes, zid, zoh = load_graph_record(p, K or 0)
            except Exception as e:
                print(f"Skipping {p}: {e}")
                continue
            record = {"graph": g_bytes}
            if K is not None:
                record.update({"zone_id": zid, "zone_onehot": zoh})
            yield record

    features_dict = {"graph": ds.Value("binary")}
    if K is not None:
        features_dict.update(
            {
                "zone_id": ds.Value("int32"),
                "zone_onehot": ds.Sequence(ds.Value("float32"), length=K),
            }
        )
    features = ds.Features(features_dict)
    dataset = ds.Dataset.from_generator(gen, features=features)
    # The datasets library >=4.0 removed the `with_metadata` helper.
    # Instead of attaching arbitrary metadata to the Dataset object,
    # we upload the optional `_zones_map.json` file separately below.

    dataset.push_to_hub(args.repo, token=args.token, commit_message=args.commit_message)

    api = HfApi()
    if zones_map is not None:
        api.upload_file(
            path_or_fileobj=str(zones_map_path),
            path_in_repo="_zones_map.json",
            repo_id=args.repo,
            repo_type="dataset",
            token=args.token,
            commit_message=args.commit_message,
        )

    params_path = data_dir / "_transform_params.json"
    if params_path.is_file():
        with params_path.open("r", encoding="utf-8") as pf:
            params = json.load(pf)
        card = (
            f"# Dataset {args.repo}\n\n## Generation Parameters\n" +
            "```json\n" + json.dumps(params, ensure_ascii=False, indent=2) + "\n```\n"
        )
        api.upload_file(
            path_or_fileobj=io.BytesIO(card.encode("utf-8")),
            path_in_repo="README.md",
            repo_id=args.repo,
            repo_type="dataset",
            token=args.token,
            commit_message=args.commit_message,
        )


if __name__ == "__main__":
    main()
