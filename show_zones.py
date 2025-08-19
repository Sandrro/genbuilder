"""Utility script for summarizing zone labels in graph datasets.

This script can work with two kinds of inputs:

1. A directory containing ``.gpickle`` files (NetworkX graphs).
   In this case the script will iterate over all graph files and aggregate
   statistics (number of graphs, nodes and edges) per zone.

2. A Hugging Face dataset repo containing serialized graphs. The script will
   aggregate the same statistics per zone over the dataset split.

Example usage for a directory of ``.gpickle`` graphs::

    python show_zones.py --dir /path/to/graphs --zones /path/to/_zones_map.json

If ``--zones`` is omitted the script looks for ``_zones_map.json`` inside the
provided directory.

Example usage for a Hugging Face dataset::

    python show_zones.py --repo user/dataset --split train
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import inspect
from collections import defaultdict
from typing import Dict, Optional

import networkx as nx


# ---------------------------------------------------------------------------
# Zone map utilities
# ---------------------------------------------------------------------------

def _invert_map(mapping: Dict[str, int]) -> Dict[int, str]:
    """Convert a name->id mapping to id->name."""

    return {idx: name for name, idx in mapping.items()}


def load_zones_map_local(path: str) -> Dict[int, str]:
    """Load zone mapping from a local JSON file."""

    with open(path, "r", encoding="utf-8") as f:
        mapping = json.load(f)["map"]
    return _invert_map(mapping)


def load_zones_map_hf(repo_id: str, token: Optional[str] = None) -> Dict[int, str]:
    """Download ``_zones_map.json`` from a HF dataset repo and return id->name."""

    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo_id, filename="_zones_map.json", token=token)
    return load_zones_map_local(path)


# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------

def process_gpickle_dir(directory: str, zones_map_path: Optional[str] = None) -> None:
    """Iterate over all ``.gpickle`` files in ``directory`` and print summary statistics.

    Parameters
    ----------
    directory:
        Folder containing graph files.
    zones_map_path:
        Optional path to ``_zones_map.json``.  If omitted, the file will be
        searched inside ``directory``.
    """

    if zones_map_path is None:
        candidate = os.path.join(directory, "_zones_map.json")
        zones_map_path = candidate if os.path.exists(candidate) else None

    zones_map: Dict[int, str] = (
        load_zones_map_local(zones_map_path) if zones_map_path else {}
    )

    files = sorted(f for f in os.listdir(directory) if f.endswith(".gpickle"))
    stats = defaultdict(lambda: {"graphs": 0, "nodes": 0, "edges": 0})

    for fname in files:
        path = os.path.join(directory, fname)
        with open(path, "rb") as f:
            graph = pickle.load(f)
        zone_id = graph.graph.get("zone_id")
        zone_name = zones_map.get(zone_id, f"unknown_{zone_id}")
        info = stats[zone_name]
        info["graphs"] += 1
        info["nodes"] += graph.number_of_nodes()
        info["edges"] += graph.number_of_edges()

    if files:
        print("Summary:")
        for zone, info in stats.items():
            print(
                f"{zone}: {info['graphs']} graphs, nodes={info['nodes']}, edges={info['edges']}"
            )
        total_graphs = sum(v["graphs"] for v in stats.values())
        total_nodes = sum(v["nodes"] for v in stats.values())
        total_edges = sum(v["edges"] for v in stats.values())
        print(
            f"Total graphs: {total_graphs}, total nodes: {total_nodes}, total edges: {total_edges}"
        )


def process_hf_dataset(repo_id: str, split: str, token: Optional[str]) -> None:
    """Aggregate zone statistics for graphs stored in a Hugging Face dataset."""

    from datasets import load_dataset

    load_dataset_sig = inspect.signature(load_dataset)
    dataset_kwargs = {"split": split}
    if token is not None:
        if "token" in load_dataset_sig.parameters:
            dataset_kwargs["token"] = token
        else:
            dataset_kwargs["use_auth_token"] = token
    dataset = load_dataset(repo_id, **dataset_kwargs)

    zones_map = dataset.info.metadata.get("zones_map")
    if zones_map is None:
        zones_map = load_zones_map_hf(repo_id, token)

    stats = defaultdict(lambda: {"graphs": 0, "nodes": 0, "edges": 0})

    for row in dataset:
        graph = pickle.loads(row["graph"])
        zone_id = graph.graph.get("zone_id")
        zone_name = zones_map.get(zone_id, f"unknown_{zone_id}")
        info = stats[zone_name]
        info["graphs"] += 1
        info["nodes"] += graph.number_of_nodes()
        info["edges"] += graph.number_of_edges()

    print("Summary:")
    for zone, info in stats.items():
        print(
            f"{zone}: {info['graphs']} graphs, nodes={info['nodes']}, edges={info['edges']}"
        )
    total_graphs = sum(v["graphs"] for v in stats.values())
    total_nodes = sum(v["nodes"] for v in stats.values())
    total_edges = sum(v["edges"] for v in stats.values())
    print(
        f"Total graphs: {total_graphs}, total nodes: {total_nodes}, total edges: {total_edges}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Show aggregated zone statistics for a graph dataset"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--dir", help="Directory containing .gpickle graph files", metavar="PATH"
    )
    group.add_argument(
        "--repo", help="HF dataset repo id, e.g. 'user/dataset'", metavar="ID"
    )
    parser.add_argument(
        "--split", default="train", help="Dataset split to load (HF datasets only)"
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Hugging Face auth token for private repos (HF datasets only)",
    )
    parser.add_argument(
        "--zones", default=None, help="Path to _zones_map.json (gpickle only)"
    )
    args = parser.parse_args()

    if args.dir is not None:
        process_gpickle_dir(args.dir, args.zones)
    else:
        process_hf_dataset(args.repo, args.split, args.token)


if __name__ == "__main__":
    main()

