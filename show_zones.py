"""Utility script for inspecting zone labels in graph datasets.

This script can work with two kinds of inputs:

1. A directory containing ``.gpickle`` files (NetworkX graphs).
   In this case the script will iterate over all graph files, print their
   zone labels and basic statistics such as number of nodes and edges.

2. A Hugging Face dataset repo containing serialized graphs. This retains the
   original behaviour of the script but now also prints the same statistics
   per graph.

Example usage for a directory of ``.gpickle`` graphs::

    python show_zones.py --dir /path/to/graphs --zones /path/to/_zones_map.json

If ``--zones`` is omitted the script looks for ``_zones_map.json`` inside the
provided directory.

Example usage for a Hugging Face dataset::

    python show_zones.py --repo user/dataset --split train
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import pickle
from collections import Counter
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
# Processing helpers
# ---------------------------------------------------------------------------

def _print_stats_per_graph(name: str, graph: nx.Graph, zones_map: Dict[int, str]) -> str:
    """Return a formatted string describing a graph's zone and size."""

    zone_id = graph.graph.get("zone_id")
    zone_name = zones_map.get(zone_id, f"unknown_{zone_id}")
    n_nodes = graph.number_of_nodes()
    n_edges = graph.number_of_edges()
    return f"{name}: {zone_name} (id {zone_id}), nodes={n_nodes}, edges={n_edges}"


def process_gpickle_dir(directory: str, zones_map_path: Optional[str] = None) -> None:
    """Iterate over all ``.gpickle`` files in ``directory`` and print zone labels.

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
    stats = Counter()
    total_nodes = 0
    total_edges = 0

    for fname in files:
        path = os.path.join(directory, fname)
        with open(path, "rb") as f:
            graph = pickle.load(f)
        print(_print_stats_per_graph(fname, graph, zones_map))
        zone_id = graph.graph.get("zone_id")
        zone_name = zones_map.get(zone_id, f"unknown_{zone_id}")
        stats[zone_name] += 1
        total_nodes += graph.number_of_nodes()
        total_edges += graph.number_of_edges()

    if files:
        print("\nSummary:")
        for zone, count in stats.items():
            print(f"{zone}: {count} graphs")
        print(
            f"Total graphs: {len(files)}, total nodes: {total_nodes}, total edges: {total_edges}"
        )


def process_hf_dataset(repo_id: str, split: str, token: Optional[str]) -> None:
    """Print zone labels for graphs stored in a Hugging Face dataset."""

    from datasets import load_dataset

    zones_map = load_zones_map_hf(repo_id, token)

    load_dataset_sig = inspect.signature(load_dataset)
    dataset_kwargs = {"split": split}
    if token is not None:
        if "token" in load_dataset_sig.parameters:
            dataset_kwargs["token"] = token
        else:
            dataset_kwargs["use_auth_token"] = token
    dataset = load_dataset(repo_id, **dataset_kwargs)

    stats = Counter()
    total_nodes = 0
    total_edges = 0

    for i, row in enumerate(dataset):
        graph = pickle.loads(row["graph"])
        print(_print_stats_per_graph(str(i), graph, zones_map))
        zone_id = graph.graph.get("zone_id")
        zone_name = zones_map.get(zone_id, f"unknown_{zone_id}")
        stats[zone_name] += 1
        total_nodes += graph.number_of_nodes()
        total_edges += graph.number_of_edges()

    print("\nSummary:")
    for zone, count in stats.items():
        print(f"{zone}: {count} graphs")
    print(
        f"Total graphs: {len(dataset)}, total nodes: {total_nodes}, total edges: {total_edges}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print zone labels and basic stats for a graph dataset"
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

