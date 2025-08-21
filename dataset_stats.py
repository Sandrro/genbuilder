"""Compute and save basic statistics for a graph dataset.

This utility iterates over all ``.gpickle`` files in a directory and
extracts a few aggregated statistics that might be useful for model
training. The resulting information is written to a JSON file. In
addition to overall counts the script also groups the same metrics by
functional zone labels (taken from the ``graph.graph['zone']`` or
``graph.graph['zone_id']`` attributes).

Usage::

    python dataset_stats.py --dir /path/to/graphs --out stats.json
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import Dict, List

import numpy as np
import networkx as nx


@dataclass
class StatBucket:
    """Aggregated statistics for a list of numeric values."""

    min: float
    max: float
    mean: float

    @classmethod
    def from_list(cls, values: List[float]) -> "StatBucket":
        arr = np.asarray(values, dtype=np.float64)
        return cls(min=float(arr.min()), max=float(arr.max()), mean=float(arr.mean()))


def process_directory(directory: str) -> dict:
    """Compute statistics for graphs stored as ``.gpickle`` files."""

    files = [f for f in os.listdir(directory) if f.endswith(".gpickle")]
    node_counts: List[int] = []
    edge_counts: List[int] = []
    areas: List[float] = []

    # per-zone aggregations
    zone_graphs: Dict[str, int] = defaultdict(int)
    zone_nodes: Dict[str, List[int]] = defaultdict(list)
    zone_edges: Dict[str, List[int]] = defaultdict(list)
    zone_areas: Dict[str, List[float]] = defaultdict(list)

    for fname in files:
        path = os.path.join(directory, fname)
        with open(path, "rb") as f:
            graph: nx.Graph = pickle.load(f)

        n_nodes = graph.number_of_nodes()
        n_edges = graph.number_of_edges()
        node_counts.append(n_nodes)
        edge_counts.append(n_edges)

        size_x = nx.get_node_attributes(graph, "size_x")
        size_y = nx.get_node_attributes(graph, "size_y")
        area = None
        if size_x and size_y:
            area = sum(
                float(size_x[n]) * float(size_y.get(n, 0.0)) for n in graph.nodes
            )
            areas.append(area)

        zone_label = graph.graph.get("zone")
        if zone_label is None:
            zone_label = graph.graph.get("zone_id", "unknown")
        zone_label = str(zone_label)

        zone_graphs[zone_label] += 1
        zone_nodes[zone_label].append(n_nodes)
        zone_edges[zone_label].append(n_edges)
        if area is not None:
            zone_areas[zone_label].append(area)

    stats = {
        "graphs": len(files),
        "nodes": asdict(StatBucket.from_list(node_counts)) if node_counts else {},
        "edges": asdict(StatBucket.from_list(edge_counts)) if edge_counts else {},
    }
    if areas:
        stats["area"] = asdict(StatBucket.from_list(areas))

    if zone_graphs:
        by_zone = {}
        for zone, gcount in zone_graphs.items():
            info = {
                "graphs": gcount,
                "nodes": asdict(StatBucket.from_list(zone_nodes[zone]))
                if zone_nodes[zone]
                else {},
                "edges": asdict(StatBucket.from_list(zone_edges[zone]))
                if zone_edges[zone]
                else {},
            }
            if zone_areas[zone]:
                info["area"] = asdict(StatBucket.from_list(zone_areas[zone]))
            by_zone[zone] = info
        stats["by_zone"] = by_zone

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Save dataset statistics to a JSON file"
    )
    parser.add_argument("--dir", required=True, help="Directory with .gpickle graphs")
    parser.add_argument("--out", required=True, help="Where to write resulting JSON")
    args = parser.parse_args()

    stats = process_directory(args.dir)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)


if __name__ == "__main__":
    main()
