#!/usr/bin/env python3
"""Download processed graphs from a HF dataset and export one block per zone.

The script relies on the ``huggingface_hub`` library to fetch the remote dataset
snapshot.  After downloading, it inspects the ``_zones_map.json`` file to
discover available functional zones and copies a single graph (city block) for
each zone into an output directory.

Example:
    python test.py --repo user/dataset --out blocks

This will download the dataset ``user/dataset`` and write ``<zone>.gpickle``
files into the ``blocks`` directory, one for each zone listed in
``_zones_map.json``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Set

import networkx as nx
from huggingface_hub import snapshot_download


def download_dataset(repo: str, dest: Path) -> None:
    """Use ``huggingface_hub`` to download ``repo`` into ``dest``."""
    dest.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo,
        repo_type="dataset",
        local_dir=dest,
        local_dir_use_symlinks=False,
    )


def load_zones_map(path: Path) -> Dict[int, str]:
    """Return a ``zone_id -> zone_name`` mapping from ``_zones_map.json``."""
    mapping = json.loads(path.read_text(encoding="utf-8"))["map"]
    return {idx: name for name, idx in mapping.items()}


def export_one_block_per_zone(dataset_dir: Path, out_dir: Path) -> int:
    """Copy the first encountered graph of each zone into ``out_dir``.

    Parameters
    ----------
    dataset_dir:
        Directory containing the downloaded HF dataset snapshot.
    out_dir:
        Destination directory for exported graphs.

    Returns
    -------
    int
        Number of exported graphs.
    """
    zones_map_path = dataset_dir / "_zones_map.json"
    zones = load_zones_map(zones_map_path)
    processed_dir = dataset_dir / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    exported: Set[str] = set()
    for gfile in sorted(processed_dir.glob("*.gpickle")):
        graph = nx.read_gpickle(gfile)
        zone_id = graph.graph.get("zone_id")
        zone_name = zones.get(zone_id)
        if zone_name and zone_name not in exported:
            nx.write_gpickle(graph, out_dir / f"{zone_name}.gpickle")
            exported.add(zone_name)
        if len(exported) == len(zones):
            break
    return len(exported)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, help="HF dataset repository id")
    parser.add_argument(
        "--out", type=Path, default=Path("blocks"), help="Output directory"
    )
    parser.add_argument(
        "--cache", type=Path, default=Path("hf_dataset"), help="Download cache"
    )
    args = parser.parse_args()

    download_dataset(args.repo, args.cache)
    count = export_one_block_per_zone(args.cache, args.out)
    print(f"Exported {count} blocks into {args.out}")


if __name__ == "__main__":
    main()
