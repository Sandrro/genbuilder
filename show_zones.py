import argparse
import json
import pickle
from typing import Dict

try:
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download
except ModuleNotFoundError as e:
    raise SystemExit("Required packages not found. Install 'datasets' and 'huggingface_hub'.") from e


def load_zones_map(repo_id: str) -> Dict[int, str]:
    """Download `_zones_map.json` from the dataset repo and return id->name mapping."""
    path = hf_hub_download(repo_id, filename="_zones_map.json")
    with open(path, "r", encoding="utf-8") as f:
        mapping = json.load(f)["map"]
    # `_zones_map.json` stores name -> id; invert it
    return {idx: name for name, idx in mapping.items()}


def main(repo_id: str, split: str) -> None:
    zones_map = load_zones_map(repo_id)
    dataset = load_dataset(repo_id, split=split)

    for i, row in enumerate(dataset):
        graph = pickle.loads(row["graph"])
        zone_id = graph.graph.get("zone_id")
        zone_name = zones_map.get(zone_id, f"unknown_{zone_id}")
        print(f"{i}: {zone_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Print zone labels for each block in a HF dataset")
    parser.add_argument("--repo", required=True, help="HF dataset repo id, e.g. 'user/dataset'")
    parser.add_argument("--split", default="train", help="Dataset split to load")
    args = parser.parse_args()
    main(args.repo, args.split)
