import argparse
import inspect
import json
import pickle
from typing import Dict, Optional

try:
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download
except ModuleNotFoundError as e:
    raise SystemExit("Required packages not found. Install 'datasets' and 'huggingface_hub'.") from e


def load_zones_map(repo_id: str, token: Optional[str] = None) -> Dict[int, str]:
    """Download `_zones_map.json` from the dataset repo and return id->name mapping."""
    path = hf_hub_download(repo_id, filename="_zones_map.json", token=token)
    with open(path, "r", encoding="utf-8") as f:
        mapping = json.load(f)["map"]
    # `_zones_map.json` stores name -> id; invert it
    return {idx: name for name, idx in mapping.items()}


def main(repo_id: str, split: str, token: Optional[str] = None) -> None:
    zones_map = load_zones_map(repo_id, token)

    load_dataset_sig = inspect.signature(load_dataset)
    dataset_kwargs = {"split": split}
    if token is not None:
        if "token" in load_dataset_sig.parameters:
            dataset_kwargs["token"] = token
        else:
            dataset_kwargs["use_auth_token"] = token
    dataset = load_dataset(repo_id, **dataset_kwargs)

    for i, row in enumerate(dataset):
        graph = pickle.loads(row["graph"])
        zone_id = graph.graph.get("zone_id")
        zone_name = zones_map.get(zone_id, f"unknown_{zone_id}")
        print(f"{i}: {zone_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Print zone labels for each block in a HF dataset")
    parser.add_argument("--repo", required=True, help="HF dataset repo id, e.g. 'user/dataset'")
    parser.add_argument("--split", default="train", help="Dataset split to load")
    parser.add_argument("--token", default=None, help="Hugging Face auth token for private repos")
    args = parser.parse_args()
    main(args.repo, args.split, args.token)
