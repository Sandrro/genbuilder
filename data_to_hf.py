#!/usr/bin/env python3
import argparse
from huggingface_hub import HfApi
import os

def main():
    parser = argparse.ArgumentParser(description="Upload processed dataset to HuggingFace")
    parser.add_argument("--path", default="my_dataset/processed", help="Path to processed data folder")
    parser.add_argument("--repo", required=True, help="HuggingFace dataset repo id, e.g. username/dataset")
    parser.add_argument("--token", default=None, help="HuggingFace token with write permission")
    parser.add_argument("--commit_message", default="upload processed data", help="Commit message")
    args = parser.parse_args()

    if not os.path.isdir(args.path):
        raise FileNotFoundError(f"Processed dataset folder not found: {args.path}")

    api = HfApi()
    api.upload_folder(
        repo_id=args.repo,
        folder_path=args.path,
        path_in_repo=".",
        repo_type="dataset",
        token=args.token,
        commit_message=args.commit_message,
    )
    print(f"Uploaded {args.path} to {args.repo}")

if __name__ == "__main__":
    main()
