#!/usr/bin/env python3
import argparse
import os
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

def main():
    parser = argparse.ArgumentParser(description="Upload processed dataset to HuggingFace")
    parser.add_argument("--path", default="my_dataset/processed", help="Path to processed data folder")
    parser.add_argument("--repo", required=True, help="HuggingFace dataset repo id, e.g. username/dataset")
    parser.add_argument("--token", default=None, help="HuggingFace token with write permission")
    parser.add_argument("--commit_message", default="upload processed data", help="Commit message")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Upload only the first N files from the dataset folder",
    )
    parser.add_argument(
        "--large",
        action="store_true",
        help="Use resilient upload_large_folder for large datasets",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.path):
        raise FileNotFoundError(f"Processed dataset folder not found: {args.path}")

    base_path = Path(args.path)
    upload_path = base_path
    temp_dir = None

    if args.limit is not None:
        # Create a temporary directory containing only the first N files
        temp_dir = tempfile.TemporaryDirectory()
        upload_path = Path(temp_dir.name)
        files = sorted(p for p in base_path.rglob("*") if p.is_file())
        for p in files[: args.limit]:
            rel = p.relative_to(base_path)
            dest = upload_path / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dest)

    api = HfApi(token=args.token)
    try:
        if args.large:
            api.upload_large_folder(
                repo_id=args.repo,
                folder_path=str(upload_path),
                repo_type="dataset",
            )
            print(f"Uploaded large folder {upload_path} to {args.repo}")
        else:
            api.upload_folder(
                repo_id=args.repo,
                folder_path=str(upload_path),
                path_in_repo=".",
                repo_type="dataset",
                token=args.token,
                commit_message=args.commit_message,
            )
            print(f"Uploaded {upload_path} to {args.repo}")
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()

if __name__ == "__main__":
    main()
