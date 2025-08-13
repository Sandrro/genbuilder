#!/usr/bin/env python3
import os
import glob
import argparse
import subprocess
from huggingface_hub import HfApi


def run(cmd, env):
    """Run a subprocess and forward output."""
    proc = subprocess.Popen(cmd, env=env)
    proc.communicate()
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)


def main():
    parser = argparse.ArgumentParser(description="Utility to train/test and upload models")
    parser.add_argument("--config", default="train_gnn.yaml", help="Path to training YAML")
    parser.add_argument("--dataset", default="my_dataset", help="Path to dataset root")
    parser.add_argument("--epoch", default=None, help="Epoch directory name to use for testing")
    parser.add_argument("--train", action="store_true", help="Run training")
    parser.add_argument("--test", action="store_true", help="Run test after training")
    parser.add_argument("--upload_repo", default=None, help="HuggingFace repo id to upload model")
    parser.add_argument("--hf_token", default=None, help="HuggingFace token")
    args = parser.parse_args()

    env = os.environ.copy()
    env["DATASET_ROOT"] = os.path.abspath(args.dataset)
    env["TRAIN_CONFIG"] = os.path.abspath(args.config)
    if args.epoch:
        env["EPOCH_NAME"] = args.epoch

    latest_epoch = None

    if args.train:
        run(["python", "train.py"], env=env)
        epoch_dirs = glob.glob(os.path.join(os.getcwd(), "epoch", "*"))
        if epoch_dirs and not args.epoch:
            latest_epoch = max(epoch_dirs, key=os.path.getmtime)
            env["EPOCH_NAME"] = os.path.basename(latest_epoch)

    if args.test:
        if "EPOCH_NAME" not in env:
            epoch_dirs = glob.glob(os.path.join(os.getcwd(), "epoch", "*"))
            if epoch_dirs:
                latest_epoch = max(epoch_dirs, key=os.path.getmtime)
                env["EPOCH_NAME"] = os.path.basename(latest_epoch)
        run(["python", "test.py"], env=env)

    if args.upload_repo:
        if latest_epoch is None:
            epoch_dirs = glob.glob(os.path.join(os.getcwd(), "epoch", "*"))
            if epoch_dirs:
                latest_epoch = max(epoch_dirs, key=os.path.getmtime)
        if latest_epoch:
            ckpt_path = os.path.join(latest_epoch, "latest.pth")
            if os.path.isfile(ckpt_path):
                api = HfApi()
                api.upload_file(
                    path_or_fileobj=ckpt_path,
                    path_in_repo=os.path.basename(ckpt_path),
                    repo_id=args.upload_repo,
                    token=args.hf_token,
                )
                print(f"Uploaded {ckpt_path} to {args.upload_repo}")
            else:
                print(f"Checkpoint {ckpt_path} not found, skipping upload")


if __name__ == "__main__":
    main()
