#!/usr/bin/env python3
import os
import glob
import argparse
import subprocess
import logging
from huggingface_hub import HfApi, snapshot_download


logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def run(cmd, env):
    """Run a subprocess and forward output."""
    logging.info("Running command: %s", " ".join(cmd))
    proc = subprocess.Popen(cmd, env=env)
    proc.communicate()
    logging.info("Command finished with code %s", proc.returncode)
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)


def main():
    parser = argparse.ArgumentParser(description="Utility to train/test and upload models")
    parser.add_argument("--config", default="train_gnn.yaml", help="Path to training YAML")
    parser.add_argument("--dataset", default="my_dataset", help="Path to dataset root")
    parser.add_argument("--dataset_repo", default=None, help="HuggingFace dataset repo id to download from")
    parser.add_argument("--model_repo", default=None, help="HuggingFace model repo id to download for testing")
    parser.add_argument("--epoch", default=None, help="Epoch directory name to use for testing")
    parser.add_argument("--train", action="store_true", help="Run training")
    parser.add_argument("--test", action="store_true", help="Run test after training")
    parser.add_argument("--upload_repo", default=None, help="HuggingFace repo id to upload model and logs")
    parser.add_argument("--hf_token", default=None, help="HuggingFace token")
    parser.add_argument("--trial", action="store_true", help="Use only first 100 graphs for quick debug run")
    args = parser.parse_args()
    logging.info("Arguments: %s", args)

    env = os.environ.copy()
    env["DATASET_ROOT"] = os.path.abspath(args.dataset)
    env["TRAIN_CONFIG"] = os.path.abspath(args.config)
    if args.trial:
        env["TRAIN_TRIAL"] = "1"
    if args.epoch:
        env["EPOCH_NAME"] = args.epoch

    if args.dataset_repo:
        logging.info("Downloading dataset from %s", args.dataset_repo)
        snapshot_download(
            repo_id=args.dataset_repo,
            repo_type="dataset",
            local_dir=args.dataset,
            token=args.hf_token,
        )

    if args.model_repo:
        logging.info("Downloading model from %s", args.model_repo)
        model_dir = os.path.join("epoch", args.model_repo.split("/")[-1])
        snapshot_download(
            repo_id=args.model_repo,
            repo_type="model",
            local_dir=model_dir,
            token=args.hf_token,
        )
        env["EPOCH_NAME"] = os.path.basename(model_dir)

    latest_epoch = None

    if args.train:
        logging.info("Starting training stage")
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
        logging.info("Starting testing stage")
        run(["python", "test.py"], env=env)

    if args.upload_repo:
        if latest_epoch is None:
            epoch_dirs = glob.glob(os.path.join(os.getcwd(), "epoch", "*"))
            if epoch_dirs:
                latest_epoch = max(epoch_dirs, key=os.path.getmtime)
        if latest_epoch:
            logging.info("Preparing to upload artifacts from %s", latest_epoch)
            files = []
            best_ckpt = os.path.join(latest_epoch, "val_best_extacc.pth")
            if os.path.isfile(best_ckpt):
                files.append(best_ckpt)
            yaml_path = os.path.join(latest_epoch, "train_save.yaml")
            if not os.path.isfile(yaml_path):
                alt_yaml = os.path.join(latest_epoch, "resume_train_save.yaml")
                if os.path.isfile(alt_yaml):
                    yaml_path = alt_yaml
                else:
                    yaml_path = None
            if yaml_path:
                files.append(yaml_path)
            log_file = os.path.join("logs", os.path.basename(latest_epoch) + ".log")
            if os.path.isfile(log_file):
                files.append(log_file)
            if files:
                api = HfApi()
                for fp in files:
                    api.upload_file(
                        path_or_fileobj=fp,
                        path_in_repo=os.path.basename(fp),
                        repo_id=args.upload_repo,
                        repo_type="model",
                        token=args.hf_token,
                    )
                    logging.info("Uploaded %s to %s", fp, args.upload_repo)
            else:
                logging.info("No artifacts found to upload")


if __name__ == "__main__":
    main()
