import os
import glob
import subprocess

import shutil
from typing import List
from fastapi import FastAPI, HTTPException, UploadFile, File

from pydantic import BaseModel

app = FastAPI()

class TrainRequest(BaseModel):
    config: str = "train_gnn.yaml"
    dataset: str = "my_dataset"
    upload_repo: str | None = None
    hf_token: str | None = None

class TestRequest(BaseModel):
    config: str = "train_gnn.yaml"
    dataset: str = "my_dataset"
    epoch: str | None = None


@app.post("/data")
async def upload_data(files: List[UploadFile] = File(...)):
    """Upload one or more .gpickle files into the dataset's processed directory."""
    dest = os.path.join("my_dataset", "processed")
    os.makedirs(dest, exist_ok=True)
    saved = []
    for file in files:
        if not file.filename.endswith(".gpickle"):
            raise HTTPException(status_code=400, detail="only .gpickle files supported")
        target = os.path.join(dest, file.filename)
        with open(target, "wb") as f:
            shutil.copyfileobj(file.file, f)
        saved.append(file.filename)
    return {"saved": saved}

def _launch(cmd: list[str]):
    """Run command in background without blocking the API."""
    subprocess.Popen(cmd)

@app.post("/train")
def start_train(req: TrainRequest):
    cmd = [
        "python", "run_pipeline.py",
        "--config", req.config,
        "--dataset", req.dataset,
        "--train",
    ]
    if req.upload_repo:
        cmd += ["--upload_repo", req.upload_repo]
    if req.hf_token:
        cmd += ["--hf_token", req.hf_token]
    _launch(cmd)
    return {"status": "started"}

@app.post("/test")
def start_test(req: TestRequest):
    cmd = [
        "python", "run_pipeline.py",
        "--config", req.config,
        "--dataset", req.dataset,
        "--test",
    ]
    if req.epoch:
        cmd += ["--epoch", req.epoch]
    _launch(cmd)
    return {"status": "started"}

@app.get("/logs")
def list_logs():
    files = glob.glob(os.path.join("logs", "*.log"))
    names = [os.path.basename(f) for f in files]
    return {"logs": names}

@app.get("/logs/{name}")
def read_log(name: str):
    path = os.path.join("logs", name)
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="log not found")
    with open(path, "r", encoding="utf-8") as f:
        return {"log": f.read()}

@app.get("/")
def root():
    return {"status": "ok"}
