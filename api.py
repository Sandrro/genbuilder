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
    """Upload .gpickle files (multiple) and optional _zones_map.json."""
    dest_proc = os.path.join("my_dataset", "processed")
    dest_root = os.path.join("my_dataset")
    os.makedirs(dest_proc, exist_ok=True)
    os.makedirs(dest_root, exist_ok=True)
    saved: list[str] = []
    for file in files:
        fname = file.filename
        if fname.endswith(".gpickle"):
            target = os.path.join(dest_proc, fname)
            with open(target, "wb") as f:
                shutil.copyfileobj(file.file, f)
        elif fname == "_zones_map.json":
            data = await file.read()
            for target in [os.path.join(dest_root, fname), os.path.join(dest_proc, fname)]:
                with open(target, "wb") as f:
                    f.write(data)
        else:
            raise HTTPException(status_code=400, detail="only .gpickle or _zones_map.json supported")
        saved.append(fname)
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
