import os
import glob
import subprocess
import json
import tempfile

import shutil
from typing import List
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse

from pydantic import BaseModel
import logging

from inference import infer_from_geojson

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

app = FastAPI()

class TrainRequest(BaseModel):
    config: str = "train_gnn.yaml"
    dataset: str = "my_dataset"
    dataset_repo: str | None = None
    upload_repo: str | None = None
    hf_token: str | None = None

class TestRequest(BaseModel):
    config: str = "train_gnn.yaml"
    dataset: str = "my_dataset"
    dataset_repo: str | None = None
    model_repo: str | None = None
    epoch: str | None = None
    hf_token: str | None = None


@app.post("/data")
async def upload_data(files: List[UploadFile] = File(...)):
    """Upload .arrow files (multiple) and optional _zones_map.json."""
    dest_proc = os.path.join("my_dataset", "processed")
    dest_root = os.path.join("my_dataset")
    os.makedirs(dest_proc, exist_ok=True)
    os.makedirs(dest_root, exist_ok=True)
    saved: list[str] = []
    for file in files:
        fname = file.filename
        logging.info("Receiving data file %s", fname)
        if fname.endswith(".arrow"):
            target = os.path.join(dest_proc, fname)
            with open(target, "wb") as f:
                shutil.copyfileobj(file.file, f)
        elif fname == "_zones_map.json":
            data = await file.read()
            for target in [os.path.join(dest_root, fname), os.path.join(dest_proc, fname)]:
                with open(target, "wb") as f:
                    f.write(data)
        else:
            raise HTTPException(status_code=400, detail="only .arrow or _zones_map.json supported")
        saved.append(fname)
    return {"saved": saved}

@app.post("/config")
async def upload_config(file: UploadFile = File(...)):
    """Upload training configuration YAML (e.g. train_gnn.yaml)."""
    fname = file.filename or "train_gnn.yaml"
    if not fname.endswith(".yaml"):
        raise HTTPException(status_code=400, detail="only .yaml files supported")
    logging.info("Uploading config %s", fname)
    with open(fname, "wb") as f:
        shutil.copyfileobj(file.file, f)
    return {"saved": fname}

def _launch(cmd: list[str]):
    """Run command in background without blocking the API."""
    logging.info("Launching command: %s", " ".join(cmd))
    subprocess.Popen(cmd)

@app.post("/train")
def start_train(req: TrainRequest):
    logging.info("Train request: %s", req)
    cmd = [
        "python", "run_pipeline.py",
        "--config", req.config,
        "--dataset", req.dataset,
        "--train",
    ]
    if req.dataset_repo:
        cmd += ["--dataset_repo", req.dataset_repo]
    if req.upload_repo:
        cmd += ["--upload_repo", req.upload_repo]
    if req.hf_token:
        cmd += ["--hf_token", req.hf_token]
    _launch(cmd)
    return {"status": "started"}


@app.post("/infer")
async def infer_block(file: UploadFile = File(...)):
    """Generate building footprints for a block polygon.

    The uploaded file must contain a GeoJSON FeatureCollection with the block
    polygon. The response is a GeoJSON file with generated building polygons in
    the same CRS as the input.
    """
    raw = await file.read()
    try:
        geojson = json.loads(raw)
    except Exception:
        raise HTTPException(status_code=400, detail="invalid GeoJSON")

    try:
        result = infer_from_geojson(geojson)
    except Exception as e:  # pragma: no cover - safe guard
        raise HTTPException(status_code=400, detail=str(e))

    with tempfile.NamedTemporaryFile(
        mode="w", delete=False, suffix=".geojson", encoding="utf-8"
    ) as tmp:
        json.dump(result, tmp)
        tmp_path = tmp.name

    logging.info("Inference produced %d buildings", len(result.get("features", [])))
    return FileResponse(tmp_path, media_type="application/geo+json", filename="buildings.geojson")

@app.post("/test")
def start_test(req: TestRequest):
    logging.info("Test request: %s", req)
    cmd = [
        "python", "run_pipeline.py",
        "--config", req.config,
        "--dataset", req.dataset,
        "--test",
    ]
    if req.dataset_repo:
        cmd += ["--dataset_repo", req.dataset_repo]
    if req.model_repo:
        cmd += ["--model_repo", req.model_repo]
    if req.epoch:
        cmd += ["--epoch", req.epoch]
    if req.hf_token:
        cmd += ["--hf_token", req.hf_token]
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

@app.get("/logs/{name}/download")
def download_log(name: str):
    path = os.path.join("logs", name)
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="log not found")
    logging.info("Serving log file %s", name)
    return FileResponse(path, media_type="text/plain", filename=name)

@app.get("/")
def root():
    return {"status": "ok"}
