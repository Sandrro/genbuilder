"""Utility helpers for running centroid inference through the training script."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Iterable, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field, PrivateAttr


class InferParams(BaseModel):
    """Parameters that control the inference call."""

    slots: int = Field(..., ge=1, description="Number of slots for inference")
    knn: int = Field(..., ge=1, description="k value for kNN graph construction")
    e_thr: float = Field(..., description="Threshold for edge activation")
    il_thr: float = Field(..., description="Threshold for is_living probability")
    sv1_thr: float = Field(..., description="Threshold for the first service head")


_BASE_DIR = Path(__file__).resolve().parents[1]
_ARTIFACTS_DIR = Path(os.getenv("GENBUILDER_ARTIFACTS_DIR", _BASE_DIR / "artifacts"))


def _resolve_train_script() -> str:
    return os.getenv("GENBUILDER_TRAIN_SCRIPT", str(_BASE_DIR / "train.py"))


def _resolve_model_ckpt() -> str:
    env_value = os.getenv("GENBUILDER_MODEL_CKPT")
    if env_value:
        return env_value
    if _ARTIFACTS_DIR.exists():
        for pattern in ("*.pt", "*.ckpt"):
            matches = sorted(_ARTIFACTS_DIR.glob(pattern))
            if matches:
                return str(matches[0])
    return str(_ARTIFACTS_DIR / "model.pt")


def _resolve_config_path() -> Optional[str]:
    env_value = os.getenv("GENBUILDER_CONFIG_PATH")
    if env_value:
        return env_value
    candidate = _BASE_DIR / "train_gnn.yaml"
    return str(candidate)


def _resolve_device() -> str:
    return "cuda"


class CentroidRequest(BaseModel):
    """Payload accepted by the centroid generation service."""

    zone_label: str = Field(..., description="Identifier of the zone that owns the block")
    feature: Dict[str, Any] = Field(..., description="GeoJSON feature representing the block")
    infer_params: InferParams = Field(..., description="Parameters controlling inference")
    la_target: float = Field(..., description="Target living area for the block")
    floors_avg: float = Field(..., description="Average floors value for the block")

    _request_id: str = PrivateAttr()
    _zone_attr: str = PrivateAttr(default="zone")
    _train_script: str = PrivateAttr(default_factory=_resolve_train_script)
    _model_ckpt: str = PrivateAttr(default_factory=_resolve_model_ckpt)
    _config: Optional[str] = PrivateAttr(default_factory=_resolve_config_path)
    _device: str = PrivateAttr(default_factory=_resolve_device)
    _python_executable: str = PrivateAttr(default_factory=lambda: sys.executable)

    def __init__(self, **data: Any) -> None:  # type: ignore[override]
        super().__init__(**data)
        self._request_id = str(uuid4())

    @property
    def request_id(self) -> str:
        return self._request_id

    @property
    def zone_attr(self) -> str:
        return self._zone_attr

    @property
    def train_script(self) -> str:
        return self._train_script

    @property
    def model_ckpt(self) -> str:
        return self._model_ckpt

    @property
    def config(self) -> Optional[str]:
        return self._config

    @property
    def device(self) -> str:
        return self._device

    @property
    def python_executable(self) -> str:
        return self._python_executable


class CentroidResult(BaseModel):
    """Result of the centroid inference request."""

    features: List[Dict[str, Any]]


class CommandLogEntry(BaseModel):
    """Captured stdout/stderr from an inference subprocess invocation."""

    timestamp: datetime
    command: List[str]
    returncode: int
    stdout: str
    stderr: str
    success: bool


_COMMAND_LOGS: deque[CommandLogEntry] = deque(maxlen=200)
_COMMAND_LOGS_LOCK = Lock()


def _record_command_log(entry: CommandLogEntry) -> None:
    """Persist a command log entry in the in-memory history."""

    with _COMMAND_LOGS_LOCK:
        _COMMAND_LOGS.append(entry)


def get_command_logs(limit: Optional[int] = None) -> List[CommandLogEntry]:
    """Return the most recent command logs, optionally constrained by *limit*."""

    def _take_latest(items: Iterable[CommandLogEntry], count: Optional[int]) -> List[CommandLogEntry]:
        entries = list(items)
        if count is None or count >= len(entries):
            return entries
        return entries[-count:]

    with _COMMAND_LOGS_LOCK:
        snapshot = list(_COMMAND_LOGS)

    return _take_latest(snapshot, limit)


def build_inference_command(request: CentroidRequest, in_path: Path, out_path: Path) -> List[str]:
    """Builds the CLI command used to call the training script in inference mode."""

    cmd = [
        request.python_executable or sys.executable,
        request.train_script,
        "--mode",
        "infer",
        "--model-ckpt",
        request.model_ckpt,
        "--infer-geojson-in",
        str(in_path),
        "--infer-out",
        str(out_path),
        "--infer-knn",
        str(request.infer_params.knn),
        "--infer-e-thr",
        str(request.infer_params.e_thr),
        "--infer-il-thr",
        str(request.infer_params.il_thr),
        "--infer-sv1-thr",
        str(request.infer_params.sv1_thr),
        "--infer-slots",
        str(request.infer_params.slots),
        "--zone",
        request.zone_label,
    ]

    if request.config:
        cmd.extend(["--config", request.config])
    if request.device:
        cmd.extend(["--device", request.device])
    if request.la_target is not None:
        cmd.extend(["--la-target", str(request.la_target)])
    if request.floors_avg is not None:
        cmd.extend(["--floors-avg", str(request.floors_avg)])

    return cmd


def _write_feature(path: Path, feature: Dict[str, Any]) -> None:
    payload = {
        "type": "FeatureCollection",
        "features": [feature],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def run_centroid_inference(request: CentroidRequest) -> List[Dict[str, Any]]:
    """Executes the training script to obtain centroid predictions for a single block."""

    with tempfile.TemporaryDirectory(prefix="centroid_infer_") as tmp_dir:
        in_path = Path(tmp_dir) / "input.geojson"
        out_path = Path(tmp_dir) / "output.geojson"
        feature = dict(request.feature)
        feature_props = dict(feature.get("properties") or {})
        feature_props.setdefault(request.zone_attr, request.zone_label)
        feature["properties"] = feature_props
        _write_feature(in_path, feature)

        cmd = build_inference_command(request, in_path, out_path)
        result = subprocess.run(
            cmd,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        log_entry = CommandLogEntry(
            timestamp=datetime.now(timezone.utc),
            command=cmd,
            returncode=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
            success=result.returncode == 0,
        )
        _record_command_log(log_entry)
        if result.returncode != 0:
            raise RuntimeError(
                "Centroid inference command failed",
                {
                    "returncode": result.returncode,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "command": cmd,
                },
            )

        if not out_path.exists():
            raise RuntimeError("Inference output was not produced", {"command": cmd})

        try:
            output = json.loads(out_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise RuntimeError("Failed to parse inference output", {"path": str(out_path)}) from exc

        features = output.get("features", [])
        if not isinstance(features, list):
            raise RuntimeError("Malformed inference output: 'features' should be a list")

        return features


__all__ = [
    "CentroidRequest",
    "CentroidResult",
    "InferParams",
    "CommandLogEntry",
    "build_inference_command",
    "get_command_logs",
    "run_centroid_inference",
]