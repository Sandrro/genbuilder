import json
import pathlib
import sys

from fastapi.testclient import TestClient

# ensure root project directory is importable
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from api import app


def _dummy_infer_buildings(block, n=5, zone_label=None):
    minx, miny, maxx, maxy = block.bounds
    buildings = []
    if minx == maxx or miny == maxy:
        return buildings
    width = (maxx - minx) / (2 * n)
    height = (maxy - miny) / 5
    offset_factor = 0.1
    if zone_label:
        zl = str(zone_label).lower()
        if "industrial" in zl:
            offset_factor = 0.5
        elif "commercial" in zl:
            offset_factor = 0.3
    y0 = miny + (maxy - miny) * offset_factor
    for i in range(n):
        x0 = minx + i * 2 * width
        buildings.append(
            {
                "type": "Polygon",
                "coordinates": [
                    [
                        [x0, y0],
                        [x0 + width, y0],
                        [x0 + width, y0 + height],
                        [x0, y0 + height],
                        [x0, y0],
                    ]
                ],
            }
        )
    return buildings


class DummyModel:
    def __init__(self, opt=None):
        pass

    def load_state_dict(self, state):
        pass

    def eval(self):
        return self

    def infer(self, block, n=5, zone_label=None):
        from shapely.geometry import shape

        geoms = _dummy_infer_buildings(block, n, zone_label)
        return [shape(g) for g in geoms]


def _patch_dummy_model(monkeypatch):
    import types, sys

    fake_torch = types.SimpleNamespace(load=lambda *a, **k: {"model_state_dict": {}, "opt": {}})
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    fake_model = types.SimpleNamespace(BlockGenerator=DummyModel)
    monkeypatch.setitem(sys.modules, "model", fake_model)


def test_infer_block_uses_counts_and_zone(tmp_path, monkeypatch):
    """The /infer endpoint should honour provided block counts and zone labels."""

    _patch_dummy_model(monkeypatch)
    client = TestClient(app)

    input_geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"id": "b1", "zone": "residential"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[0, 0], [0, 1], [1, 1], [1, 0], [0, 0]]],
                },
            }
        ],
    }

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "model.pt").write_text("dummy")

    files = {
        "file": (
            "block.geojson",
            json.dumps(input_geojson),
            "application/geo+json",
        )
    }
    counts = {"b1": 3}

    response = client.post(
        "/infer",
        files=files,
        data={"counts": json.dumps(counts), "model_repo": str(model_dir)},
    )
    assert response.status_code == 200
    data = json.loads(response.content)
    assert data["type"] == "FeatureCollection"
    assert len(data["features"]) == 3
    assert all(f["properties"].get("zone") == "residential" for f in data["features"])


def test_infer_block_accepts_single_count(tmp_path, monkeypatch):
    """Counts parameter may be a single integer applied to all blocks."""

    _patch_dummy_model(monkeypatch)
    client = TestClient(app)

    input_geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"id": "b1", "zone": "residential"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[0, 0], [0, 1], [1, 1], [1, 0], [0, 0]]],
                },
            }
        ],
    }

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "model.pt").write_text("dummy")

    files = {
        "file": (
            "block.geojson",
            json.dumps(input_geojson),
            "application/geo+json",
        )
    }

    response = client.post(
        "/infer",
        files=files,
        data={"counts": "4", "model_repo": str(model_dir)},
    )
    assert response.status_code == 200
    data = json.loads(response.content)
    assert len(data["features"]) == 4
    assert all(f["properties"].get("zone") == "residential" for f in data["features"])

def test_infer_block_accepts_model_repo(tmp_path, monkeypatch):
    client = TestClient(app)
    input_geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"id": "b1", "zone": "residential"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[0, 0], [0, 1], [1, 1], [1, 0], [0, 0]]],
                },
            }
        ],
    }
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "model.pt").write_text("dummy")
    files = {
        "file": ("block.geojson", json.dumps(input_geojson), "application/geo+json"),
    }
    _patch_dummy_model(monkeypatch)

    response = client.post(
        "/infer",
        files=files,
        data={"counts": "1", "model_repo": str(model_dir)},
    )
    assert response.status_code == 200
    data = json.loads(response.content)
    assert data["type"] == "FeatureCollection"
    assert len(data["features"]) == 1
    assert data["features"][0]["properties"].get("zone") == "residential"


def test_infer_block_forwards_hf_token(tmp_path, monkeypatch):
    client = TestClient(app)
    input_geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"id": "b1", "zone": "residential"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[0, 0], [0, 1], [1, 1], [1, 0], [0, 0]]],
                },
            }
        ],
    }
    files = {"file": ("block.geojson", json.dumps(input_geojson), "application/geo+json")}
    model_dir = tmp_path / "remote"
    model_dir.mkdir()
    (model_dir / "model.pt").write_text("dummy")

    def fake_download(repo, filename, token=None):  # pragma: no cover - simple shim
        assert token == "secret"
        return str(model_dir / filename)

    monkeypatch.setattr("inference.hf_hub_download", fake_download)
    monkeypatch.setattr("inference.list_repo_files", lambda repo, token=None: ["model.pt"])

    _patch_dummy_model(monkeypatch)

    response = client.post(
        "/infer",
        files=files,
        data={
            "counts": "1",
            "model_repo": "some/repo",
            "hf_token": "secret",
        },
    )
    assert response.status_code == 200
    data = json.loads(response.content)
    assert len(data["features"]) == 1


def test_infer_block_requires_model_repo(tmp_path):
    client = TestClient(app)
    input_geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"id": "b1", "zone": "residential"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[0, 0], [0, 1], [1, 1], [1, 0], [0, 0]]],
                },
            }
        ],
    }
    files = {
        "file": (
            "block.geojson",
            json.dumps(input_geojson),
            "application/geo+json",
        )
    }
    response = client.post("/infer", files=files, data={"counts": "1", "model_repo": ""})
    assert response.status_code == 400
