import json
import pathlib
import sys

from shapely.geometry import Polygon, shape, mapping
from shapely import affinity

# ensure root project directory is importable
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from inference import infer_from_geojson


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
        rect = Polygon(
            [(x0, y0), (x0 + width, y0), (x0 + width, y0 + height), (x0, y0 + height)]
        )
        buildings.append(rect)
    return buildings


class DummyModel:
    def infer(self, block, n=5, zone_label=None):
        return _dummy_infer_buildings(block, n, zone_label)


def test_infer_clips_and_rotates_buildings():
    """Buildings must stay within the (rotated) block and remain rectangles."""

    square = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    block = affinity.rotate(square, 45, origin=(0.5, 0.5))
    geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"id": "b1", "zone": "residential"},
                "geometry": mapping(block),
            }
        ],
    }

    result = infer_from_geojson(geojson, block_counts=2, model=DummyModel())
    assert result["type"] == "FeatureCollection"
    assert len(result["features"]) == 2

    for feat in result["features"]:
        building = shape(feat["geometry"])
        assert building.difference(block).area < 1e-9
        # canonical transform should keep buildings rectangular (5 coords incl. repeat)
        assert len(list(building.exterior.coords)) == 5
        assert feat["properties"].get("zone") == "residential"
def test_infer_requires_model():
    block = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"id": "b1", "zone": "residential"},
                "geometry": mapping(block),
            }
        ],
    }
    try:
        infer_from_geojson(geojson, block_counts=1)
    except ValueError:
        pass
    else:  # pragma: no cover - failure case
        assert False, "expected ValueError when model not supplied"


def test_infer_accepts_model_repo(tmp_path, monkeypatch):
    block = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"id": "b1", "zone": "residential"},
                "geometry": mapping(block),
            }
        ],
    }

    import types, sys

    fake_torch = types.SimpleNamespace(load=lambda *a, **k: {"model_state_dict": {}, "opt": {}})
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    class Model(DummyModel):
        def __init__(self, opt):
            pass

        def load_state_dict(self, state):
            pass

        def eval(self):
            return self

    fake_model = types.SimpleNamespace(BlockGenerator=Model)
    monkeypatch.setitem(sys.modules, "model", fake_model)

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "model.pt").write_text("dummy")

    result = infer_from_geojson(
        geojson,
        block_counts=1,
        model_repo=str(model_dir),
    )
    assert len(result["features"]) == 1


def test_infer_passes_hf_token(tmp_path, monkeypatch):
    block = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"id": "b1", "zone": "residential"},
                "geometry": mapping(block),
            }
        ],
    }

    import types, sys

    fake_torch = types.SimpleNamespace(load=lambda *a, **k: {"model_state_dict": {}, "opt": {}})
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    class Model(DummyModel):
        def __init__(self, opt):
            pass

        def load_state_dict(self, state):
            pass

        def eval(self):
            return self

    fake_model = types.SimpleNamespace(BlockGenerator=Model)
    monkeypatch.setitem(sys.modules, "model", fake_model)

    model_dir = tmp_path / "remote"
    model_dir.mkdir()
    (model_dir / "model.pt").write_text("dummy")

    def fake_download(repo, filename, token=None):  # pragma: no cover - simple shim
        assert token == "secret"
        return str(model_dir / filename)

    monkeypatch.setattr("inference.hf_hub_download", fake_download)
    monkeypatch.setattr(
        "inference.list_repo_files", lambda repo, token=None: ["model.pt"]
    )

    result = infer_from_geojson(
        geojson,
        block_counts=1,
        model_repo="some/repo",
        hf_token="secret",
    )
    assert len(result["features"]) == 1
