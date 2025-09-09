import json
import pathlib
import sys
import types
from shapely.geometry import Polygon, mapping

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))


def _dummy_infer_buildings(block, n=5, zone_label=None):
    minx, miny, maxx, maxy = block.bounds
    buildings = []
    if minx == maxx or miny == maxy:
        return buildings
    width = (maxx - minx) / (2 * n)
    height = (maxy - miny) / 5
    y0 = miny + (maxy - miny) * 0.1
    for i in range(n):
        x0 = minx + i * 2 * width
        rect = Polygon([(x0, y0), (x0 + width, y0), (x0 + width, y0 + height), (x0, y0 + height)])
        buildings.append(rect)
    return buildings


class DummyModel:
    def __init__(self, *args, **kwargs):
        pass

    def load_state_dict(self, state):
        pass

    def eval(self):
        return self

    def infer(self, block, n=5, zone_label=None):
        return _dummy_infer_buildings(block, n, zone_label)


def test_cli_generates_buildings(tmp_path, monkeypatch):
    fake_torch = types.SimpleNamespace(load=lambda *a, **k: {"model_state_dict": {}, "opt": {}})
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    fake_model = types.SimpleNamespace(BlockGenerator=DummyModel)
    monkeypatch.setitem(sys.modules, "model", fake_model)

    import inference_cli

    block = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    blocks_geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"id": "b1", "zone": "residential"},
                "geometry": mapping(block),
            }
        ],
    }
    blocks_path = tmp_path / "blocks.geojson"
    blocks_path.write_text(json.dumps(blocks_geojson))

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "model.pt").write_text("dummy")

    out_path = tmp_path / "out.geojson"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "inference_cli.py",
            "--blocks",
            str(blocks_path),
            "--model-repo",
            str(model_dir),
            "--n-buildings",
            "2",
            "--output",
            str(out_path),
        ],
    )

    inference_cli.main()

    result = json.loads(out_path.read_text())
    assert result["type"] == "FeatureCollection"
    assert len(result["features"]) == 2
