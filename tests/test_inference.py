import json
import pathlib
import sys

from shapely.geometry import Polygon, shape, mapping
from shapely import affinity

# ensure root project directory is importable
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from inference import infer_from_geojson


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

    result = infer_from_geojson(geojson, block_counts=2)
    assert result["type"] == "FeatureCollection"
    assert len(result["features"]) == 2

    for feat in result["features"]:
        building = shape(feat["geometry"])
        assert building.difference(block).area < 1e-9
        # canonical transform should keep buildings rectangular (5 coords incl. repeat)
        assert len(list(building.exterior.coords)) == 5
        assert feat["properties"].get("zone") == "residential"


def test_infer_accepts_model_repo(tmp_path):
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

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "model.pt").write_text("dummy")

    result = infer_from_geojson(
        geojson,
        block_counts=1,
        model_repo=str(model_dir),
        model_file="model.pt",
    )
    assert len(result["features"]) == 1
