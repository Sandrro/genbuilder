import json
import pathlib
import sys

from shapely.geometry import Polygon, shape, mapping

# ensure root project directory is importable
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from inference import infer_from_geojson


def test_infer_clips_buildings_and_accepts_int():
    """infer_from_geojson should clip buildings to block boundaries."""

    block = Polygon([(0, 0), (1, 0), (0, 1), (0, 0)])  # right triangle
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
        assert building.within(block)
        assert feat["properties"].get("zone") == "residential"
