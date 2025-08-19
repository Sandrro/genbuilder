import json
import pathlib
import sys

from fastapi.testclient import TestClient

# ensure root project directory is importable
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from api import app


def test_infer_block_returns_geojson(monkeypatch):
    """The /infer endpoint should return the GeoJSON produced by the model."""

    client = TestClient(app)

    # minimal block polygon
    input_geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [[0, 0], [0, 1], [1, 1], [1, 0], [0, 0]]
                    ],
                },
            }
        ],
    }

    output_geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {},
                "geometry": {
                    "type": "Point",
                    "coordinates": [0, 0],
                },
            }
        ],
    }

    def fake_infer(geojson, zone_attr):
        assert geojson == input_geojson
        assert zone_attr == "func_zone"
        return output_geojson

    monkeypatch.setattr("api.infer_from_geojson", fake_infer)

    files = {
        "file": (
            "block.geojson",
            json.dumps(input_geojson),
            "application/geo+json",
        )
    }

    response = client.post("/infer", files=files, params={"zone_attr": "func_zone"})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/geo+json")
    data = json.loads(response.content)
    assert data == output_geojson

