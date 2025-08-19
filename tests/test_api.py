import json
import pathlib
import sys

from fastapi.testclient import TestClient

# ensure root project directory is importable
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from api import app


def test_infer_block_uses_counts_and_zone():
    """The /infer endpoint should honour provided block counts and zone labels."""

    client = TestClient(app)

    # minimal block polygon with id and zone label
    input_geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"id": "b1", "zone": "residential"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [[0, 0], [0, 1], [1, 1], [1, 0], [0, 0]]
                    ],
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
    counts = {"b1": 3}

    response = client.post("/infer", files=files, data={"counts": json.dumps(counts)})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/geo+json")
    data = json.loads(response.content)
    assert data["type"] == "FeatureCollection"
    assert len(data["features"]) == 3
    # every generated building should carry the zone label
    assert all(f["properties"].get("zone") == "residential" for f in data["features"])

