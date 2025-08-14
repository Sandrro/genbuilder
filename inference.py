import json
import random
from typing import List, Dict, Any
from shapely.geometry import shape, mapping, Polygon


def _dummy_infer_buildings(block: Polygon, n: int = 5) -> List[Polygon]:
    """Generate a handful of dummy rectangular buildings inside ``block``.

    This is a placeholder for the real model inference. The rectangles are
    axis-aligned and randomly placed within the bounding box of ``block``.
    """
    minx, miny, maxx, maxy = block.bounds
    buildings: List[Polygon] = []
    if minx == maxx or miny == maxy:
        return buildings
    width = (maxx - minx) / 10.0
    height = (maxy - miny) / 10.0
    for _ in range(n):
        x = random.uniform(minx, maxx - width)
        y = random.uniform(miny, maxy - height)
        rect = Polygon([(x, y), (x + width, y), (x + width, y + height), (x, y + height)])
        if block.contains(rect.centroid):
            buildings.append(rect)
    return buildings


def infer_from_geojson(geojson: Dict[str, Any]) -> Dict[str, Any]:
    """Run model inference for a block described by GeoJSON.

    Parameters
    ----------
    geojson: dict
        A GeoJSON FeatureCollection with at least one Polygon feature
        representing the block.

    Returns
    -------
    dict
        GeoJSON FeatureCollection with generated building footprints.

    Notes
    -----
    This function currently uses a dummy generator and should be replaced
    with actual model inference logic.
    """
    if not geojson.get("features"):
        raise ValueError("GeoJSON must contain at least one feature")
    geom = shape(geojson["features"][0]["geometry"])
    buildings = _dummy_infer_buildings(geom)
    features = [{"type": "Feature", "properties": {}, "geometry": mapping(b)} for b in buildings]
    result: Dict[str, Any] = {"type": "FeatureCollection", "features": features}
    if "crs" in geojson:
        result["crs"] = geojson["crs"]
    return result
