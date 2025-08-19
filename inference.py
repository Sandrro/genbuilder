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


def infer_from_geojson(geojson: Dict[str, Any], zone_attr: str) -> Dict[str, Any]:
    """Run model inference for a block described by GeoJSON.

    The function extracts functional zone labels from the provided GeoJSON
    using the ``zone_attr`` field of each feature's properties. The dummy
    inference then generates a number of buildings depending on the zone
    label and assigns the label to the generated buildings.

    Parameters
    ----------
    geojson: dict
        A GeoJSON FeatureCollection with one or more Polygon features
        representing blocks.
    zone_attr: str
        Name of the attribute containing the functional zone label.

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
    if not zone_attr:
        raise ValueError("zone_attr must be provided")

    # Mapping from zone label to number of buildings to generate
    zone_building_count = {
        "residential": 10,
        "commercial": 6,
        "industrial": 3,
    }

    result_features: List[Dict[str, Any]] = []
    for feat in geojson["features"]:
        geom = shape(feat["geometry"])
        zone_label = feat.get("properties", {}).get(zone_attr)
        n = zone_building_count.get(zone_label, 5)
        buildings = _dummy_infer_buildings(geom, n=n)
        for b in buildings:
            props = {zone_attr: zone_label} if zone_label is not None else {}
            result_features.append({"type": "Feature", "properties": props, "geometry": mapping(b)})

    result: Dict[str, Any] = {"type": "FeatureCollection", "features": result_features}
    if "crs" in geojson:
        result["crs"] = geojson["crs"]
    return result
