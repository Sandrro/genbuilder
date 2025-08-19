from typing import List, Dict, Any
from shapely.geometry import shape, mapping, Polygon


def _dummy_infer_buildings(block: Polygon, n: int = 5, zone_label: str | None = None) -> List[Polygon]:
    """Generate simple rectangular buildings inside ``block``.

    The pattern of the rectangles is deterministic and optionally depends on
    ``zone_label`` so that tests can assert expected behaviour.  The function
    always returns ``n`` buildings (unless the block is degenerate) to make
    the output predictable.
    """
    minx, miny, maxx, maxy = block.bounds
    buildings: List[Polygon] = []
    if minx == maxx or miny == maxy:
        return buildings

    # Size and vertical offset depend slightly on the zone label in order to
    # simulate different generation patterns per zone.
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
            [
                (x0, y0),
                (x0 + width, y0),
                (x0 + width, y0 + height),
                (x0, y0 + height),
            ]
        )
        buildings.append(rect)
    return buildings


def infer_from_geojson(
    geojson: Dict[str, Any],
    block_counts: Dict[str, int] | None = None,
    zone_attr: str = "zone",
) -> Dict[str, Any]:
    """Run model inference for blocks described by GeoJSON.

    Parameters
    ----------
    geojson:
        A GeoJSON FeatureCollection with Polygon features representing blocks.
    block_counts:
        Optional mapping from block identifiers to the number of buildings that
        should be generated for each block.
    zone_attr:
        Name of the property on each block feature that stores the zone label.

    Returns
    -------
    dict
        GeoJSON FeatureCollection with generated building footprints.
    """

    if not geojson.get("features"):
        raise ValueError("GeoJSON must contain at least one feature")

    features: List[Dict[str, Any]] = []
    for feat in geojson["features"]:
        geom = shape(feat["geometry"])
        props = feat.get("properties", {})
        block_id = str(props.get("id") or feat.get("id") or "")
        zone_label = props.get(zone_attr)
        n = block_counts.get(block_id, 5) if block_counts else 5
        buildings = _dummy_infer_buildings(geom, n, zone_label)
        for b in buildings:
            b_props = {zone_attr: zone_label, "block_id": block_id}
            features.append(
                {"type": "Feature", "properties": b_props, "geometry": mapping(b)}
            )

    result: Dict[str, Any] = {"type": "FeatureCollection", "features": features}
    if "crs" in geojson:
        result["crs"] = geojson["crs"]
    return result
