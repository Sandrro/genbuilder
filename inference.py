from typing import List, Dict, Any
import os
import math

from shapely.geometry import shape, mapping, Polygon
from shapely import affinity

try:  # optional dependency used only when a remote model path is provided
    from huggingface_hub import hf_hub_download
except Exception:  # pragma: no cover - library may not be installed in minimal envs
    hf_hub_download = None  # type: ignore


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


def _to_canonical(poly: Polygon) -> tuple[Polygon, dict[str, Any]]:
    """Rotate, scale and translate ``poly`` to a canonical unit frame.

    Returns the transformed polygon together with parameters required to
    invert the transform via :func:`_from_canonical`.
    """
    mrr = poly.minimum_rotated_rectangle
    coords = list(mrr.exterior.coords)
    edges = [(coords[i], coords[(i + 1) % 4]) for i in range(4)]
    lengths = [math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in edges]
    idx = max(range(4), key=lambda i: lengths[i])
    a, b = edges[idx]
    angle = math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))

    centroid = poly.centroid
    rotated = affinity.rotate(poly, -angle, origin=centroid)
    long_side = lengths[idx]
    scaled = affinity.scale(rotated, xfact=1 / long_side, yfact=1 / long_side, origin=centroid)
    shifted_centroid = scaled.centroid
    translated = affinity.translate(scaled, xoff=-shifted_centroid.x, yoff=-shifted_centroid.y)
    params = {
        "angle": angle,
        "scale": long_side,
        "origin": (centroid.x, centroid.y),
        "shift": (shifted_centroid.x, shifted_centroid.y),
    }
    return translated, params


def _from_canonical(poly: Polygon, params: dict[str, Any]) -> Polygon:
    """Apply inverse of :func:`_to_canonical` using ``params``."""
    xoff, yoff = params["shift"]
    origin = params["origin"]
    unshifted = affinity.translate(poly, xoff=xoff, yoff=yoff)
    unscaled = affinity.scale(unshifted, xfact=params["scale"], yfact=params["scale"], origin=origin)
    return affinity.rotate(unscaled, params["angle"], origin=origin)


def infer_from_geojson(
    geojson: Dict[str, Any],
    block_counts: Dict[str, int] | int | None = None,
    zone_attr: str = "zone",
    model_repo: str | None = None,
    model_file: str = "model.pt",
) -> Dict[str, Any]:
    """Run model inference for blocks described by GeoJSON.

    Parameters
    ----------
    geojson:
        A GeoJSON FeatureCollection with Polygon features representing blocks.
    block_counts:
        Either a mapping from block identifiers to the number of buildings to
        generate for each block or a single integer applied to every block. If
        omitted, 5 buildings per block are generated.
    zone_attr:
        Name of the property on each block feature that stores the zone label.

    Returns
    -------
    dict
        GeoJSON FeatureCollection with generated building footprints.
    """

    if not geojson.get("features"):
        raise ValueError("GeoJSON must contain at least one feature")

    if model_repo:
        model_path = model_repo
        if not os.path.isdir(model_repo):
            if hf_hub_download is None:
                raise RuntimeError("huggingface_hub not installed")
            model_path = hf_hub_download(model_repo, model_file)
        # Actual model loading is outside the scope of this simplified example.

    features: List[Dict[str, Any]] = []

    count_map: Dict[str, int] = {}
    default_n = 5
    if isinstance(block_counts, int):
        default_n = block_counts
    elif isinstance(block_counts, dict):
        count_map = block_counts

    for feat in geojson["features"]:
        geom = shape(feat["geometry"])
        props = feat.get("properties", {})
        block_id = str(props.get("id") or feat.get("id") or "")
        zone_label = props.get(zone_attr)
        n = count_map.get(block_id, default_n)

        canon_geom, params = _to_canonical(geom)
        buildings = _dummy_infer_buildings(canon_geom, n, zone_label)

        for b in buildings:
            world_b = _from_canonical(b, params)
            clipped = world_b.intersection(geom)
            if clipped.is_empty:
                continue
            b_props = {zone_attr: zone_label, "block_id": block_id}
            features.append(
                {"type": "Feature", "properties": b_props, "geometry": mapping(clipped)}
            )

    result: Dict[str, Any] = {"type": "FeatureCollection", "features": features}
    if "crs" in geojson:
        result["crs"] = geojson["crs"]
    return result
