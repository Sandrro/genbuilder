import argparse
import json
from typing import Dict, Any, List

from shapely.geometry import shape, mapping

from inference import infer_from_geojson


def _attach_buildings(blocks: Dict[str, Any], buildings: Dict[str, Any]) -> None:
    """Attach building polygons to corresponding blocks in-place.

    Each building is assigned to the first block whose geometry contains it.
    If a building does not lie within any block it is ignored.
    """
    block_geoms: List[tuple[Any, Dict[str, Any]]] = []
    for feat in blocks.get("features", []):
        geom = shape(feat["geometry"])
        block_geoms.append((geom, feat))

    for b in buildings.get("features", []):
        b_geom = shape(b["geometry"])
        for geom, feat in block_geoms:
            if geom.contains(b_geom):
                props = feat.setdefault("properties", {})
                lst = props.setdefault("buildings", [])
                lst.append(mapping(b_geom))
                break


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate building polygons for blocks")
    parser.add_argument("--blocks", required=True, help="Path to GeoJSON with block polygons")
    parser.add_argument("--buildings", help="Optional GeoJSON with existing buildings")
    parser.add_argument("--model-repo", required=True, help="Path or HuggingFace repo with model weights")
    parser.add_argument("--model-file", help="Specific model file name")
    parser.add_argument("--hf-token", help="HuggingFace access token")
    parser.add_argument("--n-buildings", type=int, default=5, help="Number of buildings to generate per block")
    parser.add_argument("--zone-attr", default="zone", help="Name of zone attribute in input GeoJSON")
    parser.add_argument("--output", required=True, help="Output GeoJSON file path")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--use-midaxis-inverse", action="store_true",
                        help="Inverse-warp along medial axis before mapping back")
    parser.add_argument("--buffer-margin", type=float, default=6.0,
                        help="Inner buffer of block (m) to keep away from edges")
    parser.add_argument("--min-gap", type=float, default=4.0,
                       help="Minimum gap to block boundary and between buildings (m)")
    parser.add_argument("--min-centroid", type=float, default=9.0,
                        help="Minimum centroid-to-centroid distance (m)")
    parser.add_argument("--dedupe-iou", type=float, default=0.35,
                        help="IoU threshold for NMS deduplication")
    args = parser.parse_args()

    with open(args.blocks, "r", encoding="utf-8") as f:
        blocks_geojson = json.load(f)

    if args.buildings:
        with open(args.buildings, "r", encoding="utf-8") as f:
            buildings_geojson = json.load(f)
        _attach_buildings(blocks_geojson, buildings_geojson)

    result = infer_from_geojson(
        blocks_geojson,
        block_counts=args.n_buildings,
        zone_attr=args.zone_attr,
        model_repo=args.model_repo,
        model_file=args.model_file,
        hf_token=args.hf_token,
        use_midaxis_inverse=args.use_midaxis_inverse,
        buffer_margin=args.buffer_margin,
        dedupe_iou=args.dedupe_iou,
        verbose=args.verbose,
    )

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f)


if __name__ == "__main__":  # pragma: no cover - manual execution
    main()
