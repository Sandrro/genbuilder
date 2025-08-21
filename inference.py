from typing import List, Dict, Any
import os
import math

from shapely.geometry import shape, mapping, Polygon
from shapely import affinity

try:  # optional dependency used only when a remote model path is provided
    from huggingface_hub import hf_hub_download, list_repo_files
except Exception:  # pragma: no cover - library may not be installed in minimal envs
    hf_hub_download = None  # type: ignore
    list_repo_files = None  # type: ignore

import yaml


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
    model_file: str | None = None,
    hf_token: str | None = None,
    model: Any | None = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run model inference for blocks described by GeoJSON using a trained model.

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
    model_repo:
        HuggingFace repository id or local directory containing a model. Must
        be provided when ``model`` is not given.
    model_file:
        Optional file name within ``model_repo`` to load. If omitted the first
        file with extension ``.pt`` or ``.pth`` is used.
    hf_token:
        Optional HuggingFace token used when downloading private repositories.
    model:
        Pre-instantiated model object with an ``infer`` method.  Intended for
        tests; if provided ``model_repo`` is ignored.

    Returns
    -------
    dict
        GeoJSON FeatureCollection with generated building footprints.
    """

    if not geojson.get("features"):
        raise ValueError("GeoJSON must contain at least one feature")

    if model is None:
        if not model_repo:
            raise ValueError("model_repo must be provided")
        model_path: str
        opt: Dict[str, Any] | None = None
        if os.path.isdir(model_repo):
            if model_file is None:
                candidates = [
                    f for f in os.listdir(model_repo) if f.endswith((".pt", ".pth"))
                ]
                if not candidates:
                    raise FileNotFoundError("no model weights found in directory")
                model_file = sorted(candidates)[0]
            model_path = os.path.join(model_repo, model_file)
            for yaml_name in ("train_save.yaml", "resume_train_save.yaml"):
                yaml_path = os.path.join(model_repo, yaml_name)
                if os.path.isfile(yaml_path):
                    with open(yaml_path, "r", encoding="utf-8") as fh:
                        opt = yaml.safe_load(fh) or {}
                    break
        else:
            if hf_hub_download is None:
                raise RuntimeError("huggingface_hub not installed")
            download_kwargs = {"token": hf_token} if hf_token else {}
            if model_file is None:
                if list_repo_files is None:
                    raise RuntimeError("huggingface_hub not installed")
                files = list_repo_files(model_repo, token=hf_token)
                candidates = [f for f in files if f.endswith((".pt", ".pth"))]
                if not candidates:
                    raise FileNotFoundError("no model weights found in repository")
                model_file = sorted(candidates)[0]
            model_path = hf_hub_download(model_repo, model_file, **download_kwargs)
            if opt is None:
                for yaml_name in ("train_save.yaml", "resume_train_save.yaml"):
                    try:
                        yaml_path = hf_hub_download(model_repo, yaml_name, **download_kwargs)
                    except Exception:  # pragma: no cover - file missing
                        continue
                    if not os.path.isfile(yaml_path):
                        continue
                    with open(yaml_path, "r", encoding="utf-8") as fh:
                        opt = yaml.safe_load(fh) or {}
                    break

        import torch  # pragma: no cover - torch not required for tests
        from model import BlockGenerator  # type: ignore

        state = torch.load(model_path, map_location="cpu")  # pragma: no cover - simple load
        state_dict = state.get("model_state_dict", state)
        ckpt_opt = state.get("opt")
        if ckpt_opt:
            opt = ckpt_opt
        if opt is None:
            opt = {"device": "cpu", "latent_dim": 64, "n_ft_dim": 64}
        opt.setdefault("device", "cpu")
        model = BlockGenerator(opt)
        model.load_state_dict(state_dict)
        model.eval()

    total_blocks = len(geojson["features"])
    processed_blocks = 0
    if verbose:
        print(f"Starting generation for {total_blocks} blocks")

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

        if verbose:
            print(f"Processing block {processed_blocks + 1}/{total_blocks} (id={block_id})")
            print(" - canonicalising geometry")
        canon_geom, params = _to_canonical(geom)
        if verbose:
            print(" - generating buildings")
        if not hasattr(model, "infer"):
            raise RuntimeError("Model does not provide an 'infer' method")
        buildings = model.infer(canon_geom, n=n, zone_label=zone_label)  # type: ignore[operator]
        if verbose:
            print(f" - generated {len(buildings)} buildings, transforming back and clipping")

        for b in buildings:
            world_b = _from_canonical(b, params)
            clipped = world_b.intersection(geom)
            if clipped.is_empty:
                continue
            b_props = {zone_attr: zone_label, "block_id": block_id}
            features.append(
                {"type": "Feature", "properties": b_props, "geometry": mapping(clipped)}
            )

        processed_blocks += 1
        if verbose:
            print(f"Finished block {block_id}. Progress: {processed_blocks}/{total_blocks}")

    if verbose:
        print(f"Generation finished. Processed {processed_blocks} blocks in total")

    result: Dict[str, Any] = {"type": "FeatureCollection", "features": features}
    if "crs" in geojson:
        result["crs"] = geojson["crs"]
    return result
