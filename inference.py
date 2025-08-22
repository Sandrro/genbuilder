from typing import List, Dict, Any
import os
import math

from shapely.geometry import shape, mapping, Polygon
from shapely import affinity
import numpy as np
import networkx as nx
from PIL import Image, ImageDraw

try:  # optional dependency used only when a remote model path is provided
    from huggingface_hub import hf_hub_download, list_repo_files
except Exception:  # pragma: no cover - library may not be installed in minimal envs
    hf_hub_download = None  # type: ignore
    list_repo_files = None  # type: ignore

import yaml


def _to_canonical(poly: Polygon) -> tuple[Polygon, dict[str, Any]]:
    """Rotate, scale and translate ``poly`` to a canonical unit frame.

    The block is rotated so that its longest side becomes horizontal, then it
    is scaled separately along the X and Y axes using the dimensions of the
    minimum rotated rectangle (long and short side respectively).  Finally the
    polygon is translated so that its centroid lies at the origin.

    Returns
    -------
    (Polygon, dict)
        The transformed polygon together with parameters required to invert
        the transform via :func:`_from_canonical`.
    """

    mrr = poly.minimum_rotated_rectangle
    coords = list(mrr.exterior.coords)
    edges = [(coords[i], coords[(i + 1) % 4]) for i in range(4)]
    lengths = [math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in edges]

    idx_long = max(range(4), key=lambda i: lengths[i])
    idx_short = min(range(4), key=lambda i: lengths[i])
    a, b = edges[idx_long]
    angle = math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))

    long_side = lengths[idx_long]
    short_side = lengths[idx_short] if lengths[idx_short] > 0 else 1.0

    centroid = poly.centroid
    rotated = affinity.rotate(poly, -angle, origin=centroid)
    scaled = affinity.scale(
        rotated,
        xfact=1 / long_side,
        yfact=1 / short_side,
        origin=centroid,
    )
    shifted_centroid = scaled.centroid
    translated = affinity.translate(scaled, xoff=-shifted_centroid.x, yoff=-shifted_centroid.y)
    params = {
        "angle": angle,
        "scale_x": long_side,
        "scale_y": short_side,
        "origin": (centroid.x, centroid.y),
        "shift": (shifted_centroid.x, shifted_centroid.y),
    }
    return translated, params


def _from_canonical(poly: Polygon, params: dict[str, Any]) -> Polygon:
    """Apply inverse of :func:`_to_canonical` using ``params``."""

    xoff, yoff = params["shift"]
    origin = params["origin"]
    unshifted = affinity.translate(poly, xoff=xoff, yoff=yoff)
    unscaled = affinity.scale(
        unshifted,
        xfact=params.get("scale_x", 1.0),
        yfact=params.get("scale_y", 1.0),
        origin=origin,
    )
    return affinity.rotate(unscaled, params["angle"], origin=origin)


def _block_long_side(block: Polygon) -> float:
    """Return the length of the longer side of ``block``'s minimum rectangle."""

    mrr = block.minimum_rotated_rectangle
    coords = list(mrr.exterior.coords)
    a0, a1, a2, a3 = coords[0], coords[1], coords[2], coords[3]
    e1 = math.hypot(a1[0] - a0[0], a1[1] - a0[1])
    e2 = math.hypot(a2[0] - a1[0], a2[1] - a1[1])
    return max(e1, e2)


def _rasterize_block_mask(block: Polygon, out_size: int = 64) -> np.ndarray:
    """Rasterize ``block`` polygon to a binary mask of size ``out_size``."""

    minx, miny, maxx, maxy = block.bounds
    w = maxx - minx
    h = maxy - miny
    if w <= 0 or h <= 0:
        return np.zeros((out_size, out_size), dtype=np.uint8)
    sx = (out_size - 2) / w
    sy = (out_size - 2) / h
    s = min(sx, sy)
    ox = 1 - minx * s
    oy = 1 - miny * s
    pts = [(p[0] * s + ox, p[1] * s + oy) for p in np.asarray(block.exterior.coords)]
    img = Image.new("L", (out_size, out_size), 0)
    draw = ImageDraw.Draw(img)
    draw.polygon(pts, fill=255)
    return np.array(img, dtype=np.uint8)


def _build_graph_original(
    block: Polygon,
    pos: np.ndarray,
    size: np.ndarray,
    aspect_ratio: float,
    k_nn: int = 4,
    mask_size: int = 64,
    logger: Any | None = None,
) -> nx.Graph:
    """Reimplementation of ``transform.build_graph_original`` without skgeom."""

    n = pos.shape[0]
    G = nx.Graph()
    for i in range(n):
        G.add_node(i)
        G.nodes[i]["posx"] = float(pos[i, 0])
        G.nodes[i]["posy"] = float(pos[i, 1])
        G.nodes[i]["size_x"] = float(size[i, 0])
        G.nodes[i]["size_y"] = float(size[i, 1])
        G.nodes[i]["exist"] = int(1)
        G.nodes[i]["merge"] = int(0)
        G.nodes[i]["shape"] = int(0)
        G.nodes[i]["iou"] = float(0.0)

    if n >= 2 and k_nn > 0:
        dist = pos[:, None, :] - pos[None, :, :]
        dist = np.sum(dist * dist, axis=2)
        np.fill_diagonal(dist, np.inf)
        indices = np.argsort(dist, axis=1)[:, : min(k_nn, n - 1)]
        edge_count = 0
        for i in range(n):
            for j in indices[i]:
                if i < j:
                    G.add_edge(i, int(j))
                    edge_count += 1
        if logger:
            logger.debug("  • kNN edges added: %d (k=%d)", edge_count, k_nn)

    ls = _block_long_side(block)
    G.graph["aspect_ratio"] = float(aspect_ratio)
    G.graph["long_side"] = float(ls)
    G.graph["binary_mask"] = _rasterize_block_mask(block, out_size=mask_size)
    G.graph["block_scale"] = float(ls)

    if logger:
        logger.debug(
            "  • Graph stats: nodes=%d, edges=%d, mask=%s",
            G.number_of_nodes(),
            G.number_of_edges(),
            tuple(G.graph["binary_mask"].shape),
        )

    return G


def _infer_opt_from_state(state_dict: Dict[str, Any]) -> Dict[str, int]:
    """Derive model hyperparameters from a checkpoint ``state_dict``.

    Parameters
    ----------
    state_dict:
        State dictionary containing at least ``ft_init.weight`` and
        ``d_ft_init.weight`` tensors.

    Returns
    -------
    dict
        Mapping with inferred ``n_ft_dim`` (latent channels), ``latent_dim``
        and ``N`` (max number of nodes). Missing keys result in an empty
        dictionary.
    """

    try:
        ft_shape = state_dict["ft_init.weight"].shape
        latent_ch = ft_shape[0] * 2
        N = ft_shape[1] - int(0.75 * latent_ch)
        dft_shape = state_dict["d_ft_init.weight"].shape
        latent_dim = dft_shape[1]
    except Exception:
        return {}

    return {"n_ft_dim": latent_ch, "latent_dim": latent_dim, "N": N}


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
            opt = {}

        inferred = _infer_opt_from_state(state_dict)
        opt.setdefault("n_ft_dim", inferred.get("n_ft_dim", 64))
        opt.setdefault("latent_dim", inferred.get("latent_dim", 64))
        N = opt.pop("N", inferred.get("N", 80))
        opt.setdefault("device", "cpu")

        model = BlockGenerator(opt, N)
        model.load_state_dict(state_dict)
        # ensure model parameters reside on the configured device
        if hasattr(model, "to"):
            model.to(opt["device"])
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
        # shrink block by 5m to keep a buffer from the edges
        buffered = geom.buffer(-5)
        if not buffered.is_empty and buffered.area > 0:
            geom = buffered
        elif verbose:
            print(" - buffer too large, using original geometry")
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

        world_buildings: List[Polygon] = []
        for b in buildings:
            world_b = _from_canonical(b, params)
            clipped = world_b.intersection(geom)
            if clipped.is_empty:
                continue
            world_buildings.append(clipped)
            b_props = {zone_attr: zone_label, "block_id": block_id}
            features.append(
                {"type": "Feature", "properties": b_props, "geometry": mapping(clipped)}
            )

        # best-effort reverse transform using dataset graph utilities
        if world_buildings:
            try:
                pos = np.array(
                    [[p.centroid.x, p.centroid.y] for p in world_buildings], dtype=float
                )
                size = np.array(
                    [
                        [p.bounds[2] - p.bounds[0], p.bounds[3] - p.bounds[1]]
                        for p in world_buildings
                    ],
                    dtype=float,
                )
                aspect_ratio = (
                    params.get("scale_y", 1.0) / params.get("scale_x", 1.0)
                    if params.get("scale_x")
                    else 1.0
                )
                _build_graph_original(
                    geom, pos, size, aspect_ratio, k_nn=6, mask_size=128
                )
            except Exception:
                if verbose:
                    print(" - reverse transform failed, continuing")

        processed_blocks += 1
        if verbose:
            print(f"Finished block {block_id}. Progress: {processed_blocks}/{total_blocks}")

    if verbose:
        print(f"Generation finished. Processed {processed_blocks} blocks in total")

    result: Dict[str, Any] = {"type": "FeatureCollection", "features": features}
    if "crs" in geojson:
        result["crs"] = geojson["crs"]
    return result
