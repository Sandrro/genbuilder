from typing import List, Dict, Any, Optional, Union, Iterable, Tuple
import os
import math

from shapely.geometry import shape, mapping, Polygon, MultiPolygon, base
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

# geometry library used for mid-axis based warp
try:
    import skgeom  # type: ignore
except Exception:  # pragma: no cover - optional geometry library
    skgeom = None  # type: ignore

# make_valid (shapely>=2) — с fallback на buffer(0)
try:  # pragma: no cover
    from shapely.validation import make_valid as _shp_make_valid

    def _make_valid(g: base.BaseGeometry) -> base.BaseGeometry:
        return _shp_make_valid(g)
except Exception:  # pragma: no cover
    def _make_valid(g: base.BaseGeometry) -> base.BaseGeometry:
        try:
            return g.buffer(0)
        except Exception:
            return g


# =========================
# Canonical <-> World space
# =========================

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
        xfact=1 / max(long_side, 1e-9),
        yfact=1 / max(short_side, 1e-9),
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


# ==================
# Graph construction
# ==================

def _build_graph_original(
    block: Polygon,
    pos: np.ndarray,
    size: np.ndarray,
    aspect_ratio: float,
    k_nn: int = 4,
    mask_size: int = 64,
    logger: Optional[Any] = None,
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


# ======================
# Utility / Postprocess
# ======================

def _flatten_polys(geom: base.BaseGeometry) -> Iterable[Polygon]:
    if geom.is_empty:
        return []
    if isinstance(geom, Polygon):
        return [geom]
    if isinstance(geom, MultiPolygon):
        return list(geom.geoms)
    # любой другой — пытаемся привести к полигонам
    try:
        return [g for g in geom.geoms if isinstance(g, Polygon)]  # type: ignore[attr-defined]
    except Exception:
        return []


def _polygon_area(geom: base.BaseGeometry) -> float:
    try:
        return float(geom.area)
    except Exception:
        return 0.0


def _iou(a: Polygon, b: Polygon) -> float:
    inter = a.intersection(b)
    if inter.is_empty:
        return 0.0
    u = a.union(b)
    ia = _polygon_area(inter)
    ua = _polygon_area(u)
    return (ia / ua) if ua > 0 else 0.0


def _nms(polys: List[Polygon], iou_thr: float) -> List[Polygon]:
    kept: List[Polygon] = []
    for p in polys:
        drop = False
        for q in kept:
            if _iou(p, q) >= iou_thr:
                drop = True
                break
        if not drop:
            kept.append(p)
    return kept


def _safe_clip_to_block(poly: Polygon, block: Polygon) -> List[Polygon]:
    clipped = poly.intersection(block)
    if clipped.is_empty:
        return []
    return [g for g in _flatten_polys(_make_valid(clipped)) if _polygon_area(g) > 0]


# ==============================
# Mid-axis inverse warp (optional)
# ==============================

def _try_midaxis_inverse(
    canon_block: Polygon,
    canon_buildings: List[Polygon],
    verbose: bool = False,
) -> Optional[List[Polygon]]:
    """Attempt to inverse-warp buildings along block medial axis using skgeom.

    Returns a new list of polygons on success; None on failure or if skgeom is missing.
    """
    if skgeom is None:
        return None
    try:  # pragma: no cover - heavy geometry branch
        from example_canonical_transform import (
            get_polyskeleton_longest_path,
            modified_skel_to_medaxis,
        )
        from geo_utils import (
            inverse_warp_bldg_by_midaxis,
            get_block_aspect_ratio,
        )

        def _shapely_to_skgeom(poly: Polygon):
            exterior_polyline = list(poly.exterior.coords)[:-1]
            exterior_polyline.reverse()
            return skgeom.Polygon(exterior_polyline)

        sk_blk = _shapely_to_skgeom(canon_block)
        skel = skgeom.skeleton.create_interior_straight_skeleton(sk_blk)
        _, longest_skel = get_polyskeleton_longest_path(skel, sk_blk)
        medaxis = modified_skel_to_medaxis(longest_skel, canon_block)
        aspect_rto = get_block_aspect_ratio(canon_block, medaxis)
        pos = np.array([[p.centroid.x, p.centroid.y] for p in canon_buildings], dtype=float)
        size = np.array(
            [[p.bounds[2] - p.bounds[0], p.bounds[3] - p.bounds[1]] for p in canon_buildings],
            dtype=float,
        )
        warped_buildings, _, _ = inverse_warp_bldg_by_midaxis(pos, size, medaxis, aspect_rto)
        # функция возвращает список shapely-полигонов
        out_polys = [
            _make_valid(g) for g in warped_buildings if isinstance(g, (Polygon, MultiPolygon))
        ]
        out_flat: List[Polygon] = [gg for g in out_polys for gg in _flatten_polys(g)]
        return out_flat
    except Exception as e:
        if verbose:
            print(f" - mid-axis inverse warp failed: {e}")
        return None


# ==============
# Main inference
# ==============

def _infer_opt_from_state(state_dict: Dict[str, Any]) -> Dict[str, int]:
    """Derive model hyperparameters from a checkpoint ``state_dict``."""

    try:
        ft_shape = state_dict["ft_init.weight"].shape
        latent_ch = ft_shape[0] * 2
        N = ft_shape[1] - int(0.75 * latent_ch)
        dft_shape = state_dict["d_ft_init.weight"].shape
        latent_dim = dft_shape[1]
    except Exception:
        return {}

    return {"n_ft_dim": latent_ch, "latent_dim": latent_dim, "N": N}


def _normalize_model_output(
    raw: Union[List[Polygon], Tuple[List[Polygon], Any], Any]
) -> List[Polygon]:
    """Accept a few common return shapes from model.infer and normalize to List[Polygon]."""
    if isinstance(raw, tuple) and len(raw) >= 1 and isinstance(raw[0], list):
        raw = raw[0]
    if isinstance(raw, list) and all(isinstance(p, (Polygon, MultiPolygon)) for p in raw):
        flat: List[Polygon] = []
        for g in raw:  # type: ignore[assignment]
            flat.extend(_flatten_polys(_make_valid(g)))
        return flat
    # last resort — nothing usable
    return []


def infer_from_geojson(
    geojson: Dict[str, Any],
    block_counts: Optional[Union[Dict[str, int], int]] = None,
    zone_attr: str = "zone",
    model_repo: Optional[str] = None,
    model_file: Optional[str] = None,
    hf_token: Optional[str] = None,
    model: Optional[Any] = None,
    verbose: bool = True,
    # new knobs
    buffer_margin: float = 5.0,
    min_area: float = 1.0,  # m^2
    dedupe_iou: float = 0.6,
    k_nn: int = 6,
    mask_size: int = 128,
) -> Dict[str, Any]:
    """Run model inference for blocks described by GeoJSON using a trained model.

    Parameters
    ----------
    geojson:
        FeatureCollection with Polygon features representing blocks.
    block_counts:
        Dict block_id -> n_buildings, или просто int для всех блоков.
    zone_attr:
        Имя свойства с зоной.
    model_repo / model_file / hf_token / model:
        Загрузка или передача готовой модели. Модель должна иметь метод ``infer``.

    Returns
    -------
    FeatureCollection с контурами зданий в координатах исходных блоков.
    """

    if not geojson.get("features"):
        raise ValueError("GeoJSON must contain at least one feature")

    if model is None:
        if not model_repo:
            raise ValueError("model_repo must be provided")
        model_path: str
        opt: Optional[Dict[str, Any]] = None
        if os.path.isdir(model_repo):
            if model_file is None:
                candidates = [f for f in os.listdir(model_repo) if f.endswith((".pt", ".pth"))]
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
        geom = shape(feat["geometry"])  # исходный блок
        geom = _make_valid(geom)
        # shrink block by buffer_margin to keep a buffer from the edges
        if buffer_margin > 0:
            buffered = geom.buffer(-buffer_margin)
            if not buffered.is_empty and buffered.area > 0:
                geom = _make_valid(buffered)
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

        # --- inference in canonical space ---
        if verbose:
            print(" - generating buildings (canonical space)")
        if not hasattr(model, "infer"):
            raise RuntimeError("Model does not provide an 'infer' method")

        raw_buildings = model.infer(canon_geom, n=n, zone_label=zone_label)  # type: ignore[operator]
        canon_buildings = _normalize_model_output(raw_buildings)

        if verbose:
            print(f" - generated {len(canon_buildings)} raw buildings; inverse transform...")

        # try mid-axis warp first (if available), otherwise fallback
        transformed_buildings: Optional[List[Polygon]] = _try_midaxis_inverse(
            canon_geom, canon_buildings, verbose=verbose
        )
        if transformed_buildings is None:
            transformed_buildings = canon_buildings

        # --- map back to world, clip & clean ---
        world_buildings: List[Polygon] = []
        for b in transformed_buildings:
            wb = _from_canonical(b, params)
            wb = _make_valid(wb)
            if _polygon_area(wb) < min_area:
                continue
            clipped_parts = _safe_clip_to_block(wb, geom)
            for cp in clipped_parts:
                if _polygon_area(cp) >= min_area:
                    world_buildings.append(cp)

        # dedupe near-duplicates
        if dedupe_iou > 0 and len(world_buildings) > 1:
            world_buildings = _nms(world_buildings, dedupe_iou)

        # export features + (optional) derive graph stats for debugging
        if world_buildings:
            pos = np.array([[p.centroid.x, p.centroid.y] for p in world_buildings], dtype=float)
            size = np.array(
                [[p.bounds[2] - p.bounds[0], p.bounds[3] - p.bounds[1]] for p in world_buildings],
                dtype=float,
            )
            aspect_ratio = (
                params.get("scale_y", 1.0) / params.get("scale_x", 1.0)
                if params.get("scale_x")
                else 1.0
            )
            try:
                _build_graph_original(geom, pos, size, aspect_ratio, k_nn=k_nn, mask_size=mask_size)
            except Exception:
                if verbose:
                    print(" - graph build failed (non-fatal)")

            for b in world_buildings:
                b_props = {zone_attr: zone_label, "block_id": block_id}
                features.append(
                    {"type": "Feature", "properties": b_props, "geometry": mapping(b)}
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
