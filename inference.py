from typing import List, Dict, Any, Optional, Union, Iterable, Tuple
from shapely.geometry import Point  # для _poisson_in_poly
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
    mrr = poly.minimum_rotated_rectangle
    coords = list(mrr.exterior.coords)
    edges = [(coords[i], coords[(i + 1) % 4]) for i in range(4)]
    lengths = [math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in edges]
    idx_long = max(range(4), key=lambda i: lengths[i])
    a, b = edges[idx_long]
    angle = math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))

    long_side = max(lengths) if max(lengths) > 0 else 1.0

    centroid = poly.centroid
    rotated = affinity.rotate(poly, -angle, origin=centroid)

    s = 1.0 / max(long_side, 1e-9)        # ИЗОТРОПНЫЙ масштаб
    scaled = affinity.scale(rotated, xfact=s, yfact=s, origin=centroid)

    shifted_centroid = scaled.centroid
    translated = affinity.translate(scaled, xoff=-shifted_centroid.x, yoff=-shifted_centroid.y)
    params = {
        "angle": angle,
        "scale": 1.0 / s,                 # один общий scale для обратного хода
        "origin": (centroid.x, centroid.y),
        "shift": (shifted_centroid.x, shifted_centroid.y),
    }
    return translated, params


def _from_canonical(poly: Polygon, params: dict[str, Any]) -> Polygon:
    xoff, yoff = params["shift"]
    origin = params["origin"]
    unshifted = affinity.translate(poly, xoff=xoff, yoff=yoff)
    if "scale" in params:  # новый изотропный
        unscaled = affinity.scale(unshifted, xfact=params["scale"], yfact=params["scale"], origin=origin)
    else:                  # совместимость со старым анизотропным
        unscaled = affinity.scale(unshifted,
                                  xfact=params.get("scale_x", 1.0),
                                  yfact=params.get("scale_y", 1.0),
                                  origin=origin)
    return affinity.rotate(unscaled, params["angle"], origin=origin)

def _apply_transform(poly: Polygon, params: dict[str, Any]) -> Polygon:
    rotated = affinity.rotate(poly, -params["angle"], origin=params["origin"])
    if "scale" in params:
        s = 1.0 / max(params["scale"], 1e-9)
        scaled = affinity.scale(rotated, xfact=s, yfact=s, origin=params["origin"])
    else:
        scaled = affinity.scale(rotated,
                                xfact=1 / max(params.get("scale_x", 1.0), 1e-9),
                                yfact=1 / max(params.get("scale_y", 1.0), 1e-9),
                                origin=params["origin"])
    return affinity.translate(scaled, xoff=-params["shift"][0], yoff=-params["shift"][1])


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
            # drop Z, закрыть контур, ориентация как ждёт skgeom
            ext = []
            for xyz in list(poly.exterior.coords)[:-1]:
                x, y = float(xyz[0]), float(xyz[1])
                ext.append((x, y))
            if ext[0] != ext[-1]:
                ext.append(ext[0])
            return skgeom.Polygon(list(reversed(ext)))  # важна ориентация
        
        if not canon_block.is_valid:
            try:
                from shapely.validation import make_valid
                canon_block = make_valid(canon_block)
            except Exception:
                canon_block = canon_block.buffer(0)

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

def _poisson_in_poly(poly: Polygon, r: float, k: int = 20, max_pts: Optional[int] = None):
    # простой Bridson 2D в bbox с отбрасыванием вне полигона
    import random, math
    minx, miny, maxx, maxy = poly.bounds
    cell = r / math.sqrt(2)
    gx = int((maxx - minx) / cell) + 1
    gy = int((maxy - miny) / cell) + 1
    grid = [[-1]*gy for _ in range(gx)]
    samples, active = [], []

    def grid_idx(p):
        return int((p[0]-minx)/cell), int((p[1]-miny)/cell)

    # старт в центре масс
    p0 = (float(poly.centroid.x), float(poly.centroid.y))
    samples.append(p0); active.append(0)
    gi, gj = grid_idx(p0); grid[gi][gj] = 0

    while active and (max_pts is None or len(samples) < max_pts):
        i = random.choice(active)
        ox, oy = samples[i]
        ok = False
        for _ in range(k):
            ang = random.random()*2*math.pi
            rad = r*(1+random.random())
            px, py = ox+rad*math.cos(ang), oy+rad*math.sin(ang)
            if not poly.contains(Point(px, py)):  # type: ignore
                continue
            gi, gj = grid_idx((px, py))
            good = True
            for ii in range(max(0, gi-2), min(gx, gi+3)):
                for jj in range(max(0, gj-2), min(gy, gj+3)):
                    idx = grid[ii][jj]
                    if idx >= 0:
                        dx = samples[idx][0]-px; dy = samples[idx][1]-py
                        if dx*dx+dy*dy < r*r:
                            good = False; break
                if not good: break
            if good:
                samples.append((px, py)); active.append(len(samples)-1)
                grid[gi][gj] = len(samples)-1; ok = True
                break
        if not ok:
            active.remove(i)
    return np.array(samples, dtype=float)

def _enforce_spacing(polys, block, min_gap=4.0, min_centroid=8.0, iters=10):
    # 1) внутренняя область
    inner = block.buffer(-min_gap) if min_gap>0 else block
    out = []
    for p in polys:
        c = p.centroid
        if not inner.contains(c):
            v = np.array([inner.centroid.x - c.x, inner.centroid.y - c.y], float)
            n = np.linalg.norm(v) or 1.0
            p = affinity.translate(p, xoff=float(v[0]/n*min_gap*0.5), yoff=float(v[1]/n*min_gap*0.5))
        p = _make_valid(p).intersection(inner)
        if not p.is_empty and _polygon_area(p) > 0: out.extend(_flatten_polys(p))

    # 2) лёгкое «отталкивание» центроидов
    pts = np.array([[q.centroid.x, q.centroid.y] for q in out], float)
    for _ in range(iters):
        moved = False
        for i in range(len(out)):
            for j in range(i+1, len(out)):
                d = np.linalg.norm(pts[i]-pts[j])
                if d < 1e-6:  # совпадение
                    jitter = np.random.randn(2); jitter/=np.linalg.norm(jitter)
                    pts[i]+=jitter*0.5; pts[j]-=jitter*0.5; moved=True; continue
                if d < min_centroid:
                    step = (min_centroid - d)*0.5
                    dirv = (pts[i]-pts[j])/d
                    pts[i]+=dirv*step; pts[j]-=dirv*step; moved=True
        if not moved: break
    # применить смещения и обрезать
    res=[]
    for k, p in enumerate(out):
        q = affinity.translate(p, xoff=float(pts[k][0]-p.centroid.x), yoff=float(pts[k][1]-p.centroid.y))
        q = _make_valid(q).intersection(inner)
        if not q.is_empty and _polygon_area(q)>0: res.extend(_flatten_polys(q))
    return res

def infer_from_geojson(
    geojson: Dict[str, Any],
    block_counts: Optional[Union[Dict[str, int], int]] = None,
    zone_attr: str = "zone",
    model_repo: Optional[str] = None,
    model_file: Optional[str] = None,
    hf_token: Optional[str] = None,
    model: Optional[Any] = None,
    use_midaxis_inverse: bool = False,
    verbose: bool = True,
    # new knobs
    buffer_margin: float = 5.0,
    min_area: float = 0.0,  # m^2
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

        # --- optional conditioning buildings ---
        raw_input_buildings = props.get("buildings") or []
        canon_input_buildings: List[Polygon] = []
        if isinstance(raw_input_buildings, (list, tuple)):
            for b in raw_input_buildings:
                try:
                    poly = b if isinstance(b, (Polygon, MultiPolygon)) else shape(b)
                    poly = _make_valid(poly)
                    canon_poly = _apply_transform(poly, params)
                    canon_input_buildings.extend(_flatten_polys(canon_poly))
                except Exception:
                    continue

        pos = np.empty((0, 2))
        size = np.empty((0, 2))
        aspect_ratio = 1.0
        cond_graph: Optional[nx.Graph] = None
        if canon_input_buildings:
            pos = np.array([[p.centroid.x, p.centroid.y] for p in canon_input_buildings], dtype=float)
            size = np.array(
                [[p.bounds[2] - p.bounds[0], p.bounds[3] - p.bounds[1]] for p in canon_input_buildings],
                dtype=float,
            )
            order = np.argsort(pos[:, 0])
            pos = pos[order]
            size = size[order]
            canon_input_buildings = [canon_input_buildings[i] for i in order]
            aspect_ratio = (
                params.get("scale_y", 1.0) / params.get("scale_x", 1.0)
                if params.get("scale_x")
                else 1.0
            )
            cond_graph = _build_graph_original(
                canon_geom, pos, size, aspect_ratio, k_nn=k_nn, mask_size=mask_size
            )
            cond_graph.graph["zone"] = zone_label

        # --- inference in canonical space ---
        if verbose:
            print(" - generating buildings (canonical space)")
        
        if not canon_input_buildings:
            # радиус берём как ~ ширина квартала / sqrt(n)
            bb = canon_geom.bounds
            r0 = 0.8*min(bb[2]-bb[0], bb[3]-bb[1]) / (math.sqrt(n)+1e-6)
            pos = _poisson_in_poly(canon_geom, r=max(0.03, r0), max_pts=n)
            size = np.full((pos.shape[0], 2), 0.06)  # грубая стартовая ширина/высота
            cond_graph = _build_graph_original(canon_geom, pos, size, aspect_ratio=1.0,
                                            k_nn=k_nn, mask_size=mask_size)
            cond_graph.graph["zone"] = zone_label

        infer_graph = getattr(model, "infer_graph", None)
        if verbose:
            print(" - using infer_graph(cond_graph)" if callable(infer_graph) and cond_graph is not None
                else " - using infer(block)  [no graph]")
        if callable(infer_graph) and cond_graph is not None:
            raw_buildings = infer_graph(cond_graph, n=n, zone_label=zone_label)  # type: ignore[misc]
        else:
            if not hasattr(model, "infer"):
                raise RuntimeError("Model does not provide an 'infer' method")
            raw_buildings = model.infer(canon_geom, n=n, zone_label=zone_label)  # type: ignore[operator]

        canon_buildings = _normalize_model_output(raw_buildings)

        if verbose:
            print(f" - generated {len(canon_buildings)} raw buildings; inverse transform...")

        # try mid-axis warp first (if available), otherwise fallback
        transformed_buildings: Optional[List[Polygon]] = None
        if use_midaxis_inverse:
            transformed_buildings = _try_midaxis_inverse(canon_geom, canon_buildings, verbose=verbose)
        if transformed_buildings is None:
            transformed_buildings = canon_buildings

        # --- map back to world, clip & clean ---
        world_buildings: List[Polygon] = []
        for b in transformed_buildings:
            wb = _from_canonical(b, params)
            wb = _make_valid(wb)
            if _polygon_area(wb) < min_area:
                continue
            if wb.difference(geom).area < 1e-9:
                parts = [wb]
            else:
                parts = _safe_clip_to_block(wb, geom)
            for cp in parts:
                if _polygon_area(cp) >= min_area:
                    world_buildings.append(cp)

        def zone_gap(zone):
            z = (str(zone) or "").lower()
            if "residential" in z:   return 3.5  # м до границ/между домами
            if "industrial" in z:  return 6.0
            if "business" in z:   return 4.5
            return 4.0

        def zone_centroid(zone):
            z = (str(zone) or "").lower()
            if "residential" in z:   return 8.0   # м между центрами
            if "industrial" in z:  return 12.0
            if "business" in z:   return 9.0
            return 8.0

        world_buildings = _enforce_spacing(world_buildings, geom,
                                   min_gap=zone_gap(zone_label),
                                   min_centroid=zone_centroid(zone_label),
                                   iters=12)
        # NMS строже
        world_buildings = _nms(world_buildings, iou_thr=0.35)

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
