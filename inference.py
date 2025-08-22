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


# ------------------------
# Checkpoint introspection & model loading (mirrors test.py)
# ------------------------

def _infer_model_variant_from_ckpt(state_dict) -> str:
    """Infer which model class the checkpoint corresponds to."""
    ks = set(state_dict.keys())
    if any(k.startswith("cnn_encoder.") for k in ks) or "linear1.weight" in ks:
        return "attn_ind_cnn"
    if "enc_block_scale.weight" in ks or "enc_block_scale.bias" in ks:
        return "attn_ind"
    if any(k.startswith("e_conv1") for k in ks) and any(k.startswith("d_conv1") for k in ks):
        return "attn"
    if "aggregate.weight" in ks and "d_exist_1.weight" in ks:
        return "block"
    return "naive"


def _ckpt_dec_in_features(state_dict, latent_ch: int, heads: int) -> int | None:
    """Try to read the expected decoder input width from checkpoint linear layers."""
    import torch
    Tensor = getattr(torch, "Tensor", None)
    candidates = [
        "d_exist_0.weight",
        "d_posx_0.weight",
        "d_posy_0.weight",
        "d_sizex_0.weight",
        "d_sizey_0.weight",
        "d_shape_0.weight",
        "d_iou_0.weight",
    ]
    for k in candidates:
        W = state_dict.get(k, None)
        if Tensor is not None and isinstance(W, Tensor) and W.dim() == 2:
            return int(W.shape[1])
    return None


def _infer_concat_heads_from_ckpt(state_dict, latent_ch: int, heads: int, T: int) -> bool:
    """Return True if checkpoint uses concatenation of attention heads."""
    import torch
    Tensor = getattr(torch, "Tensor", None)
    target = latent_ch * max(1, heads)
    for k in (
        "e_conv1.bias",
        "e_conv2.bias",
        "e_conv3.bias",
        "d_conv1.bias",
        "d_conv2.bias",
        "d_conv3.bias",
    ):
        b = state_dict.get(k, None)
        if Tensor is not None and isinstance(b, Tensor) and b.numel() == target and heads > 1:
            return True
    W = state_dict.get("aggregate.weight", None)
    if Tensor is not None and isinstance(W, Tensor):
        in_features = W.shape[1]
        coef = in_features // max(1, latent_ch)
        if coef >= (2 + heads):
            return True
    dec_in = _ckpt_dec_in_features(state_dict, latent_ch, heads)
    if dec_in is not None and dec_in == latent_ch * heads:
        return True
    return False


def _build_model(opt, variant: str, N: int, concat_heads: bool):
    import importlib

    model_mod = importlib.import_module("model")
    BlockGenerator = getattr(model_mod, "BlockGenerator")
    AttentionBlockGenerator = getattr(model_mod, "AttentionBlockGenerator", BlockGenerator)
    AttentionBlockGenerator_independent = getattr(
        model_mod, "AttentionBlockGenerator_independent", BlockGenerator
    )
    AttentionBlockGenerator_independent_cnn = getattr(
        model_mod, "AttentionBlockGenerator_independent_cnn", BlockGenerator
    )
    NaiveBlockGenerator = getattr(model_mod, "NaiveBlockGenerator", BlockGenerator)

    opt = dict(opt)
    opt["concat_heads"] = bool(concat_heads)
    if variant == "attn_ind_cnn":
        return AttentionBlockGenerator_independent_cnn(opt, N=N)
    if variant == "attn_ind":
        return AttentionBlockGenerator_independent(opt, N=N)
    if variant == "attn":
        return AttentionBlockGenerator(opt, N=N)
    if variant == "block":
        return BlockGenerator(opt, N=N)
    if variant == "naive":
        return NaiveBlockGenerator(opt, N=N)
    # fallback by opt
    if opt.get("is_blockplanner", False):
        return NaiveBlockGenerator(opt, N=N)
    if opt.get("is_conditional_block", False):
        if opt.get("convlayer") in opt.get("attten_net", []):
            return AttentionBlockGenerator(opt, N=N)
        return BlockGenerator(opt, N=N)
    if opt.get("convlayer") in opt.get("attten_net", []):
        if opt.get("encode_cnn", False):
            return AttentionBlockGenerator_independent_cnn(opt, N=N)
        return AttentionBlockGenerator_independent(opt, N=N)
    return BlockGenerator(opt, N=N)


def _filter_state_for_model(state_dict, model):
    import torch
    Tensor = getattr(torch, "Tensor", None)
    model_sd = getattr(model, "state_dict", lambda: {})()
    keep = {}
    for k, v in state_dict.items():
        if (
            k in model_sd
            and Tensor is not None
            and isinstance(v, Tensor)
            and v.shape == model_sd[k].shape
        ):
            keep[k] = v
    return keep


def _try_load_variants(opt, N, device, state_dict, inferred_variant: str, latent_ch: int, heads: int):
    concat_guess = _infer_concat_heads_from_ckpt(state_dict, latent_ch, heads, int(opt.get("T", 3)))
    variants = [inferred_variant]
    if inferred_variant == "attn_ind_cnn":
        variants += ["attn_ind", "attn"]
    elif inferred_variant == "attn_ind":
        variants += ["attn", "attn_ind_cnn"]
    elif inferred_variant == "attn":
        variants += ["attn_ind", "attn_ind_cnn"]

    best = None
    best_loaded = -1
    for var in variants:
        for concat in [concat_guess, not concat_guess]:
            model = _build_model(opt, var, N, concat)
            filtered = _filter_state_for_model(state_dict, model)
            model_sd = getattr(model, "state_dict", lambda: {})()
            missing = len(model_sd) - len(filtered)
            if hasattr(model, "load_state_dict"):
                try:
                    model.load_state_dict(filtered, strict=False)
                except TypeError:
                    model.load_state_dict(filtered)
            loaded = len(filtered)
            if loaded > best_loaded:
                best = model
                best_loaded = loaded

    if best is None:
        raise RuntimeError("Cannot load checkpoint into any tested architecture")
    if hasattr(best, "to"):
        best.to(device)
    return best


def _generate_with_decode(model, block: Polygon, n: int, zone_label: str | None = None):
    """Generate buildings using the model's decode path (mirrors test.py)."""
    from torch_geometric.data import Data  # local import to avoid heavy dependency if unused
    from model import block_long_side
    import torch

    device = next(model.parameters()).device if hasattr(model, "parameters") else torch.device("cpu")
    node_cnt = max(0, int(n))
    x = torch.zeros((node_cnt, 2), dtype=torch.float32, device=device)
    node_pos = torch.zeros((node_cnt, 2), dtype=torch.float32, device=device)
    node_size = torch.zeros((node_cnt, 2), dtype=torch.float32, device=device)
    b_shape = torch.zeros((node_cnt, 6), dtype=torch.float32, device=device)
    b_iou = torch.zeros((node_cnt, 1), dtype=torch.float32, device=device)
    edges = []
    for i in range(node_cnt - 1):
        edges.append([i, i + 1])
        edges.append([i + 1, i])
    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long, device=device).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long, device=device)

    data = Data(
        x=x,
        edge_index=edge_index,
        node_pos=node_pos,
        org_node_pos=node_pos.clone(),
        node_size=node_size,
        org_node_size=node_size.clone(),
        b_shape=b_shape,
        b_iou=b_iou,
    )
    data.batch = torch.zeros(node_cnt, dtype=torch.long, device=device)

    z = torch.randn(1, int(getattr(model, "latent_dim", 64)), device=device)
    try:
        exist, posx, posy, sizex, sizey, _, _ = model.decode(z, edge_index, node_cnt)
    except TypeError:
        # expect decode(z, block_condition, edge_index, node_cnt)
        bc_dim = int(getattr(model, "blockshape_latent_dim", 20) + 20)
        block_condition = torch.zeros((1, bc_dim), dtype=torch.float32, device=device)
        exist, posx, posy, sizex, sizey, _, _ = model.decode(
            z, block_condition, edge_index, node_cnt
        )

    pos = torch.cat((posx, posy), 1)
    size = torch.cat((sizex, sizey), 1)
    long_side = float(block_long_side(block))
    exist_prob = torch.sigmoid(exist).view(-1).cpu().numpy()
    pos_np = pos.cpu().numpy() * long_side
    size_np = size.cpu().numpy() * long_side
    polygons: List[Polygon] = []
    for i in range(node_cnt):
        if exist_prob[i] <= 0.5:
            continue
        cx, cy = pos_np[i]
        w, h = size_np[i]
        poly = Polygon(
            [
                (cx - w / 2.0, cy - h / 2.0),
                (cx + w / 2.0, cy - h / 2.0),
                (cx + w / 2.0, cy + h / 2.0),
                (cx - w / 2.0, cy + h / 2.0),
            ]
        )
        polygons.append(poly)
    return polygons


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

        import torch
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
        device = opt.get("device", "cpu")
        opt["device"] = device

        latent_ch = int(opt.get("n_ft_dim", 64))
        heads = int(opt.get("head_num", 1))
        inferred_variant = _infer_model_variant_from_ckpt(state_dict)
        model = _try_load_variants(opt, N, device, state_dict, inferred_variant, latent_ch, heads)
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
        if hasattr(model, "infer"):
            buildings = model.infer(canon_geom, n=n, zone_label=zone_label)  # type: ignore[operator]
        else:
            buildings = _generate_with_decode(model, canon_geom, n=n, zone_label=zone_label)
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
