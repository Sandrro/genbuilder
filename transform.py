#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Canonical transform that writes processed graphs in Apache Arrow format
and also carries zoning labels (zone, zone_id, zone_onehot) at graph-level.

Changes in this version:
  1) Output files keep the SAME basename as input .pkl (e.g., block_004431.arrow).
  2) Ability to start processing from a given basename via --start-from block_XXXXXX.
  3) Per-block hard timeout (configurable via --timeout-min, default 10). If exceeded,
     the block is skipped and a JSON line is appended to _timeouts.jsonl under out_dir
     with details.
  4) Parallel execution with a configurable worker count (--workers, default 4).
     Concurrency is handled via ThreadPoolExecutor; each task still spawns an isolated
     child process to enforce the hard timeout reliably.
  5) Deadlock fixes:
      - Use *spawn* multiprocessing context (not fork) to avoid forking from threads
        and CGAL/GEOS fork-unsafe states.
      - Child sends only a light payload (pos, size, aspect_ratio) through the queue;
        NetworkX graph + mask are built in the parent thread.
      - Parent waits on the queue with timeout (no blocking join before reading).
      - Pre-sanitize polygons (make_valid/buffer(0)) to avoid CGAL hangs on invalid input.

Output per-graph node attrs (exact names expected by original urban_dataset.py):
  - posx, posy      : float
  - size_x, size_y  : float
  - exist, merge    : int (we set exist=1, merge=0 by default)
  - shape           : int (placeholder 0)
  - iou             : float (placeholder 0.0)
Graph attrs:
  - aspect_ratio : float (from canonical midaxis warp)
  - long_side    : float (long side of min rotated rectangle of the block, meters)
  - binary_mask  : np.uint8[H,W] rasterized block mask (default H=W=64)
  - block_scale  : float (set = long_side)
  - zone, zone_id, zone_onehot

IMPORTANT: This script saves files as <input_basename>.arrow so UrbanGraphDataset
that reads '{idx}.arrow' will need basename-driven loading if you used indexed
 naming previously.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import pickle
import json
import sys
import time
import logging
from typing import List, Tuple, Optional, Dict, Any
import multiprocessing as mp
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import signal

import numpy as np
import networkx as nx
from shapely.geometry import Polygon
import shapely  # noqa: F401
import skgeom
from PIL import Image, ImageDraw
import pyarrow as pa
import pyarrow.ipc as ipc

# optional progress bar
try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None  # falls back to simple logs

# import canonical helpers from example module
from example_canonical_transform import (
    get_polyskeleton_longest_path,
    modified_skel_to_medaxis,
    warp_bldg_by_midaxis,
)

# --------------------- defaults ---------------------

DEFAULT_TIMEOUT_MIN = 10  # can be overridden by --timeout-min

# --------------------- logging utils ---------------------

def setup_logger(level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger("canonical_transform")
    if logger.handlers:
        logger.setLevel(level.upper())
        return logger
    logger.setLevel(level.upper())
    handler = logging.StreamHandler(stream=sys.stdout)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


# --------------------- helpers ---------------------

def shapely_to_skgeom_polygon(poly: Polygon):
    exterior_polyline = list(poly.exterior.coords)[:-1]
    exterior_polyline.reverse()
    return skgeom.Polygon(exterior_polyline)


def block_long_side(block: Polygon) -> float:
    mrr = block.minimum_rotated_rectangle
    coords = list(mrr.exterior.coords)
    # 5 points (last==first); take two edge lengths
    def dist(a, b):
        return float(np.hypot(b[0]-a[0], b[1]-a[1]))

    a0, a1, a2, a3 = coords[0], coords[1], coords[2], coords[3]
    e1 = dist(a0, a1)
    e2 = dist(a1, a2)
    return max(e1, e2)


def rasterize_block_mask(block: Polygon, out_size: int = 64) -> np.ndarray:
    """Rasterize block polygon to binary mask (0/255) in its own bbox frame.
    We fit the polygon's bbox to the square canvas, preserving aspect (no rotation).
    """
    minx, miny, maxx, maxy = block.bounds
    w = maxx - minx
    h = maxy - miny
    if w <= 0 or h <= 0:
        return np.zeros((out_size, out_size), dtype=np.uint8)
    sx = (out_size - 2) / w  # 1px margin
    sy = (out_size - 2) / h
    s = min(sx, sy)
    ox = 1 - minx * s
    oy = 1 - miny * s
    pts = [(p[0] * s + ox, p[1] * s + oy) for p in np.asarray(block.exterior.coords)]
    img = Image.new("L", (out_size, out_size), 0)
    draw = ImageDraw.Draw(img)
    draw.polygon(pts, fill=255)
    return np.array(img, dtype=np.uint8)


def one_hot(index: int, K: int) -> np.ndarray:
    v = np.zeros(K, dtype=np.float32)
    v[index] = 1.0
    return v


def process_one(block: Polygon, buildings: List[Polygon], logger: logging.Logger) -> Tuple[np.ndarray, np.ndarray, float]:
    t0 = time.perf_counter()
    logger.debug(
        "Skeletonizing block: area=%.2f, bounds=%s, buildings=%d",
        float(block.area),
        tuple(round(x, 3) for x in block.bounds),
        len(buildings),
    )

    sk_blk = shapely_to_skgeom_polygon(block)
    skel = skgeom.skeleton.create_interior_straight_skeleton(sk_blk)
    t1 = time.perf_counter()
    logger.debug("  • Straight skeleton done in %.3fs", t1 - t0)

    _, longest_skel = get_polyskeleton_longest_path(skel, sk_blk)
    medaxis = modified_skel_to_medaxis(longest_skel, block)
    t2 = time.perf_counter()
    logger.debug("  • Medial axis extracted in %.3fs", t2 - t1)

    pos_sorted, size_sorted, order_idx, aspect_rto = warp_bldg_by_midaxis(buildings, block, medaxis)
    t3 = time.perf_counter()
    logger.debug(
        "  • Buildings warped/sorted in %.3fs (n=%d, aspect=%.4f)",
        t3 - t2,
        len(pos_sorted),
        float(aspect_rto),
    )

    return pos_sorted, size_sorted, float(aspect_rto)


def build_graph_original(
    block: Polygon,
    pos: np.ndarray,
    size: np.ndarray,
    aspect_ratio: float,
    k_nn: int = 4,
    mask_size: int = 64,
    logger: Optional[logging.Logger] = None,
) -> nx.Graph:
    """Build nx.Graph with ORIGINAL field names expected by the repo."""
    n = pos.shape[0]
    G = nx.Graph()

    # Node attributes
    for i in range(n):
        G.add_node(i)
        G.nodes[i]["posx"] = float(pos[i, 0])
        G.nodes[i]["posy"] = float(pos[i, 1])
        G.nodes[i]["size_x"] = float(size[i, 0])
        G.nodes[i]["size_y"] = float(size[i, 1])
        G.nodes[i]["exist"] = int(1)
        G.nodes[i]["merge"] = int(0)
        G.nodes[i]["shape"] = int(0)  # placeholder class
        G.nodes[i]["iou"] = float(0.0)  # placeholder target

    # kNN edges over normalized pos
    if n >= 2:
        try:
            from sklearn.neighbors import NearestNeighbors  # lazy import for clearer errors
        except Exception as e:
            raise RuntimeError(
                "scikit-learn is required for k-NN graph. Please 'pip install scikit-learn'."
            ) from e

        nbrs = NearestNeighbors(n_neighbors=min(k_nn + 1, n)).fit(pos)
        _, indices = nbrs.kneighbors(pos)
        edge_count = 0
        for i in range(n):
            for j in indices[i][1:]:
                if i < j:
                    G.add_edge(i, int(j))
                    edge_count += 1
        if logger:
            logger.debug("  • kNN edges added: %d (k=%d)", edge_count, k_nn)

    # Graph attributes
    G.graph["aspect_ratio"] = float(aspect_ratio)
    ls = block_long_side(block)
    G.graph["long_side"] = float(ls)
    G.graph["binary_mask"] = rasterize_block_mask(block, out_size=mask_size)
    G.graph["block_scale"] = float(ls)

    if logger:
        logger.debug(
            "  • Graph stats: nodes=%d, edges=%d, mask=%s",
            G.number_of_nodes(),
            G.number_of_edges(),
            tuple(G.graph["binary_mask"].shape),
        )

    return G


# --------------------- worker for timeout isolation ---------------------

def _process_block_worker(
    block: Polygon,
    buildings: List[Polygon],
    log_level: str,
    q: "mp.queues.Queue",
) -> None:
    """Runs heavy processing inside a child process and returns lightweight result via queue.
    Puts ("ok", (pos, size, aspect_ratio)) on success; ("err", repr(e)) on failure.
    """
    try:
        wlog = setup_logger(log_level)
        wlog.propagate = False

        # Sanitize polygon to reduce CGAL issues
        try:
            if not block.is_valid:
                try:
                    from shapely import make_valid  # Shapely >= 2.0
                    block = make_valid(block)
                except Exception:
                    block = block.buffer(0)
        except Exception:
            pass

        pos, size, ar = process_one(block, buildings, wlog)
        q.put(("ok", (pos, size, ar)))
    except Exception as e:
        q.put(("err", repr(e)))


def run_block_with_timeout(
    block: Polygon,
    buildings: List[Polygon],
    timeout_secs: int,
    log_level: str,
) -> Tuple[str, Optional[Tuple[np.ndarray, np.ndarray, float]], Optional[str]]:
    """Run the worker with a hard timeout using a *spawn* context (fork-unsafe libs!).
    Wait on the queue (not join) to avoid deadlocks from full pipes.
    Returns (status, payload, error_msg), where status in {"ok","timeout","err"}.
    payload = (pos, size, aspect_ratio) on success.
    """
    ctx = mp.get_context("spawn")
    q: "mp.queues.Queue" = ctx.Queue()
    p = ctx.Process(target=_process_block_worker, args=(block, buildings, log_level, q))
    p.daemon = False
    p.start()

    try:
        status, payload = q.get(timeout=timeout_secs)
    except Exception:
        # Timeout — terminate forcefully
        if p.is_alive():
            p.terminate()
            p.join(timeout=5)
            if p.is_alive():
                try:
                    os.kill(p.pid, signal.SIGKILL)
                except Exception:
                    pass
                p.join()
        return "timeout", None, None

    # Child produced a result — ensure clean exit
    p.join(timeout=5)
    if p.is_alive():
        p.terminate()
        p.join()

    if status == "ok":
        return "ok", payload, None
    else:
        return "err", None, str(payload)


# --------------------- per-item handler (for thread pool) ---------------------

def handle_item(
    item: Tuple[str, Polygon, List[Polygon], str],
    knn: int,
    mask_size: int,
    log_level: str,
    timeout_secs: int,
) -> Dict[str, Any]:
    basename, block, buildings, zone_label = item
    t0 = time.perf_counter()

    status, payload, err = run_block_with_timeout(
        block=block,
        buildings=buildings,
        timeout_secs=timeout_secs,
        log_level=log_level,
    )

    G: Optional[nx.Graph] = None
    if status == "ok" and payload is not None:
        pos, size, ar = payload
        plog = setup_logger(log_level)
        G = build_graph_original(block, pos, size, ar, k_nn=knn, mask_size=mask_size, logger=plog)

    elapsed = time.perf_counter() - t0
    return {
        "basename": basename,
        "zone_label": zone_label,
        "status": status,  # ok | timeout | err
        "graph": G,
        "error": err,
        "elapsed": elapsed,
        "num_buildings": int(len(buildings)),
        "area": float(block.area),
        "bounds": tuple(float(x) for x in block.bounds),
    }


# --------------------- main ---------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", required=True, type=Path, help="folder with raw_geo .pkl")
    ap.add_argument("--out_dir", required=True, type=Path, help="folder for processed .arrow")
    ap.add_argument("--zones_map", type=Path, default=None, help="optional existing zones_map.json")
    ap.add_argument("--knn", type=int, default=4)
    ap.add_argument("--mask_size", type=int, default=64)
    ap.add_argument(
        "--start-from",
        dest="start_from",
        type=str,
        default=None,
        help="start processing from the given input basename (e.g., block_004431)",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=4,
        help="number of concurrent workers (default: 4)",
    )
    ap.add_argument(
        "--timeout-min",
        type=int,
        default=DEFAULT_TIMEOUT_MIN,
        help="per-block hard timeout in minutes (default: 10)",
    )
    ap.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="verbosity level",
    )
    ap.add_argument("--strict", action="store_true", help="fail fast on any item error")
    ap.add_argument("--dry-run", action="store_true", help="process but do not write outputs")
    ap.add_argument(
        "--skip-single",
        action="store_true",
        help="skip graphs with a single node and no edges",
    )
    args = ap.parse_args()

    logger = setup_logger(args.log_level)
    t_start = time.perf_counter()

    logger.info("Starting canonical transform")
    logger.info(
        "Args: raw_dir=%s | out_dir=%s | knn=%d | mask=%d | start_from=%s | workers=%d | timeout_min=%d | strict=%s | dry_run=%s | skip_single=%s",
        str(args.raw_dir),
        str(args.out_dir),
        args.knn,
        args.mask_size,
        str(args.start_from),
        args.workers,
        args.timeout_min,
        args.strict,
        args.dry_run,
        args.skip_single,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Persist transformation parameters for downstream consumers
    params_path = args.out_dir / "_transform_params.json"
    try:
        with params_path.open("w", encoding="utf-8") as fh:
            json.dump(vars(args), fh, ensure_ascii=False, indent=2)
    except Exception as e:  # pragma: no cover - logging best effort
        logger.error("Failed to write transform params: %s", e)

    # Timeout log path
    timeouts_path = args.out_dir / "_timeouts.jsonl"

    raw_files = sorted([p for p in args.raw_dir.glob("*.pkl")])
    if not raw_files:
        raise SystemExit(f"No .pkl found in {args.raw_dir}")

    # Filter by start_from if provided (lexicographic; zero-padded numbers keep order)
    if args.start_from:
        before_count = len(raw_files)
        raw_files = [p for p in raw_files if p.stem >= args.start_from]
        logger.info("start-from filter: kept %d of %d files (from '%s')", len(raw_files), before_count, args.start_from)
        if not raw_files:
            raise SystemExit(
                f"No files at or after '{args.start_from}'. Check the name (expected like 'block_004431')."
            )

    logger.info("Found %d raw files to process", len(raw_files))

    timeout_secs = max(1, int(args.timeout_min)) * 60
    logger.info("Per-block timeout: %ds (%d min)", timeout_secs, args.timeout_min)

    # Preload raw and collect zones
    # Cache holds tuples: (basename, block, buildings, zone_label)
    cache: List[Tuple[str, Polygon, List[Polygon], str]] = []
    zones: List[str] = []
    skipped_unknown = 0

    for p in raw_files:
        with open(p, "rb") as f:
            data = pickle.load(f)

        block = data["block"]
        buildings = data["buildings"]
        zone_label = str(data.get("zone", "UNKNOWN"))
        basename = p.stem

        # skip blocks with unknown zone (case-insensitive)
        if zone_label.strip().lower() == "unknown":
            logger.debug("Skipping %s: zone='unknown'", p.name)
            skipped_unknown += 1
            continue

        cache.append((basename, block, buildings, zone_label))
        zones.append(zone_label)

    if skipped_unknown:
        logger.info("Skipped %d blocks with zone='unknown'", skipped_unknown)

    # zone map
    if args.zones_map and args.zones_map.exists():
        zmap_raw = json.loads(Path(args.zones_map).read_text(encoding="utf-8"))
        zmap = zmap_raw.get("map", zmap_raw)
        logger.info("Loaded existing zone map (%d classes) from %s", len(zmap), args.zones_map)
    else:
        uniq = sorted(set(zones))
        zmap = {z: i for i, z in enumerate(uniq)}
        out_map_path = args.out_dir / "_zones_map.json"
        out_map_path.write_text(
            json.dumps({"map": zmap}, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        logger.info("Built new zone map with %d classes → %s", len(zmap), out_map_path)

    K = len(zmap)

    ok = 0
    failed = 0
    skipped_single = 0
    total_nodes = 0
    total_edges = 0

    # Threaded orchestration of per-block subprocess workers (for timeout safety)
    if tqdm is not None and logger.level <= logging.INFO:
        pbar = tqdm(total=len(cache), desc="Processing blocks", unit="blk")  # type: ignore
    else:
        pbar = None

    def finalize_result(res: Dict[str, Any]) -> None:
        nonlocal ok, failed, skipped_single, total_nodes, total_edges
        basename = res["basename"]
        zone_label = res["zone_label"]
        status = res["status"]
        G: Optional[nx.Graph] = res["graph"]
        err = res["error"]
        elapsed = res["elapsed"]

        logger.info("[%s] zone='%s' | status=%s | elapsed=%.2fs", basename, zone_label, status, elapsed)

        if status == "timeout":
            failed += 1
            try:
                rec = {
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "filename": basename,
                    "zone": zone_label,
                    "num_buildings": int(res["num_buildings"]),
                    "area": float(res["area"]),
                    "bounds": res["bounds"],
                    "knn": int(args.knn),
                    "mask_size": int(args.mask_size),
                    "timeout_sec": int(timeout_secs),
                    "status": "timeout",
                }
                with open(timeouts_path, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "")
            except Exception as log_e:
                logger.error("  ! Failed to write timeout record for %s: %s", basename, log_e)
            finally:
                if pbar is not None:
                    pbar.update(1)
            return

        if status == "err":
            failed += 1
            logger.error("  × Failed on %s: %s", basename, err)
            if pbar is not None:
                pbar.update(1)
            if args.strict:
                raise RuntimeError(f"Failed on {basename}: {err}")
            return

        try:
            zid = int(zmap[str(zone_label)])
            assert G is not None
            G.graph["zone"] = zone_label
            G.graph["zone_id"] = zid
            G.graph["zone_onehot"] = one_hot(zid, K)

            if args.skip_single and G.number_of_nodes() == 1 and G.number_of_edges() == 0:
                skipped_single += 1
                logger.info("  • Skipping singleton graph (nodes=1, edges=0)")
            else:
                if args.dry_run:
                    logger.info("  • DRY RUN: skipping write for graph %s", basename)
                else:
                    out_name = f"{basename}.arrow"
                    out_path = args.out_dir / out_name
                    data = pickle.dumps(G)
                    table = pa.table({"graph": [data]})
                    with pa.OSFile(str(out_path), "wb") as sink:
                        with ipc.new_file(sink, table.schema) as writer:
                            writer.write_table(table)
                    logger.info(
                        "  • Saved %s (nodes=%d, edges=%d)",
                        out_path.name,
                        G.number_of_nodes(),
                        G.number_of_edges(),
                    )
                ok += 1
                total_nodes += G.number_of_nodes()
                total_edges += G.number_of_edges()
        except Exception as e:
            failed += 1
            logger.error("  × Failed on %s during save/annotate: %s", basename, e)
            if args.strict:
                raise
        finally:
            if pbar is not None:
                pbar.update(1)

    # Submit tasks
    futures = []
    with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as ex:
        for item in cache:
            futures.append(ex.submit(handle_item, item, args.knn, args.mask_size, args.log_level, timeout_secs))
        for fut in as_completed(futures):
            try:
                res = fut.result()
            except Exception as e:
                failed += 1
                logger.error("  × Unexpected orchestrator error: %s", e)
                if pbar is not None:
                    pbar.update(1)
                if args.strict:
                    raise
                continue
            finalize_result(res)

    if pbar is not None:
        pbar.close()

    summary = {
        "processed": len(cache),
        "saved": ok,
        "skipped_single": skipped_single,
        "failed": failed,
        "total_nodes": total_nodes,
        "total_edges": total_edges,
    }
    logger.info(
        "Done. Processed=%d | Saved=%d | Skipped=%d | Failed=%d | out_dir=%s",
        summary["processed"],
        summary["saved"],
        summary["skipped_single"],
        summary["failed"],
        str(args.out_dir),
    )
    summary_path = args.out_dir / "_summary.json"
    try:
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("Summary stats saved to %s", summary_path)
    except Exception as e:
        logger.error("Failed to write summary stats: %s", e)
    logger.info("Total time: %.2fs", time.perf_counter() - t_start)


if __name__ == "__main__":
    # Avoid setting a global start method; we use a spawn context per task.
    main()
