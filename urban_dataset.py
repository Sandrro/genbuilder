import os
import math
import pickle
from typing import List, Optional

import numpy as np
import networkx as nx
import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.parquet as pq

import torch
import torch.nn.functional as F
from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader

from PIL import Image
import torchvision.transforms as transforms


# =============================
# Image transforms & utilities
# =============================

def __add_gaussain_noise(img, scale: float):
    """Kept original name for backward compatibility.
    Adds gaussian noise only to 0/255 pixels of a uint8 binary image.
    """
    ow, oh = img.size
    mean = 12.0 * float(scale)
    sigma = 4.0 * max(1e-8, float(scale))
    gauss = np.random.normal(mean, sigma, (oh, ow)).astype(np.float32)

    arr = np.array(img, dtype=np.float32)
    mask_0 = (arr == 0)
    mask_255 = (arr == 255)

    # add/subtract noise only where it makes sense
    arr[mask_0] = np.clip(arr[mask_0] + gauss[mask_0], 0, 255)
    arr[mask_255] = np.clip(arr[mask_255] - gauss[mask_255], 0, 255)

    return arr.astype(np.uint8)


def get_transform(noise_range: float = 0.0, noise_type: Optional[str] = None,
                  isaug: bool = False, rescale_size: int = 64):
    """Builds a deterministic torchvision transform for masks.
    Output: FloatTensor CHW in ~[-2, 2] due to Normalize(0.5, 0.25).
    """
    ops: List[transforms.Compose] = []

    if rescale_size is not None:
        ops.append(transforms.Resize((rescale_size, rescale_size), interpolation=Image.NEAREST))

    if noise_type == 'gaussian' and noise_range > 0:
        ops.append(transforms.Lambda(lambda img: __add_gaussain_noise(img, noise_range)))

    ops += [transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.25,))]

    return transforms.Compose(ops)


GRAPH_EXTENSIONS = ('.arrow', '.parquet')


def is_graph_file(filename: str) -> bool:
    return filename.lower().endswith(GRAPH_EXTENSIONS)


# =============================
# Graph feature extraction
# =============================

def get_node_attribute(g: nx.Graph, key: str, dtype, default=None):
    attr = list(nx.get_node_attributes(g, key).items())
    if len(attr) == 0:
        # fallback to default-filled array with length equal to #nodes
        n = g.number_of_nodes()
        return np.full(n, default if default is not None else 0, dtype=dtype)
    arr = np.array([v for _, v in attr], dtype=dtype)
    return arr


def get_edge_attribute(g: nx.Graph, key: str, dtype, default=None):
    attr = list(nx.get_edge_attributes(g, key).items())
    if len(attr) == 0:
        return np.zeros((0,), dtype=dtype)
    arr = np.array([v for _, v in attr], dtype=dtype)
    return arr


def graph2vector_processed(g: nx.Graph):
    num_nodes = g.number_of_nodes()

    asp_rto = g.graph.get('aspect_ratio', 1.0)
    longside = g.graph.get('long_side', 1.0)

    posx = get_node_attribute(g, 'posx', np.float64, 0.0)
    posy = get_node_attribute(g, 'posy', np.float64, 0.0)

    size_x = get_node_attribute(g, 'size_x', np.float64, 0.0)
    size_y = get_node_attribute(g, 'size_y', np.float64, 0.0)

    # Normalize sizes using the long side if available.
    # Positions are left in their original scale to avoid unintended re-scaling.
    scale = float(longside) if float(longside) > 0 else 1.0
    if scale == 1.0:
        # Fallback to maximum coordinate if long_side is not provided or invalid
        coords = np.concatenate([posx, posy, size_x, size_y]) if num_nodes > 0 else np.array([1.0])
        max_coord = float(np.max(np.abs(coords)))
        if max_coord > 0:
            scale = max_coord
    if scale > 0:
        size_x = size_x / scale
        size_y = size_y / scale
    longside = scale

    exist = get_node_attribute(g, 'exist', np.int_, 1)
    merge = get_node_attribute(g, 'merge', np.int_, 0)

    b_shape = get_node_attribute(g, 'shape', np.int_, 0)
    b_iou = get_node_attribute(g, 'iou', np.float64, 1.0)

    node_attr = np.stack((exist, merge), axis=1)

    if num_nodes == 0:
        edge_list = np.zeros((2, 0), dtype=np.int64)
        node_pos = np.zeros((0, 2), dtype=np.float32)
        node_size = np.zeros((0, 2), dtype=np.float32)
        node_idx = np.zeros((0, 2), dtype=np.float32)
    else:
        # Edge index as [2, E]
        if g.number_of_edges() == 0:
            edge_list = np.zeros((2, 0), dtype=np.int64)
        else:
            edge_list = np.array(list(g.edges()), dtype=np.int64).T  # shape [2, E]

        node_pos = np.stack((posx, posy), axis=1)
        node_size = np.stack((size_x, size_y), axis=1)
        node_idx = np.stack((np.arange(num_nodes) / max(1, num_nodes),
                             np.arange(num_nodes) / max(1, num_nodes)), axis=1)

    return node_size, node_pos, node_attr, edge_list, node_idx, asp_rto, longside, b_shape, b_iou


# =============================
# Optional transforms for PyG Data objects
# =============================

def test_graph_transform(data: Data) -> Data:
    num_nodes = data.x.shape[0]

    org_node_size = torch.as_tensor(data.node_size, dtype=torch.float32)
    node_size = org_node_size.clone()

    org_node_pos = torch.as_tensor(data.node_pos, dtype=torch.float32)
    node_pos = org_node_pos.clone()

    b_shape_gt = torch.as_tensor(data.b_shape, dtype=torch.int64)
    b_shape = torch.as_tensor(F.one_hot(b_shape_gt, num_classes=6), dtype=torch.float32)
    b_iou = torch.as_tensor(data.b_iou, dtype=torch.float32).unsqueeze(1)

    node_feature = torch.as_tensor(data.x, dtype=torch.float32)
    edge_idx = torch.as_tensor(data.edge_index, dtype=torch.long)

    node_idx = torch.as_tensor(data.node_idx, dtype=torch.float32)
    node_idx = node_idx.clone()

    long_side = torch.as_tensor(data.long_side, dtype=torch.float32).repeat(num_nodes).unsqueeze(1)
    asp_rto = torch.as_tensor(data.asp_rto, dtype=torch.float32).repeat(num_nodes).unsqueeze(1)

    long_side_gt = torch.as_tensor(data.long_side, dtype=torch.float32)
    asp_rto_gt = torch.as_tensor(data.asp_rto, dtype=torch.float32)

    blockshape_latent = torch.as_tensor(data.blockshape_latent / 10.0, dtype=torch.float32).repeat(num_nodes, 1)
    block_scale = torch.as_tensor(data.block_scale / 100.0, dtype=torch.float32).repeat(num_nodes).unsqueeze(1)

    blockshape_latent_gt = torch.as_tensor(data.blockshape_latent / 10.0, dtype=torch.float32)
    block_scale_gt = torch.as_tensor(data.block_scale / 100.0, dtype=torch.float32)

    trans_data = Data(
        x=node_feature,
        edge_index=edge_idx,
        node_pos=node_pos,
        org_node_pos=org_node_pos,
        node_size=node_size,
        org_node_size=org_node_size,
        node_idx=node_idx,
        asp_rto=asp_rto,
        long_side=long_side,
        asp_rto_gt=asp_rto_gt,
        long_side_gt=long_side_gt,
        b_shape=b_shape,
        b_iou=b_iou,
        b_shape_gt=b_shape_gt,
        blockshape_latent=blockshape_latent,
        blockshape_latent_gt=blockshape_latent_gt,
        block_scale=block_scale,
        block_scale_gt=block_scale_gt,
        block_condition=data.block_condition,
        org_binary_mask=data.org_binary_mask,
    )

    # Optional passthroughs (zone info)
    if hasattr(data, 'zone_id'):
        trans_data.zone_id = data.zone_id
    if hasattr(data, 'zone_onehot'):
        trans_data.zone_onehot = data.zone_onehot

    return trans_data


def graph_transform(data: Data) -> Data:
    num_nodes = data.x.shape[0]

    org_node_size = np.array(data.node_size)
    org_node_pos = np.array(data.node_pos)
    b_shape_org = np.array(data.b_shape)
    b_iou = np.array(data.b_iou)
    node_feature = np.array(data.x)

    if torch.rand(1).item() < 0.5:
        org_node_size = np.flip(org_node_size, 0).copy()
        org_node_pos = np.flip(org_node_pos, 0).copy()
        b_shape_org = np.flip(b_shape_org, 0).copy()
        b_iou = np.flip(b_iou, 0).copy()
        node_feature = np.flip(node_feature, 0).copy()

    org_node_size_t = torch.as_tensor(org_node_size, dtype=torch.float32)
    node_size = org_node_size_t.clone()

    org_node_pos_t = torch.as_tensor(org_node_pos, dtype=torch.float32)
    node_pos = org_node_pos_t.clone()

    b_shape_gt = torch.as_tensor(b_shape_org, dtype=torch.int64)
    b_shape = torch.as_tensor(F.one_hot(b_shape_gt, num_classes=6), dtype=torch.float32)
    b_iou_t = torch.as_tensor(b_iou, dtype=torch.float32).unsqueeze(1)

    node_feature_t = torch.as_tensor(node_feature, dtype=torch.float32)
    edge_idx = torch.as_tensor(data.edge_index, dtype=torch.long)

    node_idx = torch.as_tensor(data.node_idx, dtype=torch.float32)

    long_side = torch.as_tensor(data.long_side, dtype=torch.float32).repeat(num_nodes).unsqueeze(1)
    asp_rto = torch.as_tensor(data.asp_rto, dtype=torch.float32).repeat(num_nodes).unsqueeze(1)

    long_side_gt = torch.as_tensor(data.long_side, dtype=torch.float32)
    asp_rto_gt = torch.as_tensor(data.asp_rto, dtype=torch.float32)

    blockshape_latent = torch.as_tensor(data.blockshape_latent / 10.0, dtype=torch.float32).repeat(num_nodes, 1)
    block_scale = torch.as_tensor(data.block_scale / 100.0, dtype=torch.float32).repeat(num_nodes).unsqueeze(1)

    blockshape_latent_gt = torch.as_tensor(data.blockshape_latent / 10.0, dtype=torch.float32)
    block_scale_gt = torch.as_tensor(data.block_scale / 100.0, dtype=torch.float32)

    trans_data = Data(
        x=node_feature_t,
        edge_index=edge_idx,
        node_pos=node_pos,
        org_node_pos=org_node_pos_t,
        node_size=node_size,
        org_node_size=org_node_size_t,
        node_idx=node_idx,
        asp_rto=asp_rto,
        long_side=long_side,
        asp_rto_gt=asp_rto_gt,
        long_side_gt=long_side_gt,
        b_shape=b_shape,
        b_iou=b_iou_t,
        b_shape_gt=b_shape_gt,
        blockshape_latent=blockshape_latent,
        blockshape_latent_gt=blockshape_latent_gt,
        block_scale=block_scale,
        block_scale_gt=block_scale_gt,
        block_condition=data.block_condition,
        org_binary_mask=data.org_binary_mask,
    )

    # Optional passthroughs (zone info)
    if hasattr(data, 'zone_id'):
        trans_data.zone_id = data.zone_id
    if hasattr(data, 'zone_onehot'):
        trans_data.zone_onehot = data.zone_onehot

    return trans_data


# =============================
# Dataset
# =============================

class UrbanGraphDataset(Dataset):
    """PyG Dataset for pre-processed urban block graphs.

    The dataset can consist of individual `.arrow` files (one graph per file)
    or `.parquet` shards produced by HuggingFace Datasets where each row
    contains a pickled networkx graph under the `graph` column.

    Root directory is the directory that already contains the processed files.
    Both `raw_dir` and `processed_dir` resolve to the same path, so PyG treats
    the dataset as already processed (`process()` is a no-op).
    """

    def __init__(
        self,
        root: str,
        transform=None,
        pre_transform=None,
        cnn_transform=None,
        is_multigpu: bool = False,
        skip_single: bool = False,
    ):
        # Build absolute root first; PyG may access dirs in super().__init__
        self._root_abs = os.path.abspath(root)

        # CNN transforms for mask conditioning
        self.cnn_transforms = cnn_transform or transforms.Compose([
            transforms.Resize((64, 64), interpolation=Image.NEAREST),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.25,)),
        ])
        self.base_transform = transforms.Compose([transforms.ToTensor()])
        self.skip_single = bool(skip_single)

        # Gather all arrow/parquet files BEFORE calling super().__init__
        self.arrow_paths: List[str] = []
        self.parquet_files: List[str] = []
        for dp, _, files in os.walk(self._root_abs):
            for f in files:
                if f.lower().endswith('.parquet'):
                    self.parquet_files.append(os.path.join(dp, f))
                elif f.lower().endswith('.arrow'):
                    self.arrow_paths.append(os.path.join(dp, f))

        # Determine storage mode
        self._use_parquet = len(self.parquet_files) > 0

        if self._use_parquet:
            # Load tables and build row index mapping
            self.parquet_files.sort()
            self.parquet_tables: List[pa.Table] = []
            self.parquet_index: List[tuple[int, int]] = []
            for pf in self.parquet_files:
                table = pq.read_table(pf)
                fid = len(self.parquet_tables)
                self.parquet_tables.append(table)
                for ridx in range(table.num_rows):
                    if self.skip_single:
                        buf = table.column("graph")[ridx].as_py()
                        g = pickle.loads(buf)
                        if g.number_of_nodes() == 1 and g.number_of_edges() == 0:
                            continue
                    self.parquet_index.append((fid, ridx))
        else:
            # Stable/numeric-first sorting for arrow files
            def sort_key(p):
                name = os.path.splitext(os.path.basename(p))[0]
                return (0, int(name)) if name.isdigit() else (1, name)
            self.arrow_paths.sort(key=sort_key)
            if self.skip_single:
                filtered: List[str] = []
                for p in self.arrow_paths:
                    with pa.memory_map(p, "rb") as source:
                        table = ipc.open_file(source).read_all()
                    g = pickle.loads(table.column("graph")[0].as_py())
                    if g.number_of_nodes() == 1 and g.number_of_edges() == 0:
                        continue
                    filtered.append(p)
                self.arrow_paths = filtered

        super().__init__(self._root_abs, transform, pre_transform)

    # ---- Directories ----
    @property
    def raw_dir(self) -> str:
        return self._root_abs

    @property
    def processed_dir(self) -> str:
        return self._root_abs

    # ---- Filenames so PyG thinks dataset is already processed ----
    @property
    def raw_file_names(self):
        files = self.parquet_files if self._use_parquet else self.arrow_paths
        return [os.path.relpath(p, self.raw_dir) for p in files]

    @property
    def processed_file_names(self):
        files = self.parquet_files if self._use_parquet else self.arrow_paths
        return [os.path.relpath(p, self.processed_dir) for p in files]

    # ---- Processing is a no-op (files are already ready to load) ----
    def process(self):
        # No processing required; files are already ready to load
        pass

    # ---- PyG API ----
    def len(self) -> int:
        if self._use_parquet:
            return len(self.parquet_index)
        return len(self.arrow_paths)

    def get(self, idx: int) -> Data:
        if self._use_parquet:
            file_idx, row_idx = self.parquet_index[idx]
            table = self.parquet_tables[file_idx]
            buf = table.column("graph")[row_idx].as_py()
            gpath = f"{self.parquet_files[file_idx]}[{row_idx}]"
        else:
            # Read .arrow file by index
            gpath = self.arrow_paths[idx]
            with pa.memory_map(gpath, "rb") as source:
                table = ipc.open_file(source).read_all()
            buf = table.column("graph")[0].as_py()
        tmp_graph: nx.Graph = pickle.loads(buf)

        # --- Mask & conditioning channel ---
        mask = tmp_graph.graph.get('binary_mask', None)
        if mask is None:
            raise KeyError(f"binary_mask is missing in graph attrs for {gpath}")

        # Normalize mask array to uint8 [0, 255]
        mask_arr = np.array(mask)
        if mask_arr.dtype != np.uint8:
            # accept bool, {0,1}, floats — bring to 0..255
            mask_arr = (mask_arr.astype(np.float32) * (255.0 if mask_arr.max() <= 1.0 else 1.0)).clip(0, 255).astype(np.uint8)
        block_mask = Image.fromarray(mask_arr)

        block_scale = float(tmp_graph.graph.get('block_scale', 1.0))
        trans_image = self.cnn_transforms(block_mask)  # [1, H, W]

        # Add a per-pixel scale channel matched to CHW
        scale_value = math.log(block_scale, 20) / 2.0
        scale_channel = torch.ones_like(trans_image[:1]) * scale_value
        block_condition = torch.cat((trans_image, scale_channel), dim=0)

        # --- Vectorized graph tensors ---
        blockshape_latent = np.zeros(40, dtype=np.float32)
        node_size, node_pos, node_feature, edge_index, node_idx, asp_rto, long_side, b_shape, b_iou = \
            graph2vector_processed(tmp_graph)

        data = Data(
            x=node_feature,  # [N, 2] (exist, merge)
            node_pos=node_pos,  # [N, 2]
            edge_index=edge_index,  # [2, E]
            node_size=node_size,  # [N, 2]
            node_idx=node_idx,  # [N, 2]
            asp_rto=asp_rto,
            long_side=long_side,
            b_shape=b_shape,  # [N]
            b_iou=b_iou,  # [N]
            blockshape_latent=blockshape_latent,  # [40]
            block_scale=block_scale,
            block_condition=block_condition,  # [2, H, W]
            org_binary_mask=self.base_transform(block_mask),  # [1, H, W]
        )

        # Optional zoning info passthrough
        zid = tmp_graph.graph.get('zone_id', None)
        zoh = tmp_graph.graph.get('zone_onehot', None)
        if zid is not None:
            data.zone_id = int(zid)
        if zoh is not None:
            zoh_arr = np.asarray(zoh, dtype=np.float32)
            if zoh_arr.ndim == 1:
                zoh_arr = zoh_arr[None, :]
            data.zone_onehot = torch.tensor(zoh_arr, dtype=torch.float32)

        return data


# Example (commented):
# root = os.getcwd()
# ds = UrbanGraphDataset(os.path.join(root, 'dataset', 'synthetic'), transform=graph_transform,
#                        cnn_transform=get_transform(rescale_size=64))
# train_loader = DataLoader(ds, batch_size=6, shuffle=True)
# for batch in train_loader:
#     print(batch)
