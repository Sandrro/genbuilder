import os, pickle, random, shutil, json
import pyarrow as pa
import pyarrow.ipc as ipc
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from urban_dataset import UrbanGraphDataset, graph_transform, get_transform, test_graph_transform
from model import *
from graph_util import *
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import networkx as nx
from typing import Optional
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def write_graph_arrow(graph: nx.Graph, path: str) -> None:
    data = pickle.dumps(graph)
    table = pa.table({"graph": [data]})
    with pa.OSFile(path, "wb") as sink:
        with ipc.new_file(sink, table.schema) as writer:
            writer.write_table(table)


def read_graph_arrow(path: str) -> nx.Graph:
    with pa.memory_map(path, "rb") as source:
        table = ipc.open_file(source).read_all()
    data = table.column("graph")[0].as_py()
    return pickle.loads(data)

"""
Test script with zoning condition support (compatible with original repo outputs).
- Loads zone info (zone_onehot/zone_id) from dataset if available.
- Can override zone via CLI-ish constants ZONE_NAME or ZONE_ID below.
- Works both in reconstruction mode and sampling mode.
- NEW: Robust checkpoint compatibility.
  * Autodetects concat_heads from checkpoint.
  * Infers model variant from checkpoint.
  * Tries multiple (variant, concat_heads) combos and **loads only matching-shaped weights** to avoid size mismatch.
    Picks the combo with the most weights loaded.
"""

# ================== User overrides ==================
ZONE_NAME = "residential"   # e.g., 'residential'; if set, we will build one-hot from _zones_map.json
ZONE_ID   = None             # e.g., 3; takes precedence over ZONE_NAME
# ====================================================


# ------------------------
# Checkpoint introspection
# ------------------------

def _infer_model_variant_from_ckpt(state_dict) -> str:
    """Infer which model class the checkpoint corresponds to.
    Returns one of: 'attn_ind_cnn', 'attn_ind', 'attn', 'block', 'naive'.
    """
    ks = set(state_dict.keys())
    if any(k.startswith('cnn_encoder.') for k in ks) or 'linear1.weight' in ks:
        return 'attn_ind_cnn'
    if 'enc_block_scale.weight' in ks or 'enc_block_scale.bias' in ks:
        return 'attn_ind'
    if any(k.startswith('e_conv1') for k in ks) and any(k.startswith('d_conv1') for k in ks):
        return 'attn'
    if 'aggregate.weight' in ks and 'd_exist_1.weight' in ks:
        return 'block'
    return 'naive'


def _ckpt_dec_in_features(state_dict, latent_ch: int, heads: int) -> Optional[int]:
    """Try to read the expected decoder input width from checkpoint linear layers.
    Returns in_features if detected, else None.
    """
    candidates = ['d_exist_0.weight', 'd_posx_0.weight', 'd_posy_0.weight', 'd_sizex_0.weight', 'd_sizey_0.weight', 'd_shape_0.weight', 'd_iou_0.weight']
    for k in candidates:
        W = state_dict.get(k, None)
        if isinstance(W, torch.Tensor) and W.dim() == 2:
            return W.shape[1]
    return None


def _infer_concat_heads_from_ckpt(state_dict, latent_ch: int, heads: int, T: int) -> bool:
    """Return True if checkpoint clearly uses concatenation of attention heads in convs/decoders."""
    # Strong signal from conv biases (e.g., 256*12) or weights
    target = latent_ch * max(1, heads)
    for k in ('e_conv1.bias','e_conv2.bias','e_conv3.bias','d_conv1.bias','d_conv2.bias','d_conv3.bias'):
        b = state_dict.get(k, None)
        if isinstance(b, torch.Tensor) and b.numel() == target and heads > 1:
            return True
    # Fallback: aggregate input width heuristic
    W = state_dict.get('aggregate.weight', None)
    if isinstance(W, torch.Tensor):
        in_features = W.shape[1]
        coef = in_features // max(1, latent_ch)
        if coef >= (2 + heads):  # rough: with T≈3 and heads=12 → 38 vs no-concat ~5
            return True
    # Decoder linear input check
    dec_in = _ckpt_dec_in_features(state_dict, latent_ch, heads)
    if dec_in is not None and dec_in == latent_ch * heads:
        return True
    return False


# ------------------------
# Model building + loading
# ------------------------

def _build_model(opt, variant: str, N: int, concat_heads: bool):
    opt = dict(opt)  # shallow copy
    opt['concat_heads'] = bool(concat_heads)
    if variant == 'attn_ind_cnn':
        return AttentionBlockGenerator_independent_cnn(opt, N=N)
    if variant == 'attn_ind':
        return AttentionBlockGenerator_independent(opt, N=N)
    if variant == 'attn':
        return AttentionBlockGenerator(opt, N=N)
    if variant == 'block':
        return BlockGenerator(opt, N=N)
    if variant == 'naive':
        return NaiveBlockGenerator(opt, N=N)
    # fallback by opt
    if opt.get('is_blockplanner', False):
        return NaiveBlockGenerator(opt, N=N)
    if opt.get('is_conditional_block', False):
        if opt.get('convlayer') in opt.get('attten_net', []):
            return AttentionBlockGenerator(opt, N=N)
        return BlockGenerator(opt, N=N)
    if opt.get('convlayer') in opt.get('attten_net', []):
        if opt.get('encode_cnn', False):
            return AttentionBlockGenerator_independent_cnn(opt, N=N)
        return AttentionBlockGenerator_independent(opt, N=N)
    return BlockGenerator(opt, N=N)


def _filter_state_for_model(state_dict, model):
    """Keep only params that exactly match by name and shape to avoid size mismatch errors."""
    model_sd = model.state_dict()
    keep = {}
    skipped = []
    for k, v in state_dict.items():
        if k in model_sd and isinstance(v, torch.Tensor) and v.shape == model_sd[k].shape:
            keep[k] = v
        else:
            skipped.append(k)
    return keep, skipped


def _try_load_variants(opt, N, device, state_dict, inferred_variant: str, latent_ch: int, heads: int):
    """Try multiple (variant, concat) combos. Return the model with most weights loaded."""
    tried = []

    # Priority order: inferred variant with concat inferred → toggle concat → fallback variants
    concat_guess = _infer_concat_heads_from_ckpt(state_dict, latent_ch, heads, int(opt.get('T', 3)))
    variants = [inferred_variant]
    # add plausible fallbacks
    if inferred_variant == 'attn_ind_cnn':
        variants += ['attn_ind', 'attn']
    elif inferred_variant == 'attn_ind':
        variants += ['attn', 'attn_ind_cnn']
    elif inferred_variant == 'attn':
        variants += ['attn_ind', 'attn_ind_cnn']

    best = None
    best_loaded = -1

    for var in variants:
        for concat in [concat_guess, not concat_guess]:
            try:
                model = _build_model(opt, var, N, concat)
                filtered, skipped = _filter_state_for_model(state_dict, model)
                model.load_state_dict(filtered, strict=False)
                loaded = len(filtered)
                tried.append((var, concat, loaded, len(skipped)))
                if loaded > best_loaded:
                    best = (model, var, concat, loaded, len(skipped))
                    best_loaded = loaded
            except Exception as e:
                tried.append((var, concat, str(e)))
                continue

    if best is None:
        raise RuntimeError(f"Cannot load checkpoint into any tested architecture. Tried={tried}")

    model, var, concat, loaded, skipped = best
    print(f"[info] chosen variant={var}, concat_heads={concat} → loaded {loaded} tensors, skipped {skipped}")
    # Optional: print a couple of skipped keys for transparency
    return model


if __name__ == "__main__":
    root = os.getcwd()
    random.seed(42)

    # allow overriding via environment variables
    pth_name = os.environ.get('MODEL_CHECKPOINT', 'latest')
    dataset_path = os.environ.get('DATASET_ROOT', os.path.join(root, 'dataset'))
    epoch_name = os.environ.get('EPOCH_NAME')
    data_name = 'osm_cities'
    logging.info("Dataset root: %s", dataset_path)
    logging.info("Requested checkpoint: %s", pth_name)

    if epoch_name is None:
        epoch_root = os.path.join(root, 'epoch')
        candidates = [d for d in os.listdir(epoch_root) if os.path.isdir(os.path.join(epoch_root, d))]
        if not candidates:
            raise RuntimeError('No epoch directory found. Set EPOCH_NAME env var.')
        epoch_name = max(candidates, key=lambda d: os.path.getmtime(os.path.join(epoch_root, d)))
    logging.info("Using epoch directory: %s", epoch_name)

    gpu_ids = 0
    batch_size = 1

    is_teaser = True
    teaser_note = 'continuous_chicago'

    template_height = 4
    template_width = 30
    N = template_width * template_height

    is_reconstruct = True

    test_yaml = os.path.join(root, 'epoch', epoch_name)
    opt = read_train_yaml(test_yaml, "train_save.yaml")

    # --- zone map (for overrides) ---
    zones_map_path = os.path.join(dataset_path, 'processed', '_zones_map.json')
    zones_map = None
    if os.path.isfile(zones_map_path):
        try:
            zones_map = json.load(open(zones_map_path, 'r', encoding='utf-8'))
            zones_map = zones_map.get('map', zones_map)
        except Exception:
            zones_map = None

    # output layout
    output_num = 1e10 if is_teaser else 1000

    draw_edge = True
    draw_nonexist = False

    save_root = os.path.join(root, 'test'); os.makedirs(save_root, exist_ok=True)
    save_pth = os.path.join(save_root, 'test_' + epoch_name); os.makedirs(save_pth, exist_ok=True)

    if is_reconstruct:
        dir_name = pth_name + '_reconstruct_' + data_name
        if is_teaser:
            dir_name = pth_name + '_reconstruct_continuous'
    else:
        dir_name = pth_name + '_var_gen'

    save_pth = os.path.join(save_pth, dir_name)
    res_path = os.path.join(save_pth, 'result')
    gt_path = os.path.join(save_pth, 'gt')

    res_graph_path = os.path.join(res_path, 'graph')
    res_visual_path = os.path.join(res_path, 'visual')
    res_block_img_path = os.path.join(res_path, 'block_img')
    res_final_img_path = os.path.join(res_path, 'final')

    gt_graph_path = os.path.join(gt_path, 'graph')
    gt_visual_path = os.path.join(gt_path, 'visual')
    ex_visual_path = os.path.join(res_path, 'exist')
    gt_ex_visual_path = os.path.join(gt_path, 'exist')

    for p in [save_pth, res_path, gt_path, res_graph_path, res_visual_path, gt_graph_path, gt_visual_path, ex_visual_path, gt_ex_visual_path, res_block_img_path]:
        os.makedirs(p, exist_ok=True)

    device = torch.device('cuda:' + str(gpu_ids))
    opt['device'] = device

    # === Load checkpoint and choose best-fitting model automatically ===
    ckpt_path = os.path.join(test_yaml, pth_name + ".pth")
    logging.info("Loading checkpoint from %s", ckpt_path)
    state = torch.load(ckpt_path, map_location='cpu')

    latent_ch = int(opt.get('n_ft_dim', 256))
    heads = int(opt.get('head_num', 1))

    inferred_variant = _infer_model_variant_from_ckpt(state)
    print(f"[info] inferred model variant from checkpoint → {inferred_variant}")

    model = _try_load_variants(opt, N, device, state, inferred_variant, latent_ch, heads)
    model.to(device)
    model.eval()

    # --- dataset/loader ---
    cnn_transform = get_transform(noise_range=10.0, noise_type='gaussian', isaug=False, rescale_size=64)
    dataset = UrbanGraphDataset(dataset_path, transform=test_graph_transform, cnn_transform=cnn_transform)
    num_data = len(dataset)

    if is_teaser:
        val_idx = np.arange(num_data)
        val_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        print('Get {} graph for teaser testing.'.format(val_idx.shape[0]))
    else:
        labels = []
        have_labels = True
        try:
            for i in range(num_data):
                z = getattr(dataset[i], 'zone_id', None)
                if z is None:
                    have_labels = False
                    break
                labels.append(int(z))
        except Exception:
            have_labels = False

        if have_labels and len(set(labels)) > 1:
            rng = np.random.RandomState(42)
            by_cls = defaultdict(list)
            for idx, c in enumerate(labels):
                by_cls[c].append(idx)
            val_idx = []
            for c, idxs in by_cls.items():
                idxs = np.array(idxs)
                rng.shuffle(idxs)
                k = max(1, int(round(len(idxs) * opt['val_ratio'])))
                val_idx.extend(idxs[:k].tolist())
            val_idx = np.array(sorted(set(val_idx)))
            print('[split] Stratified by zone_id')
        else:
            val_num = int(num_data * opt['val_ratio'])
            val_idx = np.array(random.sample(range(num_data), val_num))
            print('[split] Random split')
        print('Get {} graph for validation'.format(val_idx.shape[0]))
        val_dataset = dataset[val_idx]
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # --- helper: build cond tensor from batch or override ---
    def build_cond(batch):
        # override path
        if ZONE_ID is not None and int(opt.get('cond_dim', 0)) > 0:
            K = int(opt['cond_dim'])
            oh = torch.zeros((batch.num_graphs, K), device=device)
            oh[:, int(ZONE_ID)] = 1.0
            return oh
        if ZONE_NAME is not None and zones_map and int(opt.get('cond_dim', 0)) > 0:
            zid = zones_map.get(ZONE_NAME, None)
            if zid is not None:
                K = int(opt['cond_dim'])
                oh = torch.zeros((batch.num_graphs, K), device=device)
                oh[:, int(zid)] = 1.0
                return oh
        # dataset-provided path
        cond = getattr(batch, 'zone_onehot', None)
        if cond is not None:
            return cond.to(device)
        zid = getattr(batch, 'zone_id', None)
        if zid is not None and int(opt.get('cond_dim', 0)) > 0:
            K = int(opt['cond_dim'])
            zid_t = torch.as_tensor(zid, device=device).view(-1)
            oh = torch.zeros((zid_t.numel(), K), device=device)
            oh[torch.arange(zid_t.numel(), device=device), zid_t.long()] = 1.0
            return oh
        return None

    fn_ct = 0
    z_sample_list = []

    for data in val_loader:
        if output_num is not None and fn_ct >= output_num:
            break
        print(fn_ct)
        data = data.to(device)
        edge_index = data.edge_index

        cond = build_cond(data)  # zoning condition (optional)

        if opt.get('is_blockplanner', False):
            if is_reconstruct:
                mu, log_var = model.encode(data, cond=cond)
                z_sample = model.reparameterize(mu, log_var)
            else:
                z_sample = torch.randn(batch_size, opt['latent_dim']).to(device)
            z_sample_list.append(z_sample.squeeze().detach().cpu().numpy())
            if opt.get('is_input_road', False):
                block_condition = data.block_condition.view(batch_size, 2, 64, 64)
                block_condition = model.cnn_encode(block_condition)
                exist, posx, posy, sizex, sizey, b_shape, b_iou = model.decode(z_sample, block_condition, data.edge_index)
            else:
                exist, posx, posy, sizex, sizey, b_shape, b_iou = model.decode(z_sample, edge_index)
        else:
            if opt.get('encode_cnn', False):
                if is_reconstruct:
                    mu, log_var = model.encode(data, cond=cond)
                    if opt.get('is_input_road', False):
                        block_condition = data.block_condition.view(batch_size, 2, 64, 64)
                        block_condition = model.cnn_encode(block_condition)
                    z_sample = model.reparameterize(mu, log_var)
                    z_sample_list.append(z_sample.squeeze().detach().cpu().numpy())
                else:
                    if opt.get('is_input_road', False):
                        block_condition = data.block_condition.view(batch_size, 2, 64, 64)
                        block_condition = model.cnn_encode(block_condition)
                    z_sample = torch.randn(batch_size, opt['latent_dim']).to(device)
                if opt.get('is_input_road', False):
                    exist, posx, posy, sizex, sizey, b_shape, b_iou = model.decode(z_sample, block_condition, data.edge_index)
                else:
                    exist, posx, posy, sizex, sizey, b_shape, b_iou = model.decode(z_sample, data.edge_index)
            else:
                if is_reconstruct:
                    mu, log_var = model.encode(data, cond=cond)
                    z_sample = model.reparameterize(mu, log_var)
                    block_scale = model.enc_block_scale(data.block_scale_gt.unsqueeze(1))
                    block_shape = data.blockshape_latent_gt.view(-1, model.blockshape_latent_dim)
                    block_condition = torch.cat((block_shape, block_scale), 1)
                else:
                    z_sample = torch.randn(batch_size, opt['latent_dim']).to(device)
                    block_scale = model.enc_block_scale(data.block_scale_gt.unsqueeze(1))
                    block_shape = data.blockshape_latent_gt.view(-1, model.blockshape_latent_dim)
                    block_condition = torch.cat((block_shape, block_scale), 1)
                exist, posx, posy, sizex, sizey, b_shape, b_iou = model.decode(z_sample, block_condition, data.edge_index)

        # Prepare aux tensors for writer
        asp_rto = torch.zeros_like(data.asp_rto_gt.unsqueeze(1))
        long_side = torch.zeros_like(data.long_side_gt.unsqueeze(1))

        exist_gt = data.x[:, 0].unsqueeze(1)
        pos_gt = data.org_node_pos
        size_gt = data.org_node_size

        # NOTE: Training uses BCEWithLogitsLoss; proper thresholding is sigmoid(exist) >= 0.5
        exist_prob = torch.sigmoid(exist)
        exist_bin = (exist_prob >= 0.5).to(torch.uint8)
        correct_ext = (exist_bin == data.x[:, 0].unsqueeze(1)).sum() / torch.numel(data.x[:, 0])

        exist_np = exist_bin.squeeze().detach().cpu().numpy()
        posx_np = posx.squeeze().detach().cpu().numpy()
        posy_np = posy.squeeze().detach().cpu().numpy()
        sizex_np = sizex.squeeze().detach().cpu().numpy()
        sizey_np = sizey.squeeze().detach().cpu().numpy()
        asp_rto_np = asp_rto.squeeze().detach().cpu().numpy()
        long_side_np = long_side.squeeze().detach().cpu().numpy()
        b_iou_np = b_iou.squeeze().detach().cpu().numpy()
        _, shape_pred = torch.max(b_shape, 1)
        shape_pred_np = shape_pred.detach().cpu().numpy()

        for i in range(batch_size):
            g_add = sparse_generate_graph_from_ftsarray(
                template_height, template_width,
                posx_np, posy_np, sizey_np, sizex_np,
                exist_np, asp_rto_np, long_side_np, shape_pred_np, b_iou_np
            )
            filename = str(val_idx[fn_ct]) if is_reconstruct else str(fn_ct)
            write_graph_arrow(g_add, os.path.join(res_graph_path, filename + ".arrow"))
            visual_block_graph(g_add, res_visual_path, filename, draw_edge, draw_nonexist)
            visual_existence_template(g_add, ex_visual_path, filename, coord_scale=1,
                                      template_width=template_width, template_height=template_height)
            if is_reconstruct:
                rst = os.path.join(dataset_path, 'processed', filename + ".arrow")
                dst = os.path.join(gt_graph_path, filename + '.arrow')
                g = read_graph_arrow(rst)
                shutil.copyfile(rst, dst)
            fn_ct += 1

    if is_reconstruct:
        z_sample_array = np.array(z_sample_list)
        z_mean = np.mean(z_sample_array, axis=0)
        z_std = np.std(z_sample_array, axis=0)
        xpoints = range(opt['latent_dim'])
        ypoints = z_mean.flatten(); plt.plot(xpoints, ypoints); plt.savefig(os.path.join(save_pth, 'mean.png')); plt.clf()
        ypoints = z_std.flatten();  plt.plot(xpoints, ypoints); plt.savefig(os.path.join(save_pth, 'std.png'))
        with open(os.path.join(save_pth, 'sample_stats_' + str(fn_ct)), 'wb') as f:
            pickle.dump([z_mean, z_std], f)
        with open(os.path.join(save_pth, 'z_sample_' + str(fn_ct)), 'wb') as f:
            pickle.dump([z_sample_array], f)

    print('Finish')
