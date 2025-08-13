import os
import re
import json
import shutil
import random
import logging
from time import gmtime, strftime

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.optim.lr_scheduler import MultiStepLR
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader, DataListLoader

from urban_dataset import UrbanGraphDataset, graph_transform, get_transform
from model import *
from graph_util import read_train_yaml
from graph_trainer import train, validation
from tensorboard_logger import configure, log_value
import yaml
import warnings
warnings.filterwarnings("ignore")

# ----------------------- new: helpers for renaming -----------------------

def _try_load_zones_map(dataset_root):
    candidates = [
        os.path.join(dataset_root, 'processed', '_zones_map.json'),
        os.path.join(dataset_root, '_zones_map.json'),
    ]
    for p in candidates:
        if os.path.isfile(p):
            with open(p, 'r', encoding='utf-8') as f:
                data = json.load(f)
                zmap = data.get('map', data)
                return {k: int(v) for k, v in zmap.items()}
    return None


def _natural_key(name: str):
    """Split a string into text and integer chunks for natural sorting."""
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', name)]


def _ensure_sequential_gpickles(dataset_root: str) -> None:
    """
    Ensure all .gpickle files in <dataset_root>/processed (fallback to <dataset_root>)
    are named sequentially as 0.gpickle, 1.gpickle, ... in a stable, deterministic
    order (natural sort by filename). Writes a mapping file '_file_index_map.json'.
    Two-phase rename avoids collisions.
    """
    processed_dir = os.path.join(dataset_root, 'processed')
    if not os.path.isdir(processed_dir):
        processed_dir = dataset_root
    if not os.path.isdir(processed_dir):
        return

    files = [f for f in os.listdir(processed_dir) if f.endswith('.gpickle')]
    if not files:
        return

    files_sorted = sorted(files, key=_natural_key)

    # Check if already 0..N-1 without gaps
    stems = [os.path.splitext(f)[0] for f in files_sorted]
    all_numeric = all(s.isdigit() for s in stems)
    if all_numeric:
        idxs = sorted(int(s) for s in stems)
        if idxs == list(range(len(files_sorted))):
            # Already good
            return

    # Two-phase rename: move everything to temp hidden names, then to final indices
    tmp_names = []
    orig_to_tmp = {}
    rnd = f"{random.randint(0, 1_000_000):06d}"

    for i, fname in enumerate(files_sorted):
        src = os.path.join(processed_dir, fname)
        tmp = os.path.join(processed_dir, f".__ren_{rnd}_{i}__.gpickle")
        os.replace(src, tmp)
        tmp_names.append(tmp)
        orig_to_tmp[fname] = os.path.basename(tmp)

    mapping = {}
    for i, tmp in enumerate(tmp_names):
        dst = os.path.join(processed_dir, f"{i}.gpickle")
        os.replace(tmp, dst)
        # Find original name for this tmp
        orig = None
        for k, v in orig_to_tmp.items():
            if v == os.path.basename(tmp):
                orig = k
                break
        mapping[orig] = f"{i}.gpickle"

    # Persist mapping for traceability
    map_path = os.path.join(processed_dir, '_file_index_map.json')
    try:
        with open(map_path, 'w', encoding='utf-8') as f:
            json.dump(mapping, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


# ----------------------- main training script -----------------------

if __name__ == "__main__":
    random.seed(42)  # make sure every time has the same training and validation sets

    root = os.getcwd()

    # configuration file can be overridden via env var
    cfg_file = os.environ.get("TRAIN_CONFIG", "train_gnn.yaml")
    train_opt = read_train_yaml(root, filename=cfg_file)

    # allow custom dataset location (env var takes precedence over yaml field)
    dataset_path = os.environ.get(
        "DATASET_ROOT",
        os.path.join(root, train_opt.get("dataset_root", "dataset")),
    )

    # NEW: make sure gpickles are sequentially named before dataset loads them
    _ensure_sequential_gpickles(dataset_path)
    print(train_opt)
    is_resmue = train_opt['resume']
    gpu_ids = train_opt['gpu_ids']

    if is_resmue:
        resume_epoch = train_opt['resume_epoch']
        resume_dir = train_opt['resume_dir']
        import_name = train_opt['import_name']
        opt = read_train_yaml(os.path.join(root, 'epoch', resume_dir), filename="train_save.yaml")
    else:
        opt = train_opt

    # === NEW: autodetect cond_dim from _zones_map.json (if not set) ===
    if ('cond_dim' not in opt or int(opt['cond_dim']) <= 0) and opt.get('is_conditional_block', False):
        zmap = _try_load_zones_map(dataset_path)
        if zmap is not None:
            opt['cond_dim'] = int(len(zmap))
            print(f"[info] Detected cond_dim={opt['cond_dim']} from _zones_map.json")

    notes = 'GlobalMapper'

    maxlen = opt['template_width']
    N = maxlen * opt['template_height']
    min_bldg = 0   # >
    max_bldg = N   # <=
    opt['N'] = int(N)

    fname = opt['convlayer'] + '_' + opt['aggr'] + '_dim' + str(opt['n_ft_dim'])
    data_name = 'osm_cities'
    opt['data_name'] = data_name
    print(data_name)

    device = torch.device('cuda:{}'.format(gpu_ids[0]))
    print(device)
    opt['device'] = str(device)
    start_epoch = opt['start_epoch']

    loss_dict = {}
    loss_dict['Posloss'] = nn.MSELoss(reduction='sum')
    loss_dict['ShapeCEloss'] = nn.CrossEntropyLoss(reduction='sum')
    loss_dict['Iouloss'] = nn.MSELoss(reduction='sum')
    loss_dict['ExistBCEloss'] = nn.BCEWithLogitsLoss(reduction='sum')
    loss_dict['CELoss'] = nn.CrossEntropyLoss(reduction='none')
    loss_dict['Sizeloss'] = nn.MSELoss(reduction='sum')  # nn.SmoothL1Loss
    loss_dict['ExtSumloss'] = nn.MSELoss(reduction='sum')  # nn.SmoothL1Loss

    save_pth = os.path.join(root, 'epoch'); os.makedirs(save_pth, exist_ok=True)
    log_pth = os.path.join(root, 'tensorboard'); os.makedirs(log_pth, exist_ok=True)
    logs_pth = os.path.join(root, 'logs'); os.makedirs(logs_pth, exist_ok=True)

    time_str = strftime("%Y-%m-%d %H:%M:%S", gmtime())

    if is_resmue:
        save_name = notes + fname + '_lr{}_epochs{}_batch{}'.format(opt['lr'], opt['total_epochs'], opt['batch_size'])
        save_pth = os.path.join(root, 'epoch', resume_dir)
        log_file = os.path.join(root, 'logs', resume_dir + '.log')
        tb_path = os.path.join(root, 'tensorboard', resume_dir)
    else:
        save_name = notes + fname + '_lr{}_epochs{}_batch{}_'.format(opt['lr'], opt['total_epochs'], opt['batch_size'])
        save_pth = os.path.join(root, 'epoch', save_name + time_str)
        log_file = os.path.join(root, 'logs', save_name + time_str + '.log')
        tb_path = os.path.join(root, 'tensorboard', save_name + time_str)

    if opt['save_record']:
        os.makedirs(save_pth, exist_ok=True)
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.basicConfig(
            level=logging.INFO,
            format="[%(levelname)s] %(message)s - %(filename)s %(funcName)s %(asctime)s ",
            filename=log_file)
        opt['save_path'] = save_pth
        opt['log_path'] = log_file
        opt['tensorboard_path'] = tb_path
        if is_resmue:
            yaml_fn = 'resume_train_save.yaml'
            opt['save_notes'] = save_name
        else:
            yaml_fn = 'train_save.yaml'
        with open(os.path.join(save_pth, yaml_fn), 'w') as outfile:
            yaml.dump(opt, outfile, default_flow_style=False)
        configure(tb_path, flush_secs=5)

    torch.autograd.set_detect_anomaly(True)

    cnn_transform = get_transform(noise_range=10.0, noise_type='gaussian', isaug=False, rescale_size=64)
    dataset = UrbanGraphDataset(dataset_path, transform=graph_transform, cnn_transform=cnn_transform)
    num_data = len(dataset)
    opt['num_data'] = int(num_data)
    print(num_data)

    # === NEW: stratified split by zone_id if present ===
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
        from collections import defaultdict
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
        train_mask = np.ones(num_data, dtype=bool)
        train_mask[val_idx] = False
        train_idx = np.arange(num_data)[train_mask]
        print('[split] Stratified by zone_id')
    else:
        val_num = int(num_data * opt['val_ratio'])
        val_idx = np.array(random.sample(range(num_data), val_num))
        train_idx = np.delete(np.arange(num_data), val_idx)
        print('[split] Random split')

    print('Get {} graph for training'.format(train_idx.shape[0]))
    print('Get {} graph for validation'.format(val_idx.shape[0]))

    val_dataset = dataset[val_idx]
    train_dataset = dataset[train_idx]
    val_loader = DataLoader(val_dataset, batch_size=opt['batch_size'], shuffle=False, num_workers=8)
    train_loader = DataLoader(train_dataset, batch_size=opt['batch_size'], shuffle=True, num_workers=8)

    if opt['is_blockplanner']:
        model = NaiveBlockGenerator(opt, N=N)
    elif opt['is_conditional_block']:
        if opt['convlayer'] in opt['attten_net']:
            model = AttentionBlockGenerator(opt, N=N)
        else:
            model = BlockGenerator(opt, N=N)
    else:
        if opt['convlayer'] in opt['attten_net']:
            if opt['encode_cnn']:
                print('attention net')
                model = AttentionBlockGenerator_independent_cnn(opt, N=N)
            else:
                model = AttentionBlockGenerator_independent(opt, N=N)

    if is_resmue:
        start_epoch = resume_epoch
        print('import from {}'.format(os.path.join(root, 'epoch', resume_dir, import_name + '.pth')))
        model.load_state_dict(torch.load(os.path.join(root, 'epoch', resume_dir, import_name + '.pth'), map_location=device))

    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=float(opt['lr']), weight_decay=1e-6)
    scheduler = MultiStepLR(optimizer, milestones=[(opt['total_epochs'] - start_epoch) * 0.6, (opt['total_epochs'] - start_epoch) * 0.8], gamma=0.3)

    best_val_acc = None
    best_train_acc = None
    best_train_loss = None
    best_val_loss = None
    best_val_geo_loss = None

    print('Start Training...')
    logging.info('Start Training...' )

    for epoch in range(start_epoch, opt['total_epochs']):
        t_acc, t_loss = train(model, epoch, train_loader, device, opt, loss_dict, optimizer, scheduler)
        v_acc, v_loss, v_loss_geo = validation(model, epoch, val_loader, device, opt, loss_dict, scheduler)

        if opt['save_record']:
            if best_train_acc is None or t_acc >= best_train_acc:
                best_train_acc = t_acc
            if best_train_loss is None or t_loss <= best_train_loss:
                best_train_loss = t_loss
            if best_val_acc is None or v_acc >= best_val_acc:
                best_val_acc = v_acc
                filn = os.path.join(save_pth, "val_best_extacc.pth")
                torch.save(model.state_dict(), filn)
            if best_val_loss is None or v_loss <= best_val_loss:
                best_val_loss = v_loss
                filn = os.path.join(save_pth, "val_least_loss_all.pth")
                torch.save(model.state_dict(), filn)
            if best_val_geo_loss is None or v_loss_geo <= best_val_geo_loss:
                best_val_geo_loss = v_loss_geo
                filn = os.path.join(save_pth, "val_least_loss_geo.pth")
                torch.save(model.state_dict(), filn)
            if epoch % opt['save_epoch'] == 0:
                filn = os.path.join(save_pth, str(epoch) + "_save.pth")
                torch.save(model.state_dict(), filn)
            logging.info('Epoch: {:03d}, Train Loss: {:.7f}, Train exist accuracy: {:.7f}, Valid Loss: {:.7f}, Valid exist accuracy: {:.7f}, valid geo loss {:.7f}'.format(epoch, t_loss, t_acc, v_loss, v_acc, v_loss_geo) )
            print('Epoch: {:03d}, Train Loss: {:.7f}, Train exist accuracy: {:.7f}, Valid Loss: {:.7f}, Valid exist accuracy: {:.7f}, valid geo loss {:.7f}'.format(epoch, t_loss, t_acc, v_loss, v_acc, v_loss_geo) )
            filn = os.path.join(save_pth, "latest.pth")
            torch.save(model.state_dict(), filn)

    if opt['save_record']:
        logging.info('Least Train Loss: {:.7f}, Best Train exist accuracy: {:.7f}, Least Valid Loss: {:.7f}, Best Valid exist accuracy: {:.7f}, best valid geo loss {:.7f}'.format(best_train_loss, best_train_acc, best_val_loss, best_val_acc, best_val_geo_loss))
        print('Least Train Loss: {:.7f}, Best Train exist accuracy: {:.7f}, Least Valid Loss: {:.7f}, Best Valid exist accuracy: {:.7f}, best valid geo loss {:.7f}'.format(best_train_loss, best_train_acc, best_val_loss, best_val_acc, best_val_geo_loss))
