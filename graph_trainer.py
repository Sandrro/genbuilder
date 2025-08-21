import torch, os
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
from tensorboard_logger import configure, log_value
from time import gmtime, strftime
import warnings
import logging
from tqdm.auto import tqdm
warnings.filterwarnings("ignore")


def _make_zone_condition(batch, opt, device):
    """Build per-graph conditioning tensor from available features.

    Concatenates zoning one-hot, road features and template embeddings if
    present. Prints the shape and basic statistics of each component and the
    final concatenated tensor to ease debugging.
    """

    def _to_tensor(v):
        t = torch.as_tensor(v, dtype=torch.float32, device=device)
        if t.dim() == 1:
            t = t.unsqueeze(1)
        return t

    def _log_stats(name, t):
        print(f"{name}:", t.shape)
        flat = t.view(-1).float()
        print(
            f"  mean={float(flat.mean()):.4f} std={float(flat.std()):.4f} "
            f"min={float(flat.min()):.4f} max={float(flat.max()):.4f}"
        )

    parts = []

    zone_onehot = getattr(batch, 'zone_onehot', None)
    if zone_onehot is None:
        zid = getattr(batch, 'zone_id', None)
        if zid is not None and 'cond_dim' in opt and int(opt['cond_dim']) > 0:
            K = int(opt['cond_dim'])
            zid_t = torch.as_tensor(zid, device=device).view(-1)
            zone_onehot = torch.zeros((zid_t.numel(), K), device=device)
            zone_onehot[torch.arange(zid_t.numel(), device=device), zid_t.long()] = 1.0
    if zone_onehot is not None:
        zone_onehot = _to_tensor(zone_onehot)
        _log_stats('zone_onehot', zone_onehot)
        parts.append(zone_onehot)

    road_feats = getattr(batch, 'road_feats', None)
    if road_feats is not None:
        road_feats = _to_tensor(road_feats)
        _log_stats('road_feats', road_feats)
        parts.append(road_feats)

    templ_part = getattr(batch, 'template_flat_or_cnn', None)
    if templ_part is not None:
        templ_part = _to_tensor(templ_part)
        _log_stats('template_flat_or_cnn', templ_part)
        parts.append(templ_part)

    if not parts:
        return None

    node_cond = torch.cat(parts, dim=1)
    _log_stats('node_cond total', node_cond)
    return node_cond


def train(model, epoch, train_loader, device, opt, loss_dict, optimizer, scheduler):
    model.train()
    loss_sum = 0
    ext_acc = 0
    iter_ct = 0
    zone_correct = defaultdict(int)
    zone_total = defaultdict(int)
    geo_pos = geo_size = geo_shape = geo_iou = 0.0
    batch_size = opt['batch_size']
    num_data = opt['num_data']

    for batch_idx, data in enumerate(tqdm(train_loader, desc=f"Train {epoch+1}", leave=False)):
        data = data.to(device)
        cond = _make_zone_condition(data, opt, device)  # --- NEW ---

        optimizer.zero_grad()

        # модель должна принимать cond=...
        exist, pos, size, mu, log_var, b_shape, b_iou = model(data, cond=cond)

        exist_gt = data.x[:, 0].unsqueeze(1)
        pos_gt = data.org_node_pos
        size_gt = data.org_node_size
        b_shape_gt = data.b_shape_gt
        b_iou_gt = data.b_iou

        exist_out = torch.ge(torch.sigmoid(exist), 0.5).type(torch.uint8)
        extsum_loss = loss_dict['ExtSumloss'](torch.sum(exist_out), torch.sum(exist_gt))
        exist_loss = loss_dict['ExistBCEloss'](exist, exist_gt)
        pos_loss = loss_dict['Posloss'](pos, pos_gt)
        size_loss = loss_dict['Sizeloss'](size, size_gt)
        shape_loss = loss_dict['ShapeCEloss'](b_shape, b_shape_gt)
        iou_loss = loss_dict['Iouloss'](b_iou, b_iou_gt)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

        loss = opt['exist_weight'] * exist_loss + opt['pos_weight'] * pos_loss + opt['kld_weight'] * kld_loss + \
               size_loss * opt['size_weight'] + extsum_loss * opt['extsum_weight'] + \
               opt['shape_weight'] * shape_loss + opt['iou_weight'] * iou_loss

        loss.backward()
        loss_sum += loss.item()
        geo_pos += pos_loss.item()
        geo_size += size_loss.item()
        geo_shape += shape_loss.item()
        geo_iou += iou_loss.item()
        optimizer.step()
        scheduler.step()

        if opt['save_record']:
            step = int(epoch * (num_data / train_loader.batch_size) + batch_idx)
            log_value('train/all_loss', loss.item(), step)
            log_value('train/exist_loss', exist_loss.item(), step)
            log_value('train/pos_loss', pos_loss.item(), step)
            log_value('train/size_loss', size_loss.item(), step)
            log_value('train/kld_loss', kld_loss.item(), step)
            log_value('train/extsum_loss', extsum_loss, step)
            log_value('train/shape_loss', shape_loss.item(), step)
            log_value('train/bldg_iou_loss', iou_loss.item(), step)

        correct_mask = (exist_out == data.x[:, 0].unsqueeze(1))
        correct_ext = correct_mask.sum() / torch.numel(data.x[:, 0])
        ext_acc += correct_ext
        iter_ct += 1

        # per-zone accuracy
        zone_ids = getattr(data, 'zone_id', None)
        if zone_ids is not None:
            for g_idx, zid in enumerate(zone_ids.tolist()):
                node_mask = (data.batch == g_idx)
                zone_correct[zid] += correct_mask[node_mask].sum().item()
                zone_total[zid] += int(node_mask.sum().item())

        logging.debug(
            "train epoch %d batch %d/%d loss=%.6f",
            epoch,
            batch_idx,
            len(train_loader),
            loss.item(),
        )

    zone_acc = {z: zone_correct[z] / zone_total[z] for z in zone_correct}
    geo_breakdown = {
        'pos': geo_pos / float(iter_ct),
        'size': geo_size / float(iter_ct),
        'shape': geo_shape / float(iter_ct),
        'iou': geo_iou / float(iter_ct),
    }

    return ext_acc / float(iter_ct), loss_sum / float(iter_ct), zone_acc, geo_breakdown



def validation(model, epoch, val_loader, device, opt, loss_dict, scheduler):
    with torch.no_grad():
        model.eval()
        loss_all = 0
        ext_acc = 0
        iter_ct = 0
        batch_size = opt['batch_size']
        num_data = opt['num_data']
        loss_geo = 0.0
        zone_correct = defaultdict(int)
        zone_total = defaultdict(int)
        geo_pos = geo_size = geo_shape = geo_iou = 0.0

        for batch_idx, data in enumerate(tqdm(val_loader, desc=f"Val {epoch+1}", leave=False)):
            data = data.to(device)
            cond = _make_zone_condition(data, opt, device)  # --- NEW ---

            exist, pos, size, mu, log_var, b_shape, b_iou = model(data, cond=cond)

            exist_gt = data.x[:, 0].unsqueeze(1)
            pos_gt = data.org_node_pos
            size_gt = data.org_node_size
            b_shape_gt = data.b_shape_gt
            b_iou_gt = data.b_iou

            exist_loss = loss_dict['ExistBCEloss'](exist, exist_gt)
            pos_loss = loss_dict['Posloss'](pos, pos_gt)
            size_loss = loss_dict['Sizeloss'](size, size_gt)
            kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
            exist_out = torch.ge(exist, 0.5).type(torch.uint8)
            extsum_loss = loss_dict['ExtSumloss'](torch.sum(exist_out), torch.sum(exist_gt))
            shape_loss = loss_dict['ShapeCEloss'](b_shape, b_shape_gt)
            iou_loss = loss_dict['Iouloss'](b_iou, b_iou_gt)

            loss = opt['exist_weight'] * exist_loss + opt['pos_weight'] * pos_loss + opt['kld_weight'] * kld_loss + \
                   size_loss * opt['size_weight'] + extsum_loss * opt['extsum_weight'] + \
                   opt['shape_weight'] * shape_loss + opt['iou_weight'] * iou_loss

            loss_all += loss.item()
            loss_geo += (pos_loss.item() + size_loss.item())
            geo_pos += pos_loss.item()
            geo_size += size_loss.item()
            geo_shape += shape_loss.item()
            geo_iou += iou_loss.item()

            if opt['save_record']:
                step = int(epoch * (num_data / batch_size) + batch_idx)
                log_value('val/val_all_loss', loss.item(), step)
                log_value('val/val_exist_loss', exist_loss.item(), step)
                log_value('val/val_pos_loss', pos_loss.item(), step)
                log_value('val/val_size_loss', size_loss.item(), step)
                log_value('val/val_kld_loss', kld_loss.item(), step)
                log_value('val/val_extsum_loss', extsum_loss, step)
                log_value('val/val_shape_loss', shape_loss.item(), step)
                log_value('val/val_bldg_iou_loss', iou_loss.item(), step)

            correct_mask = (exist_out == data.x[:, 0].unsqueeze(1))
            correct_ext = correct_mask.sum() / torch.numel(data.x[:, 0])
            ext_acc += correct_ext
            iter_ct += 1

            zone_ids = getattr(data, 'zone_id', None)
            if zone_ids is not None:
                for g_idx, zid in enumerate(zone_ids.tolist()):
                    node_mask = (data.batch == g_idx)
                    zone_correct[zid] += correct_mask[node_mask].sum().item()
                    zone_total[zid] += int(node_mask.sum().item())

            logging.debug(
                "val epoch %d batch %d/%d loss=%.6f",
                epoch,
                batch_idx,
                len(val_loader),
                loss.item(),
            )

    zone_acc = {z: zone_correct[z] / zone_total[z] for z in zone_correct}
    geo_breakdown = {
        'pos': geo_pos / float(iter_ct),
        'size': geo_size / float(iter_ct),
        'shape': geo_shape / float(iter_ct),
        'iou': geo_iou / float(iter_ct),
    }

    return ext_acc / float(iter_ct), loss_all / float(iter_ct), loss_geo / float(iter_ct), zone_acc, geo_breakdown
