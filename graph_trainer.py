import torch, os
import torch.nn.functional as F
import numpy as np
from tensorboard_logger import configure, log_value
from time import gmtime, strftime
import warnings
warnings.filterwarnings("ignore")


def _make_zone_condition(batch, opt, device):
    """Build per-graph zoning condition tensor (B, K) from batch.
    Prefers batch.zone_onehot; falls back to one-hot from zone_id if present.
    Returns None if no labels.
    """
    cond = getattr(batch, 'zone_onehot', None)
    if cond is not None:
        # cond может быть списком или тензором; приведём к (B, K) тензору на device
        if isinstance(cond, (list, tuple)):
            cond = torch.stack([torch.as_tensor(c, dtype=torch.float32) for c in cond], dim=0)
        cond = cond.to(device)
        return cond

    zid = getattr(batch, 'zone_id', None)
    if zid is None or 'cond_dim' not in opt or int(opt['cond_dim']) <= 0:
        return None
    K = int(opt['cond_dim'])
    zid_t = torch.as_tensor(zid, device=device).view(-1)
    cond = torch.zeros((zid_t.numel(), K), device=device)
    cond[torch.arange(zid_t.numel(), device=device), zid_t.long()] = 1.0
    return cond


def train(model, epoch, train_loader, device, opt, loss_dict, optimizer, scheduler):
    model.train()
    loss_sum = 0
    ext_acc = 0
    iter_ct = 0
    batch_size = opt['batch_size']
    num_data = opt['num_data']

    for batch_idx, data in enumerate(train_loader):
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

        correct_ext = (exist_out == data.x[:, 0].unsqueeze(1)).sum() / torch.numel(data.x[:, 0])
        ext_acc += correct_ext
        iter_ct += 1

    return ext_acc / float(iter_ct), loss_sum / float(iter_ct)



def validation(model, epoch, val_loader, device, opt, loss_dict, scheduler):
    with torch.no_grad():
        model.eval()
        loss_all = 0
        ext_acc = 0
        iter_ct = 0
        batch_size = opt['batch_size']
        num_data = opt['num_data']
        loss_geo = 0.0

        for batch_idx, data in enumerate(val_loader):
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

            correct_ext = (exist_out == data.x[:, 0].unsqueeze(1)).sum() / torch.numel(data.x[:, 0])
            ext_acc += correct_ext
            iter_ct += 1

    return ext_acc / float(iter_ct), loss_all / float(iter_ct), loss_geo / float(iter_ct)
