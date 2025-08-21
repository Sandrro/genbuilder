import os
import json
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
from torch.nn.parameter import UninitializedParameter
from shapely.geometry import Polygon


class NaiveMsgPass(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='mean')
        self.lin = nn.Linear(in_channels * 2, out_channels)

    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j):
        tmp = torch.cat([x_i, x_j], dim=1)
        return self.lin(tmp)


class BlockGenerator(nn.Module):
    def __init__(self, opt, N=80, T=3, frequency_num=32, f_act='relu'):
        super().__init__()
        self.N = int(N)
        self.device = opt['device']
        self.latent_dim = int(opt['latent_dim'])
        self.latent_ch = int(opt['n_ft_dim'])
        self.T = int(T)
        self.blockshape_latent_dim = int(opt.get('block_latent_dim', 20))

        # Zoning conditioning
        self.cond_proj = nn.Sequential(
            nn.LazyLinear(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, self.latent_ch // 2),
        )

        # Pooling
        aggr = opt.get('aggr', 'Mean')
        if aggr == 'Mean':
            self.global_pool = torch_geometric.nn.global_mean_pool
        elif aggr == 'Max':
            self.global_pool = torch_geometric.nn.global_max_pool
        elif aggr == 'Add':
            self.global_pool = torch_geometric.nn.global_add_pool
        elif aggr == 'GlobalAttention':
            self.global_pool = torch_geometric.nn.GlobalAttention
        elif aggr == 'Set2Set':
            self.global_pool = torch_geometric.nn.Set2Set
        elif aggr == 'GraphMultisetTransformer':
            self.global_pool = torch_geometric.nn.GraphMultisetTransformer
        else:
            self.global_pool = torch_geometric.nn.global_mean_pool

        # Conv layer type
        conv_name = opt.get('convlayer', 'GCNConv')
        if conv_name == 'NaiveMsgPass':
            self.convlayer = NaiveMsgPass
        else:
            self.convlayer = getattr(torch_geometric.nn, conv_name, torch_geometric.nn.GCNConv)

        # Encoders
        self.ex_init = nn.Linear(2, self.latent_ch // 4)
        ft_in = (self.latent_ch // 4) + self.N + (self.latent_ch // 2)
        self.ft_init = nn.Linear(ft_in, self.latent_ch // 2)
        self.pos_init = nn.Linear(2, self.latent_ch // 2)
        self.size_init = nn.Linear(2, self.latent_ch // 2)

        # GNN encoder stack
        self.e_conv1 = self.convlayer(int(self.latent_ch * 2.0), self.latent_ch)
        self.e_conv2 = self.convlayer(self.latent_ch, self.latent_ch)
        self.e_conv3 = self.convlayer(self.latent_ch, self.latent_ch)

        # Decoder GNN stack
        self.d_conv1 = self.convlayer(self.latent_ch + self.N, self.latent_ch)
        self.d_conv2 = self.convlayer(self.latent_ch, self.latent_ch)
        self.d_conv3 = self.convlayer(self.latent_ch, self.latent_ch)

        self.d_ft_init = nn.Linear(self.latent_dim, self.latent_ch * self.N)

        # Dropout probability for residual blocks
        self.dropout_p = float(opt.get('dropout_p', 0.1))
        self.dropout = nn.Dropout(self.dropout_p)

        # Aggregate: g0(2*ch) + g1..g3(ch each) → 5*ch
        self.aggregate = nn.Linear(int(self.latent_ch * (2.0 + 3.0)), self.latent_dim)

        self.fc_mu = nn.Linear(self.latent_dim, self.latent_dim)
        self.fc_var = nn.Linear(self.latent_dim, self.latent_dim)

        # Decoders to targets
        self.d_exist_0 = nn.Linear(self.latent_ch, self.latent_ch)
        self.d_exist_1 = nn.Linear(self.latent_ch, 1)
        self.d_posx_0 = nn.Linear(self.latent_ch, self.latent_ch)
        self.d_posx_1 = nn.Linear(self.latent_ch, 1)
        self.d_posy_0 = nn.Linear(self.latent_ch, self.latent_ch)
        self.d_posy_1 = nn.Linear(self.latent_ch, 1)
        self.d_sizex_0 = nn.Linear(self.latent_ch, self.latent_ch)
        self.d_sizex_1 = nn.Linear(self.latent_ch, 1)
        self.d_sizey_0 = nn.Linear(self.latent_ch, self.latent_ch)
        self.d_sizey_1 = nn.Linear(self.latent_ch, 1)

        # Extra encoders/decoders
        self.enc_shape = nn.Linear(6, self.latent_ch // 4)
        self.enc_iou = nn.Linear(1, self.latent_ch // 4)
        self.d_shape_0 = nn.Linear(self.latent_ch, self.latent_ch)
        self.d_shape_1 = nn.Linear(self.latent_ch, 6)
        self.d_iou_0 = nn.Linear(self.latent_ch, self.latent_ch)
        self.d_iou_1 = nn.Linear(self.latent_ch, 1)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                w = getattr(m, "weight", None)
                if isinstance(w, UninitializedParameter):
                    continue
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))

    # ---------- helpers ----------
    def _one_hot_nodes(self, B: int, node_cnt: int):
        """Return one-hot encodings for the existing nodes in the batch.

        Args:
            B: Number of graphs in the batch.
            node_cnt: Total number of nodes across the batch.

        The output has ``self.N`` columns so that downstream linear layers
        expecting a fixed width still work, but only the rows corresponding to
        real nodes are returned.
        """
        eye = torch.eye(self.N, dtype=torch.float32, device=self.device)
        return eye.repeat(B, 1)[:node_cnt]

    # ---------- VAE bits ----------
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def _conv_dropout_residual(self, conv, x, edge_index):
        """Apply conv → ReLU → Dropout → (optional) residual."""
        out = self.dropout(F.relu(conv(x, edge_index)))
        if out.shape == x.shape:
            out = x + out
        return out

    # ---------- encoder/decoder ----------
    def encode(self, data, cond=None):
        x, edge_index = data.x, data.edge_index
        B = int(data.ptr.shape[0] - 1) if hasattr(data, 'ptr') and data.ptr is not None else 1

        # Node-wise attributes (may arrive as (B,N,feat) lists) → flatten to (B*N,feat)
        def flat2d(v, feat_dim):
            t = torch.as_tensor(v, dtype=torch.float32, device=self.device)
            if t.dim() == 3:  # (B,N,feat)
                t = t.view(B * self.N, t.size(-1))
            elif t.dim() == 1:  # (B*N,) → (B*N,feat)
                t = t.view(-1, feat_dim)
            return t

        pos_org = flat2d(getattr(data, 'org_node_pos', getattr(data, 'node_pos', None)), 2)
        size_org = flat2d(getattr(data, 'org_node_size', getattr(data, 'node_size', None)), 2)
        b_shape = flat2d(getattr(data, 'b_shape', None), 6)
        b_iou = flat2d(getattr(data, 'b_iou', None), 1)

        b_shape = self.enc_shape(b_shape)
        b_iou = self.enc_iou(b_iou)
        shape_feature = torch.cat((b_shape, b_iou), 1)

        pos = F.relu(self.pos_init(pos_org))
        size = F.relu(self.size_init(size_org))

        x = self.ex_init(x)
        node_cnt = x.size(0)
        one_hot = self._one_hot_nodes(B, node_cnt)

        x = torch.cat([x, one_hot], 1)
        if cond is not None:
            node_cond = cond[data.batch] if hasattr(data, 'batch') else cond.repeat(x.size(0), 1)
            cond_emb = self.cond_proj(node_cond)
        else:
            cond_emb = torch.zeros(x.size(0), self.latent_ch // 2, device=self.device)
        x = torch.cat([x, cond_emb], 1)

        ft = F.relu(self.ft_init(x))

        n0 = torch.cat((shape_feature, size, pos, ft), 1)
        n1 = self._conv_dropout_residual(self.e_conv1, n0, edge_index)
        n2 = self._conv_dropout_residual(self.e_conv2, n1, edge_index)
        n3 = self._conv_dropout_residual(self.e_conv3, n2, edge_index)

        g0 = self.global_pool(n0, data.batch)
        g1 = self.global_pool(n1, data.batch)
        g2 = self.global_pool(n2, data.batch)
        g3 = self.global_pool(n3, data.batch)

        g = torch.cat((g0, g1, g2, g3), 1)
        zhid = self.aggregate(g)
        mu = self.fc_mu(zhid)
        log_var = self.fc_var(zhid)
        return [mu, log_var]

    def decode(self, z, edge_index, node_cnt):
        B = z.size(0)
        z = self.d_ft_init(z).view(B * self.N, -1)[:node_cnt]
        one_hot = self._one_hot_nodes(B, node_cnt)
        z = torch.cat([z, one_hot], 1)

        d1 = self._conv_dropout_residual(self.d_conv1, z, edge_index)
        d2 = self._conv_dropout_residual(self.d_conv2, d1, edge_index)
        d3 = self._conv_dropout_residual(self.d_conv3, d2, edge_index)

        exist = self.d_exist_1(F.relu(self.d_exist_0(d3)))
        posx = self.d_posx_1(F.relu(self.d_posx_0(d3)))
        posy = self.d_posy_1(F.relu(self.d_posy_0(d3)))
        sizex = self.d_sizex_1(F.relu(self.d_sizex_0(d3)))
        sizey = self.d_sizey_1(F.relu(self.d_sizey_0(d3)))
        b_shape = self.d_shape_1(F.relu(self.d_shape_0(d3)))
        b_iou = self.d_iou_1(F.relu(self.d_iou_0(d3)))
        return exist, posx, posy, sizex, sizey, b_shape, b_iou

    def forward(self, data, cond=None):
        mu, log_var = self.encode(data, cond=cond)
        z = self.reparameterize(mu, log_var)
        node_cnt = data.x.size(0)
        exist, px, py, sx, sy, bshape, biou = self.decode(z, data.edge_index, node_cnt)
        pos = torch.cat((px, py), 1)
        size = torch.cat((sx, sy), 1)
        return exist, pos, size, mu, log_var, bshape, biou

    # --------------------------------------------------------------
    def infer(self, block: Polygon, n: int = 5, zone_label: str | None = None) -> List[Polygon]:
        """Generate ``n`` building polygons inside ``block`` in canonical frame.

        Parameters
        ----------
        block:
            Canonical block polygon (already rotated/scaled/centred).
        n:
            Number of buildings to attempt to generate.
        zone_label:
            Optional zoning label used for conditioning when a zone mapping is
            available.
        """

        from transform import block_long_side  # avoid heavy import at module load

        long_side = float(block_long_side(block))

        # --- build minimal input graph ---
        node_cnt = int(max(0, n))
        x = torch.zeros((node_cnt, 2), dtype=torch.float32, device=self.device)
        node_pos = torch.zeros((node_cnt, 2), dtype=torch.float32, device=self.device)
        node_size = torch.zeros((node_cnt, 2), dtype=torch.float32, device=self.device)
        b_shape = torch.zeros((node_cnt, 6), dtype=torch.float32, device=self.device)
        b_iou = torch.zeros((node_cnt, 1), dtype=torch.float32, device=self.device)

        # simple chain edges (bidirectional)
        edges: list[list[int]] = []
        for i in range(node_cnt - 1):
            edges.append([i, i + 1])
            edges.append([i + 1, i])
        if edges:
            edge_index = torch.tensor(edges, dtype=torch.long, device=self.device).t().contiguous()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=self.device)

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
        data.batch = torch.zeros(node_cnt, dtype=torch.long, device=self.device)

        # --- zoning conditioning ---
        cond = None
        cond_dim = 0
        try:
            cond_dim = int(self.cond_proj[0].weight.shape[1])
        except Exception:
            cond_dim = 0

        if cond_dim > 0:
            cond = torch.zeros((1, cond_dim), dtype=torch.float32, device=self.device)
            if zone_label is not None:
                if not hasattr(self, "_zone_map_loaded"):
                    self._zone_map_loaded = True
                    self._zone_map: dict[str, int] = {}
                    for p in ("_zones_map.json", "processed/_zones_map.json"):
                        if os.path.isfile(p):
                            try:
                                with open(p, "r", encoding="utf-8") as fh:
                                    raw = json.load(fh)
                                mapping = raw.get("map", raw)
                                self._zone_map = {str(k): int(v) for k, v in mapping.items()}
                            except Exception:
                                self._zone_map = {}
                            break
                zid = self._zone_map.get(str(zone_label)) if hasattr(self, "_zone_map") else None
                if zid is not None and 0 <= int(zid) < cond_dim:
                    cond[0, int(zid)] = 1.0

        # --- run model ---
        self.eval()
        with torch.no_grad():
            exist, pos, size, _, _, _, _ = self.forward(data, cond=cond)

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


class AttentionBlockGenerator(BlockGenerator):
    def __init__(self, opt, N=120, T=3, frequency_num=32, f_act='relu'):
        super().__init__(opt, N, T, frequency_num, f_act)
        self.head = int(opt.get('head_num', 1))
        self.concat_heads = bool(opt.get('concat_heads', False))

        kind = opt.get('convlayer', 'GATConv')
        heady = {'GATConv','GATv2Conv','TransformerConv','GPSConv','SuperGATConv'}

        def make_conv(in_ch, out_ch):
            if kind in {'ChebConv','SAGEConv','GraphConv','GravNetConv','GatedGraphConv','ResGatedGraphConv','GCNConv'}:
                Conv = getattr(torch_geometric.nn, kind, torch_geometric.nn.GCNConv)
                if kind == 'GatedGraphConv':
                    return Conv(out_ch)
                return Conv(in_ch, out_ch)
            elif kind in heady:
                Conv = getattr(torch_geometric.nn, kind)
                return Conv(in_ch, out_ch, heads=self.head, concat=self.concat_heads)
            else:
                return torch_geometric.nn.GCNConv(in_ch, out_ch)

        # Encoder convs
        self.e_conv1 = make_conv(int(self.latent_ch * 2.0), self.latent_ch)
        last_w = self.latent_ch if not self.concat_heads else self.latent_ch * self.head
        self.e_conv2 = make_conv(last_w, self.latent_ch)
        self.e_conv3 = make_conv(last_w, self.latent_ch)

        # Decoder convs
        self.d_conv1 = make_conv(self.latent_ch + self.N, self.latent_ch)
        self.d_conv2 = make_conv(last_w, self.latent_ch)
        self.d_conv3 = make_conv(last_w, self.latent_ch)

        # Decoder heads width depends on last conv
        last_w = self.latent_ch if not self.concat_heads else self.latent_ch * self.head
        self.d_exist_1 = nn.Linear(last_w, 1)
        self.d_posx_0 = nn.Linear(last_w, self.latent_ch)
        self.d_posx_1 = nn.Linear(self.latent_ch, 1)
        self.d_posy_0 = nn.Linear(last_w, self.latent_ch)
        self.d_posy_1 = nn.Linear(self.latent_ch, 1)
        self.d_sizex_0 = nn.Linear(last_w, self.latent_ch)
        self.d_sizex_1 = nn.Linear(self.latent_ch, 1)
        self.d_sizey_0 = nn.Linear(last_w, self.latent_ch)
        self.d_sizey_1 = nn.Linear(self.latent_ch, 1)
        self.d_shape_0 = nn.Linear(last_w, self.latent_ch)
        self.d_shape_1 = nn.Linear(self.latent_ch, 6)
        self.d_iou_0 = nn.Linear(last_w, self.latent_ch)
        self.d_iou_1 = nn.Linear(self.latent_ch, 1)

        # Aggregate: g0(2*ch) + g1..g3(last_w)
        agg_in = int(self.latent_ch * 2 + 3 * (self.latent_ch if not self.concat_heads else self.latent_ch * self.head))
        self.aggregate = nn.Linear(agg_in, self.latent_dim)


class AttentionBlockGenerator_independent(AttentionBlockGenerator):
    def __init__(self, opt, N=80, T=3, frequency_num=32, f_act='relu'):
        super().__init__(opt, N, T, frequency_num, f_act)
        kind = opt.get('convlayer', 'GATConv')
        heady = {'GATConv','GATv2Conv','TransformerConv','GPSConv','SuperGATConv'}

        def make_conv(in_ch, out_ch):
            if kind in {'ChebConv','SAGEConv','GraphConv','GravNetConv','GatedGraphConv','ResGatedGraphConv','GCNConv'}:
                Conv = getattr(torch_geometric.nn, kind, torch_geometric.nn.GCNConv)
                if kind == 'GatedGraphConv':
                    return Conv(out_ch)
                return Conv(in_ch, out_ch)
            elif kind in heady:
                Conv = getattr(torch_geometric.nn, kind)
                return Conv(in_ch, out_ch, heads=self.head, concat=self.concat_heads)
            else:
                return torch_geometric.nn.GCNConv(in_ch, out_ch)

        self.e_conv1 = make_conv(int(self.latent_ch * 2.0), self.latent_ch)
        last_w = self.latent_ch if not self.concat_heads else self.latent_ch * self.head
        self.e_conv2 = make_conv(last_w, self.latent_ch)
        self.e_conv3 = make_conv(last_w, self.latent_ch)

        # Aggregate: g0 (2*ch) + g1..g3(last_w)
        agg_in = int(self.latent_ch * 2 + 3 * (self.latent_ch if not self.concat_heads else self.latent_ch * self.head))
        self.aggregate = nn.Linear(agg_in, self.latent_dim)

        # Variant-specific extras
        self.d_ft_init = nn.Linear(self.latent_dim + 40, self.latent_ch * self.N)
        self.enc_block_scale = nn.Linear(1, 20)

    # Robust encode: also flatten (B,N,*) → (B*N,*)
    def encode(self, data, cond=None):
        x, edge_index = data.x, data.edge_index
        B = int(data.ptr.shape[0] - 1) if hasattr(data, 'ptr') and data.ptr is not None else 1

        def flat2d(v, feat_dim):
            t = torch.as_tensor(v, dtype=torch.float32, device=self.device)
            if t.dim() == 3:
                t = t.view(B * self.N, t.size(-1))
            elif t.dim() == 1:
                t = t.view(-1, feat_dim)
            return t

        pos_org = flat2d(getattr(data, 'node_pos', getattr(data, 'org_node_pos', None)), 2)
        size_org = flat2d(getattr(data, 'node_size', getattr(data, 'org_node_size', None)), 2)
        b_shape = flat2d(getattr(data, 'b_shape', None), 6)
        b_iou = flat2d(getattr(data, 'b_iou', None), 1)

        b_shape = self.enc_shape(b_shape)
        b_iou = self.enc_iou(b_iou)
        shape_feature = torch.cat((b_shape, b_iou), 1)

        pos = F.relu(self.pos_init(pos_org))
        size = F.relu(self.size_init(size_org))

        x = self.ex_init(x)
        node_cnt = x.size(0)
        one_hot = self._one_hot_nodes(B, node_cnt)

        x = torch.cat([x, one_hot], 1)
        if cond is not None:
            node_cond = cond[data.batch] if hasattr(data, 'batch') else cond.repeat(x.size(0), 1)
            cond_emb = self.cond_proj(node_cond)
        else:
            cond_emb = torch.zeros(x.size(0), self.latent_ch // 2, device=self.device)
        x = torch.cat([x, cond_emb], 1)

        ft = F.relu(self.ft_init(x))

        n0 = torch.cat((shape_feature, size, pos, ft), 1)
        n1 = self._conv_dropout_residual(self.e_conv1, n0, edge_index)
        n2 = self._conv_dropout_residual(self.e_conv2, n1, edge_index)
        n3 = self._conv_dropout_residual(self.e_conv3, n2, edge_index)

        g0 = self.global_pool(n0, data.batch)
        g1 = self.global_pool(n1, data.batch)
        g2 = self.global_pool(n2, data.batch)
        g3 = self.global_pool(n3, data.batch)

        g = torch.cat((g0, g1, g2, g3), 1)
        zhid = self.aggregate(g)
        mu = self.fc_mu(zhid)
        log_var = self.fc_var(zhid)
        return [mu, log_var]

    def decode(self, z, block_condition, edge_index, node_cnt):
        B = z.shape[0]
        z = torch.cat((z, block_condition), 1)
        z = self.d_ft_init(z).view(B * self.N, -1)[:node_cnt]
        one_hot = self._one_hot_nodes(B, node_cnt)
        z = torch.cat([z, one_hot], 1)
        d1 = self._conv_dropout_residual(self.d_conv1, z, edge_index)
        d2 = self._conv_dropout_residual(self.d_conv2, d1, edge_index)
        d3 = self._conv_dropout_residual(self.d_conv3, d2, edge_index)
        exist = self.d_exist_1(d3)
        posx = self.d_posx_1(F.relu(self.d_posx_0(d3)))
        posy = self.d_posy_1(F.relu(self.d_posy_0(d3)))
        sizex = self.d_sizex_1(F.relu(self.d_sizex_0(d3)))
        sizey = self.d_sizey_1(F.relu(self.d_sizey_0(d3)))
        b_shape = self.d_shape_1(F.relu(self.d_shape_0(d3)))
        b_iou = self.d_iou_1(F.relu(self.d_iou_0(d3)))
        return exist, posx, posy, sizex, sizey, b_shape, b_iou

    def forward(self, data, cond=None):
        mu, log_var = self.encode(data, cond=cond)
        z = self.reparameterize(mu, log_var)
        B = int(data.ptr.shape[0] - 1) if hasattr(data, 'ptr') and data.ptr is not None else 1
        block_scale = self.enc_block_scale(data.block_scale_gt.unsqueeze(1))
        block_shape = data.blockshape_latent_gt.view(-1, self.blockshape_latent_dim)
        block_condition = torch.cat((block_shape, block_scale), 1)
        node_cnt = data.x.size(0)
        exist, px, py, sx, sy, bshape, biou = self.decode(z, block_condition, data.edge_index, node_cnt)
        pos = torch.cat((px, py), 1)
        size = torch.cat((sx, sy), 1)
        return exist, pos, size, mu, log_var, bshape, biou


class AttentionBlockGenerator_independent_cnn(AttentionBlockGenerator_independent):
    def __init__(self, opt, N=80, T=3, frequency_num=32, f_act='relu', bottleneck=128, image_size=64, inner_channel=80):
        super().__init__(opt, N, T, frequency_num, f_act)
        channel_num = int((image_size / 2 ** 4) ** 2 * inner_channel)
        self.inner_channel = int(inner_channel)
        self.image_size = int(image_size)
        self.linear1 = nn.Linear(channel_num, bottleneck)
        self.bottleneck = int(bottleneck)
        self.cnn_encoder = nn.Sequential(
            nn.Conv2d(2, self.inner_channel // 8, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(self.inner_channel // 8, self.inner_channel // 4, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(self.inner_channel // 4, self.inner_channel // 2, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(self.inner_channel // 2, self.inner_channel, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2)
        )
        self.d_ft_init = nn.Linear(self.latent_dim + bottleneck, self.latent_ch * self.N)

    def cnn_encode(self, x):
        x = self.cnn_encoder(x)
        x = torch.flatten(x, 1)
        return self.linear1(x)

    def forward(self, data, cond=None):
        B = int(data.ptr.shape[0] - 1) if hasattr(data, 'ptr') and data.ptr is not None else 1
        mu, log_var = self.encode(data, cond=cond)
        z = self.reparameterize(mu, log_var)
        block_condition = data.block_condition.view(B, 2, 64, 64)
        block_condition = self.cnn_encode(block_condition)
        node_cnt = data.x.size(0)
        exist, px, py, sx, sy, bshape, biou = self.decode(z, block_condition, data.edge_index, node_cnt)
        pos = torch.cat((px, py), 1)
        size = torch.cat((sx, sy), 1)
        return exist, pos, size, mu, log_var, bshape, biou
