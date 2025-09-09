from collections import defaultdict
import numpy as np
import yaml

try:
    import torch
except Exception as e:  # pragma: no cover - torch is required for runtime but may be missing in tests
    raise RuntimeError("torch is required to run this script") from e

from torch_geometric.loader import DataLoader
from huggingface_hub import hf_hub_download, snapshot_download

from urban_dataset import UrbanGraphDataset, test_graph_transform, get_transform
from model import (
    BlockGenerator,
    AttentionBlockGenerator,
    AttentionBlockGenerator_independent,
    AttentionBlockGenerator_independent_cnn,
    NaiveBlockGenerator,
)


def _load_options(repo: str) -> dict:
    cfg_path = hf_hub_download(repo_id=repo, filename="train_save.yaml")
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


def _build_model(opt: dict, repo: str, device: torch.device):
    template_h = opt.get("template_height", 1)
    template_w = opt.get("template_width", 1)
    N = template_h * template_w

    if opt.get("is_blockplanner"):
        model = NaiveBlockGenerator(opt, N=N)
    elif opt.get("is_conditional_block"):
        if opt.get("convlayer") in opt.get("attten_net", []):
            model = AttentionBlockGenerator(opt, N=N)
        else:
            model = BlockGenerator(opt, N=N)
    else:
        if opt.get("convlayer") in opt.get("attten_net", []):
            if opt.get("encode_cnn"):
                model = AttentionBlockGenerator_independent_cnn(opt, N=N)
            else:
                model = AttentionBlockGenerator_independent(opt, N=N)
        else:
            model = BlockGenerator(opt, N=N)

    weights_path = hf_hub_download(repo_id=repo, filename="model.pth")
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_repo = "Assandrro/genbuilder_2"
    data_repo = "Assandrro/genbuilder_data_128"

    opt = _load_options(model_repo)
    model = _build_model(opt, model_repo, device)

    data_dir = snapshot_download(repo_id=data_repo)
    dataset = UrbanGraphDataset(
        data_dir,
        transform=test_graph_transform,
        cnn_transform=get_transform(rescale_size=64),
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    zone_stats: dict[int, list[float]] = defaultdict(list)

    for data in loader:
        zone = int(getattr(data, "zone_id", -1))
        data = data.to(device)
        edge_index = data.edge_index
        node_cnt = data.x.size(0)

        with torch.no_grad():
            if opt.get("is_blockplanner"):
                mu, log_var = model.encode(data)
                z = model.reparameterize(mu, log_var)
                if opt.get("is_input_road"):
                    block_condition = data.block_condition.view(1, 2, 64, 64)
                    block_condition = model.cnn_encode(block_condition)
                    exist, _, _, _, _, _, _ = model.decode(z, block_condition, edge_index, node_cnt)
                else:
                    exist, _, _, _, _, _, _ = model.decode(z, edge_index, node_cnt)
            else:
                mu, log_var = model.encode(data)
                z = model.reparameterize(mu, log_var)
                exist, _, _, _, _, _, _ = model.decode(z, edge_index, node_cnt)

        exist_gt = data.x[:, 0]
        exist_pred = (exist.squeeze() > 0.5).float()
        acc = (exist_pred == exist_gt).float().mean().item()
        zone_stats[zone].append(acc)

    for zid, values in zone_stats.items():
        print(f"Zone {zid}: accuracy={np.mean(values):.4f}")


if __name__ == "__main__":
    main()
