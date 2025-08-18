# GlobalMapper: Arbitrary-Shaped Urban Layout Generation
Official Pytorch Implementation of "GlobalMapper: Arbitrary-Shaped Urban Layout Generation"

[arXiv](https://arxiv.org/abs/2307.09693) | [BibTeX](#bibtex) | [Project Page](https://arking1995.github.io/GlobalMapper/)

This repo contains codes for single GPU training for 
[GlobalMapper: Arbitrary-Shaped Urban Layout Generation](https://arxiv.org/pdf/2307.09693.pdf)

**Note that this repo is lack of code comments.**

## Environment
We provide required environments in "environment.yml". But practially we suggest to use below commands for crucial dependencies:
```
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-1.9.0+cu111.html
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.9.0+cu111.html
pip install torch-geometric
```
Then you may install other dependencies like: matplotlib, yaml, pickle, etc.

## Docker & Docker Compose

For remote experiments the project ships with a `Dockerfile` and

`docker-compose.yml` that expose a FastAPI service wrapping training and
evaluation.

1. **Build the image**

   ```bash
   docker compose build
   ```

   The Dockerfile strips Python `__pycache__` directories after installing
   dependencies to keep the image compact and speed up the final
   *"unpacking"* stage.

2. **Launch the API server**

   ```bash
   docker compose up
   ```

   The service listens on `http://localhost:8000`.

3. **Upload dataset files (optional)**

   You can still upload multiple `.arrow` files (HF converts them to `.parquet` shards) and (optionally) `_zones_map.json` before training:

   ```bash
   curl -X POST http://localhost:8000/data \
        -F "files=@my_dataset/processed/quarter1.arrow" \
        -F "files=@my_dataset/processed/quarter2.arrow" \
        -F "files=@my_dataset/processed/_zones_map.json"
   ```

4. **Upload training config**

   ```bash
   curl -X POST http://localhost:8000/config \
        -F "file=@train_gnn.yaml"
   ```

5. **Start training**

   ```bash
   curl -X POST http://localhost:8000/train \
        -H 'Content-Type: application/json' \
        -d '{"dataset":"my_dataset","dataset_repo":"<hf_dataset_repo>","upload_repo":"<hf_model_repo>","config":"train_gnn.yaml","hf_token":"<token>"}'
   ```

6. **Run evaluation**

   ```bash
   curl -X POST http://localhost:8000/test \
        -H 'Content-Type: application/json' \
        -d '{"dataset":"my_dataset","dataset_repo":"<hf_dataset_repo>","model_repo":"<hf_model_repo>","config":"train_gnn.yaml","hf_token":"<token>"}'
   ```

7. **Inspect logs**

   ```bash
   curl http://localhost:8000/logs
   curl http://localhost:8000/logs/<logfile>
   ```

Artifacts and logs are written to the host `epoch/`, `logs/` and
`tensorboard/` directories thanks to volume mounts.


## Dataset
Use the `data_to_hf.py` script to upload processed graphs to a HuggingFace dataset repository:

```bash
python data_to_hf.py --repo <user/dataset> --token <hf_token>
# for large datasets
python data_to_hf.py --repo <user/dataset> --token <hf_token> --large
```

During training or testing the data can be downloaded automatically by providing the repository to `run_pipeline.py` (or via the API's `dataset_repo` field). After training, the best checkpoint and its log file can be pushed to a HuggingFace model repository using `--upload_repo` and later restored for evaluation with `--model_repo`.

"processed" folder contains preprocessed graph-represented city blocks stored as `.arrow` files (or `.parquet` shards when downloaded from HuggingFace Datasets) with pickled `networkx` graphs inside. `raw_geo` contains corresponding original building and block polygons (shapely.polygon format) of each city block (coordinates in UTM Zone projection) readable by `pickle.load()`.

Our canonical spatial transformation converts the original building polygons to the canonical version. After simple normalization by mean subtraction and std dividing, coordinates and location information are encoded as node attributes in 2D grid graphs, then saved in `processed`. Since the raw dataset is publicly accessible, we encourage users to implement their own preprocessing of original building polygons. It may facilitate better performance.


## How to train your model
After set up your training parameters in "train_gnn.yaml". Simply run
```
python train.py
```


## How to test your model
After you setup desired "dataset_path" and "epoch_name". Simply run
```
python test.py
```

## How to do canonical spatial transformation
We provide a simple example in "example_canonical_transform.py". Details are provieded in our Supplemental [Supp](https://openaccess.thecvf.com/content/ICCV2023/supplemental/He_GlobalMapper_Arbitrary-Shaped_Urban_ICCV_2023_supplemental.pdf). We encourage users to commit their own realization.


## How to do visualization
All maps in the paper are visualized by simple matplotlib draw functions that you may compose in minutes.


## BibTeX

If you use this code, please cite
```text
@InProceedings{He_2023_ICCV,
    author    = {He, Liu and Aliaga, Daniel},
    title     = {GlobalMapper: Arbitrary-Shaped Urban Layout Generation},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2023},
    pages     = {454-464}
}
```

