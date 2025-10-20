# Базовый образ NVIDIA CUDA с cuDNN (runtime), Ubuntu 22.04 — максимально дефолтная база
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 LC_ALL=C.UTF-8 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Ставим системный Python, pip и базовые системные библиотеки (Geo stack — как у вас)
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-pip python3-dev \
      build-essential \
      gdal-bin libgdal-dev \
      libgeos-dev libproj-dev \
      curl ca-certificates \
  && ln -sf /usr/bin/python3 /usr/bin/python \
  && python -m pip install --upgrade pip \
  && rm -rf /var/lib/apt/lists/*

# ==== ВЕРСИИ TORCH ПОД CUDA 12.1 (строго согласованы с PyG) ====
ARG TORCH_VER=2.1.2
ARG TORCHVISION_VER=0.16.2
ARG TORCHAUDIO_VER=2.1.2
ARG PYG_INDEX="https://data.pyg.org/whl/torch-2.1.2+cu121.html"

# Сначала — проектные зависимости для лучшего кэширования
COPY requirements-docker.txt /app/

RUN pip install -r requirements-docker.txt

# Затем — PyTorch под cu121 (официальный индекс)
RUN pip install --index-url https://download.pytorch.org/whl/cu121 \
      torch==${TORCH_VER} torchvision==${TORCHVISION_VER} torchaudio==${TORCHAUDIO_VER}

# И — бинарные колёса PyG, точно под ту же связку torch+cu121
RUN pip install \
      torch-scatter \
      torch-sparse \
      torch-cluster \
      torch-spline-conv \
      -f ${PYG_INDEX} \
  && pip install torch-geometric

# Код и конфиги
COPY service ./service
COPY scripts ./scripts
COPY train.py ./train.py
COPY inference.py ./inference.py
COPY train_gnn.yaml ./train_gnn.yaml
COPY building_shape_params.yaml ./building_shape_params.yaml
COPY ./out_2/checkpoints/artifacts ./artifacts/

ENV PYTHONPATH=/app
EXPOSE 8000

# Запуск FastAPI/uvicorn
CMD ["uvicorn", "service.main:app", "--host", "0.0.0.0", "--port", "8000"]
