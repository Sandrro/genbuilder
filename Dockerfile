FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 LC_ALL=C.UTF-8 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHON_VERSION=3.9.23

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
      curl ca-certificates wget \
      gdal-bin libgdal-dev \
      libgeos-dev libproj-dev \
      libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev \
      libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev \
  && rm -rf /var/lib/apt/lists/*

RUN curl -fsSLO https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz \
  && tar -xzf Python-${PYTHON_VERSION}.tgz \
  && cd Python-${PYTHON_VERSION} \
  && ./configure --enable-optimizations --with-ensurepip=install \
  && make -j"$(nproc)" \
  && make altinstall \
  && cd .. \
  && rm -rf Python-${PYTHON_VERSION} Python-${PYTHON_VERSION}.tgz \
  && ln -sf /usr/local/bin/python3.9 /usr/local/bin/python3 \
  && ln -sf /usr/local/bin/python3.9 /usr/local/bin/python \
  && python -m pip install --upgrade pip

ARG TORCH_VER=2.1.2
ARG TORCHVISION_VER=0.16.2
ARG TORCHAUDIO_VER=2.1.2
ARG PYG_INDEX="https://data.pyg.org/whl/torch-2.1.2+cu121.html"

COPY requirements-docker.txt /app/

RUN pip install -r requirements-docker.txt

RUN pip install --index-url https://download.pytorch.org/whl/cu121 \
      torch==${TORCH_VER} torchvision==${TORCHVISION_VER} torchaudio==${TORCHAUDIO_VER}

RUN pip install \
      torch-scatter \
      torch-sparse \
      torch-cluster \
      torch-spline-conv \
      -f ${PYG_INDEX} \
  && pip install torch-geometric

COPY service ./service
COPY scripts ./scripts
COPY train.py ./train.py
COPY inference.py ./inference.py
COPY train_gnn.yaml ./train_gnn.yaml
COPY building_shape_params.yaml ./building_shape_params.yaml
COPY ./out_2/checkpoints/artifacts ./artifacts/

ENV PYTHONPATH=/app
EXPOSE 8000

CMD ["uvicorn", "service.main:app", "--host", "0.0.0.0", "--port", "8000"]
