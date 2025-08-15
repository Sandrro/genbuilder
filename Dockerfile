FROM pytorch/pytorch:2.4.1-cuda11.8-cudnn9-runtime

SHELL ["/bin/bash","-lc"]
ENV PATH=/opt/conda/bin:$PATH
WORKDIR /app

COPY requirements.txt ./

# 1) Системные зависимости для CGAL (как у тебя уже есть)
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
      build-essential cmake libcgal-dev libcgal-qt5-dev libeigen3-dev libboost-all-dev \
      libgmp-dev libmpfr-dev \
 && rm -rf /var/lib/apt/lists/*

# 2) Современная связка для Python 3.11
ENV CMAKE_BUILD_PARALLEL_LEVEL=4
RUN python -m pip install --upgrade pip \
 && pip install --no-cache-dir "setuptools<70" wheel \
 && pip install --no-cache-dir "pybind11>=2.11,<2.13"

# 3) Ставим skgeom (можно с PEP517 или без изоляции — нам главное, чтобы взялся новый pybind11)
# Вариант А: без изоляции
RUN pip install --no-cache-dir --no-build-isolation \
      https://github.com/scikit-geometry/scikit-geometry/archive/refs/tags/0.1.2.tar.gz

# 4) PyG под torch 2.4.0 + cu118
ARG TORCH=2.4.0
ARG CUDA=cu118
RUN pip install --no-cache-dir \
      torch-scatter -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html && \
    pip install --no-cache-dir \
      torch-sparse  -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html && \
    pip install --no-cache-dir torch-geometric

# 5) Остальные зависимости (как раньше: без skgeom/PyG и без cu12-бандлов)
RUN grep -Evi '^(skgeom|torch_scatter|torch_sparse|torch-geometric|torch-scatter|torch-sparse|nvidia-.*-cu12)$' requirements.txt > requirements_pip.txt && \
    pip install --no-cache-dir -r requirements_pip.txt && \
    rm requirements_pip.txt && \
    find /opt/conda -name '__pycache__' -type d -exec rm -rf {} +

COPY . .
EXPOSE 8000 6006
ENTRYPOINT ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
