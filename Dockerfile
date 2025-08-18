FROM pytorch/pytorch:2.4.1-cuda11.8-cudnn9-runtime

SHELL ["/bin/bash","-lc"]
ENV PATH=/opt/conda/bin:$PATH
ARG DEBIAN_FRONTEND=noninteractive
WORKDIR /app

COPY requirements.txt ./

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential cmake git \
 && rm -rf /var/lib/apt/lists/*

ENV CMAKE_BUILD_PARALLEL_LEVEL=4
RUN python -m pip install --upgrade pip \
 && pip install --no-cache-dir "setuptools<70" wheel "pybind11==2.10.4"

ARG TORCH=2.4.1
ARG CUDA=cu118
RUN pip install --no-cache-dir \
      torch-scatter==2.1.2 torch-sparse==0.6.18 torch-geometric==2.5.3 \
      -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html

RUN grep -Evi '^(torch_scatter|torch_sparse|torch-geometric|torch-scatter|torch-sparse|nvidia-.*-cu12)$' requirements.txt \
    > requirements_pip.txt && \
    pip install --no-cache-dir -r requirements_pip.txt && \
    rm requirements_pip.txt && \
    find /opt/conda -name '__pycache__' -type d -exec rm -rf {} +

COPY . .
EXPOSE 8000 6006
ENTRYPOINT ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
