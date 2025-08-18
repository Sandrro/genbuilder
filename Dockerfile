FROM pytorch/pytorch:2.4.1-cuda11.8-cudnn9-runtime

SHELL ["/bin/bash","-lc"]
WORKDIR /app

COPY requirements.txt ./

ARG TORCH=2.4.1
ARG CUDA=cu118
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir \
      torch-scatter==2.1.2 torch-sparse==0.6.18 torch-geometric==2.5.3 \
      -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html && \
    pip install --no-cache-dir -r requirements.txt && \
    find /opt/conda -name '__pycache__' -type d -exec rm -rf {} +

COPY . .
EXPOSE 8000 6006
ENTRYPOINT ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
