ARG IMAGE_NAME=nvidia/cuda:11.1-devel-ubuntu20.04
FROM $IMAGE_NAME

ARG PROD_ENV=production
ARG COMPUTE_KERNEL=gpu
ARG DEBIAN_FRONTEND=noninteractive
ARG COMMIT_HASH=none

ENV PYTHONFAULTHANDLER=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=random \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    POETRY_VERSION=1.0.0 \
    COMMIT_HASH=$commit_hash \
    MLFLOW_S3_UPLOAD_EXTRA_ARGS='{"ACL": "bucket-owner-full-control"}' \
    MLFLOW_S3_ENDPOINT_URL=https://s3.eu-central-1.amazonaws.com/ \
    PYTHONPATH=/app/src \
    TOKENIZERS_PARALLELISM="false"


RUN apt-get update && apt-get install -y \
    curl \
    default-libmysqlclient-dev \
    gcc \
    g++ \
    htop \
    locales \
    python3-dev \
    git \
    python3-pip \
    && apt-get clean -y && rm -rf /var/lib/apt/lists/* 

# install poetry
RUN curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | POETRY_HOME=/opt/poetry python3 && \
    cd /usr/local/bin && \
    ln -s /opt/poetry/bin/poetry && \
    poetry config virtualenvs.create false

WORKDIR /app

COPY pyproject.toml poetry.lock /app/ 

RUN poetry install $(if [ "$PROD_ENV" = "production" ]; then echo --no-dev; fi) --no-interaction --no-ansi

RUN if [ "$COMPUTE_KERNEL" = "gpu" ]; then \
    poe force-cuda11 && \
    git clone https://github.com/NVIDIA/apex && \
    cd apex && \
    pip3 install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./ \
    ; fi

RUN python3 -m spacy download da_core_news_sm

COPY src /app/src

WORKDIR /app/src

ENTRYPOINT [ "/bin/bash" , "-c"]