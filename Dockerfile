FROM nvidia/cuda:13.0.0-devel-ubuntu24.04@sha256:1e8ac7a54c184a1af8ef2167f28fa98281892a835c981ebcddb1fad04bdd452d
COPY --from=ghcr.io/astral-sh/uv:0.13.6@sha256:af22a52f5eafef3d78e551599d405c02df61919d18028f0cea8eda93bc9986d0 /uv /usr/local/bin/uv
RUN apt-get update && apt-get install -y --no-install-recommends build-essential ca-certificates git curl && rm -rf /var/lib/apt/lists/*
ENV UV_PYTHON_INSTALL_DIR=/opt/uv-python UV_LINK_MODE=copy UV_NO_CACHE=1 CUDA_HOME=/usr/local/cuda
WORKDIR /opt/hibana
COPY . .
RUN chmod -R a+rX /opt/hibana \
    && test "$(git rev-parse HEAD)" = d474b74d5a892b2cd1cc8fc50cb17946e2a539bb \
    && uv sync --frozen
ENV PATH=/opt/hibana/.venv/bin:/usr/local/cuda/bin:$PATH
USER 65534:65534
RUN HOME=/tmp hibana --help >/dev/null
USER root
