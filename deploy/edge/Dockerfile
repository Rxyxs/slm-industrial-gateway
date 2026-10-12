# Imagen del gateway de planta (src/gateway).
#
# El gateway no ejecuta el modelo: lo consume por HTTP desde el contenedor de
# inferencia (Ollama / vLLM / trtllm-serve), que es el que necesita CUDA. Por eso
# esta imagen es CPU-only y multi-arquitectura: el mismo Dockerfile construye
# para un IPC x86_64 y para un Jetson (linux/arm64):
#
#   docker buildx build --platform linux/amd64,linux/arm64 -f deploy/edge/Dockerfile .
#
# La aceleración CUDA/TensorRT vive en docker-compose.yml (servicio `ollama` con
# reserva de GPU, o el perfil `vllm`).

ARG PYTHON_IMAGE=python:3.10-slim

FROM ${PYTHON_IMAGE} AS builder
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /build
COPY requirements-gateway.txt .
RUN pip wheel --wheel-dir /wheels -r requirements-gateway.txt

FROM ${PYTHON_IMAGE}
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8080 \
    AUDIT_DB_PATH=/var/lib/slmgw/audit/gateway_audit.duckdb
WORKDIR /app

COPY --from=builder /wheels /wheels
RUN pip install --no-index --find-links=/wheels /wheels/* && rm -rf /wheels

COPY src/gateway ./src/gateway

RUN useradd --system --uid 10001 --create-home slmgw \
    && mkdir -p /var/lib/slmgw/audit /etc/slmgw/tls \
    && chown -R slmgw:slmgw /var/lib/slmgw
USER slmgw

EXPOSE 8080
HEALTHCHECK --interval=15s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import os,urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/health', timeout=2)" || exit 1

CMD ["python", "-m", "src.gateway.serve"]
