# Purpose: Build a production-oriented Python 3.12 image for the LLM benchmark orchestrator.
# Input/Output: Takes the local source tree and produces a container with the `benchmark` CLI installed.
# Invariants: Runtime writes happen only in `/app/results`; configuration stays external under `/config`.
# Debugging: Rebuild with `docker build --no-cache .` if dependencies or copied assets look stale.

FROM python:3.12-slim AS runtime

ARG BUILD_DATE=unknown
ARG VCS_REF=unknown
ARG VERSION=0.2.0

LABEL org.opencontainers.image.title="LLM Benchmark" \
      org.opencontainers.image.description="Production-oriented Docker and Unraid friendly benchmark orchestrator for OpenAI-compatible LLM endpoints." \
      org.opencontainers.image.source="https://github.com/Feberdin/LLM-Benchmark" \
      org.opencontainers.image.url="https://github.com/Feberdin/LLM-Benchmark" \
      org.opencontainers.image.documentation="https://github.com/Feberdin/LLM-Benchmark#readme" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.created="${BUILD_DATE}" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.version="${VERSION}"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    BENCHMARK_CONFIG=/config/config.example.yaml \
    RESULTS_DIR=/app/results \
    LOG_LEVEL=INFO

WORKDIR /app

RUN apt-get update \
    && apt-get install --yes --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md CONTRIBUTING.md LICENSE ./
COPY src ./src
COPY docker/entrypoint.sh /entrypoint.sh
COPY fixtures ./fixtures
COPY unraid ./unraid

RUN pip install --upgrade pip \
    && pip install .

RUN mkdir -p /config /app/tests /app/results \
    && cp -R /app/fixtures/tests/. /app/tests/ \
    && cp /app/fixtures/config/config.example.yaml /config/config.example.yaml \
    && cp /app/fixtures/config/config.unraid.example.yaml /config/config.unraid.example.yaml \
    && chmod +x /entrypoint.sh \
    && useradd --create-home --shell /usr/sbin/nologin benchmark \
    && chown -R benchmark:benchmark /config /app/tests /app/results /entrypoint.sh

USER benchmark

EXPOSE 8080

VOLUME ["/config", "/app/tests", "/app/results"]

ENTRYPOINT ["/entrypoint.sh"]
CMD []
