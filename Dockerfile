FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

# CPU-only PyTorch keeps the image small; swap the index URL for a CUDA build if needed.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY configs ./configs
RUN pip install ".[netcdf]"

# Mount a volume at /app/data and /app/runs to keep downloads and results.
ENTRYPOINT ["flarenet"]
CMD ["--help"]
