# pyplant: one image, two ways to run it (see docker-compose.yml).
#
#   docker build -t pyplant .
#   docker run --rm -p 8050:8050 pyplant          # the workbench GUI
#
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Dependencies first, in their own layer, so editing source does not reinstall
# scikit-learn and DuckDB on every build.
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY pyproject.toml README.md LICENSE ./
COPY pyplant ./pyplant
RUN pip install --no-deps .

# Run as an unprivileged user. /data holds the workspace: config, datasets,
# trained detectors and the historian database. Mount a volume there to keep
# them across container restarts.
RUN useradd --create-home --uid 1000 pyplant \
    && mkdir -p /data \
    && chown pyplant:pyplant /data
USER pyplant
VOLUME ["/data"]

EXPOSE 8050

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8050/', timeout=4)"

# The workbench supervises the plant, PLC, historian and HMI as subprocesses
# inside this container, on ports 5400-5405 that never leave it. Only the GUI
# is published.
CMD ["pyplant-workbench", "--host", "0.0.0.0", "--port", "8050", \
     "--workspace", "/data/workspace", "--base-port", "5400"]
