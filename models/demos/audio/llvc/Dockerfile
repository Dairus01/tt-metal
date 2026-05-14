# Dockerfile for Koyeb dashboard deployment of LLVC device tests
# Uses the official tt-metalium base image (ttnn + tt-metal pre-installed)
FROM ghcr.io/tenstorrent/tt-metal/tt-metalium-ubuntu-22.04-release-amd64:latest-rc

WORKDIR /app

# Install Python deps for testing and the tiny HTTP server
RUN pip install --no-cache-dir flask pytest gitpython loguru

# Only copy LLVC code (.dockerignore hides everything else)
COPY models/demos/audio/llvc /app/tt-metal/models/demos/audio/llvc

WORKDIR /app/tt-metal
ENV PYTHONPATH=/app/tt-metal

# Port for the HTTP health/check/run endpoints
EXPOSE 8080

CMD ["python", "models/demos/audio/llvc/koyeb_server.py"]
