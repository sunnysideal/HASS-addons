ARG BUILD_FROM=ghcr.io/home-assistant/amd64-base-python:3.19
FROM ${BUILD_FROM}

ENV \
    PYTHONUNBUFFERED=1 \
    LANG=C.UTF-8

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Prepare app directory
RUN mkdir -p /opt/app

# Copy application files
COPY requirements.txt /opt/app/requirements.txt
COPY mqtt_publisher.py /opt/app/mqtt_publisher.py
COPY run.sh /run.sh

# Install Python dependencies and prepare runtime
RUN python -m pip install --no-cache-dir -r /opt/app/requirements.txt \
    && chmod a+x /run.sh

WORKDIR /opt/app

CMD ["/run.sh"]
