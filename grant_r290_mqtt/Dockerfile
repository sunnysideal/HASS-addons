ARG BUILD_FROM=ghcr.io/home-assistant/amd64-base:3.19
FROM ${BUILD_FROM}

ENV \
    PYTHONUNBUFFERED=1 \
    LANG=C.UTF-8

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Install Python and dependencies
RUN apk add --no-cache python3 py3-pip \
    && python3 -m ensurepip \
    && pip3 install --no-cache-dir --upgrade pip \
    && mkdir -p /opt/app

# Copy application files
COPY requirements.txt /opt/app/requirements.txt
COPY upstream/mqtt_publisher.py /opt/app/mqtt_publisher.py
COPY run.sh /run.sh

# Install Python dependencies and prepare runtime
RUN pip3 install --no-cache-dir -r /opt/app/requirements.txt \
    && chmod a+x /run.sh

WORKDIR /opt/app

CMD ["/run.sh"]
