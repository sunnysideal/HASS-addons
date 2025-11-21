# https://developers.home-assistant.io/docs/add-ons/configuration#add-on-dockerfile
ARG BUILD_FROM
FROM $BUILD_FROM

# Copy requirements.txt into the container
COPY requirements.txt /tmp/

# Install the required Python packages
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Copy root filesystem
COPY rootfs /

RUN chmod a+x /run.sh
