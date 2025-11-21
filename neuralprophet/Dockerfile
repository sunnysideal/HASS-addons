# https://developers.home-assistant.io/docs/add-ons/configuration#add-on-dockerfile
ARG BUILD_FROM
FROM $BUILD_FROM

# Install build dependencies for compiling Python packages
RUN apk add --no-cache --virtual .build-deps \
    gcc \
    g++ \
    musl-dev \
    linux-headers \
    freetype-dev \
    libpng-dev \
    && apk add --no-cache \
    freetype \
    libpng

# Copy requirements.txt into the container
COPY requirements.txt /tmp/

# Install the required Python packages
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Remove build dependencies to reduce image size
RUN apk del .build-deps

# Copy root filesystem
COPY rootfs /

# Make service scripts executable
RUN chmod a+x /etc/services.d/neuralprophet/run /etc/services.d/neuralprophet/finish