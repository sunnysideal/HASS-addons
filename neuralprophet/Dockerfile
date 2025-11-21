# https://developers.home-assistant.io/docs/add-ons/configuration#add-on-dockerfile
ARG BUILD_FROM
FROM $BUILD_FROM

# Note: If using pre-built base image, uncomment below and comment out BUILD_FROM above
ARG BUILD_ARCH=amd64
FROM ghcr.io/sunnysideal/neuralprophet-base:${BUILD_ARCH}

# Copy root filesystem
COPY rootfs /

# Make service scripts executable
RUN chmod a+x /etc/services.d/neuralprophet/run /etc/services.d/neuralprophet/finish