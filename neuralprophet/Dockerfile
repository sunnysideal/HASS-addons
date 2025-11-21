# https://developers.home-assistant.io/docs/add-ons/configuration#add-on-dockerfile
ARG BUILD_FROM
FROM $BUILD_FROM

# Copy root filesystem
COPY rootfs /

# Make service scripts executable
RUN chmod a+x /etc/services.d/neuralprophet/run /etc/services.d/neuralprophet/finish