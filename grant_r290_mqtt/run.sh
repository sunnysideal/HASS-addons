#!/usr/bin/with-contenv bashio
# shellcheck shell=bash
set -euo pipefail

log() {
    echo "[addon] $*"
}

# Load configuration
ECONET_ENDPOINT=$(bashio::config 'econet_endpoint')
MQTT_HOST=$(bashio::config 'mqtt_host')
MQTT_PORT=$(bashio::config 'mqtt_port')
MQTT_USERNAME=$(bashio::config 'mqtt_username')
MQTT_PASSWORD=$(bashio::config 'mqtt_password')
MQTT_TOPIC_PREFIX=$(bashio::config 'mqtt_topic_prefix')
POLLING_INTERVAL=$(bashio::config 'polling_interval')
HA_DISCOVERY_MESSAGES=$(bashio::config 'ha_discovery_messages')
HA_DISCOVERY_NAME=$(bashio::config 'ha_discovery_name')
USE_MQTT_FROM_SUPERVISOR=$(bashio::config 'use_mqtt_from_supervisor')
PUBLISH_EDIT_PARAMS=$(bashio::config 'publish_edit_params')
ENABLE_COMMAND_TOPIC=$(bashio::config 'enable_command_topic')

if [[ -z "${ECONET_ENDPOINT}" ]]; then
    bashio::log.fatal "Configuration option 'econet_endpoint' is required"
fi

# Fallback to supervisor MQTT service if configured and host not provided
if bashio::var.has_value "${USE_MQTT_FROM_SUPERVISOR}" \
    && bashio::var.true "${USE_MQTT_FROM_SUPERVISOR}" \
    && bashio::services.available "mqtt"; then
    if ! bashio::var.has_value "${MQTT_HOST}"; then
        MQTT_HOST=$(bashio::services mqtt "host")
        MQTT_PORT=$(bashio::services mqtt "port")
        MQTT_USERNAME=$(bashio::services mqtt "username")
        MQTT_PASSWORD=$(bashio::services mqtt "password")
        log "Using MQTT service from Supervisor: ${MQTT_HOST}:${MQTT_PORT}"
    fi
fi

if ! bashio::var.has_value "${MQTT_HOST}"; then
    bashio::log.fatal "MQTT host not set; provide 'mqtt_host' or enable Supervisor MQTT service."
fi

export MQTT_HOST MQTT_PORT MQTT_USERNAME MQTT_PASSWORD \
    MQTT_TOPIC_PREFIX POLLING_INTERVAL HA_DISCOVERY_MESSAGES \
    HA_DISCOVERY_NAME ECONET_ENDPOINT PUBLISH_EDIT_PARAMS \
    ENABLE_COMMAND_TOPIC

log "Starting mqtt_publisher.py"
exec python3 /opt/app/mqtt_publisher.py
