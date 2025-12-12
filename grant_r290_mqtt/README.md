# Grant R290 MQTT Home Assistant Add-on

Home Assistant add-on wrapper for [`econet-mqtt-publisher`](https://github.com/mikewhitby/econet-mqtt-publisher). It polls the Grant R290 (Econet) controller and publishes metrics to MQTT with Home Assistant discovery.

The core publishing code was taken from https://github.com/mikewhitby/econet-mqtt-publisher.  
I hope to contribute back with the sensors I have added.

## Configuration

Configure the add-on in the Supervisor UI. Required option: `econet_endpoint` (controller IP/hostname).  
MQTT settings can come from the Supervisor MQTT service or be set manually.

Options:
- `econet_endpoint` (required): Econet controller IP/hostname (no protocol).
- `mqtt_host` / `mqtt_port` / `mqtt_username` / `mqtt_password`: MQTT broker settings. If left blank the Supervisor MQTT service credentials are used.
- `mqtt_topic_prefix`: Topic prefix, default `econet/`.
- `polling_interval`: Seconds between polls, default `10`.
- `ha_discovery_name`: Device name for discovery.

