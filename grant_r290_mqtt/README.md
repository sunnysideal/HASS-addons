# Grant R290 MQTT Home Assistant Add-on

Home Assistant add-on wrapper for [`econet-mqtt-publisher`](https://github.com/mikewhitby/econet-mqtt-publisher). It polls the Grant R290 (Econet) controller and publishes metrics to MQTT with Home Assistant discovery.

## Configuration

Configure the add-on in the Supervisor UI. Required option: `econet_endpoint` (controller IP/hostname). MQTT settings can come from the Supervisor MQTT service or be set manually.

Options:
- `econet_endpoint` (required): Econet controller IP/hostname (no protocol).
- `mqtt_host` / `mqtt_port` / `mqtt_username` / `mqtt_password`: MQTT broker settings. If left blank and `use_mqtt_from_supervisor` is true, the Supervisor MQTT service credentials are used.
- `mqtt_topic_prefix`: Topic prefix, default `econet/`.
- `polling_interval`: Seconds between polls, default `10`.
- `ha_discovery_messages`: Enable MQTT discovery, default `true`.
- `ha_discovery_name`: Device name for discovery.
- `publish_edit_params`: Always enabled; publishes values from `/econet/editParams` each poll to mapped topics.
- `enable_command_topic`: Always enabled; subscribes to edit param command topics and `/econet/newParam`.

### Command topic format (when enabled)
- Topic: `mqtt_topic_prefix/command/set_param`
- Payload: JSON `{"name": "238", "value": 20.5}` or text `238=20.5`
- Result is published to `mqtt_topic_prefix/command/result`

## Build / Run

This is a local add-on repository. Add the folder as a repository in Home Assistant, then install the add-on. Architectures supported: `amd64`, `aarch64`.
