#!/usr/bin/env python3

import os
import sys
import time
import json
import signal
import logging
import requests
from typing import Dict, Any, Optional
import paho.mqtt.client as mqtt
from requests.auth import HTTPBasicAuth

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class EconetMQTTPublisher:
    def __init__(self):
        # Load configuration from environment variables
        self.mqtt_host = os.getenv('MQTT_HOST', 'localhost')
        self.mqtt_port = int(os.getenv('MQTT_PORT', '1883'))
        self.mqtt_username = os.getenv('MQTT_USERNAME', '')
        self.mqtt_password = os.getenv('MQTT_PASSWORD', '')
        self.mqtt_topic_prefix = os.getenv('MQTT_TOPIC_PREFIX', 'econet/')
        self.mqtt_keepalive = int(os.getenv('MQTT_KEEPALIVE', '60'))
        self.econet_endpoint = os.getenv('ECONET_ENDPOINT')
        self.polling_interval = int(os.getenv('POLLING_INTERVAL', '10'))
        self.ha_discovery = os.getenv('HA_DISCOVERY_MESSAGES', 'true').lower() == 'true'
        self.ha_discovery_name = os.getenv('HA_DISCOVERY_NAME', 'Grant R290')
        self.ha_expire_after_seconds = int(os.getenv('HA_EXPIRE_AFTER_SECONDS', '0'))
        self.ha_expire_multiplier = int(os.getenv('HA_EXPIRE_MULTIPLIER', '4'))
        # Always publish editParams and enable command topic; no opt-out flag
        self.publish_edit_params = True
        self.enable_command_topic = True

        # Validate required configuration
        if not self.econet_endpoint:
            logger.error("ECONET_ENDPOINT environment variable is required")
            sys.exit(1)

        # Ensure topic prefix ends with /
        if not self.mqtt_topic_prefix.endswith('/'):
            self.mqtt_topic_prefix += '/'

        # MQTT client setup
        self.mqtt_client = mqtt.Client()
        if self.mqtt_username:
            self.mqtt_client.username_pw_set(self.mqtt_username, self.mqtt_password)

        self.mqtt_client.on_connect = self._on_mqtt_connect
        self.mqtt_client.on_disconnect = self._on_mqtt_disconnect
        self.mqtt_client.on_message = self._on_mqtt_message

        # MQTT availability (LWT) setup
        # Use a single availability topic for this publisher
        self.availability_topic = f"{self.mqtt_topic_prefix}availability"
        # Set Last Will to offline, retained, so HA marks entities unavailable if we disconnect unexpectedly
        self.mqtt_client.will_set(self.availability_topic, payload="offline", retain=True)

        # Econet credentials (fixed as per instructions)
        self.econet_auth = HTTPBasicAuth('admin', 'admin')

        # Additional topics for edit params and command channel
        self.edit_params_topic = f"{self.mqtt_topic_prefix}edit_params"
        self.command_topic = f"{self.mqtt_topic_prefix}command/set_param"
        self.command_response_topic = f"{self.mqtt_topic_prefix}command/result"
        self.edit_command_base_topic = f"{self.mqtt_topic_prefix}command/edit_params"

        # Topic mapping: topic_name -> JSON path
        self.topic_mappings = {
            'ashp_ambient_air_temp': ['curr', 'AxenOutdoorTemp'],
            'ashp_circuit1_calculated_set_temp': ['tilesParams', 29, 0, 0],
            'ashp_compressor_freq': ['curr', 'AxenCompressorFreq'],
            'ashp_fan_speed': ['tilesParams', 3, 0, 0],
            'ashp_flow_temp': ['curr', 'AxenOutgoingTemp'],
            'ashp_flow_rate': ['curr', 'currentFlow'],
            'ashp_outlet_water_pressure': ['tilesParams', 76, 0, 0],
            'ashp_pump_active': ['curr', 'AxenUpperPump'],
            'ashp_return_temp': ['curr', 'AxenReturnTemp'],
            'ashp_target_temp': ['curr', 'HeatSourceCalcPresetTemp'],
            'ashp_work_state': ['curr', 'AxenWorkState'],
            'circuit1_thermostat': ['curr', 'Circuit1thermostat'],
            
            'dhw_temp': ['curr', 'TempCWU'],
            'outdoor_temp': ['curr', 'TempWthr'],
            'three_way_valve_state': ['curr', 'flapValveStates']
        }

        # editParams mapping: topic_name -> JSON path inside editParams payload.
        # Extend to include the fields you care about. Paths index into the JSON
        # structure returned by /econet/editParams.
        self.edit_params_mappings = {
            # Version / status
            #'editparams_version': ['editableParamsVer'],
            #'editparams_conf_done': ['confDone'],
            # Domestic hot water
            'dhw_setpoint': ['data', '103', 'value'],
            'dhw_legionella_setpoint': ['data', '136', 'value'],
            'dhw_legionella_day': ['data', '137', 'value'],
            'dhw_legionella_hour': ['data', '138', 'value'],
            # Buffer (bufor)
            #'buffer_setpoint': ['data', '183', 'value'],
            #'buffer_temp_start_hydraulic': ['data', '190', 'value'],
            # Circuit 1
            # Work_mode / Schedule ?
            'circuit1_work_state': ['data', '236', 'value'],
            #'circuit1_comfort_temp': ['data', '238', 'value'],
            #'circuit1_eco_temp': ['data', '239', 'value']
            'circuit1_signal_from_thermostat': ['informationParams','95', 1,0,0],
            'circuit1_signal_from_thermostat': ['informationParams','95', 1,0,0],
            'ashp_power': ['informationParams', '211', 1, 0, 0],
            'ashp_thermal_power': ['informationParams', '212', 1, 0, 0]
        }
        # Editable params that should be controllable via MQTT/HA.
        # Each entry defines the parameter ID used by /econet/newParam and metadata
        # for Home Assistant discovery.
        self.edit_params_control_mappings = [
            {
                "topic": "dhw_setpoint",
                "param_id": "103",
                "value_path": ['data', '103', 'value'],
                "min": 35,
                "max": 65,
                "step": 1,
                "unit": "C",
                "device_class": "temperature",
                "icon": "mdi:water-thermometer"
            },
            
            {
                "topic": "circuit1_comfort_temp",
                "param_id": "238",
                "value_path": ['data', '238', 'value'],
                "min": 10,
                "max": 35,
                "step": 0.1,
                "unit": "C",
                "device_class": "temperature",
                "icon": "mdi:home-thermometer"
            },
            {
                "topic": "circuit1_eco_temp",
                "param_id": "239",
                "value_path": ['data', '239', 'value'],
                "min": 10,
                "max": 35,
                "step": 0.1,
                "unit": "C",
                "device_class": "temperature",
                "icon": "mdi:home-thermometer-outline"
            },
            {
                "topic": "circuit1_down_hysteresis",
                "param_id": "240",
                "value_path": ['data', '240', 'value'],
                "min": 0,
                "max": 5,
                "step": 0.1,
                "unit": "C",
                "device_class": "temperature",
                "icon": "mdi:thermometer-alert"
            },
            {
                "topic": "circuit1_curve",
                "param_id": "273",
                "value_path": ['data', '273', 'value'],
                "min": 0,
                "max": 4,
                "step": 0.1,
                "icon": "mdi:chart-line"
            },
            {
                "topic": "circuit1_curve_shift",
                "param_id": "275",
                "value_path": ['data', '275', 'value'],
                "min": -20,
                "max": 20,
                "step": 1,
                "icon": "mdi:chart-line-variant"
            }
        ]
        # Command topics for edit param controls (populated in _init_edit_control_topics)
        self.edit_control_topic_map = {}

        # Home Assistant discovery metadata
        self.ha_discovery_configs = {
            'ashp_power': {
                'name': 'ASHP Power',
                'device_class': 'power',
                'state_class': 'measurement',
                'unit_of_measurement': 'kW',
                'icon': 'mdi:flash'
            },
            'ashp_thermal_power': {
                'name': 'ASHP Thermal Power',
                'device_class': 'power',
                'state_class': 'measurement',
                'unit_of_measurement': 'kW',
                'icon': 'mdi:fire'
            },
            'ashp_ambient_air_temp': {
                'name': 'ASHP Ambient Air Temperature',
                'device_class': 'temperature',
                'state_class': 'measurement',
                'unit_of_measurement': '°C',
                'icon': 'mdi:thermometer'
            },
            'ashp_circuit1_calculated_set_temp': {
                'name': 'ASHP Circuit 1 Calculated Set Temperature',
                'device_class': 'temperature',
                'state_class': 'measurement',
                'unit_of_measurement': '°C',
                'icon': 'mdi:thermometer'
            },
            'ashp_compressor_freq': {
                'name': 'ASHP Compressor Frequency',
                'device_class': 'frequency',
                'state_class': 'measurement',
                'unit_of_measurement': 'Hz',
                'icon': 'mdi:sine-wave'
            },
            'ashp_fan_speed': {
                'name': 'ASHP Fan Speed',
                'state_class': 'measurement',
                'unit_of_measurement': 'rpm',
                'icon': 'mdi:fan'
            },
            'ashp_flow_temp': {
                'name': 'ASHP Flow Temperature',
                'device_class': 'temperature',
                'state_class': 'measurement',
                'unit_of_measurement': '°C',
                'icon': 'mdi:thermometer-chevron-up'
            },
            'ashp_flow_rate': {
                'name': 'ASHP Flow Rate',
                'device_class': 'volume_flow_rate',
                'state_class': 'measurement',
                'unit_of_measurement': 'L/min',
                'icon': 'mdi:water-pump'
            },
            'ashp_outlet_water_pressure': {
                'name': 'ASHP Outlet Water Pressure',
                'device_class': 'pressure',
                'state_class': 'measurement',
                'unit_of_measurement': 'bar',
                'icon': 'mdi:gauge'
            },
            'ashp_pump_active': {
                'name': 'ASHP Pump',
                'device_class': 'running',
                'state_class': 'measurement',
                'icon': 'mdi:pump',
                'payload_on': '1',
                'payload_off': '0'
            },
            'ashp_return_temp': {
                'name': 'ASHP Return Temperature',
                'device_class': 'temperature',
                'state_class': 'measurement',
                'unit_of_measurement': '°C',
                'icon': 'mdi:thermometer-chevron-down'
            },
            'ashp_target_temp': {
                'name': 'ASHP Target Temperature',
                'device_class': 'temperature',
                'state_class': 'measurement',
                'unit_of_measurement': '°C',
                'icon': 'mdi:thermometer'
            },
            'ashp_work_state': {
                'name': 'ASHP Work State',
                'device_class': 'running',
                'state_class': 'measurement',
                'icon': 'mdi:state-machine',
                'payload_on': '1',
                'payload_off': '0'
            },
            'circuit1_signal_from_thermostat': {
                'name': 'Circuit 1 Signal from Thermostat',
                'device_class': 'heat',
                'state_class': 'measurement',
                'icon': 'mdi:radiator',
                'payload_on': '0',   # if 1 = Heat
                'payload_off': '1'
            },
            'circuit1_thermostat': {
                'name': 'Circuit 1 Thermostat Temperature',
                'device_class': 'temperature',
                'state_class': 'measurement',
                'unit_of_measurement': '°C',
                'icon': 'mdi:thermostat'
            },
            'dhw_temp': {
                'name': 'Cylinder Temperature',
                'device_class': 'temperature',
                'state_class': 'measurement',
                'unit_of_measurement': '°C',
                'icon': 'mdi:water-thermometer'
            },
            'outdoor_temp': {
                'name': 'Outdoor Sensor Temperature',
                'device_class': 'temperature',
                'state_class': 'measurement',
                'unit_of_measurement': '°C',
                'icon': 'mdi:thermometer'
            },
            'three_way_valve_state': {
                'name': 'Three Way Valve State',
                'device_class': 'enum',
                'icon': 'mdi:valve',
                'options': ['CH', 'DHW']
            }
        }

        self.running = True

        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)

        logger.info(f"Initialized EconetMQTTPublisher with:")
        logger.info(f"  MQTT Broker: {self.mqtt_host}:{self.mqtt_port}")
        logger.info(f"  Topic Prefix: {self.mqtt_topic_prefix}")
        logger.info(f"  Econet Endpoint: {self.econet_endpoint}")
        logger.info(f"  Polling Interval: {self.polling_interval}s")
        logger.info(f"  MQTT Keepalive: {self.mqtt_keepalive}s")
        logger.info(f"  Home Assistant Discovery: {self.ha_discovery}")
        if self.ha_discovery:
            if self.ha_expire_after_seconds > 0:
                logger.info(f"  HA expire_after: {self.ha_expire_after_seconds}s (fixed)")
            else:
                logger.info(f"  HA expire_after: polling_interval * {self.ha_expire_multiplier} = {self.polling_interval * self.ha_expire_multiplier}s")
        logger.info(f"  Publish editParams: {self.publish_edit_params}")
        logger.info(f"  MQTT command topic enabled: {self.enable_command_topic}")
        if self.ha_discovery:
            logger.info(f"  HA Device Name: {self.ha_discovery_name}")
        # Build command topics for editable params
        self._init_edit_control_topics()

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.running = False

    def _init_edit_control_topics(self):
        """Assign command topics for editable params and build lookup map"""
        self.edit_control_topic_map = {}
        for entry in self.edit_params_control_mappings:
            topic = entry["topic"]
            command_topic = f"{self.edit_command_base_topic}/{topic}/set"
            entry["command_topic"] = command_topic
            self.edit_control_topic_map[command_topic] = entry

    def _on_mqtt_connect(self, client, userdata, flags, rc):
        """MQTT connection callback"""
        if rc == 0:
            logger.info("Connected to MQTT broker")
            # Mark this integration as available
            try:
                self.mqtt_client.publish(self.availability_topic, "online", retain=True)
            except Exception as e:
                logger.error(f"Failed to publish availability online status: {e}")

            # Subscribe to command topic when enabled
            if self.enable_command_topic:
                try:
                    self.mqtt_client.subscribe(self.command_topic)
                    logger.info(f"Subscribed to command topic: {self.command_topic}")
                    # Also subscribe to edit param command topics
                    for entry in self.edit_params_control_mappings:
                        self.mqtt_client.subscribe(entry["command_topic"])
                        logger.info(f"Subscribed to edit param command topic: {entry['command_topic']}")
                except Exception as e:
                    logger.error(f"Failed to subscribe to command topic: {e}")
        else:
            logger.error(f"Failed to connect to MQTT broker, return code {rc}")

    def _on_mqtt_disconnect(self, client, userdata, rc):
        """MQTT disconnection callback"""
        logger.info("Disconnected from MQTT broker")

    def _on_mqtt_message(self, client, userdata, msg):
        """MQTT message dispatcher"""
        if msg.topic == self.command_topic and self.enable_command_topic:
            self._handle_command_message(msg)
        elif msg.topic in self.edit_control_topic_map:
            self._handle_edit_command_message(msg)

    def _get_nested_value(self, data: Dict[str, Any], path: list) -> Optional[Any]:
        """Extract value from nested dictionary/list using path"""
        try:
            current = data
            for key in path:
                if isinstance(current, dict):
                    current = current[key]
                elif isinstance(current, list):
                    current = current[int(key)]
                else:
                    return None

            # If the final value is a list, take the first element
            # This handles cases like tilesParams[29][0][0] returning ['24.0', 1, 0]
            if isinstance(current, list) and len(current) > 0:
                return current[0]

            return current
        except (KeyError, IndexError, TypeError, ValueError):
            return None

    def _fetch_econet_data(self) -> Optional[Dict[str, Any]]:
        """Fetch data from Econet endpoint"""
        try:
            url = f"http://{self.econet_endpoint}/econet/regParams"
            response = requests.get(url, auth=self.econet_auth, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch data from Econet: {e}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response from Econet: {e}")
            return None

    def _fetch_edit_params(self) -> Optional[Dict[str, Any]]:
        """Fetch data from the editParams endpoint (read-only)"""
        try:
            url = f"http://{self.econet_endpoint}/econet/editParams"
            response = requests.get(url, auth=self.econet_auth, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch data from editParams: {e}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response from editParams: {e}")
            return None

    def _set_econet_param(self, name: str, value: Any) -> Optional[Any]:
        """Write a parameter via the newParam endpoint"""
        try:
            url = f"http://{self.econet_endpoint}/econet/newParam"
            params = {"newParamName": name, "newParamValue": value}
            response = requests.get(url, auth=self.econet_auth, params=params, timeout=10)
            response.raise_for_status()
            try:
                return response.json()
            except json.JSONDecodeError:
                return response.text
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to set parameter {name}: {e}")
            return None

    def _publish_ha_discovery(self):
        """Publish Home Assistant MQTT discovery messages"""
        if not self.ha_discovery:
            return

        logger.info("Publishing Home Assistant discovery messages...")
        device_info = {
            "identifiers": ["econet_mqtt_publisher"],
            "name": self.ha_discovery_name,
            "model": "Heat Pump Controller",
            "manufacturer": "Econet",
            "via_device": "econet_mqtt_publisher"
        }

        for topic_name, config in self.ha_discovery_configs.items():
            # Determine component type based on device class or sensor type
            if config.get('device_class') == 'running' or 'payload_on' in config:
                component = 'binary_sensor'
            else:
                component = 'sensor'

            # Create unique ID for the entity
            unique_id = f"econet_{topic_name}"

            # Create discovery topic
            discovery_topic = f"homeassistant/{component}/{unique_id}/config"

            # Create state topic
            state_topic = f"{self.mqtt_topic_prefix}{topic_name}"

            # Build discovery payload
            discovery_payload = {
                "name": config['name'],
                "unique_id": unique_id,
                "state_topic": state_topic,
                "device": device_info,
                "icon": config.get('icon', 'mdi:gauge'),
                # MQTT availability
                "availability_topic": self.availability_topic,
                "payload_available": "online",
                "payload_not_available": "offline"
            }

            # Only sensors (not binary_sensors) support expire_after
            if component == 'sensor':
                # Mark sensor unavailable after missed updates; fixed value wins over multiplier
                expire_after = self.ha_expire_after_seconds or (self.polling_interval * self.ha_expire_multiplier)
                discovery_payload['expire_after'] = expire_after

            # Add device class if specified
            if 'device_class' in config:
                discovery_payload['device_class'] = config['device_class']

            # Add unit of measurement if specified
            if 'unit_of_measurement' in config:
                discovery_payload['unit_of_measurement'] = config['unit_of_measurement']

            # Add state class if specified (enables long-term statistics in HA)
            if 'state_class' in config:
                discovery_payload['state_class'] = config['state_class']

            # Add binary sensor specific payloads
            if component == 'binary_sensor':
                if 'payload_on' in config:
                    discovery_payload['payload_on'] = config['payload_on']
                if 'payload_off' in config:
                    discovery_payload['payload_off'] = config['payload_off']

            # Add enum sensor specific options
            if config.get('device_class') == 'enum' and 'options' in config:
                discovery_payload['options'] = config['options']

            # Publish discovery message
            try:
                result = self.mqtt_client.publish(
                    discovery_topic,
                    json.dumps(discovery_payload),
                    retain=True
                )
                if result.rc != mqtt.MQTT_ERR_SUCCESS:
                    logger.error(f"Failed to publish discovery for {topic_name}: {result.rc}")
            except Exception as e:
                logger.error(f"Error publishing discovery for {topic_name}: {e}")

        # Publish discovery for editable param controls as number entities
        for entry in self.edit_params_control_mappings:
            topic_name = entry["topic"]
            unique_id = f"econet_{topic_name}"
            discovery_topic = f"homeassistant/number/{unique_id}/config"
            state_topic = f"{self.mqtt_topic_prefix}{topic_name}"
            command_topic = entry["command_topic"]
            discovery_payload = {
                "name": entry.get("name", topic_name.replace("_", " ").title()),
                "unique_id": unique_id,
                "state_topic": state_topic,
                "command_topic": command_topic,
                "device": device_info,
                "icon": entry.get("icon", "mdi:numeric"),
                "availability_topic": self.availability_topic,
                "payload_available": "online",
                "payload_not_available": "offline",
                "mode": "box"
            }
            if "device_class" in entry:
                discovery_payload["device_class"] = entry["device_class"]
            if "unit" in entry:
                discovery_payload["unit_of_measurement"] = entry["unit"]
            if "min" in entry:
                discovery_payload["min"] = entry["min"]
            if "max" in entry:
                discovery_payload["max"] = entry["max"]
            if "step" in entry:
                discovery_payload["step"] = entry["step"]

            try:
                result = self.mqtt_client.publish(
                    discovery_topic,
                    json.dumps(discovery_payload),
                    retain=True
                )
                if result.rc != mqtt.MQTT_ERR_SUCCESS:
                    logger.error(f"Failed to publish discovery for editable param {topic_name}: {result.rc}")
            except Exception as e:
                logger.error(f"Error publishing discovery for editable param {topic_name}: {e}")

        logger.info("Home Assistant discovery messages published")

    def _convert_valve_state(self, value: Any) -> str:
        """Convert three way valve numeric state to text"""
        if value == 0:
            return 'CH'
        elif value == 3:
            return 'DHW'
        else:
            return str(value)  # Return as string if unknown value

    def _publish_metrics(self, data: Dict[str, Any]):
        """Extract and publish all metrics to MQTT"""
        published_values = {}

        for topic_name, json_path in self.topic_mappings.items():
            value = self._get_nested_value(data, json_path)
            if value is not None:
                full_topic = f"{self.mqtt_topic_prefix}{topic_name}"
                try:
                    # Special handling for three way valve state
                    if topic_name == 'three_way_valve_state':
                        payload = self._convert_valve_state(value)
                    else:
                        # Convert value to string for MQTT publishing
                        payload = str(value)

                    result = self.mqtt_client.publish(full_topic, payload)
                    if result.rc == mqtt.MQTT_ERR_SUCCESS:
                        published_values[topic_name] = payload  # Log the converted value
                    else:
                        logger.error(f"Failed to publish {topic_name}: {result.rc}")
                except Exception as e:
                    logger.error(f"Error publishing {topic_name}: {e}")
            else:
                logger.warning(f"Could not find value for {topic_name} at path {json_path}")

        # Log all published values
        if published_values:
            logger.info(f"Published values: {published_values}")
        else:
            logger.warning("No values were published")

    def _publish_edit_params(self, data: Dict[str, Any]):
        """Publish selected editParams values using a mapping"""
        if not self.publish_edit_params:
            return

        if data is None:
            logger.warning("editParams data unavailable; skipping publish")
            return

        if not isinstance(data, dict):
            logger.error("editParams payload is not a dictionary; skipping publish")
            return

        # Build combined mapping: read-only mapped values + control mapped values
        mappings = dict(self.edit_params_mappings)
        for entry in self.edit_params_control_mappings:
            mappings[entry["topic"]] = entry["value_path"]

        if not mappings:
            logger.warning("editParams mapping is empty; nothing to publish")
            return

        published_values = {}
        for topic_name, json_path in mappings.items():
            value = self._get_nested_value(data, json_path)
            if value is None:
                logger.warning(f"Could not find editParams value for {topic_name} at path {json_path}")
                continue

            # Publish with the same prefix style as regParams metrics
            full_topic = f"{self.mqtt_topic_prefix}{topic_name}"
            try:
                payload = str(value)
                result = self.mqtt_client.publish(full_topic, payload)
                if result.rc == mqtt.MQTT_ERR_SUCCESS:
                    published_values[topic_name] = payload
                else:
                    logger.error(f"Failed to publish editParams {topic_name}: {result.rc}")
            except Exception as e:
                logger.error(f"Error publishing editParams {topic_name}: {e}")

        if published_values:
            logger.info(f"Published editParams values: {published_values}")
        else:
            logger.warning("No editParams values were published")

    def _handle_command_message(self, msg):
        """Handle incoming MQTT command message to set a parameter via newParam"""
        payload_raw = msg.payload.decode('utf-8', errors='ignore').strip()
        logger.info(f"Received command message on {msg.topic}: {payload_raw}")

        param_name = None
        param_value = None

        # Expect JSON payload {"name": "...", "value": ...}
        try:
            parsed = json.loads(payload_raw)
            param_name = str(parsed.get("name"))
            param_value = parsed.get("value")
        except json.JSONDecodeError:
            # Fallback: payload like "238=20.5"
            if "=" in payload_raw:
                parts = payload_raw.split("=", 1)
                param_name = parts[0]
                param_value = parts[1]

        if not param_name or param_value is None:
            logger.error("Invalid command payload; expected JSON with name/value or 'name=value' text")
            self._publish_command_result(success=False, message="invalid payload")
            return

        result = self._set_econet_param(param_name, param_value)
        if result is not None:
            logger.info(f"Parameter {param_name} set successfully")
            self._publish_command_result(success=True, message="ok", response=result)
        else:
            self._publish_command_result(success=False, message="request failed")

    def _handle_edit_command_message(self, msg):
        """Handle MQTT command for editable params (per-entity command topics)"""
        mapping = self.edit_control_topic_map.get(msg.topic)
        if not mapping:
            logger.error(f"Received command for unknown topic {msg.topic}")
            return

        payload_raw = msg.payload.decode('utf-8', errors='ignore').strip()
        logger.info(f"Received editParam command on {msg.topic}: {payload_raw}")

        try:
            # Accept either raw number or JSON {"value": ...}
            if payload_raw.startswith("{"):
                parsed = json.loads(payload_raw)
                value = parsed.get("value")
            else:
                value = payload_raw

            if value is None or value == "":
                raise ValueError("empty value")

            result = self._set_econet_param(mapping["param_id"], value)
            if result is not None:
                # Publish new state immediately
                state_topic = f"{self.mqtt_topic_prefix}{mapping['topic']}"
                self.mqtt_client.publish(state_topic, str(value))
                logger.info(f"Set {mapping['topic']} ({mapping['param_id']}) to {value}")
            else:
                logger.error(f"Failed to set {mapping['topic']} ({mapping['param_id']})")
        except Exception as e:
            logger.error(f"Error handling editParam command for {mapping['topic']}: {e}")

    def _publish_command_result(self, success: bool, message: str, response: Any = None):
        """Publish result of command execution"""
        if not self.enable_command_topic:
            return
        payload = {"success": success, "message": message}
        if response is not None:
            payload["response"] = response
        try:
            self.mqtt_client.publish(self.command_response_topic, json.dumps(payload))
        except Exception as e:
            logger.error(f"Failed to publish command result: {e}")

    def connect_mqtt(self):
        """Connect to MQTT broker"""
        try:
            self.mqtt_client.connect(self.mqtt_host, self.mqtt_port, self.mqtt_keepalive)
            self.mqtt_client.loop_start()
            return True
        except Exception as e:
            logger.error(f"Failed to connect to MQTT broker: {e}")
            return False

    def disconnect_mqtt(self):
        """Disconnect from MQTT broker"""
        # On clean shutdown, mark availability as offline
        try:
            self.mqtt_client.publish(self.availability_topic, "offline", retain=True)
        except Exception as e:
            logger.error(f"Failed to publish availability offline status: {e}")
        self.mqtt_client.loop_stop()
        self.mqtt_client.disconnect()

    def run(self):
        """Main execution loop"""
        logger.info("Starting Econet MQTT Publisher...")

        if not self.connect_mqtt():
            logger.error("Failed to connect to MQTT broker, exiting")
            sys.exit(1)

        # Publish Home Assistant discovery messages once on startup
        self._publish_ha_discovery()

        try:
            while self.running:
                logger.info("Polling Econet endpoint...")
                data = self._fetch_econet_data()

                if data:
                    self._publish_metrics(data)
                else:
                    logger.error("Failed to fetch data from Econet endpoint")

                # Optionally publish editParams payload
                if self.publish_edit_params:
                    edit_data = self._fetch_edit_params()
                    self._publish_edit_params(edit_data)

                # Wait for next polling interval
                for _ in range(self.polling_interval):
                    if not self.running:
                        break
                    time.sleep(1)

        except Exception as e:
            logger.error(f"Unexpected error in main loop: {e}")

        finally:
            logger.info("Shutting down...")
            self.disconnect_mqtt()
            logger.info("Shutdown complete")

def main():
    publisher = EconetMQTTPublisher()
    publisher.run()

if __name__ == "__main__":
    main()
