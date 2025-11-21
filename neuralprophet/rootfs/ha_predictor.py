import logging
import yaml
from pathlib import Path
from homeassistantapi import HomeAssistantAPI

# Set up basic logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


logger.info("Home Assistant Add-on starting...")

# Load configuration from neuralprophet.yaml in addon_config
config_path = Path('/addon_config/neuralprophet.yaml')
config = {}

try:
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f) or {}
        logger.info(f"Loaded configuration: {config}")
    else:
        logger.warning(f"Configuration file not found at {config_path}")
except Exception as e:
    logger.error(f"Failed to load configuration: {e}")

ha_api = HomeAssistantAPI()

# Example: Get all states
logger.info("Fetching all entity states...")
states = ha_api.get_states()

if states:
    logger.info(f"Found {len(states)} entities")


# Get the history sensor from config or use default
history_sensor = config.get('history_sensor')
logger.info(f"Fetching history for {history_sensor}")

history = ha_api.get_history(history_sensor)