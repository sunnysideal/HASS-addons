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

# Load configuration from neuralprophet.yaml
# Check multiple possible locations for the config file
config_path = Path('/config/neuralprophet.yaml')  # HA config directory (addon_config mapping)

config = {}
config_loaded = False

logger.info(f"Checking for config at: {config_path}")   
logger.info(f"  Exists: {config_path.exists()}")
   
try:
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f) or {}
        logger.info(f"Loaded configuration from {config_path}: {config}")
        config_loaded = True
except Exception as e:
    logger.error(f"Failed to load configuration from {config_path}: {e}")

if not config_loaded:
    logger.warning(f"Configuration file not found")
    

ha_api = HomeAssistantAPI()

# Example: Get all states
logger.info("Fetching all entity states...")
states = ha_api.get_states()

if states:
    logger.info(f"Found {len(states)} entities")


# Get the history sensor from config or use default
history_sensor = config.get('history_sensor')
logger.info(f"Fetching history for {history_sensor}")
logger.info(history_sensor)

history = ha_api.get_history(history_sensor)