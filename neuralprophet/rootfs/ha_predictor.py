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


def load_config(config_path):
    """Load configuration from YAML file
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Dictionary with configuration or empty dict if not found
    """
    config = {}
    
    logger.info(f"Checking for config at: {config_path}")   
    logger.info(f"Config Exists: {config_path.exists()}")
    
    try:
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f) or {}
            logger.info(f"Loaded configuration from {config_path}: {config}")
            return config
    except Exception as e:
        logger.error(f"Failed to load configuration from {config_path}: {e}")
    
    logger.warning(f"Configuration file not found")
    return config


def get_sensors_from_config(config):
    """Extract list of sensors from configuration
    
    Args:
        config: Configuration dictionary
        
    Returns:
        List of sensor configurations or empty list
    """
    sensors = config.get('sensors', [])
    logger.info(f"Found {len(sensors)} sensor(s) in configuration")
    return sensors


logger.info("Home Assistant Add-on starting...")

# Load configuration from neuralprophet.yaml
config_path = Path('/config/neuralprophet.yaml')  # HA config directory (addon_config mapping)
config = load_config(config_path)

# Get sensors from config
sensors = get_sensors_from_config(config)

ha_api = HomeAssistantAPI()


# Get the history sensor from config or use default
training_sensor = sensors[0].get('training_sensor')
logger.info(f"Fetching history for {training_sensor}")

training_data = ha_api.get_history(training_sensor, days=1, minimal_response = True)
logger.info(f"History data: {training_data}")