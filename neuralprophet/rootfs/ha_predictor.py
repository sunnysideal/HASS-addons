import logging
import yaml
import pandas as pd
from pathlib import Path
from datetime import datetime
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


def prepare_training_data(history_data):
    """Convert Home Assistant history data to NeuralProphet format
    
    Args:
        history_data: List of history records from HA API
        
    Returns:
        pandas DataFrame with columns 'ds' (datetime) and 'y' (value)
    """
    if not history_data or len(history_data) == 0:
        logger.error("No history data provided")
        return None
    
    # history_data is a list of lists, get the first list (single entity)
    entity_data = history_data[0] if isinstance(history_data[0], list) else history_data
    
    # Extract datetime and state values
    data_points = []
    for record in entity_data:
        try:
            # Parse the timestamp
            timestamp = pd.to_datetime(record['last_changed'])
            # Convert state to float, skip if invalid
            state_value = float(record['state'])
            data_points.append({'ds': timestamp, 'y': state_value})
        except (ValueError, KeyError) as e:
            logger.debug(f"Skipping invalid record: {e}")
            continue
    
    if not data_points:
        logger.error("No valid data points found in history")
        return None
    
    # Create DataFrame
    df = pd.DataFrame(data_points)
    logger.info(f"Prepared {len(df)} data points for training")
    logger.info(f"Date range: {df['ds'].min()} to {df['ds'].max()}")
    
    return df


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
logger.info(f"Retrieved {len(training_data[0]) if training_data and len(training_data) > 0 else 0} history records")

# Prepare data for NeuralProphet
df = prepare_training_data(training_data)
if df is not None:
    logger.info(f"Training data prepared successfully")
    logger.info(f"DataFrame shape: {df.shape}")
    logger.info(f"DataFrame head:\n{df.head()}")
else:
    logger.error("Failed to prepare training data")