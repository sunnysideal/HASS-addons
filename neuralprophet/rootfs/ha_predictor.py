import logging
from homeassistantapi import HomeAssistantAPI
# Set up basic logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


logger.info("Home Assistant Add-on starting...")

ha_api = HomeAssistantAPI()

# Example: Get all states
logger.info("Fetching all entity states...")
states = ha_api.get_states()

if states:
    logger.info(f"Found {len(states)} entities")