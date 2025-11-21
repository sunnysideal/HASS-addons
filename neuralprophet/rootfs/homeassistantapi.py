import os
import sys
import logging
import requests

# Set up basic logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HomeAssistantAPI:
    """Client for interacting with Home Assistant API"""
    
    def __init__(self, config=None):
        # Get the supervisor token from environment
        self.token = os.environ.get('SUPERVISOR_TOKEN')
        
        # Use config if provided, otherwise defaults
        if config is None:
            config = {}
        
        # Home Assistant API endpoint (from config or default)
        self.ha_url = config.get('homeassistant', {}).get('url', "http://supervisor/core/api")
        self.timeout = config.get('homeassistant', {}).get('timeout', 10)
        
        # Get settings from config
        self.max_retries = config.get('settings', {}).get('max_retries', 3)
        self.retry_delay = config.get('settings', {}).get('retry_delay', 5)
        
        # Headers for authentication
        self.headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }
        
        if not self.token:
            logger.error("SUPERVISOR_TOKEN not found in environment")
            sys.exit(1)
    
    def get_states(self):
        """Get all entity states from Home Assistant"""
        try:
            response = requests.get(
                f"{self.ha_url}/states",
                headers=self.headers,
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to get states: {e}")
            return None
    
    def get_entity_state(self, entity_id):
        """Get state of a specific entity"""
        try:
            response = requests.get(
                f"{self.ha_url}/states/{entity_id}",
                headers=self.headers,
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to get entity state: {e}")
            return None
    
    def call_service(self, domain, service, service_data=None):
        """Call a Home Assistant service"""
        try:
            response = requests.post(
                f"{self.ha_url}/services/{domain}/{service}",
                headers=self.headers,
                json=service_data or {},
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to call service: {e}")
            return None
