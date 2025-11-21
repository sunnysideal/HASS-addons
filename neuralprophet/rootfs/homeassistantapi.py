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
        logger.info(f"Initializing HomeAssistantAPI client with token {self.token}")
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
    
    def get_history(self, entity_ids=None, start_time=None, end_time=None, minimal_response=False):
        """Get historical data for entities
        
        Args:
            entity_ids: List of entity IDs or single entity ID string. If None, returns all entities.
            start_time: ISO 8601 timestamp string (e.g., '2025-11-20T00:00:00+00:00'). Defaults to 1 day ago.
            end_time: ISO 8601 timestamp string. Defaults to now.
            minimal_response: If True, returns minimal response with only state changes.
            
        Returns:
            List of historical data or None on error
        """
        try:
            # Build the URL
            if entity_ids:
                if isinstance(entity_ids, str):
                    entity_ids = [entity_ids]
                # Filter by specific entities
                filter_param = ",".join(entity_ids)
                url = f"{self.ha_url}/history/period"
            else:
                url = f"{self.ha_url}/history/period"
            
            # Add timestamp if provided
            if start_time:
                url = f"{url}/{start_time}"
            
            # Build query parameters
            params = {}
            if end_time:
                params['end_time'] = end_time
            if minimal_response:
                params['minimal_response'] = 'true'
            if entity_ids:
                params['filter_entity_id'] = ",".join(entity_ids)
            
            response = requests.get(
                url,
                headers=self.headers,
                params=params,
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to get history: {e}")
            return None
