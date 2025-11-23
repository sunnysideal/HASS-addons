
import logging
# Remove all handlers associated with the root logger object.
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)

# Optionally, set a specific log level for neuralprophet if you want less/more verbosity
logging.getLogger('neuralprophet').setLevel(logging.INFO)

"""
Local development script for NeuralProphet Home Assistant addon
Connects to a remote Home Assistant instance for testing
"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add rootfs to path so we can import the modules
rootfs_path = Path(__file__).parent.parent / 'rootfs'
sys.path.insert(0, str(rootfs_path))

# Load environment variables from .env file (in repository root)
env_file = Path(__file__).parent.parent / '.env'
load_dotenv(env_file)

# Set up environment variables for local development
os.environ['SUPERVISOR_TOKEN'] = os.getenv('HA_TOKEN', '')
os.environ['HA_URL'] = os.getenv('HA_URL', 'http://localhost:8123')
os.environ['DEV_CONFIG_PATH'] = os.getenv('DEV_CONFIG_PATH', '')

# Set config path environment variable for dev
config_file = Path(__file__).parent / 'neuralprophet.yaml'
#config_file = rootfs_path / 'neuralprophet.yaml'
if not config_file.exists():
    print(f"ERROR: Configuration file not found at: {config_file.absolute()}")
    print("Please ensure rootfs/neuralprophet.yaml exists")
    sys.exit(1)

os.environ['DEV_CONFIG_PATH'] = str(config_file.absolute())
print(f"Using config file: {config_file.absolute()}")

# Set dev database path (in dev folder, excluded from git)
dev_db_path = Path(__file__).parent / 'neuralprophet_dev.db'
os.environ['DEV_DB_PATH'] = str(dev_db_path.absolute())
print(f"Using dev database: {dev_db_path.absolute()}")

# Import after setting environment
from homeassistantapi import HomeAssistantAPI

# Override the HomeAssistantAPI to use local config
class LocalHomeAssistantAPI(HomeAssistantAPI):
    def __init__(self, config=None):
        # Don't call parent __init__ to avoid sys.exit on missing SUPERVISOR_TOKEN
        if config is None:
            config = {}
        
        # Use environment variables for local development
        self.token = os.environ.get('SUPERVISOR_TOKEN')
        self.ha_url = os.environ.get('HA_URL', 'http://localhost:8123') + '/api'
        self.timeout = config.get('homeassistant', {}).get('timeout', 10)
        
        self.max_retries = config.get('settings', {}).get('max_retries', 3)
        self.retry_delay = config.get('settings', {}).get('retry_delay', 5)
        
        self.headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }
        
        if not self.token:
            print("ERROR: HA_TOKEN not found in .env file")
            print("Please create a .env file with your Home Assistant long-lived access token")
            sys.exit(1)
        
        print(f"Connected to Home Assistant at: {self.ha_url}")

# Monkey-patch the HomeAssistantAPI
import homeassistantapi
homeassistantapi.HomeAssistantAPI = LocalHomeAssistantAPI

print("Starting NeuralProphet predictor...")
print("Press Ctrl+C to stop")
print("")

# Import and run ha_predictor - it will use the DEV_CONFIG_PATH environment variable
import ha_predictor
ha_predictor.main()
