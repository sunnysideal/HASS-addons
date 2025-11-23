# NeuralProphet Home Assistant Addon - Local Development

## Setup Instructions

### 1. Create a Long-Lived Access Token in Home Assistant

1. Open your Home Assistant instance
2. Click on your profile (bottom left)
3. Scroll down to "Long-Lived Access Tokens"
4. Click "Create Token"
5. Give it a name like "NeuralProphet Dev"
6. Copy the token (you won't see it again!)

### 2. Configure Local Environment

1. Copy the example environment file:
   ```powershell
   Copy-Item .env.example .env
   ```

2. Edit `.env` and fill in your details:
   ```
   HA_URL=http://192.168.1.100:8123
   HA_TOKEN=your_long_lived_access_token_here
   CONFIG_PATH=./rootfs/neuralprophet.yaml
   ```

### 3. Install Python Dependencies

1. Create a virtual environment:
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

2. Install dependencies:
   ```powershell
   pip install python-dotenv requests pyyaml pandas neuralprophet torch
   ```

### 4. Configure Your Sensors

Edit `rootfs/neuralprophet.yaml` with your sensor entity IDs from Home Assistant.

### 5. Run Locally

```powershell
python dev/dev_run.py
```

## Notes

- The script will connect to your Home Assistant instance using the API
- It will fetch historical data and train models locally
- Predictions will be written back to your Home Assistant instance
- Press Ctrl+C to stop the script

## Troubleshooting

**Connection Issues:**
- Make sure your Home Assistant URL is accessible from your development machine
- Check that the long-lived access token is valid
- Verify firewall settings if connecting remotely

**Import Errors:**
- Make sure all dependencies are installed in your virtual environment
- Run `pip list` to verify installed packages

**Sensor Not Found:**
- Check entity IDs in Home Assistant (Developer Tools -> States)
- Update `neuralprophet.yaml` with correct entity IDs
