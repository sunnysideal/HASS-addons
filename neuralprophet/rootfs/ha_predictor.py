import logging
import yaml
import pandas as pd
import time
from pathlib import Path
from datetime import datetime, timedelta
from homeassistantapi import HomeAssistantAPI
from neuralprophet import NeuralProphet

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


def update_prediction_entity(ha_api, entity_id, forecast_df, units=''):
    """Update a Home Assistant entity with prediction data
    
    Args:
        ha_api: HomeAssistantAPI instance
        entity_id: Entity ID to update
        forecast_df: DataFrame with forecast (must have 'ds' and 'yhat1' columns)
        units: Unit of measurement
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Get the latest prediction value
        latest_prediction = forecast_df.iloc[-1]['yhat1']
        
        # Check if latest_prediction is None or NaN
        if pd.isna(latest_prediction):
            logger.error(f"Latest prediction is NaN for {entity_id}")
            return False
        
        # Format the forecast data as attributes
        forecast_list = []
        for _, row in forecast_df.iterrows():
            # Skip rows with NaN values
            if pd.notna(row['yhat1']):
                forecast_list.append({
                    'datetime': row['ds'].isoformat(),
                    'value': float(row['yhat1'])
                })
        
        if not forecast_list:
            logger.error(f"No valid forecast data for {entity_id}")
            return False
        
        # Update the entity state with forecast as attributes
        attributes = {
            'unit_of_measurement': units,
            'friendly_name': f"{entity_id.split('.')[-1].replace('_', ' ').title()}",
            'forecast': forecast_list,  # Include all forecast predictions
            'forecast_count': len(forecast_list),
            'last_updated': datetime.now().isoformat()
        }
        
        result = ha_api.set_state(entity_id, float(latest_prediction), attributes)
        
        if result is not None:
            logger.info(f"Updated {entity_id} with prediction: {latest_prediction:.2f} {units}")
            return True
        else:
            logger.error(f"Failed to update {entity_id}")
            return False
            
    except Exception as e:
        logger.error(f"Error updating prediction entity {entity_id}: {e}", exc_info=True)
        return False


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
    skipped_count = 0
    for record in entity_data:
        try:
            # Skip if state is None, 'unknown', 'unavailable', or empty
            state = record.get('state')
            if state is None or state in ['unknown', 'unavailable', '']:
                skipped_count += 1
                continue
            
            # Parse the timestamp
            timestamp = pd.to_datetime(record['last_changed'])
            # Convert state to float, skip if invalid
            state_value = float(state)
            data_points.append({'ds': timestamp, 'y': state_value})
        except (ValueError, KeyError, TypeError) as e:
            skipped_count += 1
            logger.debug(f"Skipping invalid record: {e}")
            continue
    
    if skipped_count > 0:
        logger.info(f"Skipped {skipped_count} invalid/unavailable records")
    
    if not data_points:
        logger.error("No valid data points found in history")
        return None
    
    # Create DataFrame
    df = pd.DataFrame(data_points)
    logger.info(f"Prepared {len(df)} data points for training")
    logger.info(f"Date range: {df['ds'].min()} to {df['ds'].max()}")
    
    return df


def process_sensors(ha_api, sensors, default_history_days, default_interval_duration, default_intervals_to_predict):
    """Process all sensors and prepare training data
    
    Args:
        ha_api: HomeAssistantAPI instance
        sensors: List of sensor configurations
        default_history_days: Default number of days if not specified per sensor
        default_interval_duration: Default interval duration in minutes
        default_intervals_to_predict: Default number of intervals to predict
    """
    # Process each sensor
    for sensor_config in sensors:
        training_entity_id = sensor_config.get('training_entity_id')
        prediction_entity_id = sensor_config.get('prediction_entity_id')
        
        if not training_entity_id:
            logger.warning(f"Sensor configuration missing training_entity_id: {sensor_config}")
            continue
        
        if not prediction_entity_id:
            logger.warning(f"Sensor configuration missing prediction_entity_id: {sensor_config}")
            continue
        
        # Get configuration for this specific sensor, or use defaults
        history_days = sensor_config.get('history_days', default_history_days)
        interval_duration = sensor_config.get('interval_duration', default_interval_duration)
        intervals_to_predict = sensor_config.get('intervals_to_predict', default_intervals_to_predict)
        units = sensor_config.get('units', '')
        
        logger.info(f"Processing sensor:")
        logger.info(f"  - Training from: {training_entity_id}")
        logger.info(f"  - Prediction to: {prediction_entity_id}")
        logger.info(f"  - History: {history_days} days")
        logger.info(f"  - Interval duration: {interval_duration} minutes")
        logger.info(f"  - Intervals to predict: {intervals_to_predict}")
        logger.info(f"  - Units: {units}")
        
        # Fetch history for this sensor
        try:
            training_data = ha_api.get_history(training_entity_id, days=history_days, minimal_response=True)
            
            if not training_data or len(training_data) == 0:
                logger.error(f"No history data retrieved for {training_entity_id}")
                continue
            
            logger.info(f"Retrieved {len(training_data[0])} history records for {training_entity_id}")
            
            # Prepare data for NeuralProphet
            df = prepare_training_data(training_data)
            
            if df is not None and len(df) > 0:
                logger.info(f"Training data prepared successfully for {training_entity_id}")
                logger.info(f"  - Data points: {len(df)}")
                logger.info(f"  - Date range: {df['ds'].min()} to {df['ds'].max()}")
                logger.info(f"  - Sample data:\n{df.head()}")
                
                # Train NeuralProphet model
                logger.info(f"Training NeuralProphet model for {training_entity_id}...")
                
                # Calculate n_lags based on interval duration (96 = 48 hours for 30-min intervals)
                n_lags = int((48 * 60) / interval_duration)  # 48 hours worth of lags
                freq = f"{interval_duration}min"
                
                model = NeuralProphet(
                    daily_seasonality=True,
                    weekly_seasonality=True,
                    yearly_seasonality=True,
                    n_lags=n_lags,
                    epochs=50,  # Adjust based on your needs
                    learning_rate=0.01,
                    batch_size=32
                )
                
                # Fit the model
                metrics = model.fit(df, freq=freq)
                logger.info(f"Model training complete for {training_entity_id}")
                
                # Make future dataframe for predictions
                future = model.make_future_dataframe(df, periods=intervals_to_predict, n_historic_predictions=True)
                
                # Generate forecast
                logger.info(f"Generating {intervals_to_predict} interval predictions...")
                forecast = model.predict(future)
                
                # Extract only future predictions (not historical)
                future_forecast = forecast.tail(intervals_to_predict)
                logger.info(f"Forecast generated: {len(future_forecast)} predictions")
                logger.info(f"  - Forecast range: {future_forecast['ds'].min()} to {future_forecast['ds'].max()}")
                logger.info(f"  - Sample forecast:\n{future_forecast[['ds', 'yhat1']].head()}")
                
                # Update prediction entity in Home Assistant
                success = update_prediction_entity(ha_api, prediction_entity_id, future_forecast, units)
                
                if success:
                    logger.info(f"Successfully updated {prediction_entity_id} with forecast")
                else:
                    logger.warning(f"Failed to update {prediction_entity_id}")
                
            else:
                logger.error(f"Failed to prepare training data for {training_entity_id}")
                
        except Exception as e:
            logger.error(f"Error processing sensor {training_entity_id}: {e}", exc_info=True)
            continue


logger.info("Home Assistant NeuralProphet Add-on starting...")

# Configuration path
config_path = Path('/config/neuralprophet.yaml')  # HA config directory (addon_config mapping)

ha_api = HomeAssistantAPI()

# Main loop - run continuously at specified interval
while True:
    try:
        logger.info(f"Starting training cycle at {datetime.now()}")
        
        # Load configuration from neuralprophet.yaml (reload each cycle)
        config = load_config(config_path)
        
        # Get sensors from config
        sensors = get_sensors_from_config(config)
        
        if not sensors:
            logger.error("No sensors configured in neuralprophet.yaml")
            logger.info("Retrying in 60 seconds")
            time.sleep(60)
            continue
        
        # Get global configuration
        default_history_days = config.get('history_days', 60)  # Default 60 days for 30-min intervals
        update_interval_minutes = config.get('update_interval', 60)  # Default 60 minutes
        update_interval = update_interval_minutes * 60  # Convert to seconds
        default_interval_duration = config.get('interval_duration', 30)  # Default 30 minutes
        default_intervals_to_predict = config.get('intervals_to_predict', 48)  # Default 48 intervals
        
        logger.info(f"Configuration:")
        logger.info(f"  - Default history: {default_history_days} days")
        logger.info(f"  - Update interval: {update_interval_minutes} minutes ({update_interval} seconds)")
        logger.info(f"  - Default interval duration: {default_interval_duration} minutes")
        logger.info(f"  - Default intervals to predict: {default_intervals_to_predict}")
        logger.info(f"  - Sensors: {len(sensors)}")
        
        process_sensors(ha_api, sensors, default_history_days, default_interval_duration, default_intervals_to_predict)
        
        logger.info(f"Training cycle complete. Next run in {update_interval} seconds")
        time.sleep(update_interval)
        
    except KeyboardInterrupt:
        logger.info("Shutting down gracefully...")
        break
    except Exception as e:
        logger.error(f"Error in main loop: {e}", exc_info=True)
        logger.info(f"Retrying in {update_interval} seconds")
        time.sleep(update_interval)