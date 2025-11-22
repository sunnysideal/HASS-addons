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


def fetch_regressor_data(ha_api, regressor_entity_id, history_days):
    """Fetch regressor data from Home Assistant
    
    Args:
        ha_api: HomeAssistantAPI instance
        regressor_entity_id: Entity ID of the regressor
        history_days: Number of days of history to fetch
        
    Returns:
        pandas DataFrame with columns 'ds' (datetime) and regressor name (value)
    """
    try:
        regressor_data = ha_api.get_history(regressor_entity_id, days=history_days, minimal_response=True)
        
        if not regressor_data or len(regressor_data) == 0:
            logger.warning(f"No history data retrieved for regressor {regressor_entity_id}")
            return None
        
        entity_data = regressor_data[0] if isinstance(regressor_data[0], list) else regressor_data
        
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
                data_points.append({'ds': timestamp, regressor_entity_id: state_value})
            except (ValueError, KeyError, TypeError) as e:
                skipped_count += 1
                continue
        
        if not data_points:
            logger.warning(f"No valid data points found for regressor {regressor_entity_id}")
            return None
        
        df = pd.DataFrame(data_points)
        logger.info(f"Fetched {len(df)} data points for regressor {regressor_entity_id}")
        
        return df
        
    except Exception as e:
        logger.error(f"Error fetching regressor data for {regressor_entity_id}: {e}", exc_info=True)
        return None


def fetch_weather_forecast_regressors(ha_api, weather_entity_id, attributes, history_days):
    """Fetch weather forecast data as regressors, extracting historical values from forecast attributes
    
    Args:
        ha_api: HomeAssistantAPI instance
        weather_entity_id: Weather entity ID (e.g., 'weather.home')
        attributes: List of forecast attributes to extract (e.g., ['cloud_coverage', 'temperature'])
        history_days: Number of days of history to construct
        
    Returns:
        Dictionary mapping attribute names to DataFrames with 'ds' and attribute columns
    """
    try:
        # Get current forecast
        forecast = ha_api.get_weather_forecast(weather_entity_id)
        if not forecast:
            logger.warning(f"No forecast available from {weather_entity_id}")
            return {}
        
        # For training, we need historical values of these attributes
        # We'll fetch the weather entity's history and extract these attributes
        history = ha_api.get_history(weather_entity_id, days=history_days, minimal_response=False)
        
        if not history or len(history) == 0:
            logger.warning(f"No history data for {weather_entity_id}")
            return {}
        
        entity_data = history[0] if isinstance(history[0], list) else history
        
        # Build dataframes for each attribute
        attribute_dfs = {}
        
        for attr_name in attributes:
            data_points = []
            skipped_count = 0
            
            for record in entity_data:
                try:
                    # Extract attribute value from the record
                    attr_value = record.get('attributes', {}).get(attr_name)
                    
                    if attr_value is None:
                        skipped_count += 1
                        continue
                    
                    timestamp = pd.to_datetime(record['last_changed'])
                    data_points.append({'ds': timestamp, f"weather_{attr_name}": float(attr_value)})
                    
                except (ValueError, KeyError, TypeError) as e:
                    skipped_count += 1
                    continue
            
            if data_points:
                df = pd.DataFrame(data_points)
                attribute_dfs[f"weather_{attr_name}"] = df
                logger.info(f"Fetched {len(df)} historical values for weather attribute: {attr_name}")
            else:
                logger.warning(f"No valid data for weather attribute: {attr_name}")
        
        return attribute_dfs
        
    except Exception as e:
        logger.error(f"Error fetching weather forecast regressors: {e}", exc_info=True)
        return {}


def get_weather_forecast_future_values(ha_api, weather_entity_id, attributes, intervals_to_predict, interval_duration):
    """Get future values from weather forecast for prediction
    
    Args:
        ha_api: HomeAssistantAPI instance
        weather_entity_id: Weather entity ID
        attributes: List of attributes to extract
        intervals_to_predict: Number of intervals to predict
        interval_duration: Duration of each interval in minutes
        
    Returns:
        Dictionary mapping attribute names to lists of future values
    """
    try:
        forecast = ha_api.get_weather_forecast(weather_entity_id)
        if not forecast:
            return {}
        
        # Convert forecast to time-based values
        future_values = {f"weather_{attr}": [] for attr in attributes}
        
        current_time = datetime.now()
        
        for i in range(intervals_to_predict):
            target_time = current_time + timedelta(minutes=i * interval_duration)
            
            # Find the closest forecast period
            closest_forecast = None
            min_diff = None
            
            for fc in forecast:
                if 'datetime' in fc:
                    fc_time = pd.to_datetime(fc['datetime'])
                    diff = abs((fc_time - target_time).total_seconds())
                    if min_diff is None or diff < min_diff:
                        min_diff = diff
                        closest_forecast = fc
            
            # Extract attribute values
            if closest_forecast:
                for attr in attributes:
                    value = closest_forecast.get(attr)
                    future_values[f"weather_{attr}"].append(float(value) if value is not None else None)
            else:
                # No forecast available, use None
                for attr in attributes:
                    future_values[f"weather_{attr}"].append(None)
        
        logger.info(f"Generated {intervals_to_predict} future values from weather forecast")
        return future_values
        
    except Exception as e:
        logger.error(f"Error getting weather forecast future values: {e}", exc_info=True)
        return {}


def prepare_training_data(history_data, cumulative=False):
    """Convert Home Assistant history data to NeuralProphet format
    
    Args:
        history_data: List of history records from HA API
        cumulative: If True, compute differences for cumulative sensors (energy meters)
        
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
    
    # For cumulative sensors (energy meters), compute the difference between readings
    if cumulative:
        logger.info("Converting cumulative data to rate by computing differences")
        df = df.sort_values('ds').reset_index(drop=True)
        
        # Compute differences
        df['y_diff'] = df['y'].diff()
        
        # Handle negative differences (meter reset) by setting to 0 or NaN
        # Also handle very large jumps that indicate a reset
        median_diff = df['y_diff'].median()
        threshold = median_diff * 100  # Assume jumps > 100x median are resets
        
        df.loc[df['y_diff'] < 0, 'y_diff'] = 0  # Negative differences = 0
        df.loc[df['y_diff'] > threshold, 'y_diff'] = median_diff  # Large jumps = median
        
        # Replace y with the difference
        df['y'] = df['y_diff']
        
        # Drop the first row (no difference) and the helper column
        df = df.dropna(subset=['y'])
        df = df.drop(columns=['y_diff'])
        
        logger.info(f"Converted cumulative data: {len(df)} rate values")
    
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
        cumulative = sensor_config.get('cumulative', False)
        regressors = sensor_config.get('regressors', [])
        weather_forecast_config = sensor_config.get('weather_forecast')
        
        logger.info(f"Processing sensor:")
        logger.info(f"  - Training from: {training_entity_id}")
        logger.info(f"  - Prediction to: {prediction_entity_id}")
        logger.info(f"  - History: {history_days} days")
        logger.info(f"  - Interval duration: {interval_duration} minutes")
        logger.info(f"  - Intervals to predict: {intervals_to_predict}")
        logger.info(f"  - Units: {units}")
        logger.info(f"  - Cumulative: {cumulative}")
        logger.info(f"  - Regressors: {regressors if regressors else 'None'}")
        logger.info(f"  - Weather forecast: {weather_forecast_config.get('entity_id') if weather_forecast_config else 'None'}")
        
        # Fetch history for this sensor
        try:
            training_data = ha_api.get_history(training_entity_id, days=history_days, minimal_response=True)
            
            if not training_data or len(training_data) == 0:
                logger.error(f"No history data retrieved for {training_entity_id}")
                continue
            
            logger.info(f"Retrieved {len(training_data[0])} history records for {training_entity_id}")
            
            # Prepare data for NeuralProphet
            df = prepare_training_data(training_data, cumulative=cumulative)
            
            if df is not None and len(df) > 0:
                logger.info(f"Training data prepared successfully for {training_entity_id}")
                logger.info(f"  - Data points: {len(df)}")
                logger.info(f"  - Date range: {df['ds'].min()} to {df['ds'].max()}")
                
                # Fetch and merge regressor data if configured
                regressor_columns = []
                weather_regressor_future = {}
                
                if regressors:
                    for regressor_config in regressors:
                        # Handle both old format (string) and new format (dict)
                        if isinstance(regressor_config, str):
                            regressor_entity_id = regressor_config
                            regressor_type = 'sensor'
                        else:
                            regressor_entity_id = regressor_config.get('entity_id')
                            regressor_type = regressor_config.get('type', 'sensor')
                        
                        if regressor_type == 'sensor':
                            regressor_df = fetch_regressor_data(ha_api, regressor_entity_id, history_days)
                            if regressor_df is not None:
                                # Merge regressor data with training data on timestamp
                                df = pd.merge_asof(df.sort_values('ds'), 
                                                  regressor_df.sort_values('ds'), 
                                                  on='ds', 
                                                  direction='nearest',
                                                  tolerance=pd.Timedelta(minutes=interval_duration))
                                regressor_columns.append(regressor_entity_id)
                                logger.info(f"  - Added sensor regressor: {regressor_entity_id}")
                            else:
                                logger.warning(f"  - Failed to add sensor regressor: {regressor_entity_id}")
                
                # Fetch weather forecast regressors
                if weather_forecast_config:
                    weather_entity_id = weather_forecast_config.get('entity_id')
                    weather_attributes = weather_forecast_config.get('attributes', [])
                    
                    if weather_entity_id and weather_attributes:
                        weather_dfs = fetch_weather_forecast_regressors(ha_api, weather_entity_id, weather_attributes, history_days)
                        
                        for attr_name, weather_df in weather_dfs.items():
                            # Merge weather data with training data
                            df = pd.merge_asof(df.sort_values('ds'), 
                                              weather_df.sort_values('ds'), 
                                              on='ds', 
                                              direction='nearest',
                                              tolerance=pd.Timedelta(minutes=interval_duration))
                            regressor_columns.append(attr_name)
                            logger.info(f"  - Added weather regressor: {attr_name}")
                        
                        # Get future weather forecast values for predictions
                        weather_regressor_future = get_weather_forecast_future_values(
                            ha_api, weather_entity_id, weather_attributes, 
                            intervals_to_predict, interval_duration
                        )
                
                if regressor_columns:
                    # Drop any rows with NaN values in regressors
                    df = df.dropna()
                    logger.info(f"  - Data points after regressor merge: {len(df)}")
                
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
                
                # Add regressors to the model
                for regressor_col in regressor_columns:
                    model.add_future_regressor(name=regressor_col)
                    logger.info(f"  - Added regressor to model: {regressor_col}")
                
                # Fit the model
                metrics = model.fit(df, freq=freq)
                logger.info(f"Model training complete for {training_entity_id}")
                
                # Make future dataframe for predictions
                future = model.make_future_dataframe(df, periods=intervals_to_predict, n_historic_predictions=True)
                
                # For regressors, we need to provide future values
                if regressor_columns:
                    for regressor_col in regressor_columns:
                        # Check if this is a weather forecast regressor with future values
                        if regressor_col in weather_regressor_future and weather_regressor_future[regressor_col]:
                            # Use weather forecast future values
                            future_values = weather_regressor_future[regressor_col]
                            # Get the last n rows (future predictions)
                            future_rows = future.tail(intervals_to_predict)
                            for idx, (_, row) in enumerate(future_rows.iterrows()):
                                if idx < len(future_values) and future_values[idx] is not None:
                                    future.loc[row.name, regressor_col] = future_values[idx]
                            logger.info(f"  - Using weather forecast values for {regressor_col}")
                        else:
                            # Use the last known value for regular sensors
                            last_value = df[regressor_col].iloc[-1]
                            future[regressor_col] = future[regressor_col].fillna(last_value)
                            logger.info(f"  - Using last value {last_value:.2f} for regressor {regressor_col}")
                
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