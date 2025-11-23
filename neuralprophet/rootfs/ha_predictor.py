import logging
import yaml
import pandas as pd
import numpy as np
import time
import os
from pathlib import Path
from datetime import datetime, timedelta
from homeassistantapi import HomeAssistantAPI
from neuralprophet import NeuralProphet

import warnings

# Suppress warnings from third-party libraries
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', message='.*pkg_resources.*')

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


def update_prediction_entity(ha_api, entity_id, forecast_df, units='', non_negative=False):
    """Update a Home Assistant entity with prediction data
    
    Args:
        ha_api: HomeAssistantAPI instance
        entity_id: Entity ID to update
        forecast_df: DataFrame with forecast (must have 'ds' and 'yhat1' columns)
        units: Unit of measurement
        non_negative: If True, clip predictions to non-negative values (for energy meters, etc.)
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Clip predictions to non-negative if required (for cumulative/energy sensors)
        if non_negative:
            forecast_df = forecast_df.copy()
            negative_count = (forecast_df['yhat1'] < 0).sum()
            if negative_count > 0:
                logger.info(f"Clipping {negative_count} negative predictions to 0 for {entity_id}")
                forecast_df['yhat1'] = forecast_df['yhat1'].clip(lower=0)
        
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


def fetch_regressor_data(ha_api, regressor_entity_id, history_days, interval_minutes=30):
    """Fetch regressor data from Home Assistant and resample to regular intervals
    
    Args:
        ha_api: HomeAssistantAPI instance
        regressor_entity_id: Entity ID of the regressor
        history_days: Number of days of history to fetch
        interval_minutes: Target interval in minutes for resampling (default 30)
        
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
        
        df_raw = pd.DataFrame(data_points)
        df_raw = df_raw.sort_values('ds').reset_index(drop=True)
        
        # Resample to regular intervals
        start_time = df_raw['ds'].min().floor(f'{interval_minutes}min')
        end_time = df_raw['ds'].max().ceil(f'{interval_minutes}min')
        regular_index = pd.date_range(start=start_time, end=end_time, freq=f'{interval_minutes}min')
        
        df = pd.DataFrame({'ds': regular_index})
        df[regressor_entity_id] = None
        
        for idx, row in df.iterrows():
            target_time = row['ds']
            time_diff = (df_raw['ds'] - target_time).abs()
            closest_idx = time_diff.idxmin()
            
            if time_diff.iloc[closest_idx] <= pd.Timedelta(minutes=interval_minutes/2):
                df.at[idx, regressor_entity_id] = df_raw.at[closest_idx, regressor_entity_id]
        
        # Forward fill missing values
        df[regressor_entity_id] = df[regressor_entity_id].ffill()
        df = df.dropna(subset=[regressor_entity_id])
        
        logger.info(f"Fetched and resampled {len(df_raw)} data points to {len(df)} regular intervals for regressor {regressor_entity_id}")
        
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


def get_weather_forecast_future_values(ha_api, weather_entity_id, attribute_mappings, intervals_to_predict, interval_duration):
    """Get future values from weather forecast for prediction
    
    Args:
        ha_api: HomeAssistantAPI instance
        weather_entity_id: Weather entity ID
        attribute_mappings: List of dicts with 'attribute' and 'maps_to_sensor' keys,
                           or list of attribute names (backward compatible)
        intervals_to_predict: Number of intervals to predict
        interval_duration: Duration of each interval in minutes
        
    Returns:
        Dictionary mapping sensor entity IDs to lists of future values
    """
    try:
        # Get weather forecast
        forecast_data = ha_api.get_weather_forecast(weather_entity_id)
        
        if not forecast_data:
            logger.warning(f"No forecast data available for {weather_entity_id}")
            return {}
        
        # Parse attribute mappings (support both old and new format)
        mappings = []
        for item in attribute_mappings:
            if isinstance(item, dict):
                mappings.append({
                    'attribute': item.get('attribute'),
                    'sensor': item.get('maps_to_sensor', item.get('attribute'))
                })
            else:
                # Old format: just attribute name
                mappings.append({'attribute': item, 'sensor': item})
        
        # Convert forecast to time-based values
        current_time = pd.Timestamp.now(tz='UTC')
        result = {}
        
        # Initialize result dict with sensor IDs
        for mapping in mappings:
            result[mapping['sensor']] = []
        
        for i in range(intervals_to_predict):
            target_time = current_time + pd.Timedelta(minutes=i * interval_duration)
            
            # Find the closest forecast period
            closest_forecast = None
            min_diff = None
            
            for fc in forecast_data:
                if 'datetime' in fc:
                    fc_time = pd.to_datetime(fc['datetime'])
                    diff = abs((fc_time - target_time).total_seconds())
                    if min_diff is None or diff < min_diff:
                        min_diff = diff
                        closest_forecast = fc
            
            # Extract attribute values
            if closest_forecast:
                for mapping in mappings:
                    attr_name = mapping['attribute']
                    sensor_id = mapping['sensor']
                    value = closest_forecast.get(attr_name)
                    result[sensor_id].append(float(value) if value is not None else None)
            else:
                # No forecast available, use None
                for mapping in mappings:
                    result[mapping['sensor']].append(None)
        
        logger.info(f"Generated {intervals_to_predict} future values from weather forecast")
        for sensor_id in result.keys():
            logger.info(f"  - {sensor_id}: {len(result[sensor_id])} values")
        
        return result
        
    except Exception as e:
        logger.error(f"Error getting weather forecast future values: {e}", exc_info=True)
        return {}


def prepare_training_data(history_data, cumulative=False, interval_minutes=30):
    """Convert Home Assistant history data to NeuralProphet format with regular intervals
    
    Args:
        history_data: List of history records from HA API
        cumulative: If True, compute differences for cumulative sensors (energy meters)
        interval_minutes: Target interval in minutes for resampling (default 30)
        
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
    
    # Create DataFrame with irregular timestamps
    df_raw = pd.DataFrame(data_points)
    df_raw = df_raw.sort_values('ds').reset_index(drop=True)
    
    # Resample to regular intervals
    # Start from the first timestamp rounded down to the interval
    start_time = df_raw['ds'].min().floor(f'{interval_minutes}min')
    end_time = df_raw['ds'].max().ceil(f'{interval_minutes}min')
    
    # Create regular time index
    regular_index = pd.date_range(start=start_time, end=end_time, freq=f'{interval_minutes}min')
    
    # Create resampled dataframe
    df = pd.DataFrame({'ds': regular_index})
    
    # For each regular interval, find the closest raw data point
    df['y'] = None
    for idx, row in df.iterrows():
        target_time = row['ds']
        # Find closest data point within a tolerance window
        time_diff = (df_raw['ds'] - target_time).abs()
        closest_idx = time_diff.idxmin()
        
        # Only use if within tolerance (half the interval)
        if time_diff.iloc[closest_idx] <= pd.Timedelta(minutes=interval_minutes/2):
            df.at[idx, 'y'] = df_raw.at[closest_idx, 'y']
    
    # Forward fill missing values (carry last observation forward)
    df['y'] = df['y'].ffill()
    
    # Drop any remaining NaN at the start
    df = df.dropna(subset=['y'])
    
    logger.info(f"Resampled {len(df_raw)} irregular data points to {len(df)} regular intervals of {interval_minutes} minutes")
    
    # For cumulative sensors (energy meters), compute the difference between readings
    if cumulative:
        logger.info("Converting cumulative data to rate by computing differences")
        df = df.sort_values('ds').reset_index(drop=True)
        
        # Compute differences
        df['y_diff'] = df['y'].diff()
        
        # Handle negative differences (meter reset) by setting to 0 or NaN
        # Also handle very large jumps that indicate a reset
        median_diff = df['y_diff'].median()
        threshold = median_diff * 100 if median_diff > 0 else 1000  # Assume jumps > 100x median are resets
        
        df.loc[df['y_diff'] < 0, 'y_diff'] = 0  # Negative differences = 0
        df.loc[df['y_diff'] > threshold, 'y_diff'] = median_diff  # Large jumps = median
        
        # Replace y with the difference
        df['y'] = df['y_diff']
        
        # Drop the first row (no difference) and the helper column
        df = df.dropna(subset=['y'])
        df = df.drop(columns=['y_diff'])
        
        logger.info(f"Converted cumulative data: {len(df)} rate values")
        
        # Check if all values are the same (singular value issue)
        unique_values = df['y'].nunique()
        if unique_values <= 1:
            logger.error(f"Target variable has only {unique_values} unique value(s) after cumulative conversion. Cannot train model.")
            return None
        logger.info(f"Target variable has {unique_values} unique values (min: {df['y'].min():.4f}, max: {df['y'].max():.4f})")
        
        # For debugging: check nighttime values (should be zero or near-zero for solar)
        night_mask = (df['ds'].dt.hour >= 22) | (df['ds'].dt.hour <= 4)
        night_values = df[night_mask]['y']
        if len(night_values) > 0:
            zero_count = (night_values == 0).sum()
            logger.info(f"Training data nighttime (10pm-4am): {len(night_values)} intervals, {zero_count} zeros ({100*zero_count/len(night_values):.1f}%)")
            logger.info(f"  Non-zero nighttime values: min={night_values[night_values > 0].min():.6f}, max={night_values.max():.6f}, mean={night_values[night_values > 0].mean():.6f}")
    
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
            
            # Prepare data for NeuralProphet with regular intervals
            df = prepare_training_data(training_data, cumulative=cumulative, interval_minutes=interval_duration)
            
            if df is not None and len(df) > 0:
                logger.info(f"Training data prepared successfully for {training_entity_id}")
                logger.info(f"  - Data points: {len(df)}")
                logger.info(f"  - Date range: {df['ds'].min()} to {df['ds'].max()}")
                
                # Fetch and merge regressor data if configured
                regressor_columns = []
                regressor_future_mapping = {}  # Maps regressor entity_id to future data source
                
                if regressors:
                    for regressor_config in regressors:
                        # Handle both old format (string) and new format (dict)
                        if isinstance(regressor_config, str):
                            regressor_entity_id = regressor_config
                            regressor_type = 'sensor'
                            future_entity_id = None
                            future_entity_type = None
                            future_attribute = None
                        else:
                            regressor_entity_id = regressor_config.get('entity_id')
                            regressor_type = regressor_config.get('type', 'sensor')
                            future_entity_id = regressor_config.get('future_entity_id')
                            future_entity_type = regressor_config.get('future_entity_type')
                            future_attribute = regressor_config.get('future_attribute')
                        
                        if regressor_type == 'sensor':
                            regressor_df = fetch_regressor_data(ha_api, regressor_entity_id, history_days, interval_minutes=interval_duration)
                            if regressor_df is not None:
                                # Check if regressor has sufficient variance (not all same value)
                                regressor_values = regressor_df[regressor_entity_id].dropna()
                                if len(regressor_values.unique()) <= 1:
                                    logger.warning(f"  - Skipping regressor {regressor_entity_id}: only has {len(regressor_values.unique())} unique value(s)")
                                    continue
                                
                                # Merge regressor data with training data on timestamp
                                df = pd.merge_asof(df.sort_values('ds'), 
                                                  regressor_df.sort_values('ds'), 
                                                  on='ds', 
                                                  direction='nearest',
                                                  tolerance=pd.Timedelta(minutes=interval_duration))
                                
                                # Check variance after merge
                                merged_values = df[regressor_entity_id].dropna()
                                if len(merged_values.unique()) <= 1:
                                    logger.warning(f"  - Skipping regressor {regressor_entity_id}: no variance after merge")
                                    df = df.drop(columns=[regressor_entity_id])
                                    continue
                                
                                regressor_columns.append(regressor_entity_id)
                                logger.info(f"  - Added sensor regressor: {regressor_entity_id}")
                                
                                # Store future entity mapping if specified
                                if future_entity_id:
                                    regressor_future_mapping[regressor_entity_id] = {
                                        'entity_id': future_entity_id,
                                        'entity_type': future_entity_type,
                                        'attribute': future_attribute
                                    }
                                    logger.info(f"    → Future values from: {future_entity_id} ({future_entity_type}.{future_attribute})")
                            else:
                                logger.warning(f"  - Failed to add sensor regressor: {regressor_entity_id}")
                
                # Fetch future values for regressors with weather entity mapping
                logger.info(f"Fetching future regressor values...")
                for regressor_id, future_config in regressor_future_mapping.items():
                    future_entity_type = future_config.get('entity_type')
                    
                    if future_entity_type == 'weather':
                        # Get weather forecast
                        future_entity_id = future_config['entity_id']
                        future_attribute = future_config['attribute']
                        
                        logger.info(f"  - Getting weather forecast from {future_entity_id} for attribute '{future_attribute}'")
                        forecast_data = ha_api.get_weather_forecast(future_entity_id)
                        
                        if forecast_data:
                            logger.info(f"    → Received {len(forecast_data)} forecast periods")
                        else:
                            logger.warning(f"    → No forecast data received from {future_entity_id}")
                
                if regressor_columns:
                    # Drop any rows with NaN values in regressors
                    df = df.dropna()
                    logger.info(f"  - Data points after regressor merge: {len(df)}")
                
                logger.info(f"  - Sample data:\n{df.head()}")
                
                #Train NeuralProphet model
                logger.info(f"Training NeuralProphet model for {training_entity_id}...")
                
                freq = f"{interval_duration}min"
                
                # Use n_lags for autoregression (0 = auto-detect)
                n_lags = sensor_config.get('n_lags', 0)
                
                # For cumulative sensors (like solar), use more Fourier terms to capture
                # sharp daily patterns (e.g., zero at night, peak at midday)
                daily_seasonality_order = 20 if cumulative else 'auto'
                
                model = NeuralProphet(
                    n_lags=n_lags,
                    daily_seasonality=daily_seasonality_order,
                    weekly_seasonality=True,
                    yearly_seasonality=True,
                    epochs=100 if cumulative else 50,  # More epochs for complex patterns
                    learning_rate=0.01,
                    batch_size=32,
                    drop_missing=True
                )
                
                # Extend the dataset with future timestamps BEFORE training
                # This is the key to multi-step forecasting with NeuralProphet
                last_timestamp = df['ds'].max()
                if last_timestamp.tz is None:
                    last_timestamp = last_timestamp.tz_localize('UTC')
                
                future_timestamps = []
                for i in range(1, intervals_to_predict + 1):
                    future_timestamp = last_timestamp + timedelta(minutes=interval_duration * i)
                    future_timestamps.append({'ds': future_timestamp, 'y': np.nan})
                
                future_df = pd.DataFrame(future_timestamps)
                # Explicitly set dtype to match the dataset
                future_df['y'] = future_df['y'].astype('float64')
                extended_df = pd.concat([df, future_df], ignore_index=True)
                
                logger.info(f"  - Extended dataset with {len(future_df)} future timestamps")
                
                # If we have regressors, add them and merge their future values
                if regressor_columns:
                    # Add all regressors to the model
                    for regressor_col in regressor_columns:
                        if regressor_col in df.columns:
                            model.add_future_regressor(name=regressor_col)
                            logger.info(f"  - Added regressor to model: {regressor_col}")
                            
                            # Now merge regressor data into extended dataset
                            # Get regressor values from sensor configuration
                            regressor_entity = None
                            regressor_future_entity = None
                            regressor_future_type = None
                            regressor_future_attribute = None
                            
                            for reg_cfg in sensor_config.get('regressors', []):
                                if reg_cfg.get('name') == regressor_col:
                                    regressor_entity = reg_cfg.get('entity_id')
                                    regressor_future_entity = reg_cfg.get('future_entity_id')
                                    regressor_future_type = reg_cfg.get('future_entity_type')
                                    regressor_future_attribute = reg_cfg.get('future_attribute', 'temperature')
                                    break
                            
                            # For future timestamps, get forecast data
                            if regressor_future_entity and regressor_future_type == 'weather':
                                logger.info(f"  - Fetching weather forecast for regressor {regressor_col}")
                                forecast_data = ha_api.get_weather_forecast(regressor_future_entity)
                                
                                if forecast_data:
                                    logger.info(f"    Retrieved {len(forecast_data)} forecast periods")
                                    
                                    # Create regressor dataframe from forecast
                                    regressor_future_rows = []
                                    for forecast_entry in forecast_data:
                                        timestamp_str = forecast_entry.get('datetime')
                                        value = forecast_entry.get(regressor_future_attribute)
                                        
                                        if timestamp_str and value is not None:
                                            try:
                                                forecast_time = pd.to_datetime(timestamp_str, utc=True)
                                                # Round to interval
                                                forecast_time = forecast_time.floor(f"{interval_duration}min")
                                                regressor_future_rows.append({
                                                    'ds': forecast_time,
                                                    regressor_col: float(value)
                                                })
                                            except (ValueError, TypeError) as e:
                                                logger.warning(f"    Could not parse forecast entry: {e}")
                                                continue
                                    
                                    if regressor_future_rows:
                                        regressor_future_df = pd.DataFrame(regressor_future_rows)
                                        regressor_future_df = regressor_future_df.drop_duplicates(subset=['ds'], keep='first')
                                        
                                        # Merge into extended dataset
                                        extended_df = extended_df.merge(
                                            regressor_future_df,
                                            on='ds',
                                            how='left',
                                            suffixes=('', '_forecast')
                                        )
                                        
                                        # If we have both columns, prefer the forecast for future dates
                                        forecast_col = f"{regressor_col}_forecast"
                                        if forecast_col in extended_df.columns:
                                            # Fill regressor column with forecast values where it's NaN
                                            extended_df[regressor_col] = extended_df[regressor_col].fillna(extended_df[forecast_col])
                                            extended_df = extended_df.drop(columns=[forecast_col])
                                        
                                        logger.info(f"    Merged {len(regressor_future_df)} forecast values for {regressor_col}")
                            
                            # Forward/backward fill any remaining missing values
                            extended_df[regressor_col] = extended_df[regressor_col].ffill().bfill()
                            null_count = extended_df[regressor_col].isna().sum()
                            logger.info(f"    After fill, {regressor_col} has {null_count} nulls")
                            
                            # Check for singular value after filling
                            regressor_values_in_extended = extended_df[regressor_col].dropna()
                            unique_count = regressor_values_in_extended.nunique()
                            if unique_count <= 1:
                                logger.warning(f"    Regressor {regressor_col} has only {unique_count} unique value(s) in extended dataset. Removing from model.")
                                extended_df = extended_df.drop(columns=[regressor_col])
                                regressor_columns.remove(regressor_col)
                                continue
                            logger.info(f"    Regressor {regressor_col} has {unique_count} unique values in extended dataset")
                        else:
                            logger.warning(f"  - Regressor {regressor_col} not found in training data, skipping")
                
                # Fit the model on the extended dataset (NeuralProphet handles NaN y values)
                metrics = model.fit(extended_df, freq=freq)
                logger.info(f"Model training complete for {training_entity_id}")
                
                # Since extended_df already includes future timestamps with regressor values,
                # use periods=0 and n_historic_predictions=True (like the working example)
                logger.info(f"  - Creating future dataframe from extended dataset")
                future = model.make_future_dataframe(
                    extended_df, 
                    n_historic_predictions=True, 
                    periods=0
                )
                
                logger.info(f"  - Future dataframe created with {len(future)} rows")
                
                # Generate forecast
                logger.info(f"Generating {intervals_to_predict} interval predictions...")
                forecast = model.predict(future)
                
                logger.info(f"  - Model returned {len(forecast)} forecast rows")
                logger.info(f"  - Forecast columns: {[col for col in forecast.columns if 'yhat' in col]}")
                
                # Ensure forecast['ds'] is timezone-aware
                if not pd.api.types.is_datetime64_any_dtype(forecast['ds']):
                    forecast['ds'] = pd.to_datetime(forecast['ds'], utc=True)
                elif forecast['ds'].dt.tz is None:
                    forecast['ds'] = forecast['ds'].dt.tz_localize('UTC')
                
                # Get the last training timestamp to filter future predictions
                last_training_time = df['ds'].max()
                if last_training_time.tz is None:
                    last_training_time = last_training_time.tz_localize('UTC')
                
                # Filter for future predictions only
                future_forecast = forecast[forecast['ds'] > last_training_time].copy()
                future_forecast = future_forecast[['ds', 'yhat1']].copy()
                
                logger.info(f"  - Filtered to {len(future_forecast)} future predictions")
                
                logger.info(f"Forecast generated: {len(future_forecast)} predictions")
                logger.info(f"  - Forecast range: {future_forecast['ds'].min()} to {future_forecast['ds'].max()}")
                logger.info(f"  - Value range: {future_forecast['yhat1'].min():.4f} to {future_forecast['yhat1'].max():.4f}")
                logger.info(f"  - Sample forecast:\n{future_forecast[['ds', 'yhat1']].head()}")
                
                # For cumulative sensors, also show nighttime predictions (should be near zero)
                if cumulative and len(future_forecast) > 0:
                    # Check predictions between 10pm-4am
                    night_mask = (future_forecast['ds'].dt.hour >= 22) | (future_forecast['ds'].dt.hour <= 4)
                    night_predictions = future_forecast[night_mask]
                    if len(night_predictions) > 0:
                        logger.info(f"  - Nighttime predictions (10pm-4am): {len(night_predictions)} intervals")
                        logger.info(f"    Min: {night_predictions['yhat1'].min():.4f}, Max: {night_predictions['yhat1'].max():.4f}, Mean: {night_predictions['yhat1'].mean():.4f}")
                
                # Update prediction entity in Home Assistant
                # For cumulative sensors, clip predictions to non-negative values
                success = update_prediction_entity(ha_api, prediction_entity_id, future_forecast, units, non_negative=cumulative)
                
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

# Configuration path - use DEV_CONFIG_PATH for local development, otherwise addon path
dev_config_path = os.environ.get('DEV_CONFIG_PATH')
if dev_config_path:
    config_path = Path(dev_config_path)
    logger.info(f"Running in development mode with config: {config_path}")
else:
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