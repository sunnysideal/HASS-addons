import logging
logger = logging.getLogger(__name__)

import pvlib
import yaml
import pandas as pd
import numpy as np
import time
import os
from pathlib import Path
from datetime import datetime, timedelta
from homeassistantapi import HomeAssistantAPI
from neuralprophet import NeuralProphet
from database import Database

import warnings

# Suppress warnings from third-party libraries
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', message='.*pkg_resources.*')
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)


# === Utility/Helper Functions ===
def get_regressor_names(regressors):
    """Return a list of regressor names for logging/debugging."""
    names = []
    for r in regressors:
        if isinstance(r, str):
            names.append(r)
        elif isinstance(r, dict):
            # Prefer 'name', fallback to 'entity_id', else stringified dict
            names.append(r.get('name') or r.get('entity_id') or str(r))
        else:
            names.append(str(r))
    return names

def log_sensor_config(sensor_config):
    """Log the key configuration for a sensor in a consistent, readable way."""
    logger = logging.getLogger(__name__)
    logger.info(f"Processing sensor:")
    logger.info(f"  - Training from: {sensor_config.get('training_entity_id')}")
    logger.info(f"  - Prediction to: {sensor_config.get('prediction_entity_id')}")
    logger.info(f"  - History: {sensor_config.get('history_days')}")
    logger.info(f"  - Interval duration: {sensor_config.get('interval_duration')}")
    logger.info(f"  - Intervals to predict: {sensor_config.get('intervals_to_predict')}")
    logger.info(f"  - Units: {sensor_config.get('units')}")
    logger.info(f"  - Cumulative: {sensor_config.get('cumulative')}")
    logger.info(f"  - Database: {sensor_config.get('database')}")
    logger.info(f"  - Max age: {sensor_config.get('max_age')}")
    logger.info(f"  - Regressors: {get_regressor_names(sensor_config.get('regressors', []))}")
    wf = sensor_config.get('weather_forecast')
    logger.info(f"  - Weather forecast: {wf.get('entity_id') if wf else 'None'}")

def calculate_sun_az_el(df, latitude, longitude, elevation=0):
    """Calculate sun azimuth and elevation for each timestamp in df['ds'] using pvlib."""
    df = df.set_index('ds')
    times = pd.DatetimeIndex(df.index).tz_convert('UTC') if df.index.tz is not None else pd.DatetimeIndex(df.index).tz_localize('UTC')
    solpos = pvlib.solarposition.get_solarposition(times, latitude, longitude, elevation)
    logger.info(f"Solar position sample:\n{solpos.head()}")
    # Check index alignment before assignment
    if not df.index.equals(solpos.index):
        logger.warning(f"Index mismatch between df and solpos when assigning sun_azimuth/elevation. Assigning by position.")
        df['sun_azimuth'] = solpos['azimuth'].values
        df['sun_elevation'] = solpos['elevation'].values
    else:
        df['sun_azimuth'] = solpos['azimuth']
        df['sun_elevation'] = solpos['elevation']
    logger.info(f"Calculated sun positions for {len(df)} timestamps")
    df = df.reset_index()[['ds','sun_azimuth','sun_elevation']]
    # logger info head of df
    logger.info(f"Sun position sample:\n{df[['ds', 'sun_azimuth', 'sun_elevation']].head()}")
    return df

def load_config(config_path):
    """Load configuration from YAML file
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Dictionary with configuration or empty dict if not found
    """
    config = {}
    
    logger.debug(f"Checking for config at: {config_path}")   
    logger.debug(f"Config Exists: {config_path.exists()}")
    
    try:
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f) or {}
            logger.info(f"Loaded configuration from {config_path}: {config}")
            logger.debug(f"Config: \n {config}")
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



def fetch_regressor_data(ha_api, regressor_entity_id, history_days, interval_minutes=30, db=None, use_database=False):
    """Fetch regressor data from Home Assistant and/or database, resample to regular intervals
    Args:
        ha_api: HomeAssistantAPI instance
        regressor_entity_id: Entity ID of the regressor
        history_days: Number of days of history to fetch
        interval_minutes: Target interval in minutes for resampling (default 30)
        db: Database instance (optional)
        use_database: Whether to use the database for historic regressor data
    Returns:
        pandas DataFrame with columns 'ds' (datetime) and regressor name (value)
    """
    try:
        db_data = None
        if use_database and db is not None:
            try:
                db_data = db.get_history(regressor_entity_id)
                if db_data is not None and len(db_data) > 0:
                    logger.info(f"Retrieved {len(db_data)} rows from database for regressor {regressor_entity_id}")
            except Exception as e:
                logger.error(f"Failed to retrieve database history for regressor {regressor_entity_id}: {e}")
                db_data = None

        # Fetch recent data from Home Assistant API
        api_days = min(7, history_days) if (use_database and db_data is not None and len(db_data) > 0) else history_days
        regressor_data = ha_api.get_history(regressor_entity_id, days=api_days, minimal_response=True)

        data_points = []
        # Add database data points
        if db_data is not None and len(db_data) > 0:
            # db_data is expected to have 'ds' and regressor_entity_id columns
            for _, row in db_data.iterrows():
                try:
                    data_points.append({'ds': row['ds'], regressor_entity_id: row[regressor_entity_id]})
                except Exception:
                    continue
        # Add API data points
        if regressor_data and len(regressor_data) > 0:
            entity_data = regressor_data[0] if isinstance(regressor_data[0], list) else regressor_data
            for record in entity_data:
                try:
                    state = record.get('state')
                    if state is None or state in ['unknown', 'unavailable', '']:
                        continue
                    timestamp = pd.to_datetime(record['last_changed'])
                    state_value = float(state)
                    data_points.append({'ds': timestamp, regressor_entity_id: state_value})
                except (ValueError, KeyError, TypeError):
                    continue

        if not data_points:
            logger.warning(f"No valid data points found for regressor {regressor_entity_id}")
            return None

        df_raw = pd.DataFrame(data_points)
        df_raw = df_raw.drop_duplicates(subset=['ds'], keep='last')
        df_raw = df_raw.sort_values('ds').reset_index(drop=True)

        # Resample to regular intervals
        start_time = df_raw['ds'].min().floor(f'{interval_minutes}min')
        now = pd.Timestamp.now(tz=df_raw['ds'].dt.tz) if df_raw['ds'].dt.tz is not None else pd.Timestamp.now()
        last_completed = (now.floor(f'{interval_minutes}min') - pd.Timedelta(minutes=interval_minutes))
        max_time = df_raw['ds'].max().ceil(f'{interval_minutes}min')
        end_time = min(max_time, last_completed)
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
    if len(df_raw) > 0:
        now = pd.Timestamp.now(tz=df_raw['ds'].dt.tz) if df_raw['ds'].dt.tz is not None else pd.Timestamp.now()
        logger.info(f"Raw data timestamp range: {df_raw['ds'].min()} to {df_raw['ds'].max()} (now: {now})")
        future_count = (df_raw['ds'] > now).sum()
        if future_count > 0:
            logger.warning(f"Raw data contains {future_count} timestamps in the future.")
    
    # Resample to regular intervals
    # Start from the first timestamp rounded down to the interval
    start_time = df_raw['ds'].min().floor(f'{interval_minutes}min')
    # Never allow end_time to be in the future
    now = pd.Timestamp.now(tz=df_raw['ds'].dt.tz) if df_raw['ds'].dt.tz is not None else pd.Timestamp.now()
    # The last fully completed interval is the most recent interval strictly before 'now'
    last_completed = (now.floor(f'{interval_minutes}min') - pd.Timedelta(minutes=interval_minutes))
    max_time = df_raw['ds'].max().ceil(f'{interval_minutes}min')
    end_time = min(max_time, last_completed)
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
        median_diff = df['y_diff'].median()
        threshold = median_diff * 100 if median_diff > 0 else 1000  # Assume jumps > 100x median are resets
        df.loc[df['y_diff'] < 0, 'y_diff'] = 0  # Negative differences = 0
        df.loc[df['y_diff'] > threshold, 'y_diff'] = median_diff  # Large jumps = median
        # Shift timestamps backward by one interval so increment is labeled with the beginning
        df['ds'] = df['ds'] - pd.Timedelta(minutes=interval_minutes)
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


def process_sensors(ha_api, sensors, default_history_days, default_interval_duration, default_intervals_to_predict, db=None, db_config=None):
    """Process all sensors and prepare training data
    
    Args:
        ha_api: HomeAssistantAPI instance
        sensors: List of sensor configurations
        default_history_days: Default number of days if not specified per sensor
        default_interval_duration: Default interval duration in minutes
        default_intervals_to_predict: Default number of intervals to predict
        db: Database instance for persistent storage (optional)
        db_config: Global database configuration dict (optional)
    """
    # Process each sensor
    for sensor_config in sensors:
        # Fill in defaults for missing config values
        sensor_config = dict(sensor_config)  # shallow copy
        sensor_config.setdefault('history_days', default_history_days)
        sensor_config.setdefault('interval_duration', default_interval_duration)
        sensor_config.setdefault('intervals_to_predict', default_intervals_to_predict)
        sensor_config.setdefault('units', '')
        sensor_config.setdefault('cumulative', False)
        sensor_config.setdefault('regressors', [])
        sensor_config.setdefault('database', db_config.get('enabled', False) if db_config else False)
        sensor_config.setdefault('max_age', db_config.get('max_age', 730) if db_config else 730)

        # Validate required fields
        training_entity_id = sensor_config.get('training_entity_id')
        prediction_entity_id = sensor_config.get('prediction_entity_id')
        if not training_entity_id:
            logger.warning(f"Missing training_entity_id: {sensor_config}")
            continue
        if not prediction_entity_id:
            logger.warning(f"Missing prediction_entity_id: {sensor_config}")
            continue

        # Log config in a consistent way

  

        # Get configuration for this specific sensor, or use defaults
        history_days = sensor_config.get('history_days', default_history_days)
        interval_duration = sensor_config.get('interval_duration', default_interval_duration)
        intervals_to_predict = sensor_config.get('intervals_to_predict', default_intervals_to_predict)
        units = sensor_config.get('units', '')
        cumulative = sensor_config.get('cumulative', False)
        regressors = sensor_config.get('regressors', [])
        

        # Check if database is enabled for this sensor
        # Per-sensor config overrides global config
        use_database = sensor_config.get('database', db_config.get('enabled', False) if db_config else False)
        max_age = sensor_config.get('max_age', db_config.get('max_age', 730) if db_config else 730)
        
        # Initialize database table if enabled
        if use_database and db:
            try:
                db.create_table(training_entity_id)
                # Cleanup old data
                db.cleanup_table(training_entity_id, max_age)
            except Exception as e:
                logger.error(f"Database initialization failed for {training_entity_id}: {e}")
                use_database = False
        
        # Fetch history for this sensor
        try:
            # Get historical data from database if enabled
            db_data = None
            if use_database and db:
                try:
                    db_data = db.get_history(training_entity_id)
                    if len(db_data) > 0:
                        logger.info(f"Retrieved {len(db_data)} rows from database for {training_entity_id}")
                        logger.info(f"  - DB date range: {db_data['ds'].min()} to {db_data['ds'].max()}")
                except Exception as e:
                    logger.error(f"Failed to retrieve database history: {e}")
                    db_data = None
            
            # Fetch recent data from Home Assistant API
            # Use shorter period if we have database data (e.g., last 7 days)
            api_days = min(7, history_days) if (use_database and db_data is not None and len(db_data) > 0) else history_days
            
            training_data = ha_api.get_history(training_entity_id, days=api_days, minimal_response=True)
            
            if not training_data or len(training_data) == 0:
                logger.error(f"No history data retrieved from API for {training_entity_id}")
                # If we have database data, use it
                if db_data is not None and len(db_data) > 0:
                    logger.info(f"Using {len(db_data)} data points from database only")
                    df = db_data
                else:
                    continue
            else:
                logger.info(f"Retrieved {len(training_data[0])} history records from API for {training_entity_id}")
                
                # Prepare data for NeuralProphet with regular intervals
                df_api = prepare_training_data(training_data, cumulative=cumulative, interval_minutes=interval_duration)
                
                # Store only real historical data (not future) in database if enabled and available
                if use_database and df_api is not None:
                    try:
                        now = pd.Timestamp.now(tz=df_api['ds'].dt.tz) if df_api['ds'].dt.tz is not None else pd.Timestamp.now()
                        logger.info(f"About to write to DB for {training_entity_id}: max ds={df_api['ds'].max()}, now={now}, shape={df_api.shape}")
                        
                        db.store_history(training_entity_id, df_api, db_data)
                        logger.info(f"Stored new data in database for {training_entity_id}")
                    except Exception as e:
                        logger.error(f"Failed to store history in database: {e}")
                # Combine database and API data if both available
                if use_database and db_data is not None and len(db_data) > 0 and df_api is not None:
                    combined = pd.concat([db_data, df_api], ignore_index=True)
                    combined = combined.drop_duplicates(subset=['ds'], keep='last')
                    combined = combined.sort_values('ds').reset_index(drop=True)
                    df = combined
                    logger.info(f"Combined database and API data: {len(db_data)} + {len(df_api)} = {len(df)} unique points")
                else:
                    df = df_api
            
            if df is not None and len(df) > 0:
                logger.info(f"Training data prepared successfully for {training_entity_id}")
                logger.info(f"  - Data points: {len(df)}")
                logger.info(f"  - Date range: {df['ds'].min()} to {df['ds'].max()}")

                # Diagnostic: Show last 10 rows and NaN status of df before merging regressors
                logger.info("[DIAG] Last 10 rows of df before merging regressors:")
                logger.info(f"\n{df.tail(10)}")
                logger.info("[DIAG] NaN status in last 10 rows of df before merging regressors:")
                logger.info(f"\n{df.tail(10).isna()}")

                # Fetch and merge regressor data if configured
                regressor_columns = []
                regressor_future_mapping = {}  # Maps regressor entity_id to future data source
                # (No regressor registration here; handled after merging below)

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

                # Diagnostic: Show last 10 rows and NaN status of df after merging regressors
                logger.info("[DIAG] Last 10 rows of df after merging regressors:")
                logger.info(f"\n{df.tail(10)}")
                logger.info("[DIAG] NaN status in last 10 rows of df after merging regressors:")
                logger.info(f"\n{df.tail(10).isna()}")

                regressor_columns = []
                regressor_future_mapping = {}  # Maps regressor entity_id to future data source
                sun_regressor_params = None
                if regressors:
                    for regressor_config in regressors:
                        logger.info(f"Processing regressor config: {regressor_config}")
                        # Handle both old format (string) and new format (dict)
                        if isinstance(regressor_config, str):
                            regressor_entity_id = regressor_config
                            regressor_type = 'sensor'
                            future_entity_id = None
                            future_entity_type = None
                            future_attribute = None
                            latitude = None
                            longitude = None
                            elevation = 0
                            is_calculated_sun = False
                        else:
                            regressor_entity_id = regressor_config.get('entity_id')
                            regressor_type = regressor_config.get('type', 'sensor')
                            future_entity_id = regressor_config.get('future_entity_id')
                            future_entity_type = regressor_config.get('future_entity_type')
                            future_attribute = regressor_config.get('future_attribute')
                            latitude = regressor_config.get('latitude')
                            longitude = regressor_config.get('longitude')
                            elevation = regressor_config.get('elevation', 0)
                            is_calculated_sun = regressor_type == 'calculated' and regressor_entity_id == 'sun.azel'
                            logger.info(f"  - Detected calculated sun regressor: {is_calculated_sun}")
                        if is_calculated_sun:
                            # If latitude/longitude not specified, fetch from Home Assistant config
                            if latitude is None or longitude is None:
                                logger.info("No latitude/longitude specified for sun regressor, fetching from Home Assistant config...")
                                loc = ha_api.get_location()
                                if loc is not None:
                                    latitude, longitude, elevation = loc
                                    logger.info(f"Using Home Assistant location: lat={latitude}, lon={longitude}, elev={elevation}")
                                else:
                                    logger.error("Failed to fetch Home Assistant location for sun regressor. Skipping sun azimuth/elevation regressor.")
                                    continue
                            logger.info(f"Calculating sun azimuth/elevation for lat={latitude}, lon={longitude}, elev={elevation}")
                            sun_df = pd.DataFrame({'ds': df['ds']})
                            sun_df = calculate_sun_az_el(sun_df, latitude, longitude, elevation)
                            df['sun_azimuth'] = sun_df['sun_azimuth']
                            df['sun_elevation'] = sun_df['sun_elevation']
                            regressor_columns.extend(['sun_azimuth', 'sun_elevation'])
                            sun_regressor_params = (latitude, longitude, elevation)
                        elif regressor_type == 'sensor':
                            reg_name = regressor_config.get('name') or regressor_entity_id
                            reg_df = fetch_regressor_data(
                                ha_api,
                                regressor_entity_id,
                                history_days,
                                interval_duration,
                                db=db,
                                use_database=use_database
                            )
                            # Write regressor history to the database if enabled and available
                            if use_database and db is not None and reg_df is not None and regressor_entity_id in reg_df.columns:
                                reg_table_id = f"{regressor_entity_id}"
                                # Prepare DataFrame with 'ds' and 'y' columns for store_history
                                reg_store_df = reg_df.rename(columns={regressor_entity_id: 'y'})[['ds', 'y']]
                                try:
                                    prev_reg_data = db.get_history(reg_table_id)
                                    db.store_history(reg_table_id, reg_store_df, prev_reg_data)
                                    logger.info(f"Stored regressor history for {reg_table_id} in database.")
                                except Exception as e:
                                    logger.error(f"Failed to store regressor history for {reg_table_id}: {e}")
                            if reg_df is not None and regressor_entity_id in reg_df.columns:
                                logger.info(f"Merging regressor {regressor_entity_id} as {reg_name} into main DataFrame.")
                                # Rename column to reg_name before merging
                                reg_df = reg_df.rename(columns={regressor_entity_id: reg_name})
                                df = df.merge(reg_df, on='ds', how='left')
                                regressor_columns.append(reg_name)
                            else:
                                logger.warning(f"No data found for regressor {regressor_entity_id}, skipping merge.")
                        elif regressor_type == 'weather':
                            # Weather regressor logic placeholder
                            pass
                        else:
                            logger.warning(f"Unknown regressor type: {regressor_type}")
                            #continue
                


                # --- Forward/backward fill all regressor columns in historic df before NaN check/trimming ---
                regressor_cols_to_fill = [col for col in df.columns if col not in ['ds', 'y']]
                if regressor_cols_to_fill:
                    logger.info(f"[DIAG] Forward/backward filling regressor columns in historic df before NaN check: {regressor_cols_to_fill}")
                    for col in regressor_cols_to_fill:
                        df[col] = df[col].ffill().bfill()
                        nulls = df[col].isna().sum()
                        logger.info(f"[DIAG] After fill, {col} has {nulls} nulls in historic df")

                # --- Trim trailing NaNs in historic data before extending with future rows ---
                required_cols = ['y'] + lagged_regressors if 'lagged_regressors' in locals() and lagged_regressors else ['y']
                # Also include all regressor columns present in df
                if 'regressor_columns' in locals() and regressor_columns:
                    required_cols += regressor_columns
                required_cols = [col for col in required_cols if col in df.columns]
                tail_n = 20
                logger.info(f"[DIAG] Checking for missing data in last {tail_n} rows of historic df before extending. Columns: {required_cols}")
                tail_df = df[required_cols].tail(tail_n)
                nan_summary = tail_df.isna().sum()
                logger.info(f"[DIAG] NaN count in last {tail_n} rows of historic df before extending: {nan_summary.to_dict()}")
                if nan_summary.sum() > 0:
                    logger.warning(f"[DIAG] There are missing values in the last {tail_n} rows of historic df before extending. Will trim trailing NaNs.")
                else:
                    logger.info(f"[DIAG] No missing values in the last {tail_n} rows of historic df before extending.")
                # Trim trailing rows with NaNs in any required column
                if required_cols:
                    mask = df[required_cols].notna().all(axis=1)
                else:
                    logger.warning(f"No required columns found in df for NaN trimming. Skipping mask.")
                    mask = pd.Series([True] * len(df), index=df.index)
                if not mask.all():
                    last_valid_idx = mask[::-1].idxmax()
                    if last_valid_idx < len(df) - 1:
                        logger.warning(f"Dropping {len(df) - 1 - last_valid_idx} trailing rows with NaNs in required columns before extending future.")
                        df = df.iloc[:last_valid_idx + 1]

                # Diagnostic: Print the final timestamp of the training data after merging and trimming
                if len(df) > 0:
                    logger.info(f"[DIAG] Final available training data timestamp after merging and trimming: {df['ds'].max()}")
                else:
                    logger.warning(f"[DIAG] No training data available after merging and trimming.")

                # --- Now extend the dataset with future timestamps ---
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

                # Always calculate sun azimuth/elevation for future if needed
                if sun_regressor_params is not None:
                    lat, lon, elev = sun_regressor_params
                    future_df = calculate_sun_az_el(future_df, lat, lon, elev)
                    logger.info(f"future_df tail after sun azimuth/elevation calculation:\n{future_df.tail()}\n...")

                # Diagnostic: Show last 10 rows and NaN status of df after extending with future rows
                logger.info("[DIAG] Last 10 rows of df after extending with future rows:")
                logger.info(f"\n{pd.concat([df, future_df], ignore_index=True).tail(10)}")
                logger.info("[DIAG] NaN status in last 10 rows of df after extending with future rows:")
                logger.info(f"\n{pd.concat([df, future_df], ignore_index=True).tail(10).isna()}")
                extended_df = pd.concat([df, future_df], ignore_index=True)
                logger.info(f"  - Extended dataset with {len(future_df)} future timestamps")

                # If we have regressors, add them and merge their future values
                if regressor_columns:
                    # Only merge and process regressor data here; registration happens after all merging/filling
                    # Make a copy of regressor_columns to avoid modifying while iterating
                    regressor_columns_to_process = regressor_columns.copy()
                    for regressor_col in regressor_columns_to_process:
                        if regressor_col not in extended_df.columns:
                            logger.warning(f"  - Regressor {regressor_col} not found in extended dataset, skipping")
                            continue
                        # Get regressor values from sensor configuration
                        regressor_entity = None
                        regressor_future_entity = None
                        regressor_future_type = None
                        regressor_future_attribute = None
                        regressor_name = regressor_col
                        for reg_cfg in sensor_config.get('regressors', []):
                            # Support both string and dict config
                            if isinstance(reg_cfg, dict) and (reg_cfg.get('name') == regressor_col or reg_cfg.get('entity_id') == regressor_col):
                                regressor_entity = reg_cfg.get('entity_id')
                                regressor_future_entity = reg_cfg.get('future_entity_id')
                                regressor_future_type = reg_cfg.get('future_entity_type')
                                regressor_future_attribute = reg_cfg.get('future_attribute', 'temperature')
                                regressor_name = reg_cfg.get('name') or regressor_col
                                break
                        # For future timestamps, get forecast data
                        if regressor_future_entity and regressor_future_type == 'weather':
                            logger.info(f"  - Fetching weather forecast for regressor {regressor_name}")
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
                                                regressor_name: float(value)
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
                                    forecast_col = f"{regressor_name}_forecast"
                                    if forecast_col in extended_df.columns:
                                        # Fill regressor column with forecast values where it's NaN
                                        extended_df[regressor_name] = extended_df[regressor_name].fillna(extended_df[forecast_col])
                                        extended_df = extended_df.drop(columns=[forecast_col])
                                    logger.info(f"    Merged {len(regressor_future_df)} forecast values for {regressor_name}")
                        # Forward/backward fill any remaining missing values, but only if column still exists
                        if regressor_name in extended_df.columns:
                            # Always forward-fill and backward-fill to cover missing values at the end (HA only updates on change)
                            extended_df[regressor_name] = extended_df[regressor_name].ffill().bfill()
                            null_count = extended_df[regressor_name].isna().sum()
                            logger.info(f"    After forward/backward fill, {regressor_name} has {null_count} nulls")
                            # Check for singular value after filling
                            regressor_values_in_extended = extended_df[regressor_name].dropna()
                            unique_count = regressor_values_in_extended.nunique()
                            if unique_count <= 1:
                                logger.warning(f"    Regressor {regressor_name} has only {unique_count} unique value(s) in extended dataset. Removing from model.")
                                extended_df = extended_df.drop(columns=[regressor_name])
                                regressor_columns.remove(regressor_name)
                                continue
                            logger.info(f"    Regressor {regressor_name} has {unique_count} unique values in extended dataset")
                        else:
                            logger.warning(f"    Regressor column {regressor_name} disappeared after merge/fill, skipping.")

                # Dynamically determine regressors for this sensor/model
                regressors = [col for col in extended_df.columns if col not in ['ds', 'y'] and not extended_df[col].isna().all()]
                logger.info("Regressors being used for prediction:")
                for reg in regressors:
                    unique_count = extended_df[reg].nunique(dropna=True)
                    null_count = extended_df[reg].isna().sum()
                    logger.info(f"  - {reg}: unique values={unique_count}, nulls={null_count}, dtype={extended_df[reg].dtype}")

                # Initialize NeuralProphet model and frequency string (must be after interval_duration is set)
                freq = f"{sensor_config['interval_duration']}min"
                model = NeuralProphet(
                    yearly_seasonality=True,
                    weekly_seasonality=True,
                    daily_seasonality=True,
                    n_lags=0,
                    n_forecasts=1,
                    learning_rate=1.0,
                    epochs=100,
                    batch_size=32,
                    loss_func="Huber"
                    )
                
                # Add regressors, supporting n_lags per regressor from YAML
                lagged_regressors = []
                future_regressors = []
                for reg in regressors:
                    # Find regressor config dict if available
                    reg_cfg = None
                    for r in sensor_config.get('regressors', []):
                        if (isinstance(r, dict) and (r.get('entity_id') == reg or r.get('name') == reg)) or (isinstance(r, str) and r == reg):
                            reg_cfg = r if isinstance(r, dict) else None
                            break
                    n_lags = 0
                    if reg_cfg is not None:
                        n_lags = reg_cfg.get('n_lags', 0)
                    if n_lags and int(n_lags) > 0:
                        model.add_lagged_regressor(reg, n_lags=int(n_lags))
                        lagged_regressors.append(reg)
                        logger.info(f"Added lagged regressor: {reg} with n_lags={n_lags}")
                    else:
                        model.add_future_regressor(reg)
                        future_regressors.append(reg)
                        logger.info(f"Added future regressor: {reg}")

                # For training, include all regressors
                fit_cols = ['ds', 'y'] + lagged_regressors + future_regressors
                fit_df = extended_df[fit_cols][extended_df['y'].notna()]
                logger.info(f"Training data columns: {fit_cols}")
                # Diagnostic: Show last 10 rows and NaN status before model.fit
                logger.info("Last 10 rows of fit_df before model.fit:")
                logger.info(f"\n{fit_df.tail(10)}")
                logger.info("NaN status in last 10 rows:")
                logger.info(f"\n{fit_df[fit_cols].tail(10).isna()}")
                metrics = model.fit(fit_df, freq=freq)

                
                # Since extended_df already includes future timestamps with regressor values,
                # Use extended_df directly for prediction (already has future rows and regressor values)
                logger.info(f"  - Using extended_df directly for prediction (already has future rows and regressor values)")
                logger.info(f"[DIAG] extended_df shape: {extended_df.shape}, columns: {list(extended_df.columns)}")
                logger.info(f"[DIAG] NaN count in extended_df: {extended_df.isna().sum().to_dict()}")
                # Only include columns needed for prediction
                allowed_cols = ['ds', 'y'] + [reg for reg in regressors if reg in extended_df.columns]
                future_predict_df = extended_df[allowed_cols].copy()
                logger.info("Prediction input regressor details:")
                for reg in regressors:
                    if reg in future_predict_df.columns:
                        unique_count = future_predict_df[reg].nunique(dropna=True)
                        null_count = future_predict_df[reg].isna().sum()
                        logger.info(f"  - {reg}: unique values={unique_count}, nulls={null_count}, dtype={future_predict_df[reg].dtype}")
                logger.info(f"Generating {intervals_to_predict} interval predictions...")
                forecast = model.predict(future_predict_df)
                # Log summary statistics for regressor columns in the forecast DataFrame
                logger.info("Forecast regressor summary statistics:")
                for reg in regressors:
                    if reg in forecast.columns:
                        col = forecast[reg]
                        logger.info(f"  - {reg}: min={col.min()}, max={col.max()}, mean={col.mean()}, unique={col.nunique()}, nulls={col.isna().sum()}")
                logger.info(f"Forecast sample with regressors:\n{forecast[['ds', 'yhat1'] + [r for r in regressors if r in forecast.columns]].head()}\n...")
                
                logger.info(f"  - Model returned {len(forecast)} forecast rows")
                logger.info(f"  - Forecast columns: {[col for col in forecast.columns if 'yhat' in col]}")
                
                # Ensure forecast['ds'] is timezone-aware
                if not pd.api.types.is_datetime64_any_dtype(forecast['ds']):
                    forecast['ds'] = pd.to_datetime(forecast['ds'], utc=True)
                elif forecast['ds'].dt.tz is None:
                    forecast['ds'] = forecast['ds'].dt.tz_localize('UTC')
                

                # Diagnostic: Show last 10 timestamps of the final training DataFrame


                # --- 1. Fix diagnostic to use df_raw for the latest sensor timestamp ---
                latest_raw_sensor_time = None
                if 'df_raw' in locals() and not df_raw.empty and 'ds' in df_raw.columns:
                    latest_raw_sensor_time = df_raw['ds'].max()
                    logger.info(f"[DIAG] Latest timestamp from raw sensor data: {latest_raw_sensor_time}")
                else:
                    logger.warning("[DIAG] Could not determine latest timestamp from raw sensor data (df_raw not found)")
                logger.info("[DIAG] Last 10 timestamps of final training DataFrame (after merging/filling/trimming):")
                logger.info(f"\n{df['ds'].tail(10).to_list()}")

                # --- 2. Patch: Always extend training DataFrame to latest sensor timestamp, forward-filling regressors as needed ---
                # Only if latest_raw_sensor_time is available and after the last timestamp in df
                if latest_raw_sensor_time is not None:
                    last_df_time = df['ds'].max()
                    if last_df_time < latest_raw_sensor_time:
                        logger.warning(f"[PATCH] Extending training DataFrame from {last_df_time} to {latest_raw_sensor_time} by forward-filling.")
                        # Create new index covering all intervals up to latest_raw_sensor_time
                        freq = pd.infer_freq(df['ds']) or f"{interval_duration}min"
                        full_index = pd.date_range(start=df['ds'].min(), end=latest_raw_sensor_time, freq=freq, tz=df['ds'].dt.tz)
                        df = df.set_index('ds').reindex(full_index).reset_index().rename(columns={'index': 'ds'})
                        # Forward-fill all columns except 'ds'
                        for col in df.columns:
                            if col != 'ds':
                                df[col] = df[col].ffill().bfill()
                        logger.info(f"[PATCH] Training DataFrame now extends to {df['ds'].max()} (was {last_df_time})")

                # Get the last training timestamp to filter future predictions
                last_training_time = df['ds'].max()
                if last_training_time.tz is None:
                    last_training_time = last_training_time.tz_localize('UTC')

                # Filter for future predictions only (strictly after last training time)
                future_forecast = forecast[forecast['ds'] > last_training_time].copy()
                future_forecast = future_forecast[['ds', 'yhat1']].copy()

                logger.info(f"[DIAG] Last training time: {last_training_time}")
                if not future_forecast.empty:
                    logger.info(f"[DIAG] First forecasted interval published: {future_forecast['ds'].iloc[0]}")
                else:
                    logger.warning(f"[DIAG] No future forecast intervals found after last training time {last_training_time}")

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


def main():
    logger.info("Home Assistant NeuralProphet Add-on starting...")

    # Configuration path - use DEV_CONFIG_PATH for local development, otherwise addon path
    dev_config_path = os.environ.get('DEV_CONFIG_PATH')
    if dev_config_path:
        config_path = Path(dev_config_path)
        logger.info(f"Running in development mode with config: {config_path}")
    else:
        # Always resolve dev config relative to the script location
        script_dir = Path(__file__).parent.parent  # rootfs/..
        dev_yaml = script_dir / 'dev' / 'neuralprophet.yaml'
        if os.environ.get('DEV_MODE', '').lower() == 'true' or dev_yaml.exists():
            config_path = dev_yaml.resolve()
            logger.info(f"Development mode: Using config from {config_path}")
        else:
            config_path = Path('/config/neuralprophet.yaml')  # HA config directory (addon_config mapping)

    ha_api = HomeAssistantAPI()

    # Initialize database connection (will be used if enabled in config)
    db = None

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
            # Get database configuration
            db_config = config.get('database', {})
            db_enabled = db_config.get('enabled', False)
            db_path = db_config.get('path', '/config/neuralprophet.db')
            # Override with dev database path if in development mode
            if dev_config_path and os.environ.get('DEV_DB_PATH'):
                db_path = os.environ.get('DEV_DB_PATH')
                logger.info(f"Development mode: Using dev database path")
            logger.info(f"Configuration:")
            logger.info(f"  - Default history: {default_history_days} days")
            logger.info(f"  - Update interval: {update_interval_minutes} minutes ({update_interval} seconds)")
            logger.info(f"  - Default interval duration: {default_interval_duration} minutes")
            logger.info(f"  - Default intervals to predict: {default_intervals_to_predict}")
            logger.info(f"  - Sensors: {len(sensors)}")
            logger.info(f"  - Database: {'Enabled' if db_enabled else 'Disabled'}")
            if db_enabled:
                logger.info(f"    → Path: {db_path}")
                logger.info(f"    → Max age: {db_config.get('max_age', 730)} days")
            # Check if any sensor has database: true (per-sensor override)
            if not db_enabled:
                db_enabled = any(s.get('database', False) for s in sensors)
                if db_enabled:
                    logger.info("Database enabled due to per-sensor override.")
            # Initialize database if enabled
            if db_enabled:
                try:
                    if db is None:
                        db = Database(db_path)
                        logger.info(f"Database initialized: {db_path}")
                except Exception as e:
                    logger.error(f"Failed to initialize database: {e}")
                    db = None
            process_sensors(ha_api, sensors, default_history_days, default_interval_duration, default_intervals_to_predict, db=db, db_config=db_config)
            logger.info(f"Training cycle complete. Next run in {update_interval} seconds")
            # If running in development mode, only run once
            if dev_config_path is not None and str(dev_config_path).strip() != "":
                logger.info("Development mode detected: exiting after one cycle.")
                break
            else:
                time.sleep(update_interval)
        except KeyboardInterrupt:
            logger.info("Shutting down gracefully...")
            if db:
                db.close()
            break
        except Exception as e:
            logger.error(f"Error in main loop: {e}", exc_info=True)
            logger.info(f"Retrying in {update_interval} seconds")
            time.sleep(update_interval)