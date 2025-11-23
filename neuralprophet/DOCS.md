# Home Assistant Add-on: NeuralProphet

## How to use

This add-on uses NeuralProphet to forecast sensor values in Home Assistant based on historical data and optional regressors (other sensors).

### Configuration

Create a configuration file at `/addon_configs/local_neuralprophet/neuralprophet.yaml` with your sensor settings.

#### Basic Configuration

```yaml
# Default number of days of history to fetch (can be overridden per sensor)
history_days: 60

# Update interval in minutes (how often to retrain models)
update_interval: 60

# Default interval duration in minutes (can be overridden per sensor)
interval_duration: 30

# Default number of intervals to predict (can be overridden per sensor)
intervals_to_predict: 48
```

#### Database Configuration (Optional)

Enable persistent database storage to retain training data beyond Home Assistant's history limitations:

```yaml
database:
  enabled: false  # Set to true to enable database storage
  max_age: 730    # Maximum days to retain in database (default: 2 years)
  path: /config/neuralprophet.db  # Database file path (stored in addon_configs)
```

**Note**: The `/config` path maps to `/addon_configs/local_neuralprophet/` on your Home Assistant system, so the database will be persisted with your addon configuration.

**Benefits of Database Storage:**
- **Longer training periods**: Store years of data beyond HA's typical 60-720 day limit
- **Incremental updates**: Only fetches recent data from HA API, reducing load
- **Data persistence**: Survives HA restarts and history purges
- **Automatic cleanup**: Removes data older than `max_age` days

When enabled:
1. Historical data is loaded from the database
2. Recent data (last 7 days) is fetched from Home Assistant API
3. Data is combined, deduplicated, and resampled to regular intervals
4. New data points are stored in the database for future use
5. Old data beyond `max_age` is automatically removed

#### Sensor Configuration

Each sensor can be configured individually:

```yaml
sensors:
  - training_entity_id: sensor.my_sensor
    prediction_entity_id: sensor.my_sensor_prediction
    history_days: 720   # Use 2 years of historical data
    interval_duration: 30  # 30-minute intervals
    intervals_to_predict: 48  # Predict 24 hours (48 * 30min)
    units: "kWh"
    cumulative: true  # Set to true for energy meters that always increase
    database: true  # Enable database for this sensor (overrides global setting)
    max_age: 1095  # Keep 3 years of data (overrides global max_age)
    regressors:
      - entity_id: sensor.outdoor_temperature
        type: sensor
        future_entity_id: weather.home
        future_entity_type: weather
        future_attribute: temperature
```

**Sensor Options:**
- `training_entity_id`: Source sensor to train on
- `prediction_entity_id`: Destination sensor for predictions
- `history_days`: Days of history to fetch from HA API (ignored if database has more)
- `interval_duration`: Time interval in minutes between predictions
- `intervals_to_predict`: Number of future intervals to predict
- `units`: Unit of measurement for the prediction entity
- `cumulative`: Set to `true` for energy meters (applies non-negative clipping)
- `database`: Enable/disable database storage for this sensor (optional)
- `max_age`: Maximum days to retain for this sensor (optional)
- `regressors`: List of additional sensors to use as predictors (optional)

#### Cumulative Sensors (Energy Meters)

For cumulative sensors like solar production or energy consumption:
- Set `cumulative: true`
- Predictions use 20 Fourier terms for enhanced daily seasonality
- Negative predictions are clipped to 0
- Training uses 100 epochs for better pattern learning

#### Regressors

Add external factors to improve predictions:

```yaml
regressors:
  - entity_id: sensor.outdoor_temperature
    type: sensor
    future_entity_id: weather.home
    future_entity_type: weather
    future_attribute: temperature
  - entity_id: sensor.cloud_coverage
    type: sensor
    future_entity_id: weather.home
    future_entity_type: weather
    future_attribute: cloud_coverage
```

**Regressor Options:**
- `entity_id`: Historical sensor to use as predictor
- `type`: Always `sensor` for now
- `future_entity_id`: Entity providing future values (e.g., weather forecast)
- `future_entity_type`: Type of future entity (e.g., `weather`)
- `future_attribute`: Attribute to extract from future entity

Regressors are automatically validated for sufficient variance and are stored in the database when enabled.

### Example Configuration

See the included `neuralprophet.yaml` file for a complete example with:
- Solar power prediction with UV index and cloud coverage regressors
- Database storage enabled for 3 years of data
- 30-minute intervals with 72-hour forecasts

### Viewing Predictions

Predictions are stored as sensor attributes:
- `state`: Latest predicted value
- `forecast`: Array of all predicted values with timestamps
- `unit_of_measurement`: Units from configuration

Access in Home Assistant:
```yaml
# View in Lovelace card
type: entities
entities:
  - sensor.solar_power_prediction_two
```

### Troubleshooting

**Database Issues:**
- Check logs for "Database initialized" message
- Verify `/config/neuralprophet.db` exists
- Increase `max_age` if data is being cleaned up too aggressively

**Prediction Issues:**
- Ensure sufficient historical data (at least 30 days recommended)
- Verify regressors have variance (not constant values)
- Check that sensors are regularly updated (no large gaps)

**Performance:**
- With database enabled, only recent data is fetched from HA
- First run with database may take longer to initialize
- Subsequent runs are faster with incremental updates
```
