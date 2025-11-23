# Home Assistant Add-on: NeuralProphet

_Forecast Home Assistant sensor values using NeuralProphet machine learning._

![Supports aarch64 Architecture][aarch64-shield]
![Supports amd64 Architecture][amd64-shield]

## Features

- **Multi-step forecasting**: Generate unique predictions for each time interval
- **Regressor support**: Use weather forecasts and other sensors to improve accuracy
- **Cumulative sensor handling**: Special optimization for energy meters and solar panels
- **Database storage**: Retain years of training data beyond Home Assistant's history limits
- **Automatic resampling**: Ensures regular time intervals for accurate predictions
- **Flexible configuration**: Per-sensor customization of intervals, history, and regressors

## Database Storage

Enable persistent SQLite storage to overcome Home Assistant API limitations:

```yaml
database:
  enabled: true
  max_age: 730  # Keep 2 years of data
  path: /config/neuralprophet.db
```

**Benefits:**
- Train on years of data (not limited to HA's 60-720 day history)
- Incremental updates (only fetch recent data from HA API)
- Data persists through restarts and history purges
- Automatic cleanup of old data
- Separate storage for sensors and regressors

**Per-sensor overrides:**
```yaml
sensors:
  - training_entity_id: sensor.solar_production
    database: true  # Enable for this sensor
    max_age: 1095  # Keep 3 years
```

## Quick Start

1. Install the add-on
2. Create `/addon_configs/local_neuralprophet/neuralprophet.yaml`:
```yaml
database:
  enabled: true
  max_age: 730

sensors:
  - training_entity_id: sensor.my_sensor
    prediction_entity_id: sensor.my_sensor_prediction
    interval_duration: 30
    intervals_to_predict: 48
```
3. Start the add-on
4. View predictions in Home Assistant at `sensor.my_sensor_prediction`

## Documentation

See [DOCS.md](DOCS.md) for full configuration details including:
- Database setup and configuration
- Regressor configuration (weather forecasts, etc.)
- Cumulative sensor optimization
- Troubleshooting guide

[aarch64-shield]: https://img.shields.io/badge/aarch64-yes-green.svg
[amd64-shield]: https://img.shields.io/badge/amd64-yes-green.svg
```
