
## 0.2.4

- Add `previous_forecast` attribute to prediction sensor: stores the forecast for the start of a new day (00:00) for later comparison with actuals. If not a new day, preserves the previous value.
- Add `actual` attribute to prediction sensor: stores the most recent `intervals_to_predict` values of cleaned training (actual) data for easier comparison in Home Assistant.
- Add `dev_environment` marker attribute to sensor when running in development mode for clear environment identification.
- Refactor: Move non-negative clipping for cumulative sensors out of `update_prediction_entity` and into `process_sensors` for better separation of concerns.
- Improve robustness and validation in `update_prediction_entity` (checks for valid data, attributes, and error handling).

<!-- https://developers.home-assistant.io/docs/add-ons/presentation#keeping-a-changelog -->


## 0.2.3

- Ensure regressor tables are created before storing regressor history in the database (fixes regressor DB persistence in production)

## 0.2.2

- Add `if __name__ == "__main__": main()` to ha_predictor.py to ensure main logic runs when executed as a script (fixes add-on not running in production).

## 0.2.1

- Add `pvlib` to requirements.txt for sun position calculation support (required for sun azimuth/elevation regressors)

## 0.1.6

- Add SQLite database storage for persistent training history
- Enable multi-year training periods beyond Home Assistant API limits
- Implement incremental data updates (fetch only recent data when database enabled)
- Add automatic cleanup of old data based on configurable max_age
- Store regressor data in separate database tables
- Add global and per-sensor database configuration options
- Update documentation with database setup and configuration guide

## 0.1.5

- Fix multi-step forecasting by implementing extended dataset approach from working examples
- Add data resampling to ensure regular time intervals matching prediction frequency
- Add regressor variance validation to prevent singular value errors
- Add non-negative prediction clipping for cumulative sensors (energy meters)
- Increase Fourier terms (20) and epochs (100) for cumulative sensors to better capture daily patterns
- Add training data statistics logging including nighttime value analysis
- Fix timezone handling throughout prediction pipeline

## 0.1.4

- Add regressor support for using additional sensors as predictive features
- Add weather forecast integration to use Home Assistant weather entities as regressors
- Add cumulative sensor handling for energy meters (computes rate from always-increasing values)
- Support both sensor history and weather forecast attributes as training data
- Dynamic configuration reload on each training cycle

## 0.1.3

- Fix None state handling in prepare_training_data
- Skip unavailable, unknown, and invalid sensor states
- Improved error handling for sensor data processing

## 0.1.2

- Add NeuralProphet model training and prediction
- Support multiple sensors with configurable parameters
- Add interval duration and prediction count configuration
- Support per-sensor history days override
- Implement continuous training loop with configurable update intervals
- Add separate training and prediction entity IDs

## 0.1.1

- Add pre-built base image support for faster builds
- Configure GitHub Container Registry for base images
- Add workflow for building base images with Python dependencies

## 0.1.0

- Initial release
- Read sensor configuration from YAML
- Fetch historical data from Home Assistant
- Load data into pandas DataFrames
