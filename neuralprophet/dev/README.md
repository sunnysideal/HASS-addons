# Development Files

This folder contains development and testing files that are not included in the production add-on.

## Files

- **`test_database.py`**: Comprehensive test suite for the Database class
  - 17 unit and integration tests covering all database operations
  - Run with: `python dev/test_database.py`

- **`dev_run.py`**: Local development runner
  - Allows running the add-on locally for development/testing
  - Run with: `python dev/dev_run.py`
  - Database files created during development are stored here (excluded from git)

- **`neuralprophet_dev.db`**: Development database (auto-created, not committed)
  - Created when running locally with database enabled
  - Excluded from git via `*.db` in .gitignore

## Running Tests

From the repository root:

```bash
# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Run database tests
python dev/test_database.py
```

All tests should pass with output like:
```
test_cleanup_regressor_table ... ok
test_cleanup_table ... ok
test_create_table ... ok
...
Ran 17 tests in 1.029s

OK
```

## Local Development

To run the add-on locally:

```bash
# Set up development environment (if not already done)
.\setup-dev.ps1

# Run locally
python dev/dev_run.py
```

See `DEV_SETUP.md` in the repository root for complete setup instructions.
