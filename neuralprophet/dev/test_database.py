"""
Test suite for Database class in database.py

Tests database operations including table creation, data storage,
retrieval, and cleanup functionality.
"""

import os
import sys
import sqlite3
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np

# Add the rootfs directory to Python path for importing database module
rootfs_path = Path(__file__).parent.parent / 'rootfs'
sys.path.insert(0, str(rootfs_path))

from database import Database


class TestDatabase(unittest.TestCase):
    """Test cases for Database class"""
    
    def setUp(self):
        """Create temporary database for each test"""
        self.temp_db = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.db')
        self.temp_db.close()
        self.db_path = self.temp_db.name
        self.db = Database(self.db_path)
        
    def tearDown(self):
        """Clean up temporary database"""
        self.db.close()
        if os.path.exists(self.db_path):
            os.unlink(self.db_path)
    
    def test_database_initialization(self):
        """Test database connection initialization"""
        self.assertIsNotNone(self.db.con)
        self.assertIsNotNone(self.db.cur)
        self.assertEqual(self.db.db_path, self.db_path)
        
    def test_sanitize_table_name(self):
        """Test table name sanitization"""
        self.assertEqual(
            self.db._sanitize_table_name('sensor.my_sensor'),
            'sensor_my_sensor'
        )
        self.assertEqual(
            self.db._sanitize_table_name('sensor.test-sensor-123'),
            'sensor_test_sensor_123'
        )
        
    def test_create_table(self):
        """Test table creation"""
        entity_id = 'sensor.test_sensor'
        self.db.create_table(entity_id)
        
        # Verify table exists
        table_name = self.db._sanitize_table_name(entity_id)
        self.db.cur.execute(
            f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'"
        )
        result = self.db.cur.fetchone()
        self.assertIsNotNone(result)
        self.assertEqual(result[0], table_name)
        
    def test_get_history_empty(self):
        """Test getting history from empty database"""
        entity_id = 'sensor.test_sensor'
        self.db.create_table(entity_id)
        
        df = self.db.get_history(entity_id)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 0)
        self.assertListEqual(list(df.columns), ['ds', 'y'])
        
    def test_get_history_nonexistent_table(self):
        """Test getting history from non-existent table"""
        df = self.db.get_history('sensor.nonexistent')
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 0)
        
    def test_store_and_retrieve_history(self):
        """Test storing and retrieving sensor data"""
        entity_id = 'sensor.temperature'
        self.db.create_table(entity_id)
        
        # Create test data
        now = datetime.now()
        test_data = pd.DataFrame([
            {'ds': now - timedelta(hours=2), 'y': 20.5},
            {'ds': now - timedelta(hours=1), 'y': 21.0},
            {'ds': now, 'y': 21.5},
        ])
        
        # Store data
        result = self.db.store_history(entity_id, test_data)
        self.assertIsInstance(result, pd.DataFrame)
        
        # Retrieve data
        df = self.db.get_history(entity_id)
        self.assertEqual(len(df), 3)
        self.assertAlmostEqual(df.iloc[0]['y'], 20.5)
        self.assertAlmostEqual(df.iloc[1]['y'], 21.0)
        self.assertAlmostEqual(df.iloc[2]['y'], 21.5)
        
    def test_store_incremental_updates(self):
        """Test incremental data storage (no duplicates)"""
        entity_id = 'sensor.power'
        self.db.create_table(entity_id)
        
        now = datetime.now()
        
        # First batch
        batch1 = pd.DataFrame([
            {'ds': now - timedelta(hours=3), 'y': 100.0},
            {'ds': now - timedelta(hours=2), 'y': 110.0},
        ])
        self.db.store_history(entity_id, batch1)
        
        # Second batch with one duplicate and one new
        batch2 = pd.DataFrame([
            {'ds': now - timedelta(hours=2), 'y': 110.0},  # Duplicate
            {'ds': now - timedelta(hours=1), 'y': 120.0},  # New
            {'ds': now, 'y': 130.0},  # New
        ])
        prev_data = self.db.get_history(entity_id)
        self.db.store_history(entity_id, batch2, prev_data)
        
        # Verify only unique entries stored
        df = self.db.get_history(entity_id)
        self.assertEqual(len(df), 4)  # Should have 4 unique timestamps
        
    def test_store_history_with_nan_values(self):
        """Test that NaN values are not stored"""
        entity_id = 'sensor.test'
        self.db.create_table(entity_id)
        
        now = datetime.now()
        test_data = pd.DataFrame([
            {'ds': now - timedelta(hours=2), 'y': 10.0},
            {'ds': now - timedelta(hours=1), 'y': np.nan},  # Should be skipped
            {'ds': now, 'y': 12.0},
        ])
        
        self.db.store_history(entity_id, test_data)
        
        df = self.db.get_history(entity_id)
        self.assertEqual(len(df), 2)  # Only 2 valid values
        
    def test_store_regressor_data(self):
        """Test storing regressor data"""
        entity_id = 'sensor.solar'
        regressor_name = 'cloud_coverage'
        
        now = datetime.now()
        test_data = pd.DataFrame([
            {'ds': now - timedelta(hours=2), regressor_name: 50.0},
            {'ds': now - timedelta(hours=1), regressor_name: 40.0},
            {'ds': now, regressor_name: 30.0},
        ])
        
        result = self.db.store_regressor(entity_id, regressor_name, test_data)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 3)
        self.assertIn(regressor_name, result.columns)
        
    def test_store_regressor_incremental(self):
        """Test incremental regressor storage"""
        entity_id = 'sensor.solar'
        regressor_name = 'uv_index'
        
        now = datetime.now()
        
        # First batch
        batch1 = pd.DataFrame([
            {'ds': now - timedelta(hours=2), regressor_name: 5.0},
        ])
        result1 = self.db.store_regressor(entity_id, regressor_name, batch1)
        self.assertEqual(len(result1), 1)
        
        # Second batch with overlap
        batch2 = pd.DataFrame([
            {'ds': now - timedelta(hours=2), regressor_name: 5.0},  # Duplicate
            {'ds': now - timedelta(hours=1), regressor_name: 7.0},  # New
        ])
        result2 = self.db.store_regressor(entity_id, regressor_name, batch2, result1)
        self.assertEqual(len(result2), 2)
        
    def test_cleanup_table(self):
        """Test old data cleanup"""
        entity_id = 'sensor.test_cleanup'
        self.db.create_table(entity_id)
        
        now = datetime.now()
        
        # Add data spanning 10 days
        test_data = pd.DataFrame([
            {'ds': now - timedelta(days=10), 'y': 1.0},
            {'ds': now - timedelta(days=8), 'y': 2.0},
            {'ds': now - timedelta(days=5), 'y': 3.0},
            {'ds': now - timedelta(days=2), 'y': 4.0},
            {'ds': now, 'y': 5.0},
        ])
        
        self.db.store_history(entity_id, test_data)
        
        # Verify all data stored
        df_before = self.db.get_history(entity_id)
        self.assertEqual(len(df_before), 5)
        
        # Cleanup data older than 7 days
        self.db.cleanup_table(entity_id, max_age_days=7)
        
        # Verify old data removed
        df_after = self.db.get_history(entity_id)
        self.assertEqual(len(df_after), 3)  # Only last 3 entries within 7 days
        
    def test_cleanup_regressor_table(self):
        """Test regressor data cleanup"""
        entity_id = 'sensor.solar'
        regressor_name = 'temperature'
        
        now = datetime.now()
        
        # Add data spanning multiple days
        test_data = pd.DataFrame([
            {'ds': now - timedelta(days=100), regressor_name: 10.0},
            {'ds': now - timedelta(days=50), regressor_name: 15.0},
            {'ds': now - timedelta(days=20), regressor_name: 20.0},
            {'ds': now, regressor_name: 25.0},
        ])
        
        self.db.store_regressor(entity_id, regressor_name, test_data)
        
        # Cleanup data older than 30 days
        self.db.cleanup_regressor_table(entity_id, regressor_name, max_age_days=30)
        
        # Verify cleanup worked
        table_name = self.db._sanitize_table_name(f"{entity_id}_{regressor_name}")
        self.db.cur.execute(f"SELECT COUNT(*) FROM {table_name}")
        count = self.db.cur.fetchone()[0]
        self.assertEqual(count, 2)  # Only 2 entries within 30 days
        
    def test_database_persistence(self):
        """Test that data persists after closing and reopening"""
        entity_id = 'sensor.persistent'
        
        # Store data
        self.db.create_table(entity_id)
        test_data = pd.DataFrame([
            {'ds': datetime.now(), 'y': 42.0},
        ])
        self.db.store_history(entity_id, test_data)
        self.db.close()
        
        # Reopen database
        db2 = Database(self.db_path)
        df = db2.get_history(entity_id)
        self.assertEqual(len(df), 1)
        self.assertAlmostEqual(df.iloc[0]['y'], 42.0)
        db2.close()
        
    def test_multiple_sensors(self):
        """Test handling multiple sensors in same database"""
        sensors = [
            'sensor.temperature',
            'sensor.humidity',
            'sensor.pressure'
        ]
        
        now = datetime.now()
        
        # Store data for each sensor
        for i, sensor in enumerate(sensors):
            self.db.create_table(sensor)
            test_data = pd.DataFrame([
                {'ds': now, 'y': float(i * 10)},
            ])
            self.db.store_history(sensor, test_data)
        
        # Verify each sensor has correct data
        for i, sensor in enumerate(sensors):
            df = self.db.get_history(sensor)
            self.assertEqual(len(df), 1)
            self.assertAlmostEqual(df.iloc[0]['y'], float(i * 10))
            
    def test_datetime_handling(self):
        """Test proper datetime parsing and storage"""
        entity_id = 'sensor.datetime_test'
        self.db.create_table(entity_id)
        
        # Test with various datetime formats
        now = datetime.now()
        test_data = pd.DataFrame([
            {'ds': pd.Timestamp(now - timedelta(hours=1)), 'y': 1.0},
            {'ds': now, 'y': 2.0},
        ])
        
        self.db.store_history(entity_id, test_data)
        df = self.db.get_history(entity_id)
        
        # Verify timestamps are datetime objects
        self.assertEqual(len(df), 2)
        for timestamp in df['ds']:
            self.assertIsInstance(timestamp, (pd.Timestamp, datetime))


class TestDatabaseIntegration(unittest.TestCase):
    """Integration tests for database with larger datasets"""
    
    def setUp(self):
        """Create temporary database"""
        self.temp_db = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.db')
        self.temp_db.close()
        self.db_path = self.temp_db.name
        self.db = Database(self.db_path)
        
    def tearDown(self):
        """Clean up"""
        self.db.close()
        if os.path.exists(self.db_path):
            os.unlink(self.db_path)
    
    def test_large_dataset(self):
        """Test with large dataset (simulating months of data)"""
        entity_id = 'sensor.large_test'
        self.db.create_table(entity_id)
        
        # Generate 90 days of 30-minute interval data (4320 points)
        start_date = datetime.now() - timedelta(days=90)
        timestamps = [start_date + timedelta(minutes=30*i) for i in range(4320)]
        values = [20.0 + 5.0 * np.sin(i * 0.01) for i in range(4320)]  # Synthetic data
        
        test_data = pd.DataFrame({
            'ds': timestamps,
            'y': values
        })
        
        # Store in batches (simulating incremental updates)
        batch_size = 500
        for i in range(0, len(test_data), batch_size):
            batch = test_data.iloc[i:i+batch_size]
            self.db.store_history(entity_id, batch)
        
        # Verify all data stored
        df = self.db.get_history(entity_id)
        self.assertEqual(len(df), 4320)
        
        # Test cleanup removes old data
        self.db.cleanup_table(entity_id, max_age_days=30)
        df_after = self.db.get_history(entity_id)
        self.assertLess(len(df_after), 4320)
        self.assertGreater(len(df_after), 1400)  # ~30 days of 30-min data
        
    def test_concurrent_sensor_updates(self):
        """Test multiple sensors being updated"""
        sensors = [f'sensor.test_{i}' for i in range(10)]
        now = datetime.now()
        
        # Create tables
        for sensor in sensors:
            self.db.create_table(sensor)
        
        # Add data for all sensors
        for sensor in sensors:
            data = pd.DataFrame([
                {'ds': now - timedelta(hours=i), 'y': float(i)}
                for i in range(100)
            ])
            self.db.store_history(sensor, data)
        
        # Verify all sensors have data
        for sensor in sensors:
            df = self.db.get_history(sensor)
            self.assertEqual(len(df), 100)


def run_tests():
    """Run all tests"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestDatabase))
    suite.addTests(loader.loadTestsFromTestCase(TestDatabaseIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Return exit code
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    sys.exit(run_tests())
