"""
SQLite database manager for storing sensor training history.

Stores resampled sensor and regressor data incrementally to enable
longer training periods than Home Assistant API can provide.
"""

import logging
import sqlite3
import pandas as pd
from datetime import datetime, timedelta


logger = logging.getLogger(__name__)


class Database:

    
    
    """SQLite database manager for storing sensor training history.
    
    Stores resampled sensor and regressor data incrementally to enable
    longer training periods than Home Assistant API can provide.
    """
    
    def __init__(self, db_path='/config/neuralprophet.db'):
        """Initialize database connection.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.con = None
        self.cur = None
        self._connect()
    
    def _connect(self):
        """Establish database connection."""
        try:
            self.con = sqlite3.connect(self.db_path)
            self.cur = self.con.cursor()
            logger.info(f"Connected to database: {self.db_path}")
        except Exception as e:
            logger.error(f"Failed to connect to database {self.db_path}: {e}")
            raise
    
    def close(self):
        """Close database connection."""
        if self.con:
            self.con.close()
            logger.info("Database connection closed")
    
    def _sanitize_table_name(self, entity_id):
        """Convert entity_id to valid table name.
        
        Args:
            entity_id: Entity ID like 'sensor.my_sensor'
            
        Returns:
            Table name like 'sensor_my_sensor'
        """
        return entity_id.replace('.', '_').replace('-', '_')
    
    def create_table(self, entity_id):
        """Create table for storing sensor data if it doesn't exist.
        
        Args:
            entity_id: Entity ID to create table for
        """
        table_name = self._sanitize_table_name(entity_id)
        try:
            self.cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    timestamp TEXT PRIMARY KEY,
                    value REAL
                )
            """)
            self.con.commit()
            logger.info(f"Created/verified table: {table_name}")
        except Exception as e:
            logger.error(f"Failed to create table {table_name}: {e}")
            raise
    
    def initialize_sensor_table(self, entity_id, max_age):
        """Create table and cleanup old data for a sensor entity."""
        try:
            self.create_table(entity_id)
            self.cleanup_table(entity_id, max_age)
        except Exception as e:
            logger.error(f"Database initialization failed for {entity_id}: {e}")
            raise

    def get_history(self, entity_id):
        """Retrieve all historical data for a sensor.
        
        Args:
            entity_id: Entity ID to get history for
            
        Returns:
            DataFrame with 'ds' and 'y' columns, or empty DataFrame
        """
        table_name = self._sanitize_table_name(entity_id)
        try:
            self.cur.execute(f"SELECT timestamp, value FROM {table_name} ORDER BY timestamp")
            rows = self.cur.fetchall()
            
            if not rows:
                logger.info(f"No history found in database for {entity_id}")
                return pd.DataFrame(columns=['ds', 'y'])
            
            df = pd.DataFrame(rows, columns=['ds', 'y'])
            df['ds'] = pd.to_datetime(df['ds'])
            logger.info(f"Retrieved {len(df)} rows from database for {entity_id}")
            return df
        except sqlite3.OperationalError as e:
            # Table doesn't exist yet
            logger.info(f"Table for {entity_id} doesn't exist yet: {e}")
            return pd.DataFrame(columns=['ds', 'y'])
        except Exception as e:
            logger.error(f"Failed to get history for {entity_id}: {e}")
            return pd.DataFrame(columns=['ds', 'y'])
    
    def store_history(self, entity_id, new_data, prev_data=None):
        """Store new sensor data incrementally.
        
        Only stores data points not already in the database.
        
        Args:
            entity_id: Entity ID to store data for
            new_data: DataFrame with 'ds' and 'y' columns
            prev_data: Previously stored data (optional, for optimization)
            
        Returns:
            Updated DataFrame combining previous and new data
        """
        table_name = self._sanitize_table_name(entity_id)
        
        if prev_data is None:
            prev_data = self.get_history(entity_id)
        
        if new_data is None or len(new_data) == 0:
            logger.info(f"No new data to store for {entity_id}")
            return prev_data
        
        added_rows = 0
        # Convert to strings for comparison
        prev_timestamps = set(prev_data['ds'].astype(str).values) if len(prev_data) > 0 else set()
        
        try:
            for _, row in new_data.iterrows():
                timestamp = str(row['ds'])
                value = row['y']
                
                # Skip NaN values and duplicates
                if pd.isna(value) or timestamp in prev_timestamps:
                    continue
                
                self.cur.execute(
                    f"INSERT OR REPLACE INTO {table_name} (timestamp, value) VALUES (?, ?)",
                    (timestamp, float(value))
                )
                prev_timestamps.add(timestamp)
                added_rows += 1
            
            self.con.commit()
            logger.info(f"Added {added_rows} new rows to database for {entity_id}")
            
            # Return updated data
            return self.get_history(entity_id)
        except Exception as e:
            logger.error(f"Failed to store history for {entity_id}: {e}")
            self.con.rollback()
            return prev_data
    
    
    def cleanup_table(self, entity_id, max_age_days):
        """Remove data older than max_age_days.
        
        Args:
            entity_id: Entity ID to clean up
            max_age_days: Maximum age in days to keep
        """
        table_name = self._sanitize_table_name(entity_id)
        cutoff_date = (datetime.now() - timedelta(days=max_age_days)).isoformat()
        
        try:
            self.cur.execute(
                f"DELETE FROM {table_name} WHERE timestamp < ?",
                (cutoff_date,)
            )
            deleted = self.cur.rowcount
            self.con.commit()
            
            if deleted > 0:
                logger.info(f"Cleaned up {deleted} old rows from {entity_id}")
        except sqlite3.OperationalError:
            # Table doesn't exist
            pass
        except Exception as e:
            logger.error(f"Failed to cleanup table {table_name}: {e}")
    
