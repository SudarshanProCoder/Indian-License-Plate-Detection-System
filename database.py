import sqlite3
from datetime import datetime

class Database:
    def __init__(self, db_name='license_plates.db'):
        self.db_name = db_name
        self.init_db()

    def init_db(self):
        """Initialize the database with required tables"""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        
        # Create plates table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS plates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                plate_number TEXT NOT NULL,
                state TEXT NOT NULL,
                detection_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                image_path TEXT
            )
        ''')
        
        conn.commit()
        conn.close()

    def add_plate(self, plate_number, state, image_path=None):
        """Add a new license plate record to the database"""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO plates (plate_number, state, detection_date, image_path)
            VALUES (?, ?, ?, ?)
        ''', (plate_number, state, datetime.now(), image_path))
        
        conn.commit()
        conn.close()

    def get_all_plates(self):
        """Retrieve all license plate records"""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT plate_number, state, detection_date, image_path
            FROM plates
            ORDER BY detection_date DESC
        ''')
        
        plates = cursor.fetchall()
        conn.close()
        
        return plates

    def get_plates_by_state(self, state):
        """Retrieve license plates by state"""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT plate_number, state, detection_date, image_path
            FROM plates
            WHERE state = ?
            ORDER BY detection_date DESC
        ''', (state,))
        
        plates = cursor.fetchall()
        conn.close()
        
        return plates

    def get_plate_by_number(self, plate_number):
        """Retrieve a specific license plate by number"""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT plate_number, state, detection_date, image_path
            FROM plates
            WHERE plate_number = ?
        ''', (plate_number,))
        
        plate = cursor.fetchone()
        conn.close()
        
        return plate 