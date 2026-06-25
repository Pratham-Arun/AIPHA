"""
MongoDB Connection Module
─────────────────────────
Manages the connection to the MongoDB server using a singleton pattern.
"""

from pymongo import MongoClient
from pymongo.database import Database
import config


class MongoConnection:
    """
    Singleton class for MongoDB connection management.
    Ensures only one MongoClient is created during the application lifecycle.
    """
    _instance = None
    _client: MongoClient = None
    _db: Database = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MongoConnection, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """Initialize the MongoDB client and database objects."""
        print("  Connecting to MongoDB...")
        try:
            self._client = MongoClient(config.MONGODB_URI, serverSelectionTimeoutMS=5000)
            self._db = self._client[config.DATABASE_NAME]
            # Test connection
            self._client.admin.command('ping')
            
            # Print Environment Diagnostics
            print("\nEnvironment")
            print(f"{config.APP_ENV.capitalize()}")
            print("Connected")
            db_type = "Local MongoDB" if "localhost" in config.MONGODB_URI or "127.0.0.1" in config.MONGODB_URI else "MongoDB Atlas"
            if db_type == "Local MongoDB":
                # Assuming localhost:27017 format
                try:
                    host_port = config.MONGODB_URI.split("mongodb://")[1]
                    print(host_port)
                except:
                    print("localhost:27017")
            print("Database")
            print(f"{config.DATABASE_NAME}")
        except Exception as e:
            raise ConnectionError(f"Failed to connect to MongoDB at {config.MONGODB_URI}: {e}")

    def get_database(self) -> Database:
        """Return the database instance."""
        return self._db

    def get_client(self) -> MongoClient:
        """Return the client instance."""
        return self._client

    @classmethod
    def validate_connection(cls):
        """
        Static method to validate the connection without keeping the instance.
        Used during startup checks.
        """
        instance = cls()
        instance.get_database() # Ensure it's reachable

