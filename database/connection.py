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
            self._client = MongoClient(config.MONGO_URI, serverSelectionTimeoutMS=5000)
            self._db = self._client[config.MONGO_DB_NAME]
            # Test connection
            self._client.admin.command('ping')
            print(f"  Connected to database: {config.MONGO_DB_NAME}")
        except Exception as e:
            raise ConnectionError(f"Failed to connect to MongoDB at {config.MONGO_URI}: {e}")

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

