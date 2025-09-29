"""
Database connection and configuration for MongoDB.
"""

import motor.motor_asyncio
from pymongo.errors import ServerSelectionTimeoutError

from core.config import settings


class Database:
    """Database connection manager."""

    client: motor.motor_asyncio.AsyncIOMotorClient = None
    database: motor.motor_asyncio.AsyncIOMotorDatabase = None

    @classmethod
    async def connect_db(cls):
        """Create database connection."""
        try:
            cls.client = motor.motor_asyncio.AsyncIOMotorClient(settings.mongodb_url)
            cls.database = cls.client[settings.database_name]

            # Test the connection
            await cls.client.admin.command('ping')
            print(f"Connected to MongoDB: {settings.database_name}")
        except ServerSelectionTimeoutError:
            print("Failed to connect to MongoDB. Please ensure MongoDB is running.")
            raise
        except Exception as e:
            print(f"Database connection error: {e}")
            raise

    @classmethod
    async def disconnect_db(cls):
        """Close database connection."""
        if cls.client:
            cls.client.close()
            print("Disconnected from MongoDB")

    @classmethod
    def get_database(cls):
        """Get database instance."""
        if cls.database is None:
            raise RuntimeError("Database not connected. Call connect_db() first.")
        return cls.database

    @classmethod
    def get_collection(cls, collection_name: str):
        """Get collection instance."""
        database = cls.get_database()
        return database[collection_name]


# Global database instance
db = Database()

