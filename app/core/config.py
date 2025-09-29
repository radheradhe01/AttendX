"""
Configuration settings for the AttendX application.
"""

import json
import os
from typing import List, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    # Database Configuration
    mongodb_url: str = Field(default="mongodb://localhost:27017", env="MONGODB_URL")
    database_name: str = Field(default="attendx", env="DATABASE_NAME")

    # JWT Configuration
    secret_key: str = Field(default="your-super-secret-key-change-this-in-production", env="SECRET_KEY")
    algorithm: str = Field(default="HS256", env="ALGORITHM")
    access_token_expire_minutes: int = Field(default=30, env="ACCESS_TOKEN_EXPIRE_MINUTES")

    # Server Configuration
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8000, env="PORT")
    debug: bool = Field(default=True, env="DEBUG")

    # CORS Configuration
    allowed_origins: List[str] = Field(
        default=[
            "http://localhost:3000",
            "http://127.0.0.1:3000",
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://host.docker.internal:3000",
            "http://host.docker.internal:5173",
            "http://172.17.0.1:3000",
            "http://172.17.0.1:5173",
            "http://192.168.65.1:3000",
            "http://192.168.65.1:5173"
        ],
        env="ALLOWED_ORIGINS"
    )

    @field_validator('allowed_origins', mode='before')
    @classmethod
    def parse_allowed_origins(cls, v):
        # Handle the case where Pydantic tries to parse as JSON first
        if isinstance(v, str) and v.startswith('[') and v.endswith(']'):
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                pass

        if isinstance(v, str):
            # Handle comma-separated string from environment
            if v.strip():
                return [origin.strip() for origin in v.split(',')]
            else:
                return []
        return v

    # File Upload Configuration
    max_file_size: int = Field(default=10485760, env="MAX_FILE_SIZE")  # 10MB
    upload_dir: str = Field(default="uploads", env="UPLOAD_DIR")

    # Facial Recognition Configuration
    face_recognition_threshold: float = Field(default=0.6, env="FACE_RECOGNITION_THRESHOLD")
    max_face_registrations: int = Field(default=5, env="MAX_FACE_REGISTRATIONS")

    # Location Configuration
    location_radius_km: float = Field(default=0.5, env="LOCATION_RADIUS_KM")

    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()
