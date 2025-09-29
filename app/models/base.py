"""
Base models and common types for the application.
"""

from datetime import datetime
from typing import Optional

from beanie import Document
from pydantic import BaseModel, Field
from enum import Enum


class BaseDocument(Document):
    """Base document class with common fields."""

    id: Optional[str] = Field(default=None, alias="_id")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    class Settings:
        is_root = True
        validate_on_save = True

    def save(self, *args, **kwargs):
        """Override save to update timestamp."""
        self.updated_at = datetime.utcnow()
        super().save(*args, **kwargs)


class BaseModel(BaseModel):
    """Base Pydantic model."""

    class Config:
        from_attributes = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class Location(BaseModel):
    """Geographic location model."""

    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    accuracy: Optional[float] = None
    address: Optional[str] = None


class DeviceInfo(BaseModel):
    """Device information model."""

    user_agent: Optional[str] = None
    ip_address: Optional[str] = None
    platform: Optional[str] = None


class OrganizationSettings(BaseModel):
    """Organization-specific settings."""

    allowed_radius_km: float = Field(default=0.5, ge=0.1, le=10)
    require_face_recognition: bool = Field(default=True)
    track_work_hours: bool = Field(default=True)
    auto_approve_attendance: bool = Field(default=False)
    max_daily_attendance: Optional[int] = Field(default=None, ge=1)
    working_hours_start: Optional[str] = None  # "09:00"
    working_hours_end: Optional[str] = None    # "17:00"


class UserRole(str, Enum):
    """User role enumeration."""

    SUPER_ADMIN = "super_admin"
    ORG_ADMIN = "org_admin"
    MANAGER = "manager"
    EMPLOYEE = "employee"


class AttendanceStatus(str, Enum):
    """Attendance status enumeration."""

    PRESENT = "present"
    LATE = "late"
    ABSENT = "absent"
    ON_LEAVE = "on_leave"


class WorkSessionStatus(str, Enum):
    """Work session status enumeration."""

    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
