"""
Attendance model definitions.
"""

from typing import Optional

from beanie import Indexed
from pydantic import BaseModel, Field

from .base import AttendanceStatus, BaseDocument, DeviceInfo, Location


class Attendance(BaseDocument):
    """Attendance record document model."""

    user_id: Indexed(str)
    organization_id: Indexed(str)
    timestamp: str = Field(default_factory=lambda: str)  # ISO format timestamp

    # Location information
    location: Location
    location_verified: bool = Field(default=False)
    location_distance_km: Optional[float] = None
    approved_location_id: Optional[str] = None

    # Facial recognition
    face_verified: bool = Field(default=False)
    confidence_score: Optional[float] = Field(None, ge=0, le=1)
    face_image_id: Optional[str] = None

    # Device information
    device_info: DeviceInfo

    # Status and notes
    status: AttendanceStatus = Field(default=AttendanceStatus.PRESENT)
    notes: Optional[str] = Field(None, max_length=500)
    approved_by: Optional[str] = None  # Manager/ Admin ID
    approved_at: Optional[str] = None

    # Work session tracking
    work_session_id: Optional[str] = None
    work_hours_logged: Optional[float] = None  # Hours worked this session

    class Settings:
        name = "attendance_records"


class AttendanceCreate(BaseModel):
    """Attendance creation model."""

    location: Location
    notes: Optional[str] = Field(None, max_length=500)


class AttendanceUpdate(BaseModel):
    """Attendance update model."""

    status: Optional[AttendanceStatus] = None
    notes: Optional[str] = Field(None, max_length=500)
    approved_by: Optional[str] = None


class AttendanceResponse(BaseModel):
    """Attendance response model."""

    id: str
    user_id: str
    organization_id: str
    timestamp: str
    location: Location
    location_verified: bool
    location_distance_km: Optional[float]
    approved_location_id: Optional[str]
    face_verified: bool
    confidence_score: Optional[float]
    device_info: DeviceInfo
    status: AttendanceStatus
    notes: Optional[str]
    approved_by: Optional[str]
    approved_at: Optional[str]
    work_session_id: Optional[str]
    work_hours_logged: Optional[float]
    created_at: str
    updated_at: str


class AttendanceSummary(BaseModel):
    """Attendance summary for reporting."""

    user_id: str
    user_name: str
    total_days: int
    present_days: int
    late_days: int
    absent_days: int
    average_hours_per_day: Optional[float]
    attendance_percentage: float

