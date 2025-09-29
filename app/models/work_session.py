"""
Work session model definitions for remote work tracking.
"""

from typing import List, Optional

from beanie import Indexed
from pydantic import BaseModel, Field

from models.base import BaseDocument, WorkSessionStatus


class BreakPeriod(BaseModel):
    """Break period within a work session."""

    start_time: str  # ISO format
    end_time: Optional[str] = None  # None if break is ongoing
    duration_minutes: Optional[float] = None


class WorkSession(BaseDocument):
    """Work session document model."""

    user_id: Indexed(str)
    organization_id: Indexed(str)

    # Session timing
    start_time: str  # ISO format
    end_time: Optional[str] = None
    duration_minutes: Optional[float] = None

    # Break tracking
    breaks: List[BreakPeriod] = Field(default_factory=list)
    total_break_minutes: float = Field(default=0)

    # Status and tracking
    status: WorkSessionStatus = Field(default=WorkSessionStatus.ACTIVE)
    is_remote: bool = Field(default=True)

    # Attendance linking
    attendance_record_id: Optional[str] = None

    # Productivity metrics (optional)
    keystrokes_count: Optional[int] = Field(default=0)
    mouse_clicks_count: Optional[int] = Field(default=0)
    screenshots_count: Optional[int] = Field(default=0)
    productivity_score: Optional[float] = Field(None, ge=0, le=100)

    # Session metadata
    device_info: Optional[dict] = None
    ip_address: Optional[str] = None

    class Settings:
        name = "work_sessions"


class WorkSessionCreate(BaseModel):
    """Work session creation model."""

    is_remote: bool = Field(default=True)
    attendance_record_id: Optional[str] = None


class WorkSessionUpdate(BaseModel):
    """Work session update model."""

    end_time: Optional[str] = None
    status: Optional[WorkSessionStatus] = None
    productivity_score: Optional[float] = Field(None, ge=0, le=100)


class WorkSessionResponse(BaseModel):
    """Work session response model."""

    id: str
    user_id: str
    organization_id: str
    start_time: str
    end_time: Optional[str]
    duration_minutes: Optional[float]
    breaks: List[BreakPeriod]
    total_break_minutes: float
    status: WorkSessionStatus
    is_remote: bool
    attendance_record_id: Optional[str]
    keystrokes_count: Optional[int]
    mouse_clicks_count: Optional[int]
    screenshots_count: Optional[int]
    productivity_score: Optional[float]
    device_info: Optional[dict]
    ip_address: Optional[str]
    created_at: str
    updated_at: str


class WorkSessionSummary(BaseModel):
    """Work session summary for reporting."""

    user_id: str
    user_name: str
    total_sessions: int
    total_hours: float
    average_session_hours: float
    total_breaks: int
    total_break_hours: float
    average_productivity_score: Optional[float]

