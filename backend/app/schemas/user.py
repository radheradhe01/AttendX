"""
Pydantic schemas for user-related operations
"""

from pydantic import BaseModel, EmailStr, validator
from typing import Optional, List
from datetime import datetime
from enum import Enum


class UserRole(str, Enum):
    ADMIN = "admin"
    MANAGER = "manager"
    EMPLOYEE = "employee"
    STUDENT = "student"


class AttendanceType(str, Enum):
    CHECK_IN = "check_in"
    CHECK_OUT = "check_out"
    BREAK_START = "break_start"
    BREAK_END = "break_end"


# Organization Schemas
class OrganizationBase(BaseModel):
    name: str
    description: Optional[str] = None
    address: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    radius: float = 100.0


class OrganizationCreate(OrganizationBase):
    pass


class OrganizationUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    address: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    radius: Optional[float] = None
    is_active: Optional[bool] = None


class Organization(OrganizationBase):
    id: int
    is_active: bool
    created_at: datetime
    updated_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True


# User Schemas
class UserBase(BaseModel):
    email: EmailStr
    username: str
    full_name: str
    role: UserRole = UserRole.EMPLOYEE
    organization_id: int


class UserCreate(UserBase):
    password: str
    
    @validator('password')
    def validate_password(cls, v):
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters long')
        return v


class UserUpdate(BaseModel):
    email: Optional[EmailStr] = None
    username: Optional[str] = None
    full_name: Optional[str] = None
    role: Optional[UserRole] = None
    is_active: Optional[bool] = None


class UserProfileUpdate(BaseModel):
    full_name: Optional[str] = None
    email: Optional[EmailStr] = None


class User(UserBase):
    id: int
    is_active: bool
    is_verified: bool
    profile_image_path: Optional[str] = None
    created_at: datetime
    updated_at: Optional[datetime] = None
    last_login: Optional[datetime] = None
    organization: Organization
    
    class Config:
        from_attributes = True


class UserInDB(User):
    hashed_password: str


# Authentication Schemas
class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    username: Optional[str] = None


class LoginRequest(BaseModel):
    username: str
    password: str


class PasswordChange(BaseModel):
    current_password: str
    new_password: str
    
    @validator('new_password')
    def validate_new_password(cls, v):
        if len(v) < 8:
            raise ValueError('New password must be at least 8 characters long')
        return v


# Attendance Schemas
class AttendanceLocation(BaseModel):
    latitude: float
    longitude: float
    accuracy: Optional[float] = None
    address: Optional[str] = None


class AttendanceCreate(BaseModel):
    attendance_type: AttendanceType
    location: AttendanceLocation
    image_data: Optional[str] = None  # Base64 encoded image


class AttendanceRecord(BaseModel):
    id: int
    user_id: int
    organization_id: int
    attendance_type: AttendanceType
    timestamp: datetime
    latitude: float
    longitude: float
    address: Optional[str] = None
    accuracy: Optional[float] = None
    face_confidence: Optional[float] = None
    is_verified: bool
    verification_notes: Optional[str] = None
    captured_image_path: Optional[str] = None
    
    class Config:
        from_attributes = True


class AttendanceResponse(BaseModel):
    success: bool
    message: str
    attendance_record: Optional[AttendanceRecord] = None
    confidence_score: Optional[float] = None


# Session Schemas
class AttendanceSession(BaseModel):
    id: int
    user_id: int
    organization_id: int
    check_in_time: datetime
    check_out_time: Optional[datetime] = None
    total_hours: Optional[float] = None
    check_in_latitude: Optional[float] = None
    check_in_longitude: Optional[float] = None
    check_out_latitude: Optional[float] = None
    check_out_longitude: Optional[float] = None
    is_active: bool
    status: str
    created_at: datetime
    updated_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True


# Statistics Schemas
class AttendanceStats(BaseModel):
    total_attendance: int
    check_ins: int
    check_outs: int
    total_hours: float
    average_hours_per_day: float
    attendance_rate: float


class UserAttendanceSummary(BaseModel):
    user: User
    stats: AttendanceStats
    recent_records: List[AttendanceRecord]
