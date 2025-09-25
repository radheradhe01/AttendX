"""
User model for the attendance system
"""

from sqlalchemy import Column, Integer, String, DateTime, Boolean, Text, ForeignKey, Float
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.database import Base


class Organization(Base):
    """Organization model"""
    __tablename__ = "organizations"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), unique=True, index=True, nullable=False)
    description = Column(Text)
    address = Column(Text)
    latitude = Column(Float)  # Organization's main location latitude
    longitude = Column(Float)  # Organization's main location longitude
    radius = Column(Float, default=100.0)  # Allowed radius in meters
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    users = relationship("User", back_populates="organization")
    attendance_records = relationship("AttendanceRecord", back_populates="organization")


class User(Base):
    """User model"""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    username = Column(String(100), unique=True, index=True, nullable=False)
    full_name = Column(String(255), nullable=False)
    hashed_password = Column(String(255), nullable=False)
    role = Column(String(50), default="employee")  # admin, manager, employee, student
    is_active = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)
    
    # Organization relationship
    organization_id = Column(Integer, ForeignKey("organizations.id"), nullable=False)
    organization = relationship("Organization", back_populates="users")
    
    # Face recognition data
    face_encoding = Column(Text)  # JSON string of face encoding
    profile_image_path = Column(String(500))
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    last_login = Column(DateTime(timezone=True))
    
    # Relationships
    attendance_records = relationship("AttendanceRecord", back_populates="user")


class AttendanceRecord(Base):
    """Attendance record model"""
    __tablename__ = "attendance_records"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    organization_id = Column(Integer, ForeignKey("organizations.id"), nullable=False)
    
    # Attendance data
    attendance_type = Column(String(50), default="check_in")  # check_in, check_out, break_start, break_end
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    
    # Location data
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    address = Column(Text)  # Human-readable address
    accuracy = Column(Float)  # GPS accuracy in meters
    
    # Verification data
    face_confidence = Column(Float)  # Face recognition confidence score
    is_verified = Column(Boolean, default=False)
    verification_notes = Column(Text)
    
    # Image data
    captured_image_path = Column(String(500))
    
    # Relationships
    user = relationship("User", back_populates="attendance_records")
    organization = relationship("Organization", back_populates="attendance_records")


class AttendanceSession(Base):
    """Attendance session for tracking work hours"""
    __tablename__ = "attendance_sessions"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    organization_id = Column(Integer, ForeignKey("organizations.id"), nullable=False)
    
    # Session data
    check_in_time = Column(DateTime(timezone=True), nullable=False)
    check_out_time = Column(DateTime(timezone=True))
    total_hours = Column(Float)  # Calculated total hours
    
    # Location data
    check_in_latitude = Column(Float)
    check_in_longitude = Column(Float)
    check_out_latitude = Column(Float)
    check_out_longitude = Column(Float)
    
    # Status
    is_active = Column(Boolean, default=True)
    status = Column(String(50), default="active")  # active, completed, incomplete
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    user = relationship("User")
    organization = relationship("Organization")
