"""
User model definitions.
"""

from typing import List, Optional

from beanie import Indexed
from pydantic import BaseModel, EmailStr, Field

from .base import BaseDocument, UserRole


class User(BaseDocument):
    """User document model."""

    organization_id: Indexed(str)
    email: Indexed(str, unique=True) = Field(..., pattern=r"^[^@]+@[^@]+\.[^@]+$")
    name: str = Field(..., min_length=1, max_length=100)
    role: UserRole = Field(default=UserRole.EMPLOYEE)
    is_active: bool = Field(default=True)
    last_login: Optional[str] = None

    # Facial recognition data
    face_encodings: List[bytes] = Field(default_factory=list)
    face_image_id: Optional[str] = None  # Reference to GridFS image
    face_registration_count: int = Field(default=0)

    # Profile information
    phone: Optional[str] = Field(None, max_length=20)
    department: Optional[str] = Field(None, max_length=50)
    position: Optional[str] = Field(None, max_length=50)

    class Settings:
        name = "users"


class UserCreate(BaseModel):
    """User creation model."""

    organization_id: str
    email: EmailStr
    name: str = Field(..., min_length=1, max_length=100)
    role: UserRole = Field(default=UserRole.EMPLOYEE)
    phone: Optional[str] = Field(None, max_length=20)
    department: Optional[str] = Field(None, max_length=50)
    position: Optional[str] = Field(None, max_length=50)


class UserUpdate(BaseModel):
    """User update model."""

    name: Optional[str] = Field(None, min_length=1, max_length=100)
    role: Optional[UserRole] = None
    is_active: Optional[bool] = None
    phone: Optional[str] = Field(None, max_length=20)
    department: Optional[str] = Field(None, max_length=50)
    position: Optional[str] = Field(None, max_length=50)


class UserResponse(BaseModel):
    """User response model."""

    id: str
    organization_id: str
    email: str
    name: str
    role: UserRole
    is_active: bool
    last_login: Optional[str]
    face_registration_count: int
    phone: Optional[str]
    department: Optional[str]
    position: Optional[str]
    created_at: str
    updated_at: str


class UserProfile(BaseModel):
    """User profile information."""

    id: str
    email: str
    name: str
    role: UserRole
    phone: Optional[str]
    department: Optional[str]
    position: Optional[str]
    organization_name: Optional[str] = None

