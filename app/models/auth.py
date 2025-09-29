"""
Authentication model definitions.
"""

from typing import Optional

from pydantic import BaseModel, EmailStr, Field

from models.base import UserRole


class Token(BaseModel):
    """JWT token response model."""

    access_token: str
    token_type: str = "bearer"
    expires_in: int  # seconds


class TokenData(BaseModel):
    """Token data model."""

    user_id: Optional[str] = None
    email: Optional[str] = None
    role: Optional[UserRole] = None
    organization_id: Optional[str] = None


class UserLogin(BaseModel):
    """User login model."""

    email: EmailStr
    password: str = Field(..., min_length=6, max_length=100)


class UserRegister(BaseModel):
    """User registration model."""

    email: EmailStr
    password: str = Field(..., min_length=6, max_length=100)
    name: str = Field(..., min_length=1, max_length=100)
    organization_id: str
    role: UserRole = Field(default=UserRole.EMPLOYEE)


class PasswordChange(BaseModel):
    """Password change model."""

    current_password: str = Field(..., min_length=6, max_length=100)
    new_password: str = Field(..., min_length=6, max_length=100)


class PasswordReset(BaseModel):
    """Password reset model."""

    email: EmailStr


class PasswordResetConfirm(BaseModel):
    """Password reset confirmation model."""

    token: str
    new_password: str = Field(..., min_length=6, max_length=100)


class LoginResponse(BaseModel):
    """Login response model."""

    user: dict  # User information
    token: Token
    organization: dict  # Organization information


class AuthenticatedUser(BaseModel):
    """Authenticated user information."""

    id: str
    email: str
    name: str
    role: UserRole
    organization_id: str
    organization_name: Optional[str] = None
    permissions: list[str] = Field(default_factory=list)

