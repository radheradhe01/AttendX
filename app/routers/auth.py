"""
Authentication router for login, registration, and token management.
"""

from datetime import timedelta

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm

from core.security import (
    generate_user_token,
    get_user_from_token,
    hash_password,
    verify_password
)
from models.auth import (
    AuthenticatedUser,
    LoginResponse,
    PasswordChange,
    PasswordReset,
    PasswordResetConfirm,
    Token,
    UserLogin,
    UserRegister
)

router = APIRouter()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")


@router.post("/login", response_model=LoginResponse)
async def login(login_data: UserLogin):
    """User login endpoint."""
    # TODO: Implement actual user authentication
    # For now, return a mock successful login
    user_data = {
        "id": "user_id_placeholder",
        "email": login_data.email,
        "name": "User Name",
        "role": "employee",
        "organization_id": "org_id_placeholder"
    }

    token_data = {
        "sub": user_data["id"],
        "email": user_data["email"],
        "role": user_data["role"],
        "organization_id": user_data["organization_id"]
    }

    access_token = generate_user_token(
        user_data["id"],
        user_data["email"],
        user_data["role"],
        user_data["organization_id"]
    )

    return LoginResponse(
        user=user_data,
        token=Token(
            access_token=access_token,
            token_type="bearer",
            expires_in=1800  # 30 minutes
        ),
        organization={"id": user_data["organization_id"], "name": "Organization Name"}
    )


@router.post("/register", response_model=LoginResponse)
async def register(user_data: UserRegister):
    """User registration endpoint."""
    # TODO: Implement actual user registration
    # For now, return a mock successful registration
    user_data_dict = {
        "id": "new_user_id",
        "email": user_data.email,
        "name": user_data.name,
        "role": user_data.role,
        "organization_id": user_data.organization_id
    }

    token_data = {
        "sub": user_data_dict["id"],
        "email": user_data_dict["email"],
        "role": user_data_dict["role"],
        "organization_id": user_data_dict["organization_id"]
    }

    access_token = generate_user_token(
        user_data_dict["id"],
        user_data_dict["email"],
        user_data_dict["role"],
        user_data_dict["organization_id"]
    )

    return LoginResponse(
        user=user_data_dict,
        token=Token(
            access_token=access_token,
            token_type="bearer",
            expires_in=1800
        ),
        organization={"id": user_data.organization_id, "name": "Organization Name"}
    )


@router.post("/refresh", response_model=Token)
async def refresh_token(current_token: str = Depends(oauth2_scheme)):
    """Refresh access token."""
    # TODO: Implement token refresh logic
    user_data = get_user_from_token(current_token)
    if not user_data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )

    new_token = generate_user_token(
        user_data["user_id"],
        user_data["email"],
        user_data["role"],
        user_data["organization_id"]
    )

    return Token(
        access_token=new_token,
        token_type="bearer",
        expires_in=1800
    )


@router.post("/change-password")
async def change_password(
    password_data: PasswordChange,
    current_token: str = Depends(oauth2_scheme)
):
    """Change user password."""
    # TODO: Implement password change logic
    user_data = get_user_from_token(current_token)
    if not user_data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )

    # Verify current password and update to new password
    # This is a placeholder - implement actual password verification

    return {"message": "Password changed successfully"}


@router.post("/reset-password")
async def reset_password_request(reset_data: PasswordReset):
    """Request password reset."""
    # TODO: Implement password reset request
    # Send reset email with token
    return {"message": "Password reset email sent"}


@router.post("/reset-password-confirm")
async def reset_password_confirm(reset_data: PasswordResetConfirm):
    """Confirm password reset."""
    # TODO: Implement password reset confirmation
    # Verify token and update password
    return {"message": "Password reset successfully"}


@router.get("/me", response_model=AuthenticatedUser)
async def get_current_user(current_token: str = Depends(oauth2_scheme)):
    """Get current authenticated user information."""
    user_data = get_user_from_token(current_token)
    if not user_data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )

    # TODO: Get full user information from database
    return AuthenticatedUser(
        id=user_data["user_id"],
        email=user_data["email"],
        name="User Name",  # TODO: Get from database
        role=user_data["role"],
        organization_id=user_data["organization_id"],
        organization_name="Organization Name"  # TODO: Get from database
    )

