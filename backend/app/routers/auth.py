"""
Authentication router for login, registration, and token management
"""

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from datetime import timedelta
from typing import Dict, Any

from app.database import get_db
from app.schemas.user import UserCreate, User, Token, LoginRequest, PasswordChange
from app.services.auth_service import AuthService
from app.services.face_recognition_service import FaceRecognitionService
from app.models.user import User as UserModel
from loguru import logger

router = APIRouter()
auth_service = AuthService()
face_service = FaceRecognitionService()

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/auth/login")


def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)) -> UserModel:
    """Get current authenticated user"""
    user = auth_service.get_current_user(db, token)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


@router.post("/register", response_model=Dict[str, Any])
async def register_user(
    user_data: UserCreate,
    db: Session = Depends(get_db)
):
    """Register a new user"""
    try:
        # Check if user already exists
        existing_user = db.query(UserModel).filter(
            (UserModel.email == user_data.email) | 
            (UserModel.username == user_data.username)
        ).first()
        
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="User with this email or username already exists"
            )
        
        # Create new user
        hashed_password = auth_service.get_password_hash(user_data.password)
        
        new_user = UserModel(
            email=user_data.email,
            username=user_data.username,
            full_name=user_data.full_name,
            hashed_password=hashed_password,
            role=user_data.role,
            organization_id=user_data.organization_id
        )
        
        db.add(new_user)
        db.commit()
        db.refresh(new_user)
        
        logger.info(f"User registered successfully: {new_user.username}")
        
        return {
            "message": "User registered successfully",
            "user_id": new_user.id,
            "username": new_user.username
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Registration error: {e}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Registration failed"
        )


@router.post("/login", response_model=Token)
async def login_user(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db)
):
    """Login user and return access token"""
    try:
        user = auth_service.authenticate_user(db, form_data.username, form_data.password)
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Create access token
        token_data = auth_service.create_user_token(user)
        
        logger.info(f"User logged in successfully: {user.username}")
        
        return Token(
            access_token=token_data["access_token"],
            token_type=token_data["token_type"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Login failed"
        )


@router.post("/login-json", response_model=Token)
async def login_user_json(
    login_data: LoginRequest,
    db: Session = Depends(get_db)
):
    """Login user with JSON data"""
    try:
        user = auth_service.authenticate_user(db, login_data.username, login_data.password)
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Create access token
        token_data = auth_service.create_user_token(user)
        
        logger.info(f"User logged in successfully: {user.username}")
        
        return Token(
            access_token=token_data["access_token"],
            token_type=token_data["token_type"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Login failed"
        )


@router.get("/me", response_model=User)
async def get_current_user_info(
    current_user: UserModel = Depends(get_current_user)
):
    """Get current user information"""
    return current_user


@router.post("/change-password")
async def change_password(
    password_data: PasswordChange,
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Change user password"""
    try:
        success = auth_service.change_password(
            db, 
            current_user, 
            password_data.current_password, 
            password_data.new_password
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid current password"
            )
        
        return {"message": "Password changed successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Password change error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Password change failed"
        )


@router.post("/refresh-token", response_model=Token)
async def refresh_access_token(
    current_user: UserModel = Depends(get_current_user)
):
    """Refresh access token"""
    try:
        # Create new token for current user
        token_data = auth_service.create_user_token(current_user)
        
        return Token(
            access_token=token_data["access_token"],
            token_type=token_data["token_type"]
        )
        
    except Exception as e:
        logger.error(f"Token refresh error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Token refresh failed"
        )


@router.post("/logout")
async def logout_user(
    current_user: UserModel = Depends(get_current_user)
):
    """Logout user (client-side token removal)"""
    logger.info(f"User logged out: {current_user.username}")
    return {"message": "Logged out successfully"}


@router.get("/verify-token")
async def verify_token(
    current_user: UserModel = Depends(get_current_user)
):
    """Verify if token is valid"""
    return {
        "valid": True,
        "user": {
            "id": current_user.id,
            "username": current_user.username,
            "role": current_user.role
        }
    }
