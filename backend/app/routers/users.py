"""
User management router
"""

from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File
from sqlalchemy.orm import Session
from typing import List, Optional
import base64
import os
import uuid

from app.database import get_db
from app.schemas.user import User, UserCreate, UserUpdate, UserProfileUpdate
from app.models.user import User as UserModel, Organization
from app.services.auth_service import AuthService
from app.services.face_recognition_service import FaceRecognitionService
from app.routers.auth import get_current_user
from loguru import logger

router = APIRouter()
auth_service = AuthService()
face_service = FaceRecognitionService()


def save_profile_image(image_data: str, user_id: int) -> str:
    """Save profile image and return file path"""
    try:
        # Create uploads directory if it doesn't exist
        upload_dir = "uploads/profiles"
        os.makedirs(upload_dir, exist_ok=True)
        
        # Generate unique filename
        filename = f"profile_{user_id}_{uuid.uuid4().hex}.jpg"
        file_path = os.path.join(upload_dir, filename)
        
        # Decode and save image
        image_bytes = base64.b64decode(image_data)
        with open(file_path, "wb") as f:
            f.write(image_bytes)
        
        return file_path
        
    except Exception as e:
        logger.error(f"Error saving profile image: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to save profile image"
        )


@router.get("/me", response_model=User)
async def get_my_profile(
    current_user: UserModel = Depends(get_current_user)
):
    """Get current user's profile"""
    return current_user


@router.put("/me", response_model=User)
async def update_my_profile(
    profile_data: UserProfileUpdate,
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Update current user's profile"""
    try:
        # Update user fields
        if profile_data.full_name is not None:
            current_user.full_name = profile_data.full_name
        
        if profile_data.email is not None:
            # Check if email is already taken
            existing_user = db.query(UserModel).filter(
                UserModel.email == profile_data.email,
                UserModel.id != current_user.id
            ).first()
            
            if existing_user:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Email already in use"
                )
            
            current_user.email = profile_data.email
        
        db.commit()
        db.refresh(current_user)
        
        logger.info(f"Profile updated for user: {current_user.username}")
        return current_user
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating profile: {e}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update profile"
        )


@router.post("/me/profile-image")
async def upload_profile_image(
    image_data: str,  # Base64 encoded image
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Upload and set profile image with face recognition"""
    try:
        # Save profile image
        image_path = save_profile_image(image_data, current_user.id)
        
        # Extract face encoding for attendance verification
        face_encoding = face_service.encode_face_from_image(image_data)
        
        if face_encoding is None:
            # Remove saved image if face recognition failed
            if os.path.exists(image_path):
                os.remove(image_path)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No face detected in image. Please ensure your face is clearly visible."
            )
        
        # Update user record
        current_user.profile_image_path = image_path
        current_user.face_encoding = face_service.get_face_encoding_json(face_encoding)
        
        # Add to face recognition service
        face_service.add_known_face(current_user.id, face_encoding)
        
        db.commit()
        
        logger.info(f"Profile image uploaded for user: {current_user.username}")
        
        return {
            "message": "Profile image uploaded successfully",
            "image_path": image_path
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading profile image: {e}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to upload profile image"
        )


@router.delete("/me/profile-image")
async def delete_profile_image(
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Delete profile image"""
    try:
        # Remove from face recognition service
        face_service.remove_known_face(current_user.id)
        
        # Delete image file if exists
        if current_user.profile_image_path and os.path.exists(current_user.profile_image_path):
            os.remove(current_user.profile_image_path)
        
        # Update user record
        current_user.profile_image_path = None
        current_user.face_encoding = None
        
        db.commit()
        
        logger.info(f"Profile image deleted for user: {current_user.username}")
        
        return {"message": "Profile image deleted successfully"}
        
    except Exception as e:
        logger.error(f"Error deleting profile image: {e}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete profile image"
        )


@router.get("/", response_model=List[User])
async def get_users(
    organization_id: Optional[int] = None,
    role: Optional[str] = None,
    is_active: Optional[bool] = None,
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get users (admin/manager only)"""
    try:
        # Check permissions
        if current_user.role not in ["admin", "manager"]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions"
            )
        
        query = db.query(UserModel)
        
        # Filter by organization (managers can only see their org)
        if current_user.role == "manager":
            query = query.filter(UserModel.organization_id == current_user.organization_id)
        elif organization_id:
            query = query.filter(UserModel.organization_id == organization_id)
        
        # Apply filters
        if role:
            query = query.filter(UserModel.role == role)
        
        if is_active is not None:
            query = query.filter(UserModel.is_active == is_active)
        
        users = query.all()
        
        return users
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting users: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get users"
        )


@router.get("/{user_id}", response_model=User)
async def get_user(
    user_id: int,
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get user by ID (admin/manager only)"""
    try:
        # Check permissions
        if current_user.role not in ["admin", "manager"]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions"
            )
        
        user = db.query(UserModel).filter(UserModel.id == user_id).first()
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        # Managers can only see users from their organization
        if current_user.role == "manager" and user.organization_id != current_user.organization_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions"
            )
        
        return user
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting user: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get user"
        )


@router.put("/{user_id}", response_model=User)
async def update_user(
    user_id: int,
    user_data: UserUpdate,
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Update user (admin/manager only)"""
    try:
        # Check permissions
        if current_user.role not in ["admin", "manager"]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions"
            )
        
        user = db.query(UserModel).filter(UserModel.id == user_id).first()
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        # Managers can only update users from their organization
        if current_user.role == "manager" and user.organization_id != current_user.organization_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions"
            )
        
        # Update user fields
        if user_data.email is not None:
            # Check if email is already taken
            existing_user = db.query(UserModel).filter(
                UserModel.email == user_data.email,
                UserModel.id != user_id
            ).first()
            
            if existing_user:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Email already in use"
                )
            
            user.email = user_data.email
        
        if user_data.username is not None:
            # Check if username is already taken
            existing_user = db.query(UserModel).filter(
                UserModel.username == user_data.username,
                UserModel.id != user_id
            ).first()
            
            if existing_user:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Username already in use"
                )
            
            user.username = user_data.username
        
        if user_data.full_name is not None:
            user.full_name = user_data.full_name
        
        if user_data.role is not None:
            # Only admins can change roles
            if current_user.role != "admin":
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Only admins can change user roles"
                )
            user.role = user_data.role
        
        if user_data.is_active is not None:
            user.is_active = user_data.is_active
        
        db.commit()
        db.refresh(user)
        
        logger.info(f"User updated: {user.username}")
        return user
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating user: {e}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update user"
        )


@router.delete("/{user_id}")
async def delete_user(
    user_id: int,
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Delete user (admin only)"""
    try:
        # Check permissions
        if current_user.role != "admin":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only admins can delete users"
            )
        
        user = db.query(UserModel).filter(UserModel.id == user_id).first()
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        # Don't allow deleting self
        if user.id == current_user.id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot delete your own account"
            )
        
        # Remove from face recognition service
        face_service.remove_known_face(user_id)
        
        # Delete profile image if exists
        if user.profile_image_path and os.path.exists(user.profile_image_path):
            os.remove(user.profile_image_path)
        
        # Delete user
        db.delete(user)
        db.commit()
        
        logger.info(f"User deleted: {user.username}")
        
        return {"message": "User deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting user: {e}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete user"
        )
