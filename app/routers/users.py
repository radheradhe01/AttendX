"""
User management router.
"""

from typing import List

from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File
from fastapi.security import HTTPBearer

from core.security import get_user_from_token
from models.user import UserCreate, UserUpdate, UserResponse
from services.user_service import UserService
from models.base import UserRole

router = APIRouter()
security = HTTPBearer()


@router.get("/", response_model=List[UserResponse])
async def get_users(
    organization_id: str = None,
    skip: int = 0,
    limit: int = 100,
    token: str = Depends(security)
):
    """Get all users."""
    user_data = get_user_from_token(token.credentials)
    if not user_data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )

    if organization_id:
        # Check if user has permission to view users in this organization
        has_permission = await UserService.check_user_permission(
            user_data["user_id"],
            organization_id
        )
        if not has_permission:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to view users in this organization"
            )

        users = await UserService.get_users_by_organization(organization_id, skip, limit)
    else:
        # Get users from user's accessible organizations
        user_orgs = await UserService.get_user_organizations(user_data["user_id"])
        if not user_orgs:
            return []

        # For simplicity, get users from first organization
        users = await UserService.get_users_by_organization(user_orgs[0], skip, limit)

    return [
        UserResponse(
            id=str(user.id),
            organization_id=user.organization_id,
            email=user.email,
            name=user.name,
            role=user.role,
            is_active=user.is_active,
            last_login=user.last_login,
            face_registration_count=user.face_registration_count,
            phone=user.phone,
            department=user.department,
            position=user.position,
            created_at=user.created_at.isoformat(),
            updated_at=user.updated_at.isoformat()
        )
        for user in users
    ]


@router.post("/", response_model=UserResponse)
async def create_user(
    user_data: UserCreate,
    token: str = Depends(security)
):
    """Create new user."""
    auth_user_data = get_user_from_token(token.credentials)
    if not auth_user_data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )

    # Check if current user has permission to create users in this organization
    has_permission = await UserService.check_user_permission(
        auth_user_data["user_id"],
        user_data.organization_id
    )
    if not has_permission:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to create users in this organization"
        )

    # Check if user with this email already exists
    existing_user = await UserService.get_user_by_email(user_data.email)
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User with this email already exists"
        )

    user = await UserService.create_user(user_data)

    return UserResponse(
        id=str(user.id),
        organization_id=user.organization_id,
        email=user.email,
        name=user.name,
        role=user.role,
        is_active=user.is_active,
        last_login=user.last_login,
        face_registration_count=user.face_registration_count,
        phone=user.phone,
        department=user.department,
        position=user.position,
        created_at=user.created_at.isoformat(),
        updated_at=user.updated_at.isoformat()
    )


@router.get("/{user_id}", response_model=UserResponse)
async def get_user(user_id: str, token: str = Depends(security)):
    """Get user by ID."""
    auth_user_data = get_user_from_token(token.credentials)
    if not auth_user_data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )

    user = await UserService.get_user(user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )

    # Check if current user has permission to view this user
    has_permission = await UserService.check_user_permission(
        auth_user_data["user_id"],
        user.organization_id
    )
    if not has_permission:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to view this user"
        )

    return UserResponse(
        id=str(user.id),
        organization_id=user.organization_id,
        email=user.email,
        name=user.name,
        role=user.role,
        is_active=user.is_active,
        last_login=user.last_login,
        face_registration_count=user.face_registration_count,
        phone=user.phone,
        department=user.department,
        position=user.position,
        created_at=user.created_at.isoformat(),
        updated_at=user.updated_at.isoformat()
    )


@router.put("/{user_id}", response_model=UserResponse)
async def update_user(
    user_id: str,
    user_data: UserUpdate,
    token: str = Depends(security)
):
    """Update user."""
    auth_user_data = get_user_from_token(token.credentials)
    if not auth_user_data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )

    user = await UserService.get_user(user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )

    # Check if current user has permission to update this user
    has_permission = await UserService.check_user_permission(
        auth_user_data["user_id"],
        user.organization_id
    )
    if not has_permission:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to update this user"
        )

    updated_user = await UserService.update_user(user_id, user_data)
    if not updated_user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )

    return UserResponse(
        id=str(updated_user.id),
        organization_id=updated_user.organization_id,
        email=updated_user.email,
        name=updated_user.name,
        role=updated_user.role,
        is_active=updated_user.is_active,
        last_login=updated_user.last_login,
        face_registration_count=updated_user.face_registration_count,
        phone=updated_user.phone,
        department=updated_user.department,
        position=updated_user.position,
        created_at=updated_user.created_at.isoformat(),
        updated_at=updated_user.updated_at.isoformat()
    )


@router.delete("/{user_id}")
async def delete_user(user_id: str, token: str = Depends(security)):
    """Delete user (soft delete)."""
    auth_user_data = get_user_from_token(token.credentials)
    if not auth_user_data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )

    user = await UserService.get_user(user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )

    # Check if current user has permission to delete this user
    has_permission = await UserService.check_user_permission(
        auth_user_data["user_id"],
        user.organization_id
    )
    if not has_permission:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to delete this user"
        )

    success = await UserService.delete_user(user_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )

    return {"message": "User deleted successfully"}


@router.post("/{user_id}/register-face")
async def register_face(
    user_id: str,
    image: UploadFile = File(...),
    token: str = Depends(security)
):
    """Register face for user."""
    auth_user_data = get_user_from_token(token.credentials)
    if not auth_user_data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )

    user = await UserService.get_user(user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )

    # Check if current user has permission to register face for this user
    has_permission = await UserService.check_user_permission(
        auth_user_data["user_id"],
        user.organization_id
    )
    if not has_permission:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to register face for this user"
        )

    # Read image data
    image_data = await image.read()

    # Validate image
    from services.face_recognition_service import FaceRecognitionService
    is_valid, reason = FaceRecognitionService.validate_face_image(image_data)
    if not is_valid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=reason
        )

    # Preprocess image
    processed_image = FaceRecognitionService.preprocess_image(image_data)

    # Extract face encodings
    face_encodings = FaceRecognitionService.extract_face_encodings(processed_image)

    if not face_encodings:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No face detected in image"
        )

    # Register the first (and should be only) face encoding
    face_encoding_bytes = face_encodings[0].tobytes()

    # Store image in GridFS (placeholder - would need actual GridFS implementation)
    image_id = f"face_image_{user_id}_{user.face_registration_count + 1}"

    # Register face with user
    success = await UserService.register_face(user_id, face_encoding_bytes, image_id)

    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Maximum number of face registrations reached"
        )

    return {
        "message": "Face registered successfully",
        "face_count": len(user.face_encodings) + 1,
        "image_id": image_id
    }


@router.get("/{user_id}/faces")
async def get_user_faces(user_id: str, token: str = Depends(security)):
    """Get user's registered faces."""
    auth_user_data = get_user_from_token(token.credentials)
    if not auth_user_data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )

    user = await UserService.get_user(user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )

    # Check if current user has permission to view this user's faces
    has_permission = await UserService.check_user_permission(
        auth_user_data["user_id"],
        user.organization_id
    )
    if not has_permission:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to view this user's faces"
        )

    return {
        "user_id": user_id,
        "face_count": len(user.face_encodings),
        "face_image_id": user.face_image_id
    }


@router.delete("/{user_id}/faces/{encoding_index}")
async def remove_face(
    user_id: str,
    encoding_index: int,
    token: str = Depends(security)
):
    """Remove a specific face encoding."""
    auth_user_data = get_user_from_token(token.credentials)
    if not auth_user_data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )

    user = await UserService.get_user(user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )

    # Check if current user has permission to manage this user's faces
    has_permission = await UserService.check_user_permission(
        auth_user_data["user_id"],
        user.organization_id
    )
    if not has_permission:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to manage this user's faces"
        )

    success = await UserService.remove_face(user_id, encoding_index)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid face encoding index"
        )

    return {"message": "Face encoding removed successfully"}

