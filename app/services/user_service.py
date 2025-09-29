"""
User service for business logic operations.
"""

from typing import List, Optional

from core.database import db
from models.user import User, UserCreate, UserUpdate, UserResponse
from models.base import UserRole


class UserService:
    """Service class for user operations."""

    @staticmethod
    async def create_user(user_data: UserCreate) -> User:
        """Create a new user."""
        user = User(
            **user_data.dict(),
            face_encodings=[],
            face_registration_count=0
        )
        await user.insert()
        return user

    @staticmethod
    async def get_user(user_id: str) -> Optional[User]:
        """Get user by ID."""
        return await User.get(user_id)

    @staticmethod
    async def get_user_by_email(email: str) -> Optional[User]:
        """Get user by email."""
        return await User.find_one({"email": email, "is_active": True})

    @staticmethod
    async def get_users_by_organization(
        organization_id: str,
        skip: int = 0,
        limit: int = 100
    ) -> List[User]:
        """Get all users in an organization."""
        return await User.find(
            {"organization_id": organization_id, "is_active": True}
        ).skip(skip).limit(limit).to_list()

    @staticmethod
    async def update_user(user_id: str, user_data: UserUpdate) -> Optional[User]:
        """Update user."""
        user = await User.get(user_id)
        if not user:
            return None

        update_data = user_data.dict(exclude_unset=True)
        for field, value in update_data.items():
            setattr(user, field, value)

        await user.save()
        return user

    @staticmethod
    async def delete_user(user_id: str) -> bool:
        """Delete user (soft delete by setting is_active=False)."""
        user = await User.get(user_id)
        if not user:
            return False

        user.is_active = False
        await user.save()
        return True

    @staticmethod
    async def register_face(user_id: str, face_encoding: bytes, image_id: str) -> bool:
        """Register face encoding for user."""
        user = await User.get(user_id)
        if not user:
            return False

        # Add new face encoding (limit to max registrations)
        if len(user.face_encodings) < 5:  # Configurable limit
            user.face_encodings.append(face_encoding)
            if not user.face_image_id:
                user.face_image_id = image_id
            user.face_registration_count += 1
            await user.save()
            return True

        return False

    @staticmethod
    async def remove_face(user_id: str, encoding_index: int) -> bool:
        """Remove a specific face encoding."""
        user = await User.get(user_id)
        if not user or encoding_index >= len(user.face_encodings):
            return False

        user.face_encodings.pop(encoding_index)
        user.face_registration_count -= 1

        # If no face encodings left, clear image reference
        if not user.face_encodings:
            user.face_image_id = None

        await user.save()
        return True

    @staticmethod
    async def update_last_login(user_id: str) -> bool:
        """Update user's last login timestamp."""
        from datetime import datetime

        user = await User.get(user_id)
        if not user:
            return False

        user.last_login = datetime.utcnow().isoformat()
        await user.save()
        return True

    @staticmethod
    async def get_user_organizations(user_id: str) -> List[str]:
        """Get organizations where user has access."""
        # Get user's primary organization
        user = await User.get(user_id)
        if not user:
            return []

        organizations = [user.organization_id]

        # If user is admin, get all organizations they administer
        if user.role in [UserRole.ORG_ADMIN, UserRole.SUPER_ADMIN]:
            admin_orgs = await User.find(
                {"role": {"$in": [UserRole.ORG_ADMIN.value, UserRole.SUPER_ADMIN.value]}},
                {"organization_id": 1}
            ).distinct("organization_id")

            for org_id in admin_orgs:
                if org_id not in organizations:
                    organizations.append(org_id)

        return organizations

    @staticmethod
    async def check_user_permission(
        user_id: str,
        target_organization_id: str,
        required_role: UserRole = UserRole.EMPLOYEE
    ) -> bool:
        """Check if user has required permission in organization."""
        user = await User.get(user_id)
        if not user or not user.is_active:
            return False

        # User can access their own organization
        if user.organization_id == target_organization_id:
            return True

        # Admin users can access any organization (in this simple implementation)
        if user.role in [UserRole.ORG_ADMIN, UserRole.SUPER_ADMIN]:
            return True

        return False
