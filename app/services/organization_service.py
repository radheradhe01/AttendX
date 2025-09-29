"""
Organization service for business logic operations.
"""

from typing import List, Optional

from core.database import db
from models.organization import (
    Organization,
    OrganizationCreate,
    OrganizationUpdate,
    OrganizationResponse
)
from models.base import UserRole


class OrganizationService:
    """Service class for organization operations."""

    @staticmethod
    async def create_organization(org_data: OrganizationCreate, created_by: str) -> Organization:
        """Create a new organization."""
        organization = Organization(
            **org_data.dict(),
            admin_ids=[created_by]
        )
        await organization.insert()
        return organization

    @staticmethod
    async def get_organization(org_id: str) -> Optional[Organization]:
        """Get organization by ID."""
        return await Organization.get(org_id)

    @staticmethod
    async def get_organizations(skip: int = 0, limit: int = 100) -> List[Organization]:
        """Get all organizations with pagination."""
        return await Organization.find().skip(skip).limit(limit).to_list()

    @staticmethod
    async def update_organization(org_id: str, org_data: OrganizationUpdate) -> Optional[Organization]:
        """Update organization."""
        organization = await Organization.get(org_id)
        if not organization:
            return None

        update_data = org_data.dict(exclude_unset=True)
        for field, value in update_data.items():
            setattr(organization, field, value)

        await organization.save()
        return organization

    @staticmethod
    async def delete_organization(org_id: str) -> bool:
        """Delete organization (soft delete by setting is_active=False)."""
        organization = await Organization.get(org_id)
        if not organization:
            return False

        organization.is_active = False
        await organization.save()
        return True

    @staticmethod
    async def add_admin(org_id: str, user_id: str) -> bool:
        """Add admin to organization."""
        organization = await Organization.get(org_id)
        if not organization:
            return False

        if user_id not in organization.admin_ids:
            organization.admin_ids.append(user_id)
            await organization.save()

        return True

    @staticmethod
    async def remove_admin(org_id: str, user_id: str) -> bool:
        """Remove admin from organization."""
        organization = await Organization.get(org_id)
        if not organization:
            return False

        if user_id in organization.admin_ids:
            organization.admin_ids.remove(user_id)
            await organization.save()

        return True

    @staticmethod
    async def get_user_organizations(user_id: str) -> List[Organization]:
        """Get organizations where user is an admin."""
        return await Organization.find(
            {"admin_ids": user_id, "is_active": True}
        ).to_list()

    @staticmethod
    async def check_user_permission(org_id: str, user_id: str, required_role: UserRole) -> bool:
        """Check if user has required permission in organization."""
        organization = await Organization.get(org_id)
        if not organization or not organization.is_active:
            return False

        # Super admin has access to all organizations
        if required_role == UserRole.SUPER_ADMIN:
            return True

        # Check if user is admin of this organization
        return user_id in organization.admin_ids

