"""
Organization management router.
"""

from typing import List

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBearer

from core.security import get_user_from_token
from models.organization import (
    OrganizationCreate,
    OrganizationUpdate,
    OrganizationResponse
)
from services.organization_service import OrganizationService

router = APIRouter()
security = HTTPBearer()


@router.get("/", response_model=List[OrganizationResponse])
async def get_organizations(
    skip: int = 0,
    limit: int = 100,
    token: str = Depends(security)
):
    """Get all organizations."""
    user_data = get_user_from_token(token.credentials)
    if not user_data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )

    organizations = await OrganizationService.get_organizations(skip, limit)
    return [
        OrganizationResponse(
            id=str(org.id),
            name=org.name,
            domain=org.domain,
            description=org.description,
            settings=org.settings,
            is_active=org.is_active,
            created_at=org.created_at.isoformat(),
            updated_at=org.updated_at.isoformat()
        )
        for org in organizations
    ]


@router.post("/", response_model=OrganizationResponse)
async def create_organization(
    org_data: OrganizationCreate,
    token: str = Depends(security)
):
    """Create new organization."""
    user_data = get_user_from_token(token.credentials)
    if not user_data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )

    organization = await OrganizationService.create_organization(
        org_data,
        user_data["user_id"]
    )

    return OrganizationResponse(
        id=str(organization.id),
        name=organization.name,
        domain=organization.domain,
        description=organization.description,
        settings=organization.settings,
        is_active=organization.is_active,
        created_at=organization.created_at.isoformat(),
        updated_at=organization.updated_at.isoformat()
    )


@router.get("/{org_id}", response_model=OrganizationResponse)
async def get_organization(org_id: str, token: str = Depends(security)):
    """Get organization by ID."""
    user_data = get_user_from_token(token.credentials)
    if not user_data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )

    # Check if user has permission to view this organization
    has_permission = await OrganizationService.check_user_permission(
        org_id,
        user_data["user_id"],
        user_data["role"]
    )

    if not has_permission:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to view this organization"
        )

    organization = await OrganizationService.get_organization(org_id)
    if not organization:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Organization not found"
        )

    return OrganizationResponse(
        id=str(organization.id),
        name=organization.name,
        domain=organization.domain,
        description=organization.description,
        settings=organization.settings,
        is_active=organization.is_active,
        created_at=organization.created_at.isoformat(),
        updated_at=organization.updated_at.isoformat()
    )


@router.put("/{org_id}", response_model=OrganizationResponse)
async def update_organization(
    org_id: str,
    org_data: OrganizationUpdate,
    token: str = Depends(security)
):
    """Update organization."""
    user_data = get_user_from_token(token.credentials)
    if not user_data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )

    # Check if user has permission to update this organization
    has_permission = await OrganizationService.check_user_permission(
        org_id,
        user_data["user_id"],
        user_data["role"]
    )

    if not has_permission:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to update this organization"
        )

    organization = await OrganizationService.update_organization(org_id, org_data)
    if not organization:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Organization not found"
        )

    return OrganizationResponse(
        id=str(organization.id),
        name=organization.name,
        domain=organization.domain,
        description=organization.description,
        settings=organization.settings,
        is_active=organization.is_active,
        created_at=organization.created_at.isoformat(),
        updated_at=organization.updated_at.isoformat()
    )


@router.delete("/{org_id}")
async def delete_organization(org_id: str, token: str = Depends(security)):
    """Delete organization (soft delete)."""
    user_data = get_user_from_token(token.credentials)
    if not user_data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )

    # Check if user has permission to delete this organization
    has_permission = await OrganizationService.check_user_permission(
        org_id,
        user_data["user_id"],
        user_data["role"]
    )

    if not has_permission:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to delete this organization"
        )

    success = await OrganizationService.delete_organization(org_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Organization not found"
        )

    return {"message": "Organization deleted successfully"}


@router.post("/{org_id}/admins/{user_id}")
async def add_admin(org_id: str, user_id: str, token: str = Depends(security)):
    """Add admin to organization."""
    user_data = get_user_from_token(token.credentials)
    if not user_data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )

    # Check if current user has permission to manage admins
    has_permission = await OrganizationService.check_user_permission(
        org_id,
        user_data["user_id"],
        user_data["role"]
    )

    if not has_permission:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to manage organization admins"
        )

    success = await OrganizationService.add_admin(org_id, user_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Organization not found"
        )

    return {"message": f"User {user_id} added as admin"}


@router.delete("/{org_id}/admins/{user_id}")
async def remove_admin(org_id: str, user_id: str, token: str = Depends(security)):
    """Remove admin from organization."""
    user_data = get_user_from_token(token.credentials)
    if not user_data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )

    # Check if current user has permission to manage admins
    has_permission = await OrganizationService.check_user_permission(
        org_id,
        user_data["user_id"],
        user_data["role"]
    )

    if not has_permission:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to manage organization admins"
        )

    success = await OrganizationService.remove_admin(org_id, user_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Organization not found"
        )

    return {"message": f"User {user_id} removed as admin"}
