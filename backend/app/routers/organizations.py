"""
Organization management router
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List, Optional

from app.database import get_db
from app.schemas.user import Organization, OrganizationCreate, OrganizationUpdate
from app.models.user import Organization as OrganizationModel, User
from app.routers.auth import get_current_user
from loguru import logger

router = APIRouter()


@router.get("/", response_model=List[Organization])
async def get_organizations(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get organizations (admin can see all, others see only their org)"""
    try:
        if current_user.role == "admin":
            # Admin can see all organizations
            organizations = db.query(OrganizationModel).all()
        else:
            # Others can only see their organization
            organizations = db.query(OrganizationModel).filter(
                OrganizationModel.id == current_user.organization_id
            ).all()
        
        return organizations
        
    except Exception as e:
        logger.error(f"Error getting organizations: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get organizations"
        )


@router.get("/{organization_id}", response_model=Organization)
async def get_organization(
    organization_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get organization by ID"""
    try:
        organization = db.query(OrganizationModel).filter(
            OrganizationModel.id == organization_id
        ).first()
        
        if not organization:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Organization not found"
            )
        
        # Check permissions
        if current_user.role != "admin" and organization.id != current_user.organization_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions"
            )
        
        return organization
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting organization: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get organization"
        )


@router.post("/", response_model=Organization)
async def create_organization(
    organization_data: OrganizationCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Create new organization (admin only)"""
    try:
        # Check permissions
        if current_user.role != "admin":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only admins can create organizations"
            )
        
        # Check if organization name already exists
        existing_org = db.query(OrganizationModel).filter(
            OrganizationModel.name == organization_data.name
        ).first()
        
        if existing_org:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Organization with this name already exists"
            )
        
        # Create new organization
        new_organization = OrganizationModel(
            name=organization_data.name,
            description=organization_data.description,
            address=organization_data.address,
            latitude=organization_data.latitude,
            longitude=organization_data.longitude,
            radius=organization_data.radius
        )
        
        db.add(new_organization)
        db.commit()
        db.refresh(new_organization)
        
        logger.info(f"Organization created: {new_organization.name}")
        
        return new_organization
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating organization: {e}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create organization"
        )


@router.put("/{organization_id}", response_model=Organization)
async def update_organization(
    organization_id: int,
    organization_data: OrganizationUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Update organization (admin only)"""
    try:
        # Check permissions
        if current_user.role != "admin":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only admins can update organizations"
            )
        
        organization = db.query(OrganizationModel).filter(
            OrganizationModel.id == organization_id
        ).first()
        
        if not organization:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Organization not found"
            )
        
        # Update organization fields
        if organization_data.name is not None:
            # Check if name is already taken by another organization
            existing_org = db.query(OrganizationModel).filter(
                OrganizationModel.name == organization_data.name,
                OrganizationModel.id != organization_id
            ).first()
            
            if existing_org:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Organization with this name already exists"
                )
            
            organization.name = organization_data.name
        
        if organization_data.description is not None:
            organization.description = organization_data.description
        
        if organization_data.address is not None:
            organization.address = organization_data.address
        
        if organization_data.latitude is not None:
            organization.latitude = organization_data.latitude
        
        if organization_data.longitude is not None:
            organization.longitude = organization_data.longitude
        
        if organization_data.radius is not None:
            organization.radius = organization_data.radius
        
        if organization_data.is_active is not None:
            organization.is_active = organization_data.is_active
        
        db.commit()
        db.refresh(organization)
        
        logger.info(f"Organization updated: {organization.name}")
        
        return organization
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating organization: {e}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update organization"
        )


@router.delete("/{organization_id}")
async def delete_organization(
    organization_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Delete organization (admin only)"""
    try:
        # Check permissions
        if current_user.role != "admin":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only admins can delete organizations"
            )
        
        organization = db.query(OrganizationModel).filter(
            OrganizationModel.id == organization_id
        ).first()
        
        if not organization:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Organization not found"
            )
        
        # Check if organization has users
        user_count = db.query(User).filter(
            User.organization_id == organization_id
        ).count()
        
        if user_count > 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot delete organization with existing users"
            )
        
        # Delete organization
        db.delete(organization)
        db.commit()
        
        logger.info(f"Organization deleted: {organization.name}")
        
        return {"message": "Organization deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting organization: {e}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete organization"
        )


@router.get("/{organization_id}/users", response_model=List[User])
async def get_organization_users(
    organization_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get users in organization"""
    try:
        # Check permissions
        if current_user.role not in ["admin", "manager"]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions"
            )
        
        # Check if organization exists
        organization = db.query(OrganizationModel).filter(
            OrganizationModel.id == organization_id
        ).first()
        
        if not organization:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Organization not found"
            )
        
        # Managers can only see users from their organization
        if current_user.role == "manager" and organization_id != current_user.organization_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions"
            )
        
        users = db.query(User).filter(
            User.organization_id == organization_id
        ).all()
        
        return users
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting organization users: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get organization users"
        )
