"""
Location model definitions for approved attendance locations.
"""

from typing import Optional

from beanie import Indexed
from pydantic import BaseModel, Field

from .base import BaseDocument


class ApprovedLocation(BaseDocument):
    """Approved location document model."""

    organization_id: Indexed(str)
    name: str = Field(..., min_length=1, max_length=100)
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    radius_km: float = Field(default=0.1, ge=0.01, le=10)
    address: Optional[str] = Field(None, max_length=200)
    description: Optional[str] = Field(None, max_length=200)
    is_active: bool = Field(default=True)
    created_by: str  # User ID who created this location

    class Settings:
        name = "approved_locations"


class LocationCreate(BaseModel):
    """Location creation model."""

    name: str = Field(..., min_length=1, max_length=100)
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    radius_km: float = Field(default=0.1, ge=0.01, le=10)
    address: Optional[str] = Field(None, max_length=200)
    description: Optional[str] = Field(None, max_length=200)


class LocationUpdate(BaseModel):
    """Location update model."""

    name: Optional[str] = Field(None, min_length=1, max_length=100)
    latitude: Optional[float] = Field(None, ge=-90, le=90)
    longitude: Optional[float] = Field(None, ge=-180, le=180)
    radius_km: Optional[float] = Field(None, ge=0.01, le=10)
    address: Optional[str] = Field(None, max_length=200)
    description: Optional[str] = Field(None, max_length=200)
    is_active: Optional[bool] = None


class LocationResponse(BaseModel):
    """Location response model."""

    id: str
    organization_id: str
    name: str
    latitude: float
    longitude: float
    radius_km: float
    address: Optional[str]
    description: Optional[str]
    is_active: bool
    created_by: str
    created_at: str
    updated_at: str


class LocationDistance(BaseModel):
    """Location with distance calculation."""

    location: LocationResponse
    distance_km: float
    within_radius: bool

