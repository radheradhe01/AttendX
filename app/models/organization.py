"""
Organization model definitions.
"""

from typing import List, Optional

from beanie import Indexed
from pydantic import BaseModel, Field

from models.base import BaseDocument, OrganizationSettings


class Organization(BaseDocument):
    """Organization document model."""

    name: Indexed(str) = Field(..., min_length=1, max_length=100)
    domain: Optional[str] = Field(None, min_length=1, max_length=100)
    description: Optional[str] = Field(None, max_length=500)
    settings: OrganizationSettings = Field(default_factory=OrganizationSettings)
    is_active: bool = Field(default=True)
    admin_ids: List[str] = Field(default_factory=list)

    class Settings:
        name = "organizations"


class OrganizationCreate(BaseModel):
    """Organization creation model."""

    name: str = Field(..., min_length=1, max_length=100)
    domain: Optional[str] = Field(None, min_length=1, max_length=100)
    description: Optional[str] = Field(None, max_length=500)
    settings: Optional[OrganizationSettings] = None


class OrganizationUpdate(BaseModel):
    """Organization update model."""

    name: Optional[str] = Field(None, min_length=1, max_length=100)
    domain: Optional[str] = Field(None, min_length=1, max_length=100)
    description: Optional[str] = Field(None, max_length=500)
    settings: Optional[OrganizationSettings] = None
    is_active: Optional[bool] = None


class OrganizationResponse(BaseModel):
    """Organization response model."""

    id: str
    name: str
    domain: Optional[str]
    description: Optional[str]
    settings: OrganizationSettings
    is_active: bool
    created_at: str
    updated_at: str

