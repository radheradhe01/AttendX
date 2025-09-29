"""
Location management router.
"""

from fastapi import APIRouter

router = APIRouter()


@router.get("/")
async def get_locations():
    """Get approved locations."""
    return {"message": "Get locations endpoint - TODO: Implement"}


@router.post("/")
async def create_location():
    """Create approved location."""
    return {"message": "Create location endpoint - TODO: Implement"}

