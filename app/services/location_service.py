"""
Location service for geolocation-based attendance validation.
"""

import math
from typing import List, Optional, Tuple

from app.models.location import ApprovedLocation, LocationCreate, LocationUpdate
from app.models.base import Location


class LocationService:
    """Service class for location operations."""

    @staticmethod
    def calculate_distance(
        lat1: float,
        lon1: float,
        lat2: float,
        lon2: float
    ) -> float:
        """
        Calculate distance between two GPS coordinates using Haversine formula.

        Args:
            lat1, lon1: First coordinate (latitude, longitude)
            lat2, lon2: Second coordinate (latitude, longitude)

        Returns:
            Distance in kilometers
        """
        # Earth's radius in kilometers
        R = 6371.0

        # Convert degrees to radians
        lat1_rad = math.radians(lat1)
        lon1_rad = math.radians(lon1)
        lat2_rad = math.radians(lat2)
        lon2_rad = math.radians(lon2)

        # Differences
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad

        # Haversine formula
        a = math.sin(dlat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        distance = R * c
        return distance

    @staticmethod
    def validate_location(
        user_location: Location,
        approved_locations: List[ApprovedLocation],
        max_radius_km: float = 0.5
    ) -> Tuple[bool, Optional[str], Optional[float]]:
        """
        Validate if user location is within approved locations.

        Args:
            user_location: User's current location
            approved_locations: List of approved locations
            max_radius_km: Maximum allowed radius from approved locations

        Returns:
            Tuple of (is_valid, approved_location_id, distance_km)
        """
        if not approved_locations:
            return True, None, 0.0  # No location restrictions

        user_lat = user_location.latitude
        user_lon = user_location.longitude

        min_distance = float('inf')
        closest_location_id = None

        for location in approved_locations:
            if not location.is_active:
                continue

            distance = LocationService.calculate_distance(
                user_lat, user_lon,
                location.latitude, location.longitude
            )

            if distance < min_distance:
                min_distance = distance
                closest_location_id = str(location.id)

        # Check if within allowed radius
        if min_distance <= max_radius_km:
            return True, closest_location_id, min_distance

        return False, closest_location_id, min_distance

    @staticmethod
    async def create_approved_location(
        org_id: str,
        location_data: LocationCreate,
        created_by: str
    ) -> ApprovedLocation:
        """Create a new approved location."""
        from core.database import db

        location = ApprovedLocation(
            organization_id=org_id,
            **location_data.dict(),
            created_by=created_by
        )
        await location.insert()
        return location

    @staticmethod
    async def get_approved_locations(organization_id: str) -> List[ApprovedLocation]:
        """Get all approved locations for an organization."""
        return await ApprovedLocation.find(
            {"organization_id": organization_id, "is_active": True}
        ).to_list()

    @staticmethod
    async def update_approved_location(
        location_id: str,
        location_data: LocationUpdate
    ) -> Optional[ApprovedLocation]:
        """Update approved location."""
        location = await ApprovedLocation.get(location_id)
        if not location:
            return None

        update_data = location_data.dict(exclude_unset=True)
        for field, value in update_data.items():
            setattr(location, field, value)

        await location.save()
        return location

    @staticmethod
    async def delete_approved_location(location_id: str) -> bool:
        """Delete approved location (soft delete)."""
        location = await ApprovedLocation.get(location_id)
        if not location:
            return False

        location.is_active = False
        await location.save()
        return True

    @staticmethod
    def get_location_address(latitude: float, longitude: float) -> str:
        """
        Get approximate address from coordinates.
        Note: This is a placeholder - in production, you'd use a geocoding service.
        """
        # Placeholder implementation
        return f"Location at {latitude:.4f}, {longitude:.4f}"

    @staticmethod
    def validate_coordinates(latitude: float, longitude: float) -> bool:
        """
        Validate GPS coordinates.

        Args:
            latitude: Latitude (-90 to 90)
            longitude: Longitude (-180 to 180)

        Returns:
            True if coordinates are valid
        """
        return -90 <= latitude <= 90 and -180 <= longitude <= 180
