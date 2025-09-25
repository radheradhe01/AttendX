"""
Location service for geolocation validation and processing
"""

import math
from typing import Tuple, Optional, Dict
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import requests
from loguru import logger


class LocationService:
    """Service for location-related operations"""
    
    def __init__(self):
        self.geolocator = Nominatim(user_agent="attendx_app")
        
    def calculate_distance(
        self, 
        lat1: float, 
        lon1: float, 
        lat2: float, 
        lon2: float
    ) -> float:
        """
        Calculate distance between two coordinates in meters
        
        Args:
            lat1, lon1: First coordinate
            lat2, lon2: Second coordinate
            
        Returns:
            Distance in meters
        """
        try:
            point1 = (lat1, lon1)
            point2 = (lat2, lon2)
            distance = geodesic(point1, point2).meters
            return distance
        except Exception as e:
            logger.error(f"Error calculating distance: {e}")
            return float('inf')
    
    def is_within_radius(
        self, 
        user_lat: float, 
        user_lon: float, 
        org_lat: float, 
        org_lon: float, 
        radius: float
    ) -> bool:
        """
        Check if user location is within organization's allowed radius
        
        Args:
            user_lat, user_lon: User's coordinates
            org_lat, org_lon: Organization's coordinates
            radius: Allowed radius in meters
            
        Returns:
            True if within radius, False otherwise
        """
        distance = self.calculate_distance(user_lat, user_lon, org_lat, org_lon)
        return distance <= radius
    
    def get_address_from_coordinates(
        self, 
        latitude: float, 
        longitude: float
    ) -> Optional[str]:
        """
        Get human-readable address from coordinates
        
        Args:
            latitude: Latitude coordinate
            longitude: Longitude coordinate
            
        Returns:
            Address string or None if not found
        """
        try:
            location = self.geolocator.reverse(f"{latitude}, {longitude}")
            if location:
                return location.address
            return None
        except Exception as e:
            logger.error(f"Error getting address from coordinates: {e}")
            return None
    
    def validate_location(
        self, 
        user_lat: float, 
        user_lon: float, 
        org_lat: float, 
        org_lon: float, 
        org_radius: float,
        accuracy: Optional[float] = None
    ) -> Dict:
        """
        Validate user location against organization requirements
        
        Args:
            user_lat, user_lon: User's coordinates
            org_lat, org_lon: Organization's coordinates
            org_radius: Organization's allowed radius
            accuracy: GPS accuracy in meters
            
        Returns:
            Dictionary with validation results
        """
        try:
            # Calculate distance
            distance = self.calculate_distance(user_lat, user_lon, org_lat, org_lon)
            
            # Check if within radius
            is_within_radius = distance <= org_radius
            
            # Check GPS accuracy (if provided)
            accuracy_valid = True
            if accuracy is not None and accuracy > 100:  # More than 100m accuracy is questionable
                accuracy_valid = False
                logger.warning(f"GPS accuracy is low: {accuracy}m")
            
            # Get address
            address = self.get_address_from_coordinates(user_lat, user_lon)
            
            return {
                "is_valid": is_within_radius and accuracy_valid,
                "distance": distance,
                "is_within_radius": is_within_radius,
                "accuracy_valid": accuracy_valid,
                "address": address,
                "org_radius": org_radius
            }
            
        except Exception as e:
            logger.error(f"Error validating location: {e}")
            return {
                "is_valid": False,
                "error": str(e)
            }
    
    def get_location_info(self, latitude: float, longitude: float) -> Dict:
        """
        Get comprehensive location information
        
        Args:
            latitude: Latitude coordinate
            longitude: Longitude coordinate
            
        Returns:
            Dictionary with location information
        """
        try:
            # Get address
            address = self.get_address_from_coordinates(latitude, longitude)
            
            # Get timezone (using a simple approximation)
            timezone = self._get_timezone_from_coordinates(latitude, longitude)
            
            return {
                "latitude": latitude,
                "longitude": longitude,
                "address": address,
                "timezone": timezone
            }
            
        except Exception as e:
            logger.error(f"Error getting location info: {e}")
            return {
                "latitude": latitude,
                "longitude": longitude,
                "error": str(e)
            }
    
    def _get_timezone_from_coordinates(self, lat: float, lon: float) -> str:
        """
        Get timezone from coordinates (simplified)
        
        Args:
            lat: Latitude
            lon: Longitude
            
        Returns:
            Timezone string
        """
        try:
            # This is a simplified timezone calculation
            # In production, you might want to use a more accurate service
            if -180 <= lon < -120:
                return "America/Los_Angeles"
            elif -120 <= lon < -60:
                return "America/New_York"
            elif -60 <= lon < 0:
                return "Europe/London"
            elif 0 <= lon < 60:
                return "Europe/Paris"
            elif 60 <= lon < 120:
                return "Asia/Shanghai"
            elif 120 <= lon < 180:
                return "Asia/Tokyo"
            else:
                return "UTC"
        except Exception:
            return "UTC"
    
    def calculate_work_hours(
        self, 
        check_in_lat: float, 
        check_in_lon: float, 
        check_out_lat: float, 
        check_out_lon: float
    ) -> Dict:
        """
        Calculate work hours and location consistency
        
        Args:
            check_in_lat, check_in_lon: Check-in coordinates
            check_out_lat, check_out_lon: Check-out coordinates
            
        Returns:
            Dictionary with work hours analysis
        """
        try:
            # Calculate distance between check-in and check-out locations
            location_distance = self.calculate_distance(
                check_in_lat, check_in_lon, 
                check_out_lat, check_out_lon
            )
            
            # Determine if locations are consistent (within reasonable distance)
            is_location_consistent = location_distance <= 1000  # Within 1km
            
            return {
                "location_distance": location_distance,
                "is_location_consistent": is_location_consistent,
                "check_in_address": self.get_address_from_coordinates(check_in_lat, check_in_lon),
                "check_out_address": self.get_address_from_coordinates(check_out_lat, check_out_lon)
            }
            
        except Exception as e:
            logger.error(f"Error calculating work hours: {e}")
            return {
                "error": str(e)
            }
