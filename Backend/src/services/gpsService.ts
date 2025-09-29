/**
 * GPS Service
 * Handles GPS coordinate validation and campus geofencing
 */

import { cloudConfig } from '../config/cloud';
import { logger } from '../config/logger';
import { CustomError } from '../middleware/errorHandler';

export interface GPSLocation {
  latitude: number;
  longitude: number;
  accuracy: number; // in meters
}

export interface CampusLocation {
  latitude: number;
  longitude: number;
  radius: number; // in meters
}

export interface GPSValidationResult {
  isWithinCampus: boolean;
  distance: number; // distance from campus center in meters
  accuracy: number;
  isValid: boolean;
}

/**
 * GPS validation service class
 */
export class GPSService {
  private campusLocation: CampusLocation;

  constructor() {
    this.campusLocation = {
      latitude: cloudConfig.gps.campusLatitude,
      longitude: cloudConfig.gps.campusLongitude,
      radius: cloudConfig.gps.radiusMeters,
    };
  }

  /**
   * Calculate distance between two GPS coordinates using Haversine formula
   */
  private calculateDistance(
    lat1: number,
    lon1: number,
    lat2: number,
    lon2: number
  ): number {
    const R = 6371000; // Earth's radius in meters
    const dLat = this.toRadians(lat2 - lat1);
    const dLon = this.toRadians(lon2 - lon1);
    
    const a = 
      Math.sin(dLat / 2) * Math.sin(dLat / 2) +
      Math.cos(this.toRadians(lat1)) * Math.cos(this.toRadians(lat2)) *
      Math.sin(dLon / 2) * Math.sin(dLon / 2);
    
    const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
    const distance = R * c;
    
    return distance;
  }

  /**
   * Convert degrees to radians
   */
  private toRadians(degrees: number): number {
    return degrees * (Math.PI / 180);
  }

  /**
   * Validate GPS coordinates
   */
  validateGPSLocation(location: GPSLocation): GPSValidationResult {
    try {
      logger.info('Validating GPS location:', { location });

      // Validate coordinate ranges
      if (location.latitude < -90 || location.latitude > 90) {
        throw new CustomError('Invalid latitude. Must be between -90 and 90.', 400, 'INVALID_LATITUDE');
      }

      if (location.longitude < -180 || location.longitude > 180) {
        throw new CustomError('Invalid longitude. Must be between -180 and 180.', 400, 'INVALID_LONGITUDE');
      }

      // Validate accuracy
      if (location.accuracy < 0) {
        throw new CustomError('GPS accuracy cannot be negative.', 400, 'INVALID_ACCURACY');
      }

      // Calculate distance from campus center
      const distance = this.calculateDistance(
        location.latitude,
        location.longitude,
        this.campusLocation.latitude,
        this.campusLocation.longitude
      );

      // Check if location is within campus radius
      const isWithinCampus = distance <= this.campusLocation.radius;

      // Consider accuracy in validation
      // If GPS accuracy is poor, we might allow slightly larger radius
      const adjustedRadius = this.campusLocation.radius + (location.accuracy * 0.5);
      const isWithinAdjustedCampus = distance <= adjustedRadius;

      const result: GPSValidationResult = {
        isWithinCampus: isWithinCampus || isWithinAdjustedCampus,
        distance,
        accuracy: location.accuracy,
        isValid: true,
      };

      logger.info('GPS validation result:', {
        location,
        result,
        campusLocation: this.campusLocation,
      });

      return result;

    } catch (error) {
      logger.error('GPS validation failed:', {
        location,
        error: error instanceof Error ? error.message : 'Unknown error',
      });

      if (error instanceof CustomError) {
        throw error;
      }

      throw new CustomError('GPS validation failed', 500, 'GPS_VALIDATION_ERROR');
    }
  }

  /**
   * Check if location is within campus boundaries
   */
  isWithinCampus(latitude: number, longitude: number, accuracy: number = 0): boolean {
    const location: GPSLocation = { latitude, longitude, accuracy };
    const result = this.validateGPSLocation(location);
    return result.isWithinCampus;
  }

  /**
   * Get campus location information
   */
  getCampusLocation(): CampusLocation {
    return { ...this.campusLocation };
  }

  /**
   * Update campus location (admin only)
   */
  updateCampusLocation(latitude: number, longitude: number, radius: number): void {
    // Validate new coordinates
    if (latitude < -90 || latitude > 90) {
      throw new CustomError('Invalid latitude. Must be between -90 and 90.', 400, 'INVALID_LATITUDE');
    }

    if (longitude < -180 || longitude > 180) {
      throw new CustomError('Invalid longitude. Must be between -180 and 180.', 400, 'INVALID_LONGITUDE');
    }

    if (radius <= 0) {
      throw new CustomError('Campus radius must be positive.', 400, 'INVALID_RADIUS');
    }

    this.campusLocation = { latitude, longitude, radius };

    logger.info('Campus location updated:', this.campusLocation);
  }

  /**
   * Get distance from campus center
   */
  getDistanceFromCampus(latitude: number, longitude: number): number {
    return this.calculateDistance(
      latitude,
      longitude,
      this.campusLocation.latitude,
      this.campusLocation.longitude
    );
  }

  /**
   * Validate multiple locations at once
   */
  validateMultipleLocations(locations: GPSLocation[]): GPSValidationResult[] {
    return locations.map(location => this.validateGPSLocation(location));
  }

  /**
   * Get campus boundary information for frontend
   */
  getCampusBoundaryInfo(): {
    center: { latitude: number; longitude: number };
    radius: number;
    bounds: {
      north: number;
      south: number;
      east: number;
      west: number;
    };
  } {
    const { latitude, longitude, radius } = this.campusLocation;
    
    // Calculate approximate bounds (rough calculation)
    const latOffset = radius / 111000; // Rough conversion: 1 degree ≈ 111km
    const lonOffset = radius / (111000 * Math.cos(this.toRadians(latitude)));

    return {
      center: { latitude, longitude },
      radius,
      bounds: {
        north: latitude + latOffset,
        south: latitude - latOffset,
        east: longitude + lonOffset,
        west: longitude - lonOffset,
      },
    };
  }
}

// Export singleton instance
export const gpsService = new GPSService();
