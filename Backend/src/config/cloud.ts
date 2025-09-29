/**
 * Cloud and AI service configurations
 * Handles external API configurations for face recognition, GPS, etc.
 */

export interface CloudConfig {
  faceRecognition: {
    apiUrl: string;
    apiKey: string;
    timeout: number;
  };
  gps: {
    campusLatitude: number;
    campusLongitude: number;
    radiusMeters: number;
  };
  email: {
    smtpHost: string;
    smtpPort: number;
    smtpUser: string;
    smtpPass: string;
    fromEmail: string;
    fromName: string;
  };
  googleMaps: {
    apiKey: string;
  };
}

export const cloudConfig: CloudConfig = {
  faceRecognition: {
    apiUrl: process.env.FACE_RECOGNITION_API_URL || 'http://localhost:8000/api/face-recognition',
    apiKey: process.env.FACE_RECOGNITION_API_KEY || '',
    timeout: 10000, // 10 seconds timeout
  },
  gps: {
    campusLatitude: parseFloat(process.env.CAMPUS_LATITUDE || '12.9716'),
    campusLongitude: parseFloat(process.env.CAMPUS_LONGITUDE || '77.5946'),
    radiusMeters: parseInt(process.env.CAMPUS_RADIUS_METERS || '100'),
  },
  email: {
    smtpHost: process.env.SMTP_HOST || 'smtp.gmail.com',
    smtpPort: parseInt(process.env.SMTP_PORT || '587'),
    smtpUser: process.env.SMTP_USER || '',
    smtpPass: process.env.SMTP_PASS || '',
    fromEmail: process.env.FROM_EMAIL || 'noreply@attendex.com',
    fromName: process.env.FROM_NAME || 'AttendEase System',
  },
  googleMaps: {
    apiKey: process.env.GOOGLE_MAPS_API_KEY || '',
  },
};

// Validation function to check if all required configurations are present
export const validateCloudConfig = (): { isValid: boolean; missingFields: string[] } => {
  const missingFields: string[] = [];

  if (!cloudConfig.faceRecognition.apiKey) {
    missingFields.push('FACE_RECOGNITION_API_KEY');
  }

  if (!cloudConfig.email.smtpUser) {
    missingFields.push('SMTP_USER');
  }

  if (!cloudConfig.email.smtpPass) {
    missingFields.push('SMTP_PASS');
  }

  if (!cloudConfig.googleMaps.apiKey) {
    missingFields.push('GOOGLE_MAPS_API_KEY');
  }

  return {
    isValid: missingFields.length === 0,
    missingFields,
  };
};
