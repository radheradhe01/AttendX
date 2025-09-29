import axios, { AxiosResponse } from 'axios';
import { cloudConfig } from '../config/cloud';
import { logger } from '../config/logger';
import { CustomError } from '../middleware/errorHandler';

/**
 * Face Recognition Service
 * Handles communication with external face recognition API
 */

export interface FaceRecognitionRequest {
  image: string; // Base64 encoded image
  studentId: string;
  courseId: string;
}

export interface FaceRecognitionResponse {
  isVerified: boolean;
  confidence: number;
  studentId: string;
  processingTime: number;
  error?: string;
}

export interface FaceRecognitionResult {
  isVerified: boolean;
  confidence: number;
  processingTime: number;
  error?: string;
}

/**
 * Face recognition service class
 */
export class FaceRecognitionService {
  private apiUrl: string;
  private apiKey: string;
  private timeout: number;

  constructor() {
    this.apiUrl = cloudConfig.faceRecognition.apiUrl;
    this.apiKey = cloudConfig.faceRecognition.apiKey;
    this.timeout = cloudConfig.faceRecognition.timeout;
  }

  /**
   * Verify face against student's registered face
   */
  async verifyFace(
    imageBase64: string,
    studentId: string,
    courseId: string
  ): Promise<FaceRecognitionResult> {
    const startTime = Date.now();

    try {
      logger.info('Starting face recognition for student:', { studentId, courseId });

      // Validate input
      if (!imageBase64 || !studentId || !courseId) {
        throw new CustomError('Missing required parameters for face recognition', 400, 'MISSING_PARAMETERS');
      }

      // Prepare request payload
      const requestData: FaceRecognitionRequest = {
        image: imageBase64,
        studentId,
        courseId,
      };

      // Make API call to face recognition service
      const response: AxiosResponse<FaceRecognitionResponse> = await axios.post(
        this.apiUrl,
        requestData,
        {
          headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${this.apiKey}`,
            'X-API-Key': this.apiKey,
          },
          timeout: this.timeout,
        }
      );

      const processingTime = Date.now() - startTime;

      // Validate response
      if (!response.data) {
        throw new CustomError('Invalid response from face recognition service', 500, 'INVALID_RESPONSE');
      }

      const result: FaceRecognitionResult = {
        isVerified: response.data.isVerified,
        confidence: response.data.confidence,
        processingTime,
        error: response.data.error,
      };

      logger.info('Face recognition completed:', {
        studentId,
        courseId,
        isVerified: result.isVerified,
        confidence: result.confidence,
        processingTime: result.processingTime,
      });

      return result;

    } catch (error) {
      const processingTime = Date.now() - startTime;
      
      logger.error('Face recognition failed:', {
        studentId,
        courseId,
        error: error instanceof Error ? error.message : 'Unknown error',
        processingTime,
      });

      // Handle different types of errors
      if (axios.isAxiosError(error)) {
        if (error.response) {
          // Server responded with error status
          const status = error.response.status;
          const message = error.response.data?.message || 'Face recognition service error';
          
          if (status === 401) {
            throw new CustomError('Face recognition service authentication failed', 401, 'AUTH_FAILED');
          } else if (status === 429) {
            throw new CustomError('Face recognition service rate limit exceeded', 429, 'RATE_LIMIT_EXCEEDED');
          } else if (status >= 500) {
            throw new CustomError('Face recognition service unavailable', 503, 'SERVICE_UNAVAILABLE');
          } else {
            throw new CustomError(message, status, 'FACE_RECOGNITION_ERROR');
          }
        } else if (error.request) {
          // Request was made but no response received
          throw new CustomError('Face recognition service timeout', 504, 'SERVICE_TIMEOUT');
        }
      }

      // Re-throw custom errors
      if (error instanceof CustomError) {
        throw error;
      }

      // Handle unexpected errors
      throw new CustomError('Face recognition failed', 500, 'FACE_RECOGNITION_ERROR');
    }
  }

  /**
   * Register a new face for a student
   */
  async registerFace(
    imageBase64: string,
    studentId: string,
    courseId: string
  ): Promise<{ success: boolean; message: string }> {
    try {
      logger.info('Registering face for student:', { studentId, courseId });

      const requestData = {
        image: imageBase64,
        studentId,
        courseId,
        action: 'register',
      };

      const response = await axios.post(
        `${this.apiUrl}/register`,
        requestData,
        {
          headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${this.apiKey}`,
          },
          timeout: this.timeout,
        }
      );

      logger.info('Face registration completed:', { studentId, courseId });

      return {
        success: true,
        message: 'Face registered successfully',
      };

    } catch (error) {
      logger.error('Face registration failed:', {
        studentId,
        courseId,
        error: error instanceof Error ? error.message : 'Unknown error',
      });

      if (axios.isAxiosError(error)) {
        const message = error.response?.data?.message || 'Face registration failed';
        throw new CustomError(message, error.response?.status || 500, 'FACE_REGISTRATION_ERROR');
      }

      throw new CustomError('Face registration failed', 500, 'FACE_REGISTRATION_ERROR');
    }
  }

  /**
   * Delete registered face for a student
   */
  async deleteFace(studentId: string, courseId: string): Promise<{ success: boolean; message: string }> {
    try {
      logger.info('Deleting face for student:', { studentId, courseId });

      const response = await axios.delete(
        `${this.apiUrl}/delete`,
        {
          data: { studentId, courseId },
          headers: {
            'Authorization': `Bearer ${this.apiKey}`,
          },
          timeout: this.timeout,
        }
      );

      logger.info('Face deletion completed:', { studentId, courseId });

      return {
        success: true,
        message: 'Face deleted successfully',
      };

    } catch (error) {
      logger.error('Face deletion failed:', {
        studentId,
        courseId,
        error: error instanceof Error ? error.message : 'Unknown error',
      });

      if (axios.isAxiosError(error)) {
        const message = error.response?.data?.message || 'Face deletion failed';
        throw new CustomError(message, error.response?.status || 500, 'FACE_DELETION_ERROR');
      }

      throw new CustomError('Face deletion failed', 500, 'FACE_DELETION_ERROR');
    }
  }

  /**
   * Check if face recognition service is available
   */
  async healthCheck(): Promise<{ isHealthy: boolean; message: string }> {
    try {
      const response = await axios.get(
        `${this.apiUrl}/health`,
        {
          headers: {
            'Authorization': `Bearer ${this.apiKey}`,
          },
          timeout: 5000, // 5 second timeout for health check
        }
      );

      return {
        isHealthy: response.status === 200,
        message: 'Face recognition service is healthy',
      };

    } catch (error) {
      logger.error('Face recognition service health check failed:', error);
      
      return {
        isHealthy: false,
        message: 'Face recognition service is unavailable',
      };
    }
  }
}

// Export singleton instance
export const faceRecognitionService = new FaceRecognitionService();
