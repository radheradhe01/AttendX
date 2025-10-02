import { Request, Response, NextFunction } from 'express';
import { logger } from '../config/logger';

/**
 * Global error handling middleware
 * Provides consistent error responses and logging
 */

export interface AppError extends Error {
  statusCode?: number;
  code?: string;
  isOperational?: boolean;
}

/**
 * Custom error class for application-specific errors
 */
export class CustomError extends Error implements AppError {
  public statusCode: number;
  public code: string;
  public isOperational: boolean;

  constructor(message: string, statusCode: number = 500, code: string = 'INTERNAL_ERROR') {
    super(message);
    this.statusCode = statusCode;
    this.code = code;
    this.isOperational = true;

    Error.captureStackTrace(this, this.constructor);
  }
}

/**
 * Handle different types of errors
 */
export const handleError = (error: AppError, req: Request, res: Response, next: NextFunction): void => {
  let statusCode = error.statusCode || 500;
  let message = error.message || 'Internal Server Error';
  let code = error.code || 'INTERNAL_ERROR';

  // Log error details
  logger.error('Error occurred:', {
    message: error.message,
    stack: error.stack,
    url: req.url,
    method: req.method,
    ip: req.ip,
    userAgent: req.get('User-Agent'),
    userId: req.user?.id,
  });

  // Handle specific error types
  if (error.name === 'ValidationError') {
    statusCode = 400;
    message = 'Validation Error';
    code = 'VALIDATION_ERROR';
  } else if (error.name === 'CastError') {
    statusCode = 400;
    message = 'Invalid ID format';
    code = 'INVALID_ID';
  } else if (error.name === 'MongoError' && (error as any).code === 11000) {
    statusCode = 409;
    message = 'Duplicate field value';
    code = 'DUPLICATE_ERROR';
  } else if (error.name === 'JsonWebTokenError') {
    statusCode = 401;
    message = 'Invalid token';
    code = 'INVALID_TOKEN';
  } else if (error.name === 'TokenExpiredError') {
    statusCode = 401;
    message = 'Token expired';
    code = 'TOKEN_EXPIRED';
  } else if (error.name === 'MulterError') {
    statusCode = 400;
    message = 'File upload error';
    code = 'UPLOAD_ERROR';
  }

  // Don't leak error details in production
  if (process.env.NODE_ENV === 'production' && !error.isOperational) {
    message = 'Something went wrong';
    code = 'INTERNAL_ERROR';
  }

  // Send error response
  res.status(statusCode).json({
    success: false,
    message,
    code,
    ...(process.env.NODE_ENV === 'development' && {
      stack: error.stack,
      details: error,
    }),
  });
};

/**
 * Handle 404 errors for undefined routes
 */
export const handleNotFound = (req: Request, res: Response, next: NextFunction): void => {
  const error = new CustomError(`Route ${req.originalUrl} not found`, 404, 'ROUTE_NOT_FOUND');
  next(error);
};

/**
 * Async error wrapper to catch async errors
 */
export const asyncHandler = (fn: Function) => {
  return (req: Request, res: Response, next: NextFunction) => {
    return Promise.resolve(fn(req, res, next)).catch(next);
  };
};

/**
 * Validation error handler
 */
export const handleValidationError = (errors: any[]) => {
  const formattedErrors = errors.map(error => ({
    field: error.path,
    message: error.message,
    value: error.value,
  }));

  return new CustomError('Validation failed', 400, 'VALIDATION_ERROR');
};

/**
 * Database error handler
 */
export const handleDatabaseError = (error: any): AppError => {
  if (error.name === 'ValidationError') {
    const errors = Object.values(error.errors).map((err: any) => ({
      field: err.path,
      message: err.message,
    }));
    return new CustomError('Validation failed', 400, 'VALIDATION_ERROR');
  }

  if (error.name === 'CastError') {
    return new CustomError('Invalid ID format', 400, 'INVALID_ID');
  }

  if (error.code === 11000) {
    const field = Object.keys(error.keyValue)[0];
    return new CustomError(`${field} already exists`, 409, 'DUPLICATE_ERROR');
  }

  return new CustomError('Database operation failed', 500, 'DATABASE_ERROR');
};

/**
 * Rate limit error handler
 */
export const handleRateLimitError = (req: Request, res: Response, next: NextFunction): void => {
  const error = new CustomError('Too many requests, please try again later', 429, 'RATE_LIMIT_EXCEEDED');
  next(error);
};

/**
 * File upload error handler
 */
export const handleUploadError = (error: any): AppError => {
  if (error.code === 'LIMIT_FILE_SIZE') {
    return new CustomError('File size too large', 400, 'FILE_TOO_LARGE');
  }

  if (error.code === 'LIMIT_FILE_COUNT') {
    return new CustomError('Too many files', 400, 'TOO_MANY_FILES');
  }

  if (error.code === 'LIMIT_UNEXPECTED_FILE') {
    return new CustomError('Unexpected file field', 400, 'UNEXPECTED_FILE');
  }

  return new CustomError('File upload failed', 400, 'UPLOAD_ERROR');
};

/**
 * API error handler for external service calls
 */
export const handleAPIError = (error: any, service: string): AppError => {
  if (error.response) {
    // The request was made and the server responded with a status code
    const status = error.response.status;
    const message = error.response.data?.message || `${service} service error`;
    return new CustomError(message, status, 'EXTERNAL_API_ERROR');
  } else if (error.request) {
    // The request was made but no response was received
    return new CustomError(`${service} service unavailable`, 503, 'SERVICE_UNAVAILABLE');
  } else {
    // Something happened in setting up the request
    return new CustomError(`${service} service error`, 500, 'SERVICE_ERROR');
  }
};
