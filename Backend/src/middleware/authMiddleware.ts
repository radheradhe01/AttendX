import { Request, Response, NextFunction } from 'express';
import jwt from 'jsonwebtoken';
import { User } from '../models/User';
import { logger } from '../config/logger';

/**
 * Authentication middleware for JWT token verification
 * Handles role-based access control and user authentication
 */

// Extend Express Request interface to include user data
declare global {
  namespace Express {
    interface Request {
      user?: {
        id: string;
        email: string;
        role: string;
      };
    }
  }
}

export interface JWTPayload {
  id: string;
  email: string;
  role: string;
  iat: number;
  exp: number;
}

/**
 * Middleware to verify JWT token and authenticate user
 */
export const authenticateToken = async (
  req: Request,
  res: Response,
  next: NextFunction
): Promise<void> => {
  try {
    const authHeader = req.headers.authorization;
    const token = authHeader && authHeader.split(' ')[1]; // Bearer TOKEN

    if (!token) {
      res.status(401).json({
        success: false,
        message: 'Access token is required',
        code: 'TOKEN_MISSING'
      });
      return;
    }

    const jwtSecret = process.env.JWT_SECRET;
    if (!jwtSecret) {
      logger.error('JWT_SECRET is not configured');
      res.status(500).json({
        success: false,
        message: 'Server configuration error',
        code: 'CONFIG_ERROR'
      });
      return;
    }

    // Verify the token
    const decoded = jwt.verify(token, jwtSecret) as JWTPayload;
    
    // Check if user still exists and is active
    const user = await User.findById(decoded.id).select('_id email role isActive');
    
    if (!user) {
      res.status(401).json({
        success: false,
        message: 'User not found',
        code: 'USER_NOT_FOUND'
      });
      return;
    }

    if (!user.isActive) {
      res.status(401).json({
        success: false,
        message: 'User account is deactivated',
        code: 'USER_DEACTIVATED'
      });
      return;
    }

    // Add user info to request object
    req.user = {
      id: user._id.toString(),
      email: user.email,
      role: user.role
    };

    next();
  } catch (error) {
    if (error instanceof jwt.JsonWebTokenError) {
      res.status(401).json({
        success: false,
        message: 'Invalid or expired token',
        code: 'TOKEN_INVALID'
      });
      return;
    }

    logger.error('Authentication error:', error);
    res.status(500).json({
      success: false,
      message: 'Authentication failed',
      code: 'AUTH_ERROR'
    });
  }
};

/**
 * Middleware to check if user has required role
 */
export const requireRole = (...roles: string[]) => {
  return (req: Request, res: Response, next: NextFunction): void => {
    if (!req.user) {
      res.status(401).json({
        success: false,
        message: 'Authentication required',
        code: 'AUTH_REQUIRED'
      });
      return;
    }

    if (!roles.includes(req.user.role)) {
      res.status(403).json({
        success: false,
        message: 'Insufficient permissions',
        code: 'INSUFFICIENT_PERMISSIONS',
        requiredRoles: roles,
        userRole: req.user.role
      });
      return;
    }

    next();
  };
};

/**
 * Middleware to check if user is admin
 */
export const requireAdmin = requireRole('admin');

/**
 * Middleware to check if user is faculty or admin
 */
export const requireFacultyOrAdmin = requireRole('faculty', 'admin');

/**
 * Middleware to check if user is student, faculty, or admin
 */
export const requireAnyRole = requireRole('student', 'faculty', 'admin');

/**
 * Middleware to check if user can access student data
 * Students can only access their own data, faculty and admin can access any
 */
export const canAccessStudentData = (req: Request, res: Response, next: NextFunction): void => {
  if (!req.user) {
    res.status(401).json({
      success: false,
      message: 'Authentication required',
      code: 'AUTH_REQUIRED'
    });
    return;
  }

  const { role } = req.user;
  const studentId = req.params.studentId || req.params.id;

  // Admin and faculty can access any student data
  if (role === 'admin' || role === 'faculty') {
    next();
    return;
  }

  // Students can only access their own data
  if (role === 'student') {
    if (studentId && studentId !== req.user.id) {
      res.status(403).json({
        success: false,
        message: 'You can only access your own data',
        code: 'ACCESS_DENIED'
      });
      return;
    }
    next();
    return;
  }

  res.status(403).json({
    success: false,
    message: 'Insufficient permissions',
    code: 'INSUFFICIENT_PERMISSIONS'
  });
};

/**
 * Middleware to check if user can access course data
 * Students can access courses they're enrolled in, faculty can access courses they teach
 */
export const canAccessCourseData = (req: Request, res: Response, next: NextFunction): void => {
  if (!req.user) {
    res.status(401).json({
      success: false,
      message: 'Authentication required',
      code: 'AUTH_REQUIRED'
    });
    return;
  }

  const { role } = req.user;

  // Admin can access any course data
  if (role === 'admin') {
    next();
    return;
  }

  // For faculty and students, we need to check course enrollment/assignment
  // This would typically involve a database query to verify access
  // For now, we'll allow access and let the controller handle the business logic
  next();
};

/**
 * Optional authentication middleware
 * Doesn't fail if no token is provided, but adds user info if token is valid
 */
export const optionalAuth = async (
  req: Request,
  res: Response,
  next: NextFunction
): Promise<void> => {
  try {
    const authHeader = req.headers.authorization;
    const token = authHeader && authHeader.split(' ')[1];

    if (!token) {
      next();
      return;
    }

    const jwtSecret = process.env.JWT_SECRET;
    if (!jwtSecret) {
      next();
      return;
    }

    const decoded = jwt.verify(token, jwtSecret) as JWTPayload;
    const user = await User.findById(decoded.id).select('_id email role isActive');
    
    if (user && user.isActive) {
      req.user = {
        id: user._id.toString(),
        email: user.email,
        role: user.role
      };
    }

    next();
  } catch (error) {
    // Silently continue without authentication
    next();
  }
};
