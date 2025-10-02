import { Request, Response, NextFunction } from 'express';
import jwt from 'jsonwebtoken';
import { User } from '../models/User';
import { Student } from '../models/Student';
import { Faculty } from '../models/Faculty';
import { logger } from '../config/logger';
import { CustomError, asyncHandler } from '../middleware/errorHandler';
import { validationResult } from 'express-validator';

/**
 * Authentication Controller
 * Handles user authentication, registration, and token management
 */

export interface AuthRequest extends Request {
  body: {
    email: string;
    password: string;
    role?: string;
    studentId?: string;
    facultyId?: string;
  };
}

export interface RegisterRequest extends Request {
  body: {
    email: string;
    password: string;
    role: 'student' | 'faculty' | 'admin';
    studentId?: string;
    facultyId?: string;
    firstName: string;
    lastName: string;
    phone: string;
  };
}

/**
 * Generate JWT token
 */
const generateToken = (userId: string): string => {
  const jwtSecret = process.env.JWT_SECRET;
  const jwtExpire = process.env.JWT_EXPIRE || '7d';

  if (!jwtSecret) {
    throw new CustomError('JWT secret not configured', 500, 'CONFIG_ERROR');
  }

  return jwt.sign({ id: userId }, jwtSecret, { expiresIn: jwtExpire } as any);
};

/**
 * Generate refresh token
 */
const generateRefreshToken = (userId: string): string => {
  const refreshSecret = process.env.JWT_REFRESH_SECRET;
  const refreshExpire = process.env.JWT_REFRESH_EXPIRE || '30d';

  if (!refreshSecret) {
    throw new CustomError('JWT refresh secret not configured', 500, 'CONFIG_ERROR');
  }

  return jwt.sign({ id: userId }, refreshSecret, { expiresIn: refreshExpire } as any);
};

/**
 * @desc    Register new user
 * @route   POST /api/auth/register
 * @access  Public
 */
export const register = asyncHandler(async (req: RegisterRequest, res: Response, next: NextFunction) => {
  // Check for validation errors
  const errors = validationResult(req);
  if (!errors.isEmpty()) {
    return res.status(400).json({
      success: false,
      message: 'Validation failed',
      errors: errors.array(),
      code: 'VALIDATION_ERROR'
    });
  }

  const { email, password, role, studentId, facultyId, firstName, lastName, phone } = req.body;

  logger.info('User registration attempt:', { email, role });

  // Check if user already exists
  const existingUser = await User.findOne({ email: email.toLowerCase() });
  if (existingUser) {
    return res.status(409).json({
      success: false,
      message: 'User already exists with this email',
      code: 'USER_EXISTS'
    });
  }

  // Create user
  const user = await User.create({
    email: email.toLowerCase(),
    password,
    role,
  });

  // Create role-specific profile
  if (role === 'student' && studentId) {
    await Student.create({
      user: user._id,
      studentId,
      firstName,
      lastName,
      phone,
      dateOfBirth: new Date(), // This should be provided in the request
      gender: 'other', // This should be provided in the request
      address: {
        street: '',
        city: '',
        state: '',
        zipCode: '',
        country: 'India',
      },
      emergencyContact: {
        name: '',
        relationship: '',
        phone: '',
      },
      academicInfo: {
        admissionDate: new Date(),
        currentSemester: 1,
        currentYear: 1,
        department: '',
        course: '',
        rollNumber: studentId,
        batch: '',
      },
    });
  } else if (role === 'faculty' && facultyId) {
    await Faculty.create({
      user: user._id,
      facultyId,
      firstName,
      lastName,
      phone,
      dateOfBirth: new Date(), // This should be provided in the request
      gender: 'other', // This should be provided in the request
      address: {
        street: '',
        city: '',
        state: '',
        zipCode: '',
        country: 'India',
      },
      emergencyContact: {
        name: '',
        relationship: '',
        phone: '',
      },
      professionalInfo: {
        employeeId: facultyId,
        department: '',
        designation: '',
        qualification: '',
        specialization: [],
        joiningDate: new Date(),
        experience: 0,
        salary: 0,
      },
    });
  }

  // Generate tokens
  const token = generateToken(user._id);
  const refreshToken = generateRefreshToken(user._id);

  logger.info('User registered successfully:', { userId: user._id, email, role });

  return res.status(201).json({
    success: true,
    message: 'User registered successfully',
    data: {
      user: {
        id: user._id,
        email: user.email,
        role: user.role,
      },
      token,
      refreshToken,
    },
  });
});

/**
 * @desc    Login user
 * @route   POST /api/auth/login
 * @access  Public
 */
export const login = asyncHandler(async (req: AuthRequest, res: Response, next: NextFunction) => {
  // Check for validation errors
  const errors = validationResult(req);
  if (!errors.isEmpty()) {
    return res.status(400).json({
      success: false,
      message: 'Validation failed',
      errors: errors.array(),
      code: 'VALIDATION_ERROR'
    });
  }

  const { email, password } = req.body;

  logger.info('User login attempt:', { email });

  // Find user and include password for comparison
  const user = await User.findOne({ email: email.toLowerCase() }).select('+password');
  if (!user) {
    return res.status(401).json({
      success: false,
      message: 'Invalid email or password',
      code: 'INVALID_CREDENTIALS'
    });
  }

  // Check if user is active
  if (!user.isActive) {
    return res.status(401).json({
      success: false,
      message: 'Account is deactivated',
      code: 'ACCOUNT_DEACTIVATED'
    });
  }

  // Check password
  const isPasswordValid = await user.comparePassword(password);
  if (!isPasswordValid) {
    return res.status(401).json({
      success: false,
      message: 'Invalid email or password',
      code: 'INVALID_CREDENTIALS'
    });
  }

  // Update last login
  user.lastLogin = new Date();
  await user.save();

  // Generate tokens
  const token = generateToken(user._id);
  const refreshToken = generateRefreshToken(user._id);

  logger.info('User logged in successfully:', { userId: user._id, email, role: user.role });

  return res.json({
    success: true,
    message: 'Login successful',
    data: {
      user: {
        id: user._id,
        email: user.email,
        role: user.role,
        lastLogin: user.lastLogin,
      },
      token,
      refreshToken,
    },
  });
});

/**
 * @desc    Refresh access token
 * @route   POST /api/auth/refresh
 * @access  Public
 */
export const refreshToken = asyncHandler(async (req: Request, res: Response, next: NextFunction) => {
  const { refreshToken } = req.body;

  if (!refreshToken) {
    return res.status(400).json({
      success: false,
      message: 'Refresh token is required',
      code: 'REFRESH_TOKEN_MISSING'
    });
  }

  try {
    const refreshSecret = process.env.JWT_REFRESH_SECRET;
    if (!refreshSecret) {
      throw new CustomError('JWT refresh secret not configured', 500, 'CONFIG_ERROR');
    }

    const decoded = jwt.verify(refreshToken, refreshSecret) as any;
    const user = await User.findById(decoded.id);

    if (!user || !user.isActive) {
      return res.status(400).json({
        success: false,
        message: 'Invalid refresh token',
        code: 'INVALID_REFRESH_TOKEN'
      });
    }

    // Generate new tokens
    const newToken = generateToken(user._id);
    const newRefreshToken = generateRefreshToken(user._id);

    return res.json({
      success: true,
      message: 'Token refreshed successfully',
      data: {
        token: newToken,
        refreshToken: newRefreshToken,
      },
    });

  } catch (error) {
    if (error instanceof jwt.JsonWebTokenError) {
      return res.status(400).json({
        success: false,
        message: 'Invalid refresh token',
        code: 'INVALID_REFRESH_TOKEN'
      });
    }

    throw error;
  }
});

/**
 * @desc    Get current user profile
 * @route   GET /api/auth/me
 * @access  Private
 */
export const getMe = asyncHandler(async (req: Request, res: Response, next: NextFunction) => {
  const user = await User.findById(req.user?.id).select('-__v');
  
  if (!user) {
    return res.status(400).json({ success: false, message: 'User not found', code: 'USER_NOT_FOUND' });
  }

  // Get role-specific profile
  let profile = null;
  if (user.role === 'student') {
    profile = await Student.findOne({ user: user._id }).select('-__v');
  } else if (user.role === 'faculty') {
    profile = await Faculty.findOne({ user: user._id }).select('-__v');
  }

  return res.json({
    success: true,
    data: {
      user,
      profile,
    },
  });
});

/**
 * @desc    Update user profile
 * @route   PUT /api/auth/profile
 * @access  Private
 */
export const updateProfile = asyncHandler(async (req: Request, res: Response, next: NextFunction) => {
  const { email, profilePicture } = req.body;
  const userId = req.user?.id;

  const user = await User.findById(userId);
  if (!user) {
    return res.status(400).json({ success: false, message: 'User not found', code: 'USER_NOT_FOUND' });
  }

  // Update email if provided
  if (email && email !== user.email) {
    const existingUser = await User.findOne({ email: email.toLowerCase() });
    if (existingUser) {
      return res.status(400).json({ success: false, message: 'Email already exists', code: 'EMAIL_EXISTS' });
    }
    user.email = email.toLowerCase();
  }

  // Update profile picture if provided
  if (profilePicture) {
    user.profilePicture = profilePicture;
  }

  await user.save();

  return res.json({
    success: true,
    message: 'Profile updated successfully',
    data: {
      user: {
        id: user._id,
        email: user.email,
        role: user.role,
        profilePicture: user.profilePicture,
      },
    },
  });
});

/**
 * @desc    Change password
 * @route   PUT /api/auth/change-password
 * @access  Private
 */
export const changePassword = asyncHandler(async (req: Request, res: Response, next: NextFunction) => {
  const { currentPassword, newPassword } = req.body;
  const userId = req.user?.id;

  const user = await User.findById(userId).select('+password');
  if (!user) {
    return res.status(400).json({ success: false, message: 'User not found', code: 'USER_NOT_FOUND' });
  }

  // Verify current password
  const isCurrentPasswordValid = await user.comparePassword(currentPassword);
  if (!isCurrentPasswordValid) {
    return res.status(400).json({
      success: false,
      message: 'Current password is incorrect',
      code: 'INVALID_CURRENT_PASSWORD'
    });
  }

  // Update password
  user.password = newPassword;
  await user.save();

  logger.info('Password changed successfully:', { userId });

  return res.json({
    success: true,
    message: 'Password changed successfully',
  });
});

/**
 * @desc    Logout user
 * @route   POST /api/auth/logout
 * @access  Private
 */
export const logout = asyncHandler(async (req: Request, res: Response, next: NextFunction) => {
  // In a stateless JWT system, logout is handled on the client side
  // by removing the token. However, you could implement token blacklisting
  // for additional security if needed.

  logger.info('User logged out:', { userId: req.user?.id });

  return res.json({
    success: true,
    message: 'Logged out successfully',
  });
});
