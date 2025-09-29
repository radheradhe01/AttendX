import jwt from 'jsonwebtoken';
import { logger } from '../config/logger';
import { CustomError } from '../middleware/errorHandler';

/**
 * JWT Utility Functions
 * Handles JWT token generation, verification, and management
 */

export interface JWTPayload {
  id: string;
  email?: string;
  role?: string;
  iat?: number;
  exp?: number;
}

export interface TokenPair {
  accessToken: string;
  refreshToken: string;
}

/**
 * Generate JWT access token
 */
export const generateAccessToken = (payload: Omit<JWTPayload, 'iat' | 'exp'>): string => {
  try {
    const jwtSecret = process.env.JWT_SECRET;
    const jwtExpire = process.env.JWT_EXPIRE || '7d';

    if (!jwtSecret) {
      throw new CustomError('JWT secret not configured', 500, 'CONFIG_ERROR');
    }

    return jwt.sign(payload, jwtSecret, { expiresIn: jwtExpire } as any);
  } catch (error) {
    logger.error('Error generating access token:', error);
    throw new CustomError('Failed to generate access token', 500, 'TOKEN_GENERATION_ERROR');
  }
};

/**
 * Generate JWT refresh token
 */
export const generateRefreshToken = (payload: Omit<JWTPayload, 'iat' | 'exp'>): string => {
  try {
    const refreshSecret = process.env.JWT_REFRESH_SECRET;
    const refreshExpire = process.env.JWT_REFRESH_EXPIRE || '30d';

    if (!refreshSecret) {
      throw new CustomError('JWT refresh secret not configured', 500, 'CONFIG_ERROR');
    }

    return jwt.sign(payload, refreshSecret, { expiresIn: refreshExpire } as any);
  } catch (error) {
    logger.error('Error generating refresh token:', error);
    throw new CustomError('Failed to generate refresh token', 500, 'TOKEN_GENERATION_ERROR');
  }
};

/**
 * Generate both access and refresh tokens
 */
export const generateTokenPair = (payload: Omit<JWTPayload, 'iat' | 'exp'>): TokenPair => {
  return {
    accessToken: generateAccessToken(payload),
    refreshToken: generateRefreshToken(payload),
  };
};

/**
 * Verify JWT access token
 */
export const verifyAccessToken = (token: string): JWTPayload => {
  try {
    const jwtSecret = process.env.JWT_SECRET;

    if (!jwtSecret) {
      throw new CustomError('JWT secret not configured', 500, 'CONFIG_ERROR');
    }

    return jwt.verify(token, jwtSecret) as JWTPayload;
  } catch (error) {
    if (error instanceof jwt.JsonWebTokenError) {
      throw new CustomError('Invalid access token', 401, 'INVALID_TOKEN');
    }
    if (error instanceof jwt.TokenExpiredError) {
      throw new CustomError('Access token expired', 401, 'TOKEN_EXPIRED');
    }
    
    logger.error('Error verifying access token:', error);
    throw new CustomError('Failed to verify access token', 500, 'TOKEN_VERIFICATION_ERROR');
  }
};

/**
 * Verify JWT refresh token
 */
export const verifyRefreshToken = (token: string): JWTPayload => {
  try {
    const refreshSecret = process.env.JWT_REFRESH_SECRET;

    if (!refreshSecret) {
      throw new CustomError('JWT refresh secret not configured', 500, 'CONFIG_ERROR');
    }

    return jwt.verify(token, refreshSecret) as JWTPayload;
  } catch (error) {
    if (error instanceof jwt.JsonWebTokenError) {
      throw new CustomError('Invalid refresh token', 401, 'INVALID_REFRESH_TOKEN');
    }
    if (error instanceof jwt.TokenExpiredError) {
      throw new CustomError('Refresh token expired', 401, 'REFRESH_TOKEN_EXPIRED');
    }
    
    logger.error('Error verifying refresh token:', error);
    throw new CustomError('Failed to verify refresh token', 500, 'TOKEN_VERIFICATION_ERROR');
  }
};

/**
 * Extract token from Authorization header
 */
export const extractTokenFromHeader = (authHeader: string | undefined): string | null => {
  if (!authHeader) {
    return null;
  }

  const parts = authHeader.split(' ');
  if (parts.length !== 2 || parts[0] !== 'Bearer') {
    return null;
  }

  return parts[1];
};

/**
 * Get token expiration time
 */
export const getTokenExpiration = (token: string): Date | null => {
  try {
    const decoded = jwt.decode(token) as JWTPayload;
    if (decoded && decoded.exp) {
      return new Date(decoded.exp * 1000);
    }
    return null;
  } catch (error) {
    logger.error('Error getting token expiration:', error);
    return null;
  }
};

/**
 * Check if token is expired
 */
export const isTokenExpired = (token: string): boolean => {
  const expiration = getTokenExpiration(token);
  if (!expiration) {
    return true;
  }
  return expiration < new Date();
};

/**
 * Get token payload without verification (for debugging)
 */
export const decodeToken = (token: string): JWTPayload | null => {
  try {
    return jwt.decode(token) as JWTPayload;
  } catch (error) {
    logger.error('Error decoding token:', error);
    return null;
  }
};

/**
 * Generate password reset token
 */
export const generatePasswordResetToken = (userId: string): string => {
  try {
    const jwtSecret = process.env.JWT_SECRET;
    const resetExpire = '1h'; // Password reset tokens expire in 1 hour

    if (!jwtSecret) {
      throw new CustomError('JWT secret not configured', 500, 'CONFIG_ERROR');
    }

    return jwt.sign(
      { id: userId, type: 'password_reset' },
      jwtSecret,
      { expiresIn: resetExpire }
    );
  } catch (error) {
    logger.error('Error generating password reset token:', error);
    throw new CustomError('Failed to generate password reset token', 500, 'TOKEN_GENERATION_ERROR');
  }
};

/**
 * Verify password reset token
 */
export const verifyPasswordResetToken = (token: string): JWTPayload => {
  try {
    const jwtSecret = process.env.JWT_SECRET;

    if (!jwtSecret) {
      throw new CustomError('JWT secret not configured', 500, 'CONFIG_ERROR');
    }

    const decoded = jwt.verify(token, jwtSecret) as JWTPayload;
    
    if ((decoded as any).type !== 'password_reset') {
      throw new CustomError('Invalid token type', 401, 'INVALID_TOKEN_TYPE');
    }

    return decoded;
  } catch (error) {
    if (error instanceof jwt.JsonWebTokenError) {
      throw new CustomError('Invalid password reset token', 401, 'INVALID_TOKEN');
    }
    if (error instanceof jwt.TokenExpiredError) {
      throw new CustomError('Password reset token expired', 401, 'TOKEN_EXPIRED');
    }
    
    logger.error('Error verifying password reset token:', error);
    throw new CustomError('Failed to verify password reset token', 500, 'TOKEN_VERIFICATION_ERROR');
  }
};

/**
 * Generate email verification token
 */
export const generateEmailVerificationToken = (userId: string, email: string): string => {
  try {
    const jwtSecret = process.env.JWT_SECRET;
    const verifyExpire = '24h'; // Email verification tokens expire in 24 hours

    if (!jwtSecret) {
      throw new CustomError('JWT secret not configured', 500, 'CONFIG_ERROR');
    }

    return jwt.sign(
      { id: userId, email, type: 'email_verification' },
      jwtSecret,
      { expiresIn: verifyExpire }
    );
  } catch (error) {
    logger.error('Error generating email verification token:', error);
    throw new CustomError('Failed to generate email verification token', 500, 'TOKEN_GENERATION_ERROR');
  }
};

/**
 * Verify email verification token
 */
export const verifyEmailVerificationToken = (token: string): JWTPayload => {
  try {
    const jwtSecret = process.env.JWT_SECRET;

    if (!jwtSecret) {
      throw new CustomError('JWT secret not configured', 500, 'CONFIG_ERROR');
    }

    const decoded = jwt.verify(token, jwtSecret) as JWTPayload;
    
    if ((decoded as any).type !== 'email_verification') {
      throw new CustomError('Invalid token type', 401, 'INVALID_TOKEN_TYPE');
    }

    return decoded;
  } catch (error) {
    if (error instanceof jwt.JsonWebTokenError) {
      throw new CustomError('Invalid email verification token', 401, 'INVALID_TOKEN');
    }
    if (error instanceof jwt.TokenExpiredError) {
      throw new CustomError('Email verification token expired', 401, 'TOKEN_EXPIRED');
    }
    
    logger.error('Error verifying email verification token:', error);
    throw new CustomError('Failed to verify email verification token', 500, 'TOKEN_VERIFICATION_ERROR');
  }
};
