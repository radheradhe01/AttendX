import { Router } from 'express';
import {
  register,
  login,
  refreshToken,
  getMe,
  updateProfile,
  changePassword,
  logout
} from '../controllers/authController';
import { authenticateToken } from '../middleware/authMiddleware';
import {
  validateBasicRegister,
  validateLogin,
  validateChangePassword,
  handleValidationErrors
} from '../utils/validators';

/**
 * Authentication Routes
 * Handles user authentication, registration, and profile management
 */

const router = Router();

/**
 * @route   POST /api/auth/register
 * @desc    Register new user
 * @access  Public
 */
router.post('/register', validateBasicRegister(), handleValidationErrors, register);

/**
 * @route   POST /api/auth/login
 * @desc    Login user
 * @access  Public
 */
router.post('/login', validateLogin(), handleValidationErrors, login);

/**
 * @route   POST /api/auth/refresh
 * @desc    Refresh access token
 * @access  Public
 */
router.post('/refresh', refreshToken);

/**
 * @route   GET /api/auth/me
 * @desc    Get current user profile
 * @access  Private
 */
router.get('/me', authenticateToken, getMe);

/**
 * @route   PUT /api/auth/profile
 * @desc    Update user profile
 * @access  Private
 */
router.put('/profile', authenticateToken, updateProfile);

/**
 * @route   PUT /api/auth/change-password
 * @desc    Change user password
 * @access  Private
 */
router.put('/change-password', validateChangePassword(), handleValidationErrors, authenticateToken, changePassword);

/**
 * @route   POST /api/auth/logout
 * @desc    Logout user
 * @access  Private
 */
router.post('/logout', authenticateToken, logout);

export default router;
