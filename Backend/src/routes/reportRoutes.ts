import { Router } from 'express';
import {
  generateReport,
  getReports,
  getReportById,
  downloadReport,
  deleteReport,
  getDashboardStats,
  getLowAttendanceStudents
} from '../controllers/reportController';
import { authenticateToken, requireAdmin, requireFacultyOrAdmin } from '../middleware/authMiddleware';
import {
  validateGenerateReport,
  validateMongoId,
  validatePagination,
  validateSearch,
  handleValidationErrors
} from '../utils/validators';

/**
 * Report Routes
 * Handles report generation, management, and download
 */

const router = Router();

/**
 * @route   POST /api/reports/generate
 * @desc    Generate a new report
 * @access  Private (Faculty, Admin)
 */
router.post('/generate', validateGenerateReport(), handleValidationErrors, authenticateToken, requireFacultyOrAdmin, generateReport);

/**
 * @route   GET /api/reports
 * @desc    Get all reports
 * @access  Private (Faculty, Admin)
 */
router.get('/',
  validatePagination(),
  validateSearch(),
  handleValidationErrors,
  authenticateToken,
  requireFacultyOrAdmin,
  getReports
);

/**
 * @route   GET /api/reports/:id
 * @desc    Get report by ID
 * @access  Private (Faculty, Admin)
 */
router.get('/:id',
  validateMongoId('id'),
  handleValidationErrors,
  authenticateToken,
  requireFacultyOrAdmin,
  getReportById
);

/**
 * @route   GET /api/reports/:id/download
 * @desc    Download report file
 * @access  Private (Faculty, Admin)
 */
router.get('/:id/download',
  validateMongoId('id'),
  handleValidationErrors,
  authenticateToken,
  requireFacultyOrAdmin,
  downloadReport
);

/**
 * @route   DELETE /api/reports/:id
 * @desc    Delete report
 * @access  Private (Faculty, Admin)
 */
router.delete('/:id',
  validateMongoId('id'),
  handleValidationErrors,
  authenticateToken,
  requireFacultyOrAdmin,
  deleteReport
);

/**
 * @route   GET /api/reports/dashboard
 * @desc    Get dashboard statistics
 * @access  Private (Admin)
 */
router.get('/dashboard',
  authenticateToken,
  requireAdmin,
  getDashboardStats
);

/**
 * @route   GET /api/reports/low-attendance
 * @desc    Get low attendance students
 * @access  Private (Faculty, Admin)
 */
router.get('/low-attendance',
  authenticateToken,
  requireFacultyOrAdmin,
  getLowAttendanceStudents
);

export default router;
