import { Router } from 'express';
import {
  createCourse,
  getCourses,
  getCourseById,
  updateCourse,
  deleteCourse,
  getCourseAttendanceSummary,
  getCoursesByDepartment,
  getCoursesByFaculty,
  updateAttendanceSettings
} from '../controllers/courseController';
import { authenticateToken, requireAdmin, requireFacultyOrAdmin } from '../middleware/authMiddleware';
import {
  validateCreateCourse,
  validateMongoId,
  validatePagination,
  validateSearch,
  validateDateRange,
  handleValidationErrors
} from '../utils/validators';

/**
 * Course Routes
 * Handles course-related operations and data management
 */

const router = Router();

/**
 * @route   POST /api/courses
 * @desc    Create new course
 * @access  Private (Admin)
 */
router.post('/', validateCreateCourse(), handleValidationErrors, authenticateToken, requireAdmin, createCourse);

/**
 * @route   GET /api/courses
 * @desc    Get all courses
 * @access  Private (Faculty, Admin)
 */
router.get('/',
  validatePagination(),
  validateSearch(),
  handleValidationErrors,
  authenticateToken,
  requireFacultyOrAdmin,
  getCourses
);

/**
 * @route   GET /api/courses/:id
 * @desc    Get course by ID
 * @access  Private (Student, Faculty, Admin)
 */
router.get('/:id',
  validateMongoId('id'),
  handleValidationErrors,
  authenticateToken,
  getCourseById
);

/**
 * @route   PUT /api/courses/:id
 * @desc    Update course
 * @access  Private (Admin)
 */
router.put('/:id',
  validateMongoId('id'),
  handleValidationErrors,
  authenticateToken,
  requireAdmin,
  updateCourse
);

/**
 * @route   DELETE /api/courses/:id
 * @desc    Delete course (soft delete)
 * @access  Private (Admin)
 */
router.delete('/:id',
  validateMongoId('id'),
  handleValidationErrors,
  authenticateToken,
  requireAdmin,
  deleteCourse
);

/**
 * @route   GET /api/courses/:id/attendance-summary
 * @desc    Get course attendance summary
 * @access  Private (Faculty, Admin)
 */
router.get('/:id/attendance-summary',
  validateMongoId('id'),
  validateDateRange(),
  handleValidationErrors,
  authenticateToken,
  requireFacultyOrAdmin,
  getCourseAttendanceSummary
);

/**
 * @route   PUT /api/courses/:id/attendance-settings
 * @desc    Update course attendance settings
 * @access  Private (Admin)
 */
router.put('/:id/attendance-settings',
  validateMongoId('id'),
  handleValidationErrors,
  authenticateToken,
  requireAdmin,
  updateAttendanceSettings
);

/**
 * @route   GET /api/courses/department/:department
 * @desc    Get courses by department
 * @access  Private (Faculty, Admin)
 */
router.get('/department/:department',
  authenticateToken,
  requireFacultyOrAdmin,
  getCoursesByDepartment
);

/**
 * @route   GET /api/courses/faculty/:facultyId
 * @desc    Get courses by faculty
 * @access  Private (Faculty, Admin)
 */
router.get('/faculty/:facultyId',
  authenticateToken,
  requireFacultyOrAdmin,
  getCoursesByFaculty
);

export default router;
