import { Router } from 'express';
import {
  createFaculty,
  getFaculty,
  getFacultyById,
  updateFaculty,
  deleteFaculty,
  getFacultyCourses,
  assignCourseToFaculty,
  removeCourseFromFaculty,
  getFacultyByDepartment
} from '../controllers/facultyController';
import { authenticateToken, requireAdmin, requireFacultyOrAdmin } from '../middleware/authMiddleware';
import {
  validateCreateFaculty,
  validateMongoId,
  validatePagination,
  validateSearch,
  handleValidationErrors
} from '../utils/validators';

/**
 * Faculty Routes
 * Handles faculty-related operations and data management
 */

const router = Router();

/**
 * @route   POST /api/faculty
 * @desc    Create new faculty
 * @access  Private (Admin)
 */
router.post('/', validateCreateFaculty(), handleValidationErrors, authenticateToken, requireAdmin, createFaculty);

/**
 * @route   GET /api/faculty
 * @desc    Get all faculty
 * @access  Private (Admin)
 */
router.get('/',
  validatePagination(),
  validateSearch(),
  handleValidationErrors,
  authenticateToken,
  requireAdmin,
  getFaculty
);

/**
 * @route   GET /api/faculty/:id
 * @desc    Get faculty by ID
 * @access  Private (Faculty, Admin)
 */
router.get('/:id',
  validateMongoId('id'),
  handleValidationErrors,
  authenticateToken,
  requireFacultyOrAdmin,
  getFacultyById
);

/**
 * @route   PUT /api/faculty/:id
 * @desc    Update faculty profile
 * @access  Private (Faculty, Admin)
 */
router.put('/:id',
  validateMongoId('id'),
  handleValidationErrors,
  authenticateToken,
  requireFacultyOrAdmin,
  updateFaculty
);

/**
 * @route   DELETE /api/faculty/:id
 * @desc    Delete faculty (soft delete)
 * @access  Private (Admin)
 */
router.delete('/:id',
  validateMongoId('id'),
  handleValidationErrors,
  authenticateToken,
  requireAdmin,
  deleteFaculty
);

/**
 * @route   GET /api/faculty/:id/courses
 * @desc    Get faculty's assigned courses
 * @access  Private (Faculty, Admin)
 */
router.get('/:id/courses',
  validateMongoId('id'),
  handleValidationErrors,
  authenticateToken,
  requireFacultyOrAdmin,
  getFacultyCourses
);

/**
 * @route   POST /api/faculty/:id/assign-course
 * @desc    Assign course to faculty
 * @access  Private (Admin)
 */
router.post('/:id/assign-course',
  validateMongoId('id'),
  handleValidationErrors,
  authenticateToken,
  requireAdmin,
  assignCourseToFaculty
);

/**
 * @route   DELETE /api/faculty/:id/assign-course/:courseId
 * @desc    Remove course assignment from faculty
 * @access  Private (Admin)
 */
router.delete('/:id/assign-course/:courseId',
  validateMongoId('id'),
  validateMongoId('courseId'),
  handleValidationErrors,
  authenticateToken,
  requireAdmin,
  removeCourseFromFaculty
);

/**
 * @route   GET /api/faculty/department/:department
 * @desc    Get faculty by department
 * @access  Private (Admin)
 */
router.get('/department/:department',
  authenticateToken,
  requireAdmin,
  getFacultyByDepartment
);

export default router;
