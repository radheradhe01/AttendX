import { Router } from 'express';
import {
  createStudent,
  getStudents,
  getStudentById,
  updateStudent,
  deleteStudent,
  getStudentCourses,
  getStudentAttendanceSummary,
  enrollStudentInCourse,
  unenrollStudentFromCourse
} from '../controllers/studentController';
import { authenticateToken, requireAdmin, canAccessStudentData } from '../middleware/authMiddleware';
import {
  validateCreateStudent,
  validateMongoId,
  validatePagination,
  validateSearch,
  handleValidationErrors
} from '../utils/validators';

/**
 * Student Routes
 * Handles student-related operations and data management
 */

const router = Router();

/**
 * @route   POST /api/students
 * @desc    Create new student
 * @access  Private (Admin)
 */
router.post('/', validateCreateStudent(), handleValidationErrors, authenticateToken, requireAdmin, createStudent);

/**
 * @route   GET /api/students
 * @desc    Get all students
 * @access  Private (Faculty, Admin)
 */
router.get('/',
  validatePagination(),
  validateSearch(),
  handleValidationErrors,
  authenticateToken,
  getStudents
);

/**
 * @route   GET /api/students/:id
 * @desc    Get student by ID
 * @access  Private (Student, Faculty, Admin)
 */
router.get('/:id',
  validateMongoId('id'),
  handleValidationErrors,
  authenticateToken,
  canAccessStudentData,
  getStudentById
);

/**
 * @route   PUT /api/students/:id
 * @desc    Update student profile
 * @access  Private (Student, Faculty, Admin)
 */
router.put('/:id',
  validateMongoId('id'),
  handleValidationErrors,
  authenticateToken,
  canAccessStudentData,
  updateStudent
);

/**
 * @route   DELETE /api/students/:id
 * @desc    Delete student (soft delete)
 * @access  Private (Admin)
 */
router.delete('/:id',
  validateMongoId('id'),
  handleValidationErrors,
  authenticateToken,
  requireAdmin,
  deleteStudent
);

/**
 * @route   GET /api/students/:id/courses
 * @desc    Get student's enrolled courses
 * @access  Private (Student, Faculty, Admin)
 */
router.get('/:id/courses',
  validateMongoId('id'),
  handleValidationErrors,
  authenticateToken,
  canAccessStudentData,
  getStudentCourses
);

/**
 * @route   GET /api/students/:id/attendance-summary
 * @desc    Get student's attendance summary
 * @access  Private (Student, Faculty, Admin)
 */
router.get('/:id/attendance-summary',
  validateMongoId('id'),
  handleValidationErrors,
  authenticateToken,
  canAccessStudentData,
  getStudentAttendanceSummary
);

/**
 * @route   POST /api/students/:id/enroll
 * @desc    Enroll student in a course
 * @access  Private (Admin)
 */
router.post('/:id/enroll',
  validateMongoId('id'),
  handleValidationErrors,
  authenticateToken,
  requireAdmin,
  enrollStudentInCourse
);

/**
 * @route   DELETE /api/students/:id/enroll/:courseId
 * @desc    Unenroll student from a course
 * @access  Private (Admin)
 */
router.delete('/:id/enroll/:courseId',
  validateMongoId('id'),
  validateMongoId('courseId'),
  handleValidationErrors,
  authenticateToken,
  requireAdmin,
  unenrollStudentFromCourse
);

export default router;
