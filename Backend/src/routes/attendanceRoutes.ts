import { Router } from 'express';
import {
  markAttendance,
  getStudentAttendance,
  getCourseAttendance,
  getStudentAttendanceStats,
  markAttendanceManually,
  getLowAttendanceStudents
} from '../controllers/attendanceController';
import { authenticateToken, requireFacultyOrAdmin, canAccessStudentData } from '../middleware/authMiddleware';
import {
  validateMarkAttendance,
  validateMongoId,
  validatePagination,
  validateDateRange,
  handleValidationErrors
} from '../utils/validators';

/**
 * Attendance Routes
 * Handles attendance marking, validation, and reporting
 */

const router = Router();

/**
 * @route   POST /api/attendance/mark
 * @desc    Mark attendance for a student
 * @access  Private (Student, Faculty, Admin)
 */
router.post('/mark', validateMarkAttendance(), handleValidationErrors, authenticateToken, markAttendance);

/**
 * @route   POST /api/attendance/manual
 * @desc    Manually mark attendance (Faculty/Admin only)
 * @access  Private (Faculty, Admin)
 */
router.post('/manual', authenticateToken, requireFacultyOrAdmin, markAttendanceManually);

/**
 * @route   GET /api/attendance/student/:studentId
 * @desc    Get attendance records for a student
 * @access  Private (Student, Faculty, Admin)
 */
router.get('/student/:studentId', 
  validateMongoId('studentId'),
  validatePagination(),
  validateDateRange(),
  handleValidationErrors,
  authenticateToken,
  canAccessStudentData,
  getStudentAttendance
);

/**
 * @route   GET /api/attendance/student/:studentId/stats
 * @desc    Get attendance statistics for a student
 * @access  Private (Student, Faculty, Admin)
 */
router.get('/student/:studentId/stats',
  validateMongoId('studentId'),
  handleValidationErrors,
  authenticateToken,
  canAccessStudentData,
  getStudentAttendanceStats
);

/**
 * @route   GET /api/attendance/course/:courseId
 * @desc    Get attendance records for a course
 * @access  Private (Faculty, Admin)
 */
router.get('/course/:courseId',
  validateMongoId('courseId'),
  validatePagination(),
  validateDateRange(),
  handleValidationErrors,
  authenticateToken,
  requireFacultyOrAdmin,
  getCourseAttendance
);

/**
 * @route   GET /api/attendance/low-attendance
 * @desc    Get students with low attendance
 * @access  Private (Faculty, Admin)
 */
router.get('/low-attendance',
  authenticateToken,
  requireFacultyOrAdmin,
  getLowAttendanceStudents
);

export default router;
