import { body, param, query, ValidationChain } from 'express-validator';
import { Request, Response, NextFunction } from 'express';
import { validationResult } from 'express-validator';
import { CustomError } from '../middleware/errorHandler';

/**
 * Validation utility functions
 * Provides reusable validation rules and error handling
 */

/**
 * Handle validation errors
 */
export const handleValidationErrors = (req: Request, res: Response, next: NextFunction): void => {
  const errors = validationResult(req);
  if (!errors.isEmpty()) {
    const errorMessages = errors.array().map(error => ({
      field: error.type === 'field' ? (error as any).path : 'unknown',
      message: error.msg,
      value: error.type === 'field' ? (error as any).value : undefined,
    }));

    throw new CustomError('Validation failed', 400, 'VALIDATION_ERROR');
  }
  next();
};

/**
 * Basic user registration validation rules (for auth endpoint)
 */
export const validateBasicRegister = (): ValidationChain[] => [
  body('email')
    .isEmail()
    .normalizeEmail()
    .withMessage('Please provide a valid email address'),
  body('password')
    .isLength({ min: 6 })
    .withMessage('Password must be at least 6 characters long')
    .matches(/^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)/)
    .withMessage('Password must contain at least one lowercase letter, one uppercase letter, and one number'),
  body('role')
    .isIn(['student', 'faculty', 'admin'])
    .withMessage('Role must be student, faculty, or admin'),
];

/**
 * Full registration validation rules (for student/faculty endpoints)
 */
export const validateRegister = (): ValidationChain[] => [
  body('email')
    .isEmail()
    .normalizeEmail()
    .withMessage('Please provide a valid email address'),
  body('password')
    .isLength({ min: 6 })
    .withMessage('Password must be at least 6 characters long')
    .matches(/^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)/)
    .withMessage('Password must contain at least one lowercase letter, one uppercase letter, and one number'),
  body('role')
    .isIn(['student', 'faculty', 'admin'])
    .withMessage('Role must be student, faculty, or admin'),
  body('firstName')
    .trim()
    .isLength({ min: 2, max: 50 })
    .withMessage('First name must be between 2 and 50 characters'),
  body('lastName')
    .trim()
    .isLength({ min: 2, max: 50 })
    .withMessage('Last name must be between 2 and 50 characters'),
  body('phone')
    .matches(/^\+?[\d\s\-\(\)]+$/)
    .withMessage('Please provide a valid phone number'),
];

export const validateLogin = (): ValidationChain[] => [
  body('email')
    .isEmail()
    .normalizeEmail()
    .withMessage('Please provide a valid email address'),
  body('password')
    .notEmpty()
    .withMessage('Password is required'),
];

export const validateChangePassword = (): ValidationChain[] => [
  body('currentPassword')
    .notEmpty()
    .withMessage('Current password is required'),
  body('newPassword')
    .isLength({ min: 6 })
    .withMessage('New password must be at least 6 characters long')
    .matches(/^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)/)
    .withMessage('New password must contain at least one lowercase letter, one uppercase letter, and one number'),
];

/**
 * Student validation rules
 */
export const validateCreateStudent = (): ValidationChain[] => [
  ...validateRegister(),
  body('studentId')
    .trim()
    .isLength({ min: 3, max: 20 })
    .withMessage('Student ID must be between 3 and 20 characters')
    .matches(/^[A-Z0-9]+$/)
    .withMessage('Student ID must contain only uppercase letters and numbers'),
  body('dateOfBirth')
    .isISO8601()
    .withMessage('Please provide a valid date of birth')
    .custom((value) => {
      const birthDate = new Date(value);
      const today = new Date();
      const age = today.getFullYear() - birthDate.getFullYear();
      if (age < 16 || age > 100) {
        throw new Error('Age must be between 16 and 100 years');
      }
      return true;
    }),
  body('gender')
    .isIn(['male', 'female', 'other'])
    .withMessage('Gender must be male, female, or other'),
  body('address.street')
    .trim()
    .notEmpty()
    .withMessage('Street address is required'),
  body('address.city')
    .trim()
    .notEmpty()
    .withMessage('City is required'),
  body('address.state')
    .trim()
    .notEmpty()
    .withMessage('State is required'),
  body('address.zipCode')
    .trim()
    .notEmpty()
    .withMessage('ZIP code is required'),
  body('emergencyContact.name')
    .trim()
    .notEmpty()
    .withMessage('Emergency contact name is required'),
  body('emergencyContact.relationship')
    .trim()
    .notEmpty()
    .withMessage('Emergency contact relationship is required'),
  body('emergencyContact.phone')
    .matches(/^\+?[\d\s\-\(\)]+$/)
    .withMessage('Emergency contact phone number is required'),
  body('academicInfo.admissionDate')
    .isISO8601()
    .withMessage('Please provide a valid admission date'),
  body('academicInfo.currentSemester')
    .isInt({ min: 1, max: 8 })
    .withMessage('Current semester must be between 1 and 8'),
  body('academicInfo.currentYear')
    .isInt({ min: 1, max: 4 })
    .withMessage('Current year must be between 1 and 4'),
  body('academicInfo.department')
    .trim()
    .notEmpty()
    .withMessage('Department is required'),
  body('academicInfo.course')
    .trim()
    .notEmpty()
    .withMessage('Course is required'),
  body('academicInfo.rollNumber')
    .trim()
    .isLength({ min: 3, max: 20 })
    .withMessage('Roll number must be between 3 and 20 characters'),
  body('academicInfo.batch')
    .trim()
    .notEmpty()
    .withMessage('Batch is required'),
];

/**
 * Faculty validation rules
 */
export const validateCreateFaculty = (): ValidationChain[] => [
  ...validateRegister(),
  body('facultyId')
    .trim()
    .isLength({ min: 3, max: 20 })
    .withMessage('Faculty ID must be between 3 and 20 characters')
    .matches(/^[A-Z0-9]+$/)
    .withMessage('Faculty ID must contain only uppercase letters and numbers'),
  body('dateOfBirth')
    .isISO8601()
    .withMessage('Please provide a valid date of birth')
    .custom((value) => {
      const birthDate = new Date(value);
      const today = new Date();
      const age = today.getFullYear() - birthDate.getFullYear();
      if (age < 22 || age > 70) {
        throw new Error('Age must be between 22 and 70 years');
      }
      return true;
    }),
  body('gender')
    .isIn(['male', 'female', 'other'])
    .withMessage('Gender must be male, female, or other'),
  body('address.street')
    .trim()
    .notEmpty()
    .withMessage('Street address is required'),
  body('address.city')
    .trim()
    .notEmpty()
    .withMessage('City is required'),
  body('address.state')
    .trim()
    .notEmpty()
    .withMessage('State is required'),
  body('address.zipCode')
    .trim()
    .notEmpty()
    .withMessage('ZIP code is required'),
  body('emergencyContact.name')
    .trim()
    .notEmpty()
    .withMessage('Emergency contact name is required'),
  body('emergencyContact.relationship')
    .trim()
    .notEmpty()
    .withMessage('Emergency contact relationship is required'),
  body('emergencyContact.phone')
    .matches(/^\+?[\d\s\-\(\)]+$/)
    .withMessage('Emergency contact phone number is required'),
  body('professionalInfo.employeeId')
    .trim()
    .isLength({ min: 3, max: 20 })
    .withMessage('Employee ID must be between 3 and 20 characters'),
  body('professionalInfo.department')
    .trim()
    .notEmpty()
    .withMessage('Department is required'),
  body('professionalInfo.designation')
    .trim()
    .notEmpty()
    .withMessage('Designation is required'),
  body('professionalInfo.qualification')
    .trim()
    .notEmpty()
    .withMessage('Qualification is required'),
  body('professionalInfo.specialization')
    .isArray({ min: 1 })
    .withMessage('At least one specialization is required'),
  body('professionalInfo.joiningDate')
    .isISO8601()
    .withMessage('Please provide a valid joining date'),
  body('professionalInfo.experience')
    .isInt({ min: 0, max: 50 })
    .withMessage('Experience must be between 0 and 50 years'),
  body('professionalInfo.salary')
    .isFloat({ min: 0 })
    .withMessage('Salary must be a positive number'),
];

/**
 * Course validation rules
 */
export const validateCreateCourse = (): ValidationChain[] => [
  body('courseCode')
    .trim()
    .matches(/^[A-Z]{2,4}\d{3}$/)
    .withMessage('Course code must be in format like CS101, MATH201'),
  body('courseName')
    .trim()
    .isLength({ min: 3, max: 100 })
    .withMessage('Course name must be between 3 and 100 characters'),
  body('description')
    .trim()
    .isLength({ min: 10, max: 500 })
    .withMessage('Description must be between 10 and 500 characters'),
  body('department')
    .trim()
    .notEmpty()
    .withMessage('Department is required'),
  body('credits')
    .isInt({ min: 1, max: 6 })
    .withMessage('Credits must be between 1 and 6'),
  body('semester')
    .isInt({ min: 1, max: 8 })
    .withMessage('Semester must be between 1 and 8'),
  body('year')
    .isInt({ min: 1, max: 4 })
    .withMessage('Year must be between 1 and 4'),
  body('facultyId')
    .trim()
    .notEmpty()
    .withMessage('Faculty ID is required'),
  body('schedule')
    .isArray({ min: 1 })
    .withMessage('At least one schedule slot is required'),
  body('schedule.*.dayOfWeek')
    .isIn(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
    .withMessage('Invalid day of week'),
  body('schedule.*.startTime')
    .matches(/^([0-1]?[0-9]|2[0-3]):[0-5][0-9]$/)
    .withMessage('Start time must be in HH:MM format'),
  body('schedule.*.endTime')
    .matches(/^([0-1]?[0-9]|2[0-3]):[0-5][0-9]$/)
    .withMessage('End time must be in HH:MM format'),
  body('schedule.*.room')
    .trim()
    .notEmpty()
    .withMessage('Room is required'),
  body('attendanceSettings.minimumAttendance')
    .isInt({ min: 0, max: 100 })
    .withMessage('Minimum attendance must be between 0 and 100'),
];

/**
 * Attendance validation rules
 */
export const validateMarkAttendance = (): ValidationChain[] => [
  body('studentId')
    .trim()
    .notEmpty()
    .withMessage('Student ID is required'),
  body('courseId')
    .isMongoId()
    .withMessage('Valid course ID is required'),
  body('gpsLocation.latitude')
    .isFloat({ min: -90, max: 90 })
    .withMessage('Latitude must be between -90 and 90'),
  body('gpsLocation.longitude')
    .isFloat({ min: -180, max: 180 })
    .withMessage('Longitude must be between -180 and 180'),
  body('gpsLocation.accuracy')
    .isFloat({ min: 0 })
    .withMessage('GPS accuracy must be a positive number'),
  body('faceImage')
    .notEmpty()
    .withMessage('Face image is required'),
  body('deviceInfo.deviceId')
    .trim()
    .notEmpty()
    .withMessage('Device ID is required'),
  body('deviceInfo.platform')
    .isIn(['iOS', 'Android', 'Web'])
    .withMessage('Platform must be iOS, Android, or Web'),
  body('deviceInfo.appVersion')
    .trim()
    .notEmpty()
    .withMessage('App version is required'),
  body('deviceInfo.ipAddress')
    .isIP()
    .withMessage('Valid IP address is required'),
];

/**
 * Report validation rules
 */
export const validateGenerateReport = (): ValidationChain[] => [
  body('reportType')
    .isIn(['attendance', 'academic', 'financial', 'student', 'faculty', 'course'])
    .withMessage('Invalid report type'),
  body('title')
    .trim()
    .isLength({ min: 3, max: 200 })
    .withMessage('Title must be between 3 and 200 characters'),
  body('description')
    .trim()
    .isLength({ min: 10, max: 500 })
    .withMessage('Description must be between 10 and 500 characters'),
  body('parameters.format')
    .isIn(['pdf', 'csv', 'excel'])
    .withMessage('Format must be pdf, csv, or excel'),
  body('parameters.startDate')
    .optional()
    .isISO8601()
    .withMessage('Start date must be a valid date'),
  body('parameters.endDate')
    .optional()
    .isISO8601()
    .withMessage('End date must be a valid date'),
];

/**
 * Common parameter validation rules
 */
export const validateMongoId = (paramName: string): ValidationChain => 
  param(paramName).isMongoId().withMessage(`Valid ${paramName} is required`);

export const validatePagination = (): ValidationChain[] => [
  query('page')
    .optional()
    .isInt({ min: 1 })
    .withMessage('Page must be a positive integer'),
  query('limit')
    .optional()
    .isInt({ min: 1, max: 100 })
    .withMessage('Limit must be between 1 and 100'),
];

export const validateDateRange = (): ValidationChain[] => [
  query('startDate')
    .optional()
    .isISO8601()
    .withMessage('Start date must be a valid date'),
  query('endDate')
    .optional()
    .isISO8601()
    .withMessage('End date must be a valid date'),
];

/**
 * File upload validation
 */
export const validateFileUpload = (): ValidationChain[] => [
  body('file')
    .custom((value, { req }) => {
      if (!req.file) {
        throw new Error('File is required');
      }
      
      const allowedTypes = ['image/jpeg', 'image/png', 'image/jpg'];
      if (!allowedTypes.includes(req.file.mimetype)) {
        throw new Error('File type not allowed. Only JPEG and PNG images are allowed.');
      }
      
      const maxSize = 5 * 1024 * 1024; // 5MB
      if (req.file.size > maxSize) {
        throw new Error('File size too large. Maximum size is 5MB.');
      }
      
      return true;
    }),
];

/**
 * Search validation
 */
export const validateSearch = (): ValidationChain[] => [
  query('search')
    .optional()
    .trim()
    .isLength({ min: 2, max: 100 })
    .withMessage('Search term must be between 2 and 100 characters'),
];
