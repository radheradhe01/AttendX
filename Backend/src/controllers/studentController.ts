import { Request, Response, NextFunction } from 'express';
import { Student } from '../models/Student';
import { User } from '../models/User';
import { Course } from '../models/Course';
import { Attendance } from '../models/Attendance';
import { erpService } from '../services/erpService';
import { logger } from '../config/logger';
import { CustomError, asyncHandler } from '../middleware/errorHandler';
import { validationResult } from 'express-validator';
import mongoose from 'mongoose';

/**
 * Student Controller
 * Handles student-related operations and data management
 */

export interface CreateStudentRequest extends Request {
  body: {
    email: string;
    password: string;
    studentId: string;
    firstName: string;
    lastName: string;
    dateOfBirth: string;
    gender: 'male' | 'female' | 'other';
    phone: string;
    address: {
      street: string;
      city: string;
      state: string;
      zipCode: string;
      country: string;
    };
    emergencyContact: {
      name: string;
      relationship: string;
      phone: string;
      email?: string;
    };
    academicInfo: {
      admissionDate: string;
      currentSemester: number;
      currentYear: number;
      department: string;
      course: string;
      rollNumber: string;
      batch: string;
    };
  };
}

/**
 * @desc    Create new student
 * @route   POST /api/students
 * @access  Private (Admin)
 */
export const createStudent = asyncHandler(async (req: CreateStudentRequest, res: Response, next: NextFunction) => {
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

  const {
    email,
    password,
    studentId,
    firstName,
    lastName,
    dateOfBirth,
    gender,
    phone,
    address,
    emergencyContact,
    academicInfo
  } = req.body;

  logger.info('Creating new student:', { email, studentId });

  // Check if user already exists
  const existingUser = await User.findOne({ email: email.toLowerCase() });
  if (existingUser) {
    return res.status(400).json({ success: false, message: 'User already exists with this email', code: 'USER_EXISTS' });
  }

  // Check if student ID already exists
  const existingStudent = await Student.findOne({ studentId });
  if (existingStudent) {
    return res.status(400).json({ success: false, message: 'Student ID already exists', code: 'STUDENT_ID_EXISTS' });
  }

  // Check if roll number already exists
  const existingRollNumber = await Student.findOne({ 'academicInfo.rollNumber': academicInfo.rollNumber });
  if (existingRollNumber) {
    return res.status(400).json({ success: false, message: 'Roll number already exists', code: 'ROLL_NUMBER_EXISTS' });
  }

  // Create user account
  const user = await User.create({
    email: email.toLowerCase(),
    password,
    role: 'student',
  });

  // Create student profile
  const student = await Student.create({
    user: user._id,
    studentId,
    firstName,
    lastName,
    dateOfBirth: new Date(dateOfBirth),
    gender,
    phone,
    address,
    emergencyContact,
    academicInfo: {
      ...academicInfo,
      admissionDate: new Date(academicInfo.admissionDate),
    },
  });

  // Populate the created student with user data
  await student.populate('user', 'email role isActive');

  logger.info('Student created successfully:', { studentId, userId: user._id });

  return res.status(201).json({
    success: true,
    message: 'Student created successfully',
    data: {
      student,
    },
  });
});

/**
 * @desc    Get all students
 * @route   GET /api/students
 * @access  Private (Faculty, Admin)
 */
export const getStudents = asyncHandler(async (req: Request, res: Response, next: NextFunction) => {
  const {
    page = 1,
    limit = 10,
    department,
    course,
    batch,
    semester,
    year,
    search
  } = req.query;

  const query: any = { isActive: true };

  // Apply filters
  if (department) {
    query['academicInfo.department'] = department;
  }
  if (course) {
    query['academicInfo.course'] = course;
  }
  if (batch) {
    query['academicInfo.batch'] = batch;
  }
  if (semester) {
    query['academicInfo.currentSemester'] = Number(semester);
  }
  if (year) {
    query['academicInfo.currentYear'] = Number(year);
  }
  if (search) {
    query.$or = [
      { firstName: { $regex: search, $options: 'i' } },
      { lastName: { $regex: search, $options: 'i' } },
      { studentId: { $regex: search, $options: 'i' } },
      { 'academicInfo.rollNumber': { $regex: search, $options: 'i' } },
    ];
  }

  const skip = (Number(page) - 1) * Number(limit);

  const [students, total] = await Promise.all([
    Student.find(query)
      .populate('user', 'email role isActive lastLogin')
      .select('-__v')
      .sort({ 'academicInfo.rollNumber': 1 })
      .skip(skip)
      .limit(Number(limit)),
    Student.countDocuments(query)
  ]);

  return res.json({
    success: true,
    data: {
      students,
      pagination: {
        current: Number(page),
        pages: Math.ceil(total / Number(limit)),
        total,
      },
    },
  });
});

/**
 * @desc    Get student by ID
 * @route   GET /api/students/:id
 * @access  Private (Student, Faculty, Admin)
 */
export const getStudentById = asyncHandler(async (req: Request, res: Response, next: NextFunction) => {
  const { id } = req.params;

  // Validate access - students can only view their own profile
  if (req.user?.role === 'student' && req.user.id !== id) {
    return res.status(400).json({
      success: false,
      message: 'You can only view your own profile',
      code: 'ACCESS_DENIED'
    });
  }

  const student = await Student.findOne({ _id: id, isActive: true })
    .populate('user', 'email role isActive lastLogin')
    .select('-__v');

  if (!student) {
    return res.status(400).json({ success: false, message: 'Student not found', code: 'STUDENT_NOT_FOUND' });
  }

  return res.json({
    success: true,
    data: {
      student,
    },
  });
});

/**
 * @desc    Update student profile
 * @route   PUT /api/students/:id
 * @access  Private (Student, Faculty, Admin)
 */
export const updateStudent = asyncHandler(async (req: Request, res: Response, next: NextFunction) => {
  const { id } = req.params;
  const updateData = req.body;

  // Validate access - students can only update their own profile
  if (req.user?.role === 'student' && req.user.id !== id) {
    return res.status(400).json({
      success: false,
      message: 'You can only update your own profile',
      code: 'ACCESS_DENIED'
    });
  }

  // Remove fields that shouldn't be updated directly
  delete updateData.user;
  delete updateData.studentId;
  delete updateData._id;
  delete updateData.createdAt;
  delete updateData.updatedAt;

  // Convert date strings to Date objects
  if (updateData.dateOfBirth) {
    updateData.dateOfBirth = new Date(updateData.dateOfBirth);
  }
  if (updateData.academicInfo?.admissionDate) {
    updateData.academicInfo.admissionDate = new Date(updateData.academicInfo.admissionDate);
  }

  const student = await Student.findOneAndUpdate(
    { _id: id, isActive: true },
    updateData,
    { new: true, runValidators: true }
  ).populate('user', 'email role isActive');

  if (!student) {
    return res.status(400).json({ success: false, message: 'Student not found', code: 'STUDENT_NOT_FOUND' });
  }

  logger.info('Student updated:', { studentId: student.studentId, updatedBy: req.user?.id });

  return res.json({
    success: true,
    message: 'Student updated successfully',
    data: {
      student,
    },
  });
});

/**
 * @desc    Delete student (soft delete)
 * @route   DELETE /api/students/:id
 * @access  Private (Admin)
 */
export const deleteStudent = asyncHandler(async (req: Request, res: Response, next: NextFunction) => {
  const { id } = req.params;

  const student = await Student.findOneAndUpdate(
    { _id: id, isActive: true },
    { isActive: false },
    { new: true }
  );

  if (!student) {
    return res.status(400).json({ success: false, message: 'Student not found', code: 'STUDENT_NOT_FOUND' });
  }

  // Also deactivate the user account
  await User.findByIdAndUpdate(student.user, { isActive: false });

  logger.info('Student deleted:', { studentId: student.studentId, deletedBy: req.user?.id });

  return res.json({
    success: true,
    message: 'Student deleted successfully',
  });
});

/**
 * @desc    Get student's enrolled courses
 * @route   GET /api/students/:id/courses
 * @access  Private (Student, Faculty, Admin)
 */
export const getStudentCourses = asyncHandler(async (req: Request, res: Response, next: NextFunction) => {
  const { id } = req.params;

  // Validate access
  if (req.user?.role === 'student' && req.user.id !== id) {
    return res.status(400).json({
      success: false,
      message: 'You can only view your own courses',
      code: 'ACCESS_DENIED'
    });
  }

  const student = await Student.findById(id);
  if (!student) {
    return res.status(400).json({ success: false, message: 'Student not found', code: 'STUDENT_NOT_FOUND' });
  }

  const courses = await Course.find({
    enrolledStudents: student._id,
    isActive: true
  })
    .populate('faculty', 'facultyId firstName lastName')
    .select('-enrolledStudents -__v');

  return res.json({
    success: true,
    data: {
      student: {
        id: student._id,
        studentId: student.studentId,
        name: `${student.firstName} ${student.lastName}`,
      },
      courses,
    },
  });
});

/**
 * @desc    Get student's attendance summary
 * @route   GET /api/students/:id/attendance-summary
 * @access  Private (Student, Faculty, Admin)
 */
export const getStudentAttendanceSummary = asyncHandler(async (req: Request, res: Response, next: NextFunction) => {
  const { id } = req.params;

  // Validate access
  if (req.user?.role === 'student' && req.user.id !== id) {
    return res.status(400).json({
      success: false,
      message: 'You can only view your own attendance',
      code: 'ACCESS_DENIED'
    });
  }

  const student = await Student.findById(id);
  if (!student) {
    return res.status(400).json({ success: false, message: 'Student not found', code: 'STUDENT_NOT_FOUND' });
  }

  // Get overall attendance stats
  const overallStats = await erpService.getStudentAttendanceStats(id);

  // Get course-wise attendance stats
  const courses = await Course.find({
    enrolledStudents: student._id,
    isActive: true
  }).select('_id courseCode courseName');

  const courseStats = [];
  for (const course of courses) {
    const stats = await erpService.getStudentAttendanceStats(id, course._id.toString());
    courseStats.push({
      courseId: course._id,
      courseCode: course.courseCode,
      courseName: course.courseName,
      ...stats,
    });
  }

  return res.json({
    success: true,
    data: {
      student: {
        id: student._id,
        studentId: student.studentId,
        name: `${student.firstName} ${student.lastName}`,
      },
      overallStats,
      courseStats,
    },
  });
});

/**
 * @desc    Enroll student in a course
 * @route   POST /api/students/:id/enroll
 * @access  Private (Admin)
 */
export const enrollStudentInCourse = asyncHandler(async (req: Request, res: Response, next: NextFunction) => {
  const { id } = req.params;
  const { courseId } = req.body;

  const student = await Student.findById(id);
  if (!student) {
    return res.status(400).json({ success: false, message: 'Student not found', code: 'STUDENT_NOT_FOUND' });
  }

  const course = await Course.findById(courseId);
  if (!course) {
    return res.status(400).json({ success: false, message: 'Course not found', code: 'COURSE_NOT_FOUND' });
  }

  // Check if student is already enrolled
  if (course.enrolledStudents.includes(student._id as any)) {
    return res.status(400).json({ success: false, message: 'Student is already enrolled in this course', code: 'ALREADY_ENROLLED' });
  }

  // Enroll student
  await Course.findByIdAndUpdate(courseId, {
    $push: { enrolledStudents: student._id }
  });

  logger.info('Student enrolled in course:', {
    studentId: student.studentId,
    courseCode: course.courseCode,
    enrolledBy: req.user?.id
  });

  return res.json({
    success: true,
    message: 'Student enrolled successfully',
    data: {
      student: {
        id: student._id,
        studentId: student.studentId,
        name: `${student.firstName} ${student.lastName}`,
      },
      course: {
        id: course._id,
        courseCode: course.courseCode,
        courseName: course.courseName,
      },
    },
  });
});

/**
 * @desc    Unenroll student from a course
 * @route   DELETE /api/students/:id/enroll/:courseId
 * @access  Private (Admin)
 */
export const unenrollStudentFromCourse = asyncHandler(async (req: Request, res: Response, next: NextFunction) => {
  const { id, courseId } = req.params;

  const student = await Student.findById(id);
  if (!student) {
    return res.status(400).json({ success: false, message: 'Student not found', code: 'STUDENT_NOT_FOUND' });
  }

  const course = await Course.findById(courseId);
  if (!course) {
    return res.status(400).json({ success: false, message: 'Course not found', code: 'COURSE_NOT_FOUND' });
  }

  // Unenroll student
  await Course.findByIdAndUpdate(courseId, {
    $pull: { enrolledStudents: student._id }
  });

  logger.info('Student unenrolled from course:', {
    studentId: student.studentId,
    courseCode: course.courseCode,
    unenrolledBy: req.user?.id
  });

  return res.json({
    success: true,
    message: 'Student unenrolled successfully',
  });
});
