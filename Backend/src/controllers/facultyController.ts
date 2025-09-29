import { Request, Response, NextFunction } from 'express';
import { Faculty } from '../models/Faculty';
import { User } from '../models/User';
import { Course } from '../models/Course';
import { logger } from '../config/logger';
import { CustomError, asyncHandler } from '../middleware/errorHandler';
import { validationResult } from 'express-validator';
import mongoose from 'mongoose';

/**
 * Faculty Controller
 * Handles faculty-related operations and data management
 */

export interface CreateFacultyRequest extends Request {
  body: {
    email: string;
    password: string;
    facultyId: string;
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
    professionalInfo: {
      employeeId: string;
      department: string;
      designation: string;
      qualification: string;
      specialization: string[];
      joiningDate: string;
      experience: number;
      salary: number;
    };
  };
}

/**
 * @desc    Create new faculty
 * @route   POST /api/faculty
 * @access  Private (Admin)
 */
export const createFaculty = asyncHandler(async (req: CreateFacultyRequest, res: Response, next: NextFunction) => {
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
    facultyId,
    firstName,
    lastName,
    dateOfBirth,
    gender,
    phone,
    address,
    emergencyContact,
    professionalInfo
  } = req.body;

  logger.info('Creating new faculty:', { email, facultyId });

  // Check if user already exists
  const existingUser = await User.findOne({ email: email.toLowerCase() });
  if (existingUser) {
    return res.status(400).json({ success: false, message: 'User already exists with this email', code: 'USER_EXISTS' });
  }

  // Check if faculty ID already exists
  const existingFaculty = await Faculty.findOne({ facultyId });
  if (existingFaculty) {
    return res.status(400).json({ success: false, message: 'Faculty ID already exists', code: 'FACULTY_ID_EXISTS' });
  }

  // Check if employee ID already exists
  const existingEmployee = await Faculty.findOne({ 'professionalInfo.employeeId': professionalInfo.employeeId });
  if (existingEmployee) {
    return res.status(400).json({ success: false, message: 'Employee ID already exists', code: 'EMPLOYEE_ID_EXISTS' });
  }

  // Create user account
  const user = await User.create({
    email: email.toLowerCase(),
    password,
    role: 'faculty',
  });

  // Create faculty profile
  const faculty = await Faculty.create({
    user: user._id,
    facultyId,
    firstName,
    lastName,
    dateOfBirth: new Date(dateOfBirth),
    gender,
    phone,
    address,
    emergencyContact,
    professionalInfo: {
      ...professionalInfo,
      joiningDate: new Date(professionalInfo.joiningDate),
    },
  });

  // Populate the created faculty with user data
  await faculty.populate('user', 'email role isActive');

  logger.info('Faculty created successfully:', { facultyId, userId: user._id });

  return res.status(201).json({
    success: true,
    message: 'Faculty created successfully',
    data: {
      faculty,
    },
  });
});

/**
 * @desc    Get all faculty
 * @route   GET /api/faculty
 * @access  Private (Admin)
 */
export const getFaculty = asyncHandler(async (req: Request, res: Response, next: NextFunction) => {
  const {
    page = 1,
    limit = 10,
    department,
    designation,
    search
  } = req.query;

  const query: any = { isActive: true };

  // Apply filters
  if (department) {
    query['professionalInfo.department'] = department;
  }
  if (designation) {
    query['professionalInfo.designation'] = designation;
  }
  if (search) {
    query.$or = [
      { firstName: { $regex: search, $options: 'i' } },
      { lastName: { $regex: search, $options: 'i' } },
      { facultyId: { $regex: search, $options: 'i' } },
      { 'professionalInfo.employeeId': { $regex: search, $options: 'i' } },
    ];
  }

  const skip = (Number(page) - 1) * Number(limit);

  const [faculty, total] = await Promise.all([
    Faculty.find(query)
      .populate('user', 'email role isActive lastLogin')
      .populate('assignedCourses', 'courseCode courseName')
      .select('-__v')
      .sort({ 'professionalInfo.employeeId': 1 })
      .skip(skip)
      .limit(Number(limit)),
    Faculty.countDocuments(query)
  ]);

  return res.json({
    success: true,
    data: {
      faculty,
      pagination: {
        current: Number(page),
        pages: Math.ceil(total / Number(limit)),
        total,
      },
    },
  });
});

/**
 * @desc    Get faculty by ID
 * @route   GET /api/faculty/:id
 * @access  Private (Faculty, Admin)
 */
export const getFacultyById = asyncHandler(async (req: Request, res: Response, next: NextFunction) => {
  const { id } = req.params;

  // Validate access - faculty can only view their own profile
  if (req.user?.role === 'faculty' && req.user.id !== id) {
    return res.status(400).json({
      success: false,
      message: 'You can only view your own profile',
      code: 'ACCESS_DENIED'
    });
  }

  const faculty = await Faculty.findOne({ _id: id, isActive: true })
    .populate('user', 'email role isActive lastLogin')
    .populate('assignedCourses', 'courseCode courseName description')
    .select('-__v');

  if (!faculty) {
    return res.status(400).json({ success: false, message: 'Faculty not found', code: 'FACULTY_NOT_FOUND' });
  }

  return res.json({
    success: true,
    data: {
      faculty,
    },
  });
});

/**
 * @desc    Update faculty profile
 * @route   PUT /api/faculty/:id
 * @access  Private (Faculty, Admin)
 */
export const updateFaculty = asyncHandler(async (req: Request, res: Response, next: NextFunction) => {
  const { id } = req.params;
  const updateData = req.body;

  // Validate access - faculty can only update their own profile
  if (req.user?.role === 'faculty' && req.user.id !== id) {
    return res.status(400).json({
      success: false,
      message: 'You can only update your own profile',
      code: 'ACCESS_DENIED'
    });
  }

  // Remove fields that shouldn't be updated directly
  delete updateData.user;
  delete updateData.facultyId;
  delete updateData._id;
  delete updateData.createdAt;
  delete updateData.updatedAt;

  // Convert date strings to Date objects
  if (updateData.dateOfBirth) {
    updateData.dateOfBirth = new Date(updateData.dateOfBirth);
  }
  if (updateData.professionalInfo?.joiningDate) {
    updateData.professionalInfo.joiningDate = new Date(updateData.professionalInfo.joiningDate);
  }

  const faculty = await Faculty.findOneAndUpdate(
    { _id: id, isActive: true },
    updateData,
    { new: true, runValidators: true }
  ).populate('user', 'email role isActive');

  if (!faculty) {
    return res.status(400).json({ success: false, message: 'Faculty not found', code: 'FACULTY_NOT_FOUND' });
  }

  logger.info('Faculty updated:', { facultyId: faculty.facultyId, updatedBy: req.user?.id });

  return res.json({
    success: true,
    message: 'Faculty updated successfully',
    data: {
      faculty,
    },
  });
});

/**
 * @desc    Delete faculty (soft delete)
 * @route   DELETE /api/faculty/:id
 * @access  Private (Admin)
 */
export const deleteFaculty = asyncHandler(async (req: Request, res: Response, next: NextFunction) => {
  const { id } = req.params;

  const faculty = await Faculty.findOneAndUpdate(
    { _id: id, isActive: true },
    { isActive: false },
    { new: true }
  );

  if (!faculty) {
    return res.status(400).json({ success: false, message: 'Faculty not found', code: 'FACULTY_NOT_FOUND' });
  }

  // Also deactivate the user account
  await User.findByIdAndUpdate(faculty.user, { isActive: false });

  logger.info('Faculty deleted:', { facultyId: faculty.facultyId, deletedBy: req.user?.id });

  return res.json({
    success: true,
    message: 'Faculty deleted successfully',
  });
});

/**
 * @desc    Get faculty's assigned courses
 * @route   GET /api/faculty/:id/courses
 * @access  Private (Faculty, Admin)
 */
export const getFacultyCourses = asyncHandler(async (req: Request, res: Response, next: NextFunction) => {
  const { id } = req.params;

  // Validate access
  if (req.user?.role === 'faculty' && req.user.id !== id) {
    return res.status(400).json({
      success: false,
      message: 'You can only view your own courses',
      code: 'ACCESS_DENIED'
    });
  }

  const faculty = await Faculty.findById(id);
  if (!faculty) {
    return res.status(400).json({ success: false, message: 'Faculty not found', code: 'FACULTY_NOT_FOUND' });
  }

  const courses = await Course.find({
    faculty: faculty._id,
    isActive: true
  })
    .populate('enrolledStudents', 'studentId firstName lastName')
    .select('-__v');

  return res.json({
    success: true,
    data: {
      faculty: {
        id: faculty._id,
        facultyId: faculty.facultyId,
        name: `${faculty.firstName} ${faculty.lastName}`,
      },
      courses,
    },
  });
});

/**
 * @desc    Assign course to faculty
 * @route   POST /api/faculty/:id/assign-course
 * @access  Private (Admin)
 */
export const assignCourseToFaculty = asyncHandler(async (req: Request, res: Response, next: NextFunction) => {
  const { id } = req.params;
  const { courseId } = req.body;

  const faculty = await Faculty.findById(id);
  if (!faculty) {
    return res.status(400).json({ success: false, message: 'Faculty not found', code: 'FACULTY_NOT_FOUND' });
  }

  const course = await Course.findById(courseId);
  if (!course) {
    return res.status(400).json({ success: false, message: 'Course not found', code: 'COURSE_NOT_FOUND' });
  }

  // Check if course is already assigned to this faculty
  if (course.faculty.toString() === faculty._id.toString()) {
    return res.status(400).json({ success: false, message: 'Course is already assigned to this faculty', code: 'ALREADY_ASSIGNED' });
  }

  // Check if faculty already has this course in assigned courses
  if (faculty.assignedCourses.includes(course._id as any)) {
    return res.status(400).json({ success: false, message: 'Faculty already has this course assigned', code: 'ALREADY_ASSIGNED' });
  }

  // Assign course to faculty
  await Promise.all([
    Course.findByIdAndUpdate(courseId, { faculty: faculty._id }),
    Faculty.findByIdAndUpdate(id, { $push: { assignedCourses: course._id } })
  ]);

  logger.info('Course assigned to faculty:', {
    facultyId: faculty.facultyId,
    courseCode: course.courseCode,
    assignedBy: req.user?.id
  });

  return res.json({
    success: true,
    message: 'Course assigned successfully',
    data: {
      faculty: {
        id: faculty._id,
        facultyId: faculty.facultyId,
        name: `${faculty.firstName} ${faculty.lastName}`,
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
 * @desc    Remove course assignment from faculty
 * @route   DELETE /api/faculty/:id/assign-course/:courseId
 * @access  Private (Admin)
 */
export const removeCourseFromFaculty = asyncHandler(async (req: Request, res: Response, next: NextFunction) => {
  const { id, courseId } = req.params;

  const faculty = await Faculty.findById(id);
  if (!faculty) {
    return res.status(400).json({ success: false, message: 'Faculty not found', code: 'FACULTY_NOT_FOUND' });
  }

  const course = await Course.findById(courseId);
  if (!course) {
    return res.status(400).json({ success: false, message: 'Course not found', code: 'COURSE_NOT_FOUND' });
  }

  // Remove course assignment
  await Promise.all([
    Course.findByIdAndUpdate(courseId, { $unset: { faculty: 1 } }),
    Faculty.findByIdAndUpdate(id, { $pull: { assignedCourses: course._id } })
  ]);

  logger.info('Course assignment removed from faculty:', {
    facultyId: faculty.facultyId,
    courseCode: course.courseCode,
    removedBy: req.user?.id
  });

  return res.json({
    success: true,
    message: 'Course assignment removed successfully',
  });
});

/**
 * @desc    Get faculty by department
 * @route   GET /api/faculty/department/:department
 * @access  Private (Admin)
 */
export const getFacultyByDepartment = asyncHandler(async (req: Request, res: Response, next: NextFunction) => {
  const { department } = req.params;

  const faculty = await Faculty.find({
    'professionalInfo.department': department,
    isActive: true
  })
    .populate('user', 'email role isActive')
    .populate('assignedCourses', 'courseCode courseName')
    .select('-__v')
    .sort({ 'professionalInfo.employeeId': 1 });

  return res.json({
    success: true,
    data: {
      department,
      faculty,
      count: faculty.length,
    },
  });
});
