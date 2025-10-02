import { Request, Response, NextFunction } from 'express';
import { Course } from '../models/Course';
import { Student } from '../models/Student';
import { Faculty } from '../models/Faculty';
import { Attendance } from '../models/Attendance';
import { logger } from '../config/logger';
import { CustomError, asyncHandler } from '../middleware/errorHandler';
import { validationResult } from 'express-validator';
import mongoose from 'mongoose';

/**
 * Course Controller
 * Handles course-related operations and data management
 */

export interface CreateCourseRequest extends Request {
  body: {
    courseCode: string;
    courseName: string;
    description: string;
    department: string;
    credits: number;
    semester: number;
    year: number;
    facultyId: string;
    schedule: {
      dayOfWeek: string;
      startTime: string;
      endTime: string;
      room: string;
    }[];
    attendanceSettings: {
      minimumAttendance: number;
    };
  };
}

/**
 * @desc    Create new course
 * @route   POST /api/courses
 * @access  Private (Admin)
 */
export const createCourse = asyncHandler(async (req: CreateCourseRequest, res: Response, next: NextFunction) => {
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
    courseCode,
    courseName,
    description,
    department,
    credits,
    semester,
    year,
    facultyId,
    schedule,
    attendanceSettings
  } = req.body;

  logger.info('Creating new course:', { courseCode, courseName });

  // Check if course code already exists
  const existingCourse = await Course.findOne({ courseCode });
  if (existingCourse) {
    return res.status(400).json({ success: false, message: 'Course code already exists', code: 'COURSE_CODE_EXISTS' });
  }

  // Validate faculty exists
  const faculty = await Faculty.findOne({ facultyId, isActive: true });
  if (!faculty) {
    return res.status(400).json({ success: false, message: 'Faculty not found', code: 'FACULTY_NOT_FOUND' });
  }

  // Create course
  const course = await Course.create({
    courseCode,
    courseName,
    description,
    department,
    credits,
    semester,
    year,
    faculty: faculty._id,
    schedule,
    attendanceSettings: {
      minimumAttendance: attendanceSettings.minimumAttendance,
      totalClasses: 0,
      conductedClasses: 0,
    },
  });

  // Add course to faculty's assigned courses
  await Faculty.findByIdAndUpdate(faculty._id, {
    $push: { assignedCourses: course._id }
  });

  // Populate the created course with faculty data
  await course.populate('faculty', 'facultyId firstName lastName');

  logger.info('Course created successfully:', { courseCode, courseId: course._id });

  return res.status(201).json({
    success: true,
    message: 'Course created successfully',
    data: {
      course,
    },
  });
});

/**
 * @desc    Get all courses
 * @route   GET /api/courses
 * @access  Private (Faculty, Admin)
 */
export const getCourses = asyncHandler(async (req: Request, res: Response, next: NextFunction) => {
  const {
    page = 1,
    limit = 10,
    department,
    semester,
    year,
    facultyId,
    search
  } = req.query;

  const query: any = { isActive: true };

  // Apply filters
  if (department) {
    query.department = department;
  }
  if (semester) {
    query.semester = Number(semester);
  }
  if (year) {
    query.year = Number(year);
  }
  if (facultyId) {
    const faculty = await Faculty.findOne({ facultyId, isActive: true });
    if (faculty) {
      query.faculty = faculty._id;
    }
  }
  if (search) {
    query.$or = [
      { courseCode: { $regex: search, $options: 'i' } },
      { courseName: { $regex: search, $options: 'i' } },
      { description: { $regex: search, $options: 'i' } },
    ];
  }

  const skip = (Number(page) - 1) * Number(limit);

  const [courses, total] = await Promise.all([
    Course.find(query)
      .populate('faculty', 'facultyId firstName lastName')
      .populate('enrolledStudents', 'studentId firstName lastName')
      .select('-__v')
      .sort({ courseCode: 1 })
      .skip(skip)
      .limit(Number(limit)),
    Course.countDocuments(query)
  ]);

  return res.json({
    success: true,
    data: {
      courses,
      pagination: {
        current: Number(page),
        pages: Math.ceil(total / Number(limit)),
        total,
      },
    },
  });
});

/**
 * @desc    Get course by ID
 * @route   GET /api/courses/:id
 * @access  Private (Student, Faculty, Admin)
 */
export const getCourseById = asyncHandler(async (req: Request, res: Response, next: NextFunction) => {
  const { id } = req.params;

  const course = await Course.findOne({ _id: id, isActive: true })
    .populate('faculty', 'facultyId firstName lastName')
    .populate('enrolledStudents', 'studentId firstName lastName')
    .select('-__v');

  if (!course) {
    return res.status(400).json({ success: false, message: 'Course not found', code: 'COURSE_NOT_FOUND' });
  }

  return res.json({
    success: true,
    data: {
      course,
    },
  });
});

/**
 * @desc    Update course
 * @route   PUT /api/courses/:id
 * @access  Private (Admin)
 */
export const updateCourse = asyncHandler(async (req: Request, res: Response, next: NextFunction) => {
  const { id } = req.params;
  const updateData = req.body;

  // Remove fields that shouldn't be updated directly
  delete updateData._id;
  delete updateData.createdAt;
  delete updateData.updatedAt;
  delete updateData.enrolledStudents;

  // If faculty is being changed, update faculty assignments
  if (updateData.facultyId) {
    const newFaculty = await Faculty.findOne({ facultyId: updateData.facultyId, isActive: true });
    if (!newFaculty) {
      return res.status(400).json({ success: false, message: 'Faculty not found', code: 'FACULTY_NOT_FOUND' });
    }

    const currentCourse = await Course.findById(id);
    if (currentCourse) {
      // Remove course from old faculty
      await Faculty.findByIdAndUpdate(currentCourse.faculty, {
        $pull: { assignedCourses: currentCourse._id }
      });

      // Add course to new faculty
      await Faculty.findByIdAndUpdate(newFaculty._id, {
        $push: { assignedCourses: currentCourse._id }
      });

      updateData.faculty = newFaculty._id;
      delete updateData.facultyId;
    }
  }

  const course = await Course.findOneAndUpdate(
    { _id: id, isActive: true },
    updateData,
    { new: true, runValidators: true }
  ).populate('faculty', 'facultyId firstName lastName');

  if (!course) {
    return res.status(400).json({ success: false, message: 'Course not found', code: 'COURSE_NOT_FOUND' });
  }

  logger.info('Course updated:', { courseCode: course.courseCode, updatedBy: req.user?.id });

  return res.json({
    success: true,
    message: 'Course updated successfully',
    data: {
      course,
    },
  });
});

/**
 * @desc    Delete course (soft delete)
 * @route   DELETE /api/courses/:id
 * @access  Private (Admin)
 */
export const deleteCourse = asyncHandler(async (req: Request, res: Response, next: NextFunction) => {
  const { id } = req.params;

  const course = await Course.findOneAndUpdate(
    { _id: id, isActive: true },
    { isActive: false },
    { new: true }
  );

  if (!course) {
    return res.status(400).json({ success: false, message: 'Course not found', code: 'COURSE_NOT_FOUND' });
  }

  // Remove course from faculty's assigned courses
  await Faculty.findByIdAndUpdate(course.faculty, {
    $pull: { assignedCourses: course._id }
  });

  logger.info('Course deleted:', { courseCode: course.courseCode, deletedBy: req.user?.id });

  return res.json({
    success: true,
    message: 'Course deleted successfully',
  });
});

/**
 * @desc    Get course attendance summary
 * @route   GET /api/courses/:id/attendance-summary
 * @access  Private (Faculty, Admin)
 */
export const getCourseAttendanceSummary = asyncHandler(async (req: Request, res: Response, next: NextFunction) => {
  const { id } = req.params;
  const { startDate, endDate } = req.query;

  const course = await Course.findById(id);
  if (!course) {
    return res.status(400).json({ success: false, message: 'Course not found', code: 'COURSE_NOT_FOUND' });
  }

  // Build date filter
  const dateFilter: any = { course: course._id };
  if (startDate && endDate) {
    dateFilter.date = {
      $gte: new Date(startDate as string),
      $lte: new Date(endDate as string)
    };
  }

  // Get attendance statistics
  const attendanceStats = await Attendance.aggregate([
    { $match: dateFilter },
    {
      $group: {
        _id: '$student',
        totalClasses: { $sum: 1 },
        attendedClasses: {
          $sum: {
            $cond: [{ $in: ['$status', ['present', 'late']] }, 1, 0]
          }
        }
      }
    },
    {
      $lookup: {
        from: 'students',
        localField: '_id',
        foreignField: '_id',
        as: 'studentInfo'
      }
    },
    {
      $unwind: '$studentInfo'
    },
    {
      $project: {
        studentId: '$studentInfo.studentId',
        studentName: { $concat: ['$studentInfo.firstName', ' ', '$studentInfo.lastName'] },
        totalClasses: 1,
        attendedClasses: 1,
        attendancePercentage: {
          $cond: [
            { $gt: ['$totalClasses', 0] },
            { $multiply: [{ $divide: ['$attendedClasses', '$totalClasses'] }, 100] },
            0
          ]
        }
      }
    },
    { $sort: { studentId: 1 } }
  ]);

  // Calculate overall statistics
  const overallStats = await Attendance.aggregate([
    { $match: dateFilter },
    {
      $group: {
        _id: null,
        totalRecords: { $sum: 1 },
        presentRecords: {
          $sum: {
            $cond: [{ $eq: ['$status', 'present'] }, 1, 0]
          }
        },
        lateRecords: {
          $sum: {
            $cond: [{ $eq: ['$status', 'late'] }, 1, 0]
          }
        },
        absentRecords: {
          $sum: {
            $cond: [{ $eq: ['$status', 'absent'] }, 1, 0]
          }
        }
      }
    }
  ]);

  return res.json({
    success: true,
    data: {
      course: {
        id: course._id,
        courseCode: course.courseCode,
        courseName: course.courseName,
        totalEnrolled: course.enrolledStudents.length,
      },
      attendanceStats,
      overallStats: overallStats[0] || {
        totalRecords: 0,
        presentRecords: 0,
        lateRecords: 0,
        absentRecords: 0
      },
    },
  });
});

/**
 * @desc    Get courses by department
 * @route   GET /api/courses/department/:department
 * @access  Private (Faculty, Admin)
 */
export const getCoursesByDepartment = asyncHandler(async (req: Request, res: Response, next: NextFunction) => {
  const { department } = req.params;
  const { semester, year } = req.query;

  const query: any = {
    department,
    isActive: true
  };

  if (semester) {
    query.semester = Number(semester);
  }
  if (year) {
    query.year = Number(year);
  }

  const courses = await Course.find(query)
    .populate('faculty', 'facultyId firstName lastName')
    .populate('enrolledStudents', 'studentId firstName lastName')
    .select('-__v')
    .sort({ courseCode: 1 });

  return res.json({
    success: true,
    data: {
      department,
      courses,
      count: courses.length,
    },
  });
});

/**
 * @desc    Get courses by faculty
 * @route   GET /api/courses/faculty/:facultyId
 * @access  Private (Faculty, Admin)
 */
export const getCoursesByFaculty = asyncHandler(async (req: Request, res: Response, next: NextFunction) => {
  const { facultyId } = req.params;

  const faculty = await Faculty.findOne({ facultyId, isActive: true });
  if (!faculty) {
    return res.status(400).json({ success: false, message: 'Faculty not found', code: 'FACULTY_NOT_FOUND' });
  }

  const courses = await Course.find({
    faculty: faculty._id,
    isActive: true
  })
    .populate('enrolledStudents', 'studentId firstName lastName')
    .select('-__v')
    .sort({ courseCode: 1 });

  return res.json({
    success: true,
    data: {
      faculty: {
        id: faculty._id,
        facultyId: faculty.facultyId,
        name: `${faculty.firstName} ${faculty.lastName}`,
      },
      courses,
      count: courses.length,
    },
  });
});

/**
 * @desc    Update course attendance settings
 * @route   PUT /api/courses/:id/attendance-settings
 * @access  Private (Admin)
 */
export const updateAttendanceSettings = asyncHandler(async (req: Request, res: Response, next: NextFunction) => {
  const { id } = req.params;
  const { minimumAttendance, totalClasses, conductedClasses } = req.body;

  const course = await Course.findOneAndUpdate(
    { _id: id, isActive: true },
    {
      $set: {
        'attendanceSettings.minimumAttendance': minimumAttendance,
        'attendanceSettings.totalClasses': totalClasses,
        'attendanceSettings.conductedClasses': conductedClasses,
      }
    },
    { new: true, runValidators: true }
  );

  if (!course) {
    return res.status(400).json({ success: false, message: 'Course not found', code: 'COURSE_NOT_FOUND' });
  }

  logger.info('Course attendance settings updated:', {
    courseCode: course.courseCode,
    updatedBy: req.user?.id
  });

  return res.json({
    success: true,
    message: 'Attendance settings updated successfully',
    data: {
      course: {
        id: course._id,
        courseCode: course.courseCode,
        courseName: course.courseName,
        attendanceSettings: course.attendanceSettings,
      },
    },
  });
});
