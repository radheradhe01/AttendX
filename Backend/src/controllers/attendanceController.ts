import { Request, Response, NextFunction } from 'express';
import { Attendance } from '../models/Attendance';
import { Student } from '../models/Student';
import { Course } from '../models/Course';
import { Faculty } from '../models/Faculty';
import { faceRecognitionService } from '../services/faceService';
import { gpsService } from '../services/gpsService';
import { erpService } from '../services/erpService';
import { logger } from '../config/logger';
import { CustomError, asyncHandler } from '../middleware/errorHandler';
import { validationResult } from 'express-validator';
import mongoose from 'mongoose';

/**
 * Attendance Controller
 * Handles attendance marking, validation, and reporting
 */

export interface MarkAttendanceRequest extends Request {
  body: {
    studentId: string;
    courseId: string;
    gpsLocation: {
      latitude: number;
      longitude: number;
      accuracy: number;
    };
    faceImage: string; // Base64 encoded image
    deviceInfo: {
      deviceId: string;
      platform: string;
      appVersion: string;
      ipAddress: string;
    };
    remarks?: string;
  };
}

/**
 * @desc    Mark attendance for a student
 * @route   POST /api/attendance/mark
 * @access  Private (Student, Faculty, Admin)
 */
export const markAttendance = asyncHandler(async (req: MarkAttendanceRequest, res: Response, next: NextFunction) => {
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

  const { studentId, courseId, gpsLocation, faceImage, deviceInfo, remarks } = req.body;
  const facultyId = req.user?.id;

  logger.info('Attendance marking attempt:', { studentId, courseId, facultyId });

  try {
    // Validate student exists and is active
    const student = await Student.findOne({ studentId, isActive: true });
    if (!student) {
    return res.status(404).json({
      success: false,
      message: 'Student not found or inactive',
        code: 'STUDENT_NOT_FOUND'
      });
    }

    // Validate course exists and is active
    const course = await Course.findOne({ _id: courseId, isActive: true });
    if (!course) {
    return res.status(404).json({
      success: false,
      message: 'Course not found or inactive',
        code: 'COURSE_NOT_FOUND'
      });
    }

    // Check if student is enrolled in the course
    if (!course.enrolledStudents.includes(student._id as any)) {
      return res.status(400).json({
        success: false,
        message: 'Student is not enrolled in this course',
        code: 'STUDENT_NOT_ENROLLED'
      });
    }

    // Validate GPS location
    const gpsValidation = gpsService.validateGPSLocation(gpsLocation);
    if (!gpsValidation.isWithinCampus) {
      return res.status(400).json({
        success: false,
        message: 'Location is outside campus boundaries',
        code: 'LOCATION_OUTSIDE_CAMPUS',
        data: {
          distance: gpsValidation.distance,
          campusRadius: gpsService.getCampusLocation().radius,
        }
      });
    }

    // Perform face recognition
    const faceRecognitionResult = await faceRecognitionService.verifyFace(
      faceImage,
      studentId,
      courseId
    );

    if (!faceRecognitionResult.isVerified) {
      return res.status(400).json({
        success: false,
        message: 'Face recognition failed',
        code: 'FACE_RECOGNITION_FAILED',
        data: {
          confidence: faceRecognitionResult.confidence,
          error: faceRecognitionResult.error,
        }
      });
    }

    // Check if attendance already marked for today
    const today = new Date();
    today.setHours(0, 0, 0, 0);
    const tomorrow = new Date(today);
    tomorrow.setDate(tomorrow.getDate() + 1);

    const existingAttendance = await Attendance.findOne({
      student: student._id,
      course: courseId,
      date: { $gte: today, $lt: tomorrow }
    });

    if (existingAttendance) {
    return res.status(409).json({
      success: false,
      message: 'Attendance already marked for today',
        code: 'ATTENDANCE_ALREADY_MARKED',
        data: {
          existingAttendance: {
            id: existingAttendance._id,
            time: existingAttendance.time,
            status: existingAttendance.status,
          }
        }
      });
    }

    // Determine attendance status based on time
    const currentTime = new Date();
    const courseStartTime = new Date(`${today.toISOString().split('T')[0]}T${course.schedule[0]?.startTime || '09:00'}:00`);
    const lateThreshold = new Date(courseStartTime.getTime() + 15 * 60000); // 15 minutes late

    let status: 'present' | 'late' = 'present';
    if (currentTime > lateThreshold) {
      status = 'late';
    }

    // Create attendance record
    const attendance = await Attendance.create({
      student: student._id,
      course: courseId,
      faculty: facultyId,
      date: today,
      time: currentTime.toTimeString().slice(0, 5), // HH:MM format
      status,
      gpsLocation: {
        latitude: gpsLocation.latitude,
        longitude: gpsLocation.longitude,
        accuracy: gpsLocation.accuracy,
        isWithinCampus: gpsValidation.isWithinCampus,
      },
      faceRecognition: {
        isVerified: faceRecognitionResult.isVerified,
        confidence: faceRecognitionResult.confidence,
        faceImage,
        processingTime: faceRecognitionResult.processingTime,
      },
      deviceInfo,
      remarks,
      markedBy: req.user?.role === 'student' ? 'student' : 'faculty',
      isManual: false,
    });

    // Update student's attendance statistics
    const attendanceStats = await erpService.getStudentAttendanceStats(student._id.toString(), courseId);
    await Student.findByIdAndUpdate(student._id, {
      'attendance.totalClasses': attendanceStats.totalClasses,
      'attendance.attendedClasses': attendanceStats.attendedClasses,
      'attendance.attendancePercentage': attendanceStats.attendancePercentage,
    });

    // Update course's conducted classes count
    await Course.findByIdAndUpdate(courseId, {
      $inc: { 'attendanceSettings.conductedClasses': 1 }
    });

    logger.info('Attendance marked successfully:', {
      attendanceId: attendance._id,
      studentId,
      courseId,
      status,
      validationScore: attendance.validationScore,
    });

    return res.status(201).json({
      success: true,
      message: 'Attendance marked successfully',
      data: {
        attendance: {
          id: attendance._id,
          studentId,
          courseCode: course.courseCode,
          courseName: course.courseName,
          date: attendance.date,
          time: attendance.time,
          status: attendance.status,
          validationScore: attendance.validationScore,
          isReliable: attendance.isReliable,
        },
        faceRecognition: {
          isVerified: faceRecognitionResult.isVerified,
          confidence: faceRecognitionResult.confidence,
        },
        gpsValidation: {
          isWithinCampus: gpsValidation.isWithinCampus,
          distance: gpsValidation.distance,
        },
      },
    });

  } catch (error) {
    logger.error('Error marking attendance:', error);
    
    if (error instanceof CustomError) {
      throw error;
    }
    
    throw new CustomError('Failed to mark attendance', 500, 'ATTENDANCE_MARKING_ERROR');
  }
});

/**
 * @desc    Get attendance records for a student
 * @route   GET /api/attendance/student/:studentId
 * @access  Private (Student, Faculty, Admin)
 */
export const getStudentAttendance = asyncHandler(async (req: Request, res: Response, next: NextFunction) => {
  const { studentId } = req.params;
  const { courseId, startDate, endDate, page = 1, limit = 10 } = req.query;

  // Validate student access
  if (req.user?.role === 'student' && req.user.id !== studentId) {
    return res.status(403).json({
      success: false,
      message: 'You can only view your own attendance',
      code: 'ACCESS_DENIED'
    });
  }

  const query: any = { student: studentId };
  
  if (courseId) {
    query.course = courseId;
  }
  
  if (startDate && endDate) {
    query.date = {
      $gte: new Date(startDate as string),
      $lte: new Date(endDate as string)
    };
  }

  const skip = (Number(page) - 1) * Number(limit);

  const [attendance, total] = await Promise.all([
    Attendance.find(query)
      .populate('course', 'courseCode courseName')
      .populate('faculty', 'facultyId firstName lastName')
      .sort({ date: -1, time: -1 })
      .skip(skip)
      .limit(Number(limit)),
    Attendance.countDocuments(query)
  ]);

  return res.json({
    success: true,
    data: {
      attendance,
      pagination: {
        current: Number(page),
        pages: Math.ceil(total / Number(limit)),
        total,
      },
    },
  });
});

/**
 * @desc    Get attendance records for a course
 * @route   GET /api/attendance/course/:courseId
 * @access  Private (Faculty, Admin)
 */
export const getCourseAttendance = asyncHandler(async (req: Request, res: Response, next: NextFunction) => {
  const { courseId } = req.params;
  const { date, page = 1, limit = 10 } = req.query;

  const query: any = { course: courseId };
  
  if (date) {
    const targetDate = new Date(date as string);
    targetDate.setHours(0, 0, 0, 0);
    const nextDay = new Date(targetDate);
    nextDay.setDate(nextDay.getDate() + 1);
    
    query.date = { $gte: targetDate, $lt: nextDay };
  }

  const skip = (Number(page) - 1) * Number(limit);

  const [attendance, total] = await Promise.all([
    Attendance.find(query)
      .populate('student', 'studentId firstName lastName')
      .populate('faculty', 'facultyId firstName lastName')
      .sort({ time: -1 })
      .skip(skip)
      .limit(Number(limit)),
    Attendance.countDocuments(query)
  ]);

  return res.json({
    success: true,
    data: {
      attendance,
      pagination: {
        current: Number(page),
        pages: Math.ceil(total / Number(limit)),
        total,
      },
    },
  });
});

/**
 * @desc    Get attendance statistics for a student
 * @route   GET /api/attendance/student/:studentId/stats
 * @access  Private (Student, Faculty, Admin)
 */
export const getStudentAttendanceStats = asyncHandler(async (req: Request, res: Response, next: NextFunction) => {
  const { studentId } = req.params;
  const { courseId } = req.query;

  // Validate student access
  if (req.user?.role === 'student' && req.user.id !== studentId) {
    return res.status(403).json({
      success: false,
      message: 'You can only view your own attendance statistics',
      code: 'ACCESS_DENIED'
    });
  }

  const stats = await erpService.getStudentAttendanceStats(studentId, courseId as string);

  return res.json({
    success: true,
    data: {
      studentId,
      courseId,
      stats,
    },
  });
});

/**
 * @desc    Manually mark attendance (Faculty/Admin only)
 * @route   POST /api/attendance/manual
 * @access  Private (Faculty, Admin)
 */
export const markAttendanceManually = asyncHandler(async (req: Request, res: Response, next: NextFunction) => {
  const { studentId, courseId, status, remarks } = req.body;
  const facultyId = req.user?.id;

  // Validate student exists
  const student = await Student.findOne({ studentId, isActive: true });
  if (!student) {
    return res.status(404).json({
      success: false,
      message: 'Student not found',
      code: 'STUDENT_NOT_FOUND'
    });
  }

  // Validate course exists
  const course = await Course.findOne({ _id: courseId, isActive: true });
  if (!course) {
    return res.status(404).json({
      success: false,
      message: 'Course not found',
      code: 'COURSE_NOT_FOUND'
    });
  }

  // Check if attendance already marked for today
  const today = new Date();
  today.setHours(0, 0, 0, 0);
  const tomorrow = new Date(today);
  tomorrow.setDate(tomorrow.getDate() + 1);

  const existingAttendance = await Attendance.findOne({
    student: student._id,
    course: courseId,
    date: { $gte: today, $lt: tomorrow }
  });

  if (existingAttendance) {
    return res.status(409).json({
      success: false,
      message: 'Attendance already marked for today',
      code: 'ATTENDANCE_ALREADY_MARKED'
    });
  }

  // Create manual attendance record
  const attendance = await Attendance.create({
    student: student._id,
    course: courseId,
    faculty: facultyId,
    date: today,
    time: new Date().toTimeString().slice(0, 5),
    status: status || 'present',
    gpsLocation: {
      latitude: 0,
      longitude: 0,
      accuracy: 0,
      isWithinCampus: true, // Manual attendance is always considered within campus
    },
    faceRecognition: {
      isVerified: true, // Manual attendance bypasses face recognition
      confidence: 100,
      processingTime: 0,
    },
    deviceInfo: {
      deviceId: 'manual',
      platform: 'Web',
      appVersion: '1.0.0',
      ipAddress: req.ip || '127.0.0.1',
    },
    remarks,
    markedBy: 'faculty',
    isManual: true,
  });

  // Update student's attendance statistics
  const attendanceStats = await erpService.getStudentAttendanceStats(student._id.toString(), courseId);
  await Student.findByIdAndUpdate(student._id, {
    'attendance.totalClasses': attendanceStats.totalClasses,
    'attendance.attendedClasses': attendanceStats.attendedClasses,
    'attendance.attendancePercentage': attendanceStats.attendancePercentage,
  });

  logger.info('Manual attendance marked:', {
    attendanceId: attendance._id,
    studentId,
    courseId,
    facultyId,
    status,
  });

  return res.status(201).json({
    success: true,
    message: 'Attendance marked manually',
    data: {
      attendance: {
        id: attendance._id,
        studentId,
        courseCode: course.courseCode,
        courseName: course.courseName,
        date: attendance.date,
        time: attendance.time,
        status: attendance.status,
        isManual: true,
      },
    },
  });
});

/**
 * @desc    Get low attendance students
 * @route   GET /api/attendance/low-attendance
 * @access  Private (Faculty, Admin)
 */
export const getLowAttendanceStudents = asyncHandler(async (req: Request, res: Response, next: NextFunction) => {
  const { threshold = 75 } = req.query;

  const lowAttendanceStudents = await erpService.getLowAttendanceStudents(Number(threshold));

  return res.json({
    success: true,
    data: {
      students: lowAttendanceStudents,
      threshold: Number(threshold),
      count: lowAttendanceStudents.length,
    },
  });
});
