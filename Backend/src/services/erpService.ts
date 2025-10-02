import { Student } from '../models/Student';
import { Faculty } from '../models/Faculty';
import { Course } from '../models/Course';
import { Attendance } from '../models/Attendance';
import { Report } from '../models/Report';
import { logger } from '../config/logger';
import { CustomError } from '../middleware/errorHandler';
import mongoose from 'mongoose';

/**
 * ERP Service
 * Handles core business logic for the ERP system
 */

export interface AttendanceStats {
  totalClasses: number;
  attendedClasses: number;
  attendancePercentage: number;
  status: 'good' | 'warning' | 'critical';
}

export interface AcademicPerformance {
  studentId: string;
  studentName: string;
  courseCode: string;
  courseName: string;
  attendancePercentage: number;
  averageMarks: number;
  grade: string;
}

export interface FeeCollectionStats {
  totalFees: number;
  collectedFees: number;
  pendingFees: number;
  collectionPercentage: number;
}

export interface DashboardStats {
  totalStudents: number;
  totalFaculty: number;
  totalCourses: number;
  attendanceStats: {
    todayAttendance: number;
    averageAttendance: number;
  };
  academicStats: {
    totalExams: number;
    averageMarks: number;
  };
  financialStats: FeeCollectionStats;
}

/**
 * ERP service class
 */
export class ERPService {
  /**
   * Get attendance statistics for a student
   */
  async getStudentAttendanceStats(studentId: string, courseId?: string): Promise<AttendanceStats> {
    try {
      logger.info('Getting attendance stats for student:', { studentId, courseId });

      const matchQuery: any = { student: new mongoose.Types.ObjectId(studentId) };
      if (courseId) {
        matchQuery.course = new mongoose.Types.ObjectId(courseId);
      }

      const stats = await Attendance.aggregate([
        { $match: matchQuery },
        {
          $group: {
            _id: null,
            totalClasses: { $sum: 1 },
            attendedClasses: {
              $sum: {
                $cond: [{ $in: ['$status', ['present', 'late']] }, 1, 0]
              }
            }
          }
        }
      ]);

      const result = stats[0] || { totalClasses: 0, attendedClasses: 0 };
      const attendancePercentage = result.totalClasses > 0 
        ? (result.attendedClasses / result.totalClasses) * 100 
        : 0;

      let status: 'good' | 'warning' | 'critical' = 'good';
      if (attendancePercentage < 50) {
        status = 'critical';
      } else if (attendancePercentage < 75) {
        status = 'warning';
      }

      return {
        totalClasses: result.totalClasses,
        attendedClasses: result.attendedClasses,
        attendancePercentage: Math.round(attendancePercentage * 100) / 100,
        status,
      };

    } catch (error) {
      logger.error('Error getting student attendance stats:', error);
      throw new CustomError('Failed to get attendance statistics', 500, 'ATTENDANCE_STATS_ERROR');
    }
  }

  /**
   * Get academic performance for students
   */
  async getAcademicPerformance(
    department?: string,
    semester?: number,
    year?: number
  ): Promise<AcademicPerformance[]> {
    try {
      logger.info('Getting academic performance:', { department, semester, year });

      const matchQuery: any = { isActive: true };
      if (department) matchQuery['academicInfo.department'] = department;
      if (semester) matchQuery['academicInfo.currentSemester'] = semester;
      if (year) matchQuery['academicInfo.currentYear'] = year;

      const students = await Student.find(matchQuery)
        .populate('user', 'email')
        .select('studentId firstName lastName academicInfo');

      const performanceData: AcademicPerformance[] = [];

      for (const student of students) {
        const attendanceStats = await this.getStudentAttendanceStats(student._id.toString());
        
        // Get course information for the student
        const courses = await Course.find({
          enrolledStudents: student._id,
          isActive: true
        }).select('courseCode courseName');

        for (const course of courses) {
          const courseAttendanceStats = await this.getStudentAttendanceStats(
            student._id.toString(),
            course._id.toString()
          );

          // Calculate grade based on attendance (simplified)
          let grade = 'F';
          if (courseAttendanceStats.attendancePercentage >= 90) grade = 'A';
          else if (courseAttendanceStats.attendancePercentage >= 80) grade = 'B';
          else if (courseAttendanceStats.attendancePercentage >= 70) grade = 'C';
          else if (courseAttendanceStats.attendancePercentage >= 60) grade = 'D';

          performanceData.push({
            studentId: student.studentId,
            studentName: `${student.firstName} ${student.lastName}`,
            courseCode: course.courseCode,
            courseName: course.courseName,
            attendancePercentage: courseAttendanceStats.attendancePercentage,
            averageMarks: 0, // This would come from a marks/grades system
            grade,
          });
        }
      }

      return performanceData;

    } catch (error) {
      logger.error('Error getting academic performance:', error);
      throw new CustomError('Failed to get academic performance', 500, 'ACADEMIC_PERFORMANCE_ERROR');
    }
  }

  /**
   * Get fee collection statistics
   */
  async getFeeCollectionStats(): Promise<FeeCollectionStats> {
    try {
      logger.info('Getting fee collection statistics');

      // This is a simplified implementation
      // In a real system, you would have a Fees model with actual fee data
      const totalStudents = await Student.countDocuments({ isActive: true });
      const totalFees = totalStudents * 50000; // Assuming 50k per student
      const collectedFees = totalFees * 0.85; // Assuming 85% collection rate
      const pendingFees = totalFees - collectedFees;
      const collectionPercentage = (collectedFees / totalFees) * 100;

      return {
        totalFees,
        collectedFees,
        pendingFees,
        collectionPercentage: Math.round(collectionPercentage * 100) / 100,
      };

    } catch (error) {
      logger.error('Error getting fee collection stats:', error);
      throw new CustomError('Failed to get fee collection statistics', 500, 'FEE_STATS_ERROR');
    }
  }

  /**
   * Get dashboard statistics
   */
  async getDashboardStats(): Promise<DashboardStats> {
    try {
      logger.info('Getting dashboard statistics');

      const [
        totalStudents,
        totalFaculty,
        totalCourses,
        todayAttendance,
        feeStats
      ] = await Promise.all([
        Student.countDocuments({ isActive: true }),
        Faculty.countDocuments({ isActive: true }),
        Course.countDocuments({ isActive: true }),
        this.getTodayAttendanceCount(),
        this.getFeeCollectionStats()
      ]);

      const averageAttendance = await this.getAverageAttendancePercentage();

      return {
        totalStudents,
        totalFaculty,
        totalCourses,
        attendanceStats: {
          todayAttendance,
          averageAttendance,
        },
        academicStats: {
          totalExams: 0, // This would come from an exams system
          averageMarks: 0, // This would come from a marks system
        },
        financialStats: feeStats,
      };

    } catch (error) {
      logger.error('Error getting dashboard stats:', error);
      throw new CustomError('Failed to get dashboard statistics', 500, 'DASHBOARD_STATS_ERROR');
    }
  }

  /**
   * Get today's attendance count
   */
  private async getTodayAttendanceCount(): Promise<number> {
    const today = new Date();
    today.setHours(0, 0, 0, 0);
    const tomorrow = new Date(today);
    tomorrow.setDate(tomorrow.getDate() + 1);

    return await Attendance.countDocuments({
      date: { $gte: today, $lt: tomorrow },
      status: { $in: ['present', 'late'] }
    });
  }

  /**
   * Get average attendance percentage across all students
   */
  private async getAverageAttendancePercentage(): Promise<number> {
    const stats = await Attendance.aggregate([
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
        $project: {
          attendancePercentage: {
            $cond: [
              { $gt: ['$totalClasses', 0] },
              { $multiply: [{ $divide: ['$attendedClasses', '$totalClasses'] }, 100] },
              0
            ]
          }
        }
      },
      {
        $group: {
          _id: null,
          averageAttendance: { $avg: '$attendancePercentage' }
        }
      }
    ]);

    return stats[0]?.averageAttendance || 0;
  }

  /**
   * Generate attendance report
   */
  async generateAttendanceReport(
    startDate: Date,
    endDate: Date,
    department?: string,
    courseId?: string
  ): Promise<any[]> {
    try {
      logger.info('Generating attendance report:', { startDate, endDate, department, courseId });

      const matchQuery: any = {
        date: { $gte: startDate, $lte: endDate }
      };

      if (courseId) {
        matchQuery.course = new mongoose.Types.ObjectId(courseId);
      }

      const report = await Attendance.aggregate([
        { $match: matchQuery },
        {
          $lookup: {
            from: 'students',
            localField: 'student',
            foreignField: '_id',
            as: 'studentInfo'
          }
        },
        {
          $lookup: {
            from: 'courses',
            localField: 'course',
            foreignField: '_id',
            as: 'courseInfo'
          }
        },
        {
          $unwind: '$studentInfo'
        },
        {
          $unwind: '$courseInfo'
        },
        {
          $match: department ? {
            'studentInfo.academicInfo.department': department
          } : {}
        },
        {
          $group: {
            _id: {
              studentId: '$studentInfo.studentId',
              studentName: { $concat: ['$studentInfo.firstName', ' ', '$studentInfo.lastName'] },
              courseCode: '$courseInfo.courseCode',
              courseName: '$courseInfo.courseName'
            },
            totalClasses: { $sum: 1 },
            attendedClasses: {
              $sum: {
                $cond: [{ $in: ['$status', ['present', 'late']] }, 1, 0]
              }
            }
          }
        },
        {
          $project: {
            studentId: '$_id.studentId',
            studentName: '$_id.studentName',
            courseCode: '$_id.courseCode',
            courseName: '$_id.courseName',
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
        { $sort: { studentId: 1, courseCode: 1 } }
      ]);

      return report;

    } catch (error) {
      logger.error('Error generating attendance report:', error);
      throw new CustomError('Failed to generate attendance report', 500, 'REPORT_GENERATION_ERROR');
    }
  }

  /**
   * Get low attendance students
   */
  async getLowAttendanceStudents(threshold: number = 75): Promise<any[]> {
    try {
      logger.info('Getting low attendance students:', { threshold });

      const students = await Student.find({ isActive: true })
        .populate('user', 'email')
        .select('studentId firstName lastName academicInfo user');

      const lowAttendanceStudents = [];

      for (const student of students) {
        const attendanceStats = await this.getStudentAttendanceStats(student._id.toString());
        
        if (attendanceStats.attendancePercentage < threshold) {
          lowAttendanceStudents.push({
            studentId: student.studentId,
            studentName: `${student.firstName} ${student.lastName}`,
            email: (student.user as any).email,
            department: student.academicInfo.department,
            course: student.academicInfo.course,
            attendancePercentage: attendanceStats.attendancePercentage,
            status: attendanceStats.status,
          });
        }
      }

      return lowAttendanceStudents.sort((a, b) => a.attendancePercentage - b.attendancePercentage);

    } catch (error) {
      logger.error('Error getting low attendance students:', error);
      throw new CustomError('Failed to get low attendance students', 500, 'LOW_ATTENDANCE_ERROR');
    }
  }
}

// Export singleton instance
export const erpService = new ERPService();
