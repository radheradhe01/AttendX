import mongoose, { Document, Schema } from 'mongoose';

/**
 * Attendance Model - Records student attendance with GPS and face recognition
 * Tracks attendance logs with location validation and biometric verification
 */

export interface IAttendance extends Document {
  _id: string;
  student: mongoose.Types.ObjectId;
  course: mongoose.Types.ObjectId;
  faculty: mongoose.Types.ObjectId;
  date: Date;
  time: string; // HH:MM format
  status: 'present' | 'absent' | 'late' | 'excused';
  gpsLocation: {
    latitude: number;
    longitude: number;
    accuracy: number; // in meters
    isWithinCampus: boolean;
  };
  faceRecognition: {
    isVerified: boolean;
    confidence: number; // 0-100
    faceImage?: string; // Base64 or file path
    processingTime: number; // in milliseconds
  };
  deviceInfo: {
    deviceId: string;
    platform: string; // iOS, Android, Web
    appVersion: string;
    ipAddress: string;
  };
  remarks?: string;
  markedBy: 'student' | 'faculty' | 'system';
  isManual: boolean; // true if marked manually by faculty
  createdAt: Date;
  updatedAt: Date;
  // Virtual properties
  validationScore: number;
  isReliable: boolean;
}

const attendanceSchema = new Schema<IAttendance>({
  student: {
    type: Schema.Types.ObjectId,
    ref: 'Student',
    required: [true, 'Student reference is required'],
  },
  course: {
    type: Schema.Types.ObjectId,
    ref: 'Course',
    required: [true, 'Course reference is required'],
  },
  faculty: {
    type: Schema.Types.ObjectId,
    ref: 'Faculty',
    required: [true, 'Faculty reference is required'],
  },
  date: {
    type: Date,
    required: [true, 'Date is required'],
    default: Date.now,
  },
  time: {
    type: String,
    required: [true, 'Time is required'],
    match: [/^([0-1]?[0-9]|2[0-3]):[0-5][0-9]$/, 'Time must be in HH:MM format'],
  },
  status: {
    type: String,
    enum: ['present', 'absent', 'late', 'excused'],
    required: [true, 'Status is required'],
    default: 'present',
  },
  gpsLocation: {
    latitude: {
      type: Number,
      required: [true, 'Latitude is required'],
      min: [-90, 'Latitude must be between -90 and 90'],
      max: [90, 'Latitude must be between -90 and 90'],
    },
    longitude: {
      type: Number,
      required: [true, 'Longitude is required'],
      min: [-180, 'Longitude must be between -180 and 180'],
      max: [180, 'Longitude must be between -180 and 180'],
    },
    accuracy: {
      type: Number,
      required: [true, 'GPS accuracy is required'],
      min: [0, 'Accuracy cannot be negative'],
    },
    isWithinCampus: {
      type: Boolean,
      required: [true, 'Campus location validation is required'],
    },
  },
  faceRecognition: {
    isVerified: {
      type: Boolean,
      required: [true, 'Face verification status is required'],
    },
    confidence: {
      type: Number,
      required: [true, 'Face recognition confidence is required'],
      min: [0, 'Confidence cannot be negative'],
      max: [100, 'Confidence cannot exceed 100'],
    },
    faceImage: {
      type: String,
      trim: true,
    },
    processingTime: {
      type: Number,
      required: [true, 'Processing time is required'],
      min: [0, 'Processing time cannot be negative'],
    },
  },
  deviceInfo: {
    deviceId: {
      type: String,
      required: [true, 'Device ID is required'],
      trim: true,
    },
    platform: {
      type: String,
      required: [true, 'Platform is required'],
      enum: ['iOS', 'Android', 'Web'],
    },
    appVersion: {
      type: String,
      required: [true, 'App version is required'],
      trim: true,
    },
    ipAddress: {
      type: String,
      required: [true, 'IP address is required'],
      match: [/^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$/, 'Invalid IP address format'],
    },
  },
  remarks: {
    type: String,
    trim: true,
    maxlength: [500, 'Remarks cannot exceed 500 characters'],
  },
  markedBy: {
    type: String,
    enum: ['student', 'faculty', 'system'],
    required: [true, 'Marked by field is required'],
    default: 'student',
  },
  isManual: {
    type: Boolean,
    default: false,
  },
}, {
  timestamps: true,
  toJSON: { virtuals: true },
  toObject: { virtuals: true },
});

// Compound indexes for better query performance
attendanceSchema.index({ student: 1, course: 1, date: 1 });
attendanceSchema.index({ course: 1, date: 1 });
attendanceSchema.index({ faculty: 1, date: 1 });
attendanceSchema.index({ date: 1, status: 1 });
attendanceSchema.index({ 'gpsLocation.isWithinCampus': 1 });
attendanceSchema.index({ 'faceRecognition.isVerified': 1 });

// Virtual for attendance validation score
attendanceSchema.virtual('validationScore').get(function () {
  let score = 0;
  
  // GPS validation (40 points)
  if (this.gpsLocation.isWithinCampus) {
    score += 40;
  }
  
  // Face recognition validation (60 points)
  if (this.faceRecognition.isVerified && this.faceRecognition.confidence >= 80) {
    score += 60;
  } else if (this.faceRecognition.isVerified && this.faceRecognition.confidence >= 60) {
    score += 40;
  }
  
  return score;
});

// Virtual for attendance reliability
attendanceSchema.virtual('isReliable').get(function () {
  return this.validationScore >= 80;
});

// Pre-save middleware to validate attendance data
attendanceSchema.pre('save', function (next) {
  // Check if attendance is being marked for a future date
  const today = new Date();
  today.setHours(0, 0, 0, 0);
  
  if (this.date > today) {
    return next(new Error('Cannot mark attendance for future dates'));
  }
  
  // Check if attendance is being marked too early (before 6 AM)
  const attendanceTime = new Date(`${this.date.toISOString().split('T')[0]}T${this.time}:00`);
  const sixAM = new Date(`${this.date.toISOString().split('T')[0]}T06:00:00`);
  
  if (attendanceTime < sixAM) {
    return next(new Error('Attendance cannot be marked before 6 AM'));
  }
  
  next();
});

// Static method to find attendance by student and date range
attendanceSchema.statics.findByStudentAndDateRange = function (
  studentId: string, 
  startDate: Date, 
  endDate: Date
) {
  return this.find({
    student: studentId,
    date: { $gte: startDate, $lte: endDate }
  }).populate('course', 'courseCode courseName');
};

// Static method to find attendance by course and date
attendanceSchema.statics.findByCourseAndDate = function (courseId: string, date: Date) {
  return this.find({
    course: courseId,
    date: date
  }).populate('student', 'studentId firstName lastName');
};

// Static method to get attendance statistics
attendanceSchema.statics.getAttendanceStats = function (studentId: string, courseId: string) {
  return this.aggregate([
    {
      $match: {
        student: new mongoose.Types.ObjectId(studentId),
        course: new mongoose.Types.ObjectId(courseId)
      }
    },
    {
      $group: {
        _id: '$status',
        count: { $sum: 1 }
      }
    }
  ]);
};

export const Attendance = mongoose.model<IAttendance>('Attendance', attendanceSchema);
