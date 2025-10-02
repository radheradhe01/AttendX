import mongoose, { Document, Schema } from 'mongoose';

/**
 * Course Model - Academic course/subject information
 * Contains course details, faculty assignments, and student enrollments
 */

export interface ICourse extends Document {
  _id: string;
  courseCode: string; // e.g., "CS101", "MATH201"
  courseName: string;
  description: string;
  department: string;
  credits: number;
  semester: number;
  year: number;
  faculty: mongoose.Types.ObjectId; // Assigned faculty member
  enrolledStudents: mongoose.Types.ObjectId[]; // Enrolled students
  schedule: {
    dayOfWeek: string; // Monday, Tuesday, etc.
    startTime: string; // HH:MM format
    endTime: string; // HH:MM format
    room: string;
  }[];
  attendanceSettings: {
    minimumAttendance: number; // Minimum percentage required
    totalClasses: number;
    conductedClasses: number;
  };
  isActive: boolean;
  createdAt: Date;
  updatedAt: Date;
}

const courseSchema = new Schema<ICourse>({
  courseCode: {
    type: String,
    required: [true, 'Course code is required'],
    unique: true,
    uppercase: true,
    trim: true,
    match: [/^[A-Z]{2,4}\d{3}$/, 'Course code must be in format like CS101, MATH201'],
  },
  courseName: {
    type: String,
    required: [true, 'Course name is required'],
    trim: true,
    maxlength: [100, 'Course name cannot exceed 100 characters'],
  },
  description: {
    type: String,
    required: [true, 'Course description is required'],
    trim: true,
    maxlength: [500, 'Course description cannot exceed 500 characters'],
  },
  department: {
    type: String,
    required: [true, 'Department is required'],
    trim: true,
  },
  credits: {
    type: Number,
    required: [true, 'Credits are required'],
    min: [1, 'Credits must be at least 1'],
    max: [6, 'Credits cannot exceed 6'],
  },
  semester: {
    type: Number,
    required: [true, 'Semester is required'],
    min: [1, 'Semester must be at least 1'],
    max: [8, 'Semester cannot exceed 8'],
  },
  year: {
    type: Number,
    required: [true, 'Year is required'],
    min: [1, 'Year must be at least 1'],
    max: [4, 'Year cannot exceed 4'],
  },
  faculty: {
    type: Schema.Types.ObjectId,
    ref: 'Faculty',
    required: [true, 'Faculty assignment is required'],
  },
  enrolledStudents: [{
    type: Schema.Types.ObjectId,
    ref: 'Student',
  }],
  schedule: [{
    dayOfWeek: {
      type: String,
      required: [true, 'Day of week is required'],
      enum: ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
    },
    startTime: {
      type: String,
      required: [true, 'Start time is required'],
      match: [/^([0-1]?[0-9]|2[0-3]):[0-5][0-9]$/, 'Time must be in HH:MM format'],
    },
    endTime: {
      type: String,
      required: [true, 'End time is required'],
      match: [/^([0-1]?[0-9]|2[0-3]):[0-5][0-9]$/, 'Time must be in HH:MM format'],
    },
    room: {
      type: String,
      required: [true, 'Room is required'],
      trim: true,
    },
  }],
  attendanceSettings: {
    minimumAttendance: {
      type: Number,
      required: [true, 'Minimum attendance percentage is required'],
      min: [0, 'Minimum attendance cannot be negative'],
      max: [100, 'Minimum attendance cannot exceed 100'],
      default: 75,
    },
    totalClasses: {
      type: Number,
      default: 0,
      min: [0, 'Total classes cannot be negative'],
    },
    conductedClasses: {
      type: Number,
      default: 0,
      min: [0, 'Conducted classes cannot be negative'],
    },
  },
  isActive: {
    type: Boolean,
    default: true,
  },
}, {
  timestamps: true,
  toJSON: { virtuals: true },
  toObject: { virtuals: true },
});

// Indexes for better query performance (courseCode index is already created by unique: true)
courseSchema.index({ department: 1 });
courseSchema.index({ semester: 1, year: 1 });
courseSchema.index({ faculty: 1 });
courseSchema.index({ enrolledStudents: 1 });
courseSchema.index({ isActive: 1 });

// Virtual for enrollment count
courseSchema.virtual('enrollmentCount').get(function () {
  return this.enrolledStudents.length;
});

// Virtual for attendance percentage
courseSchema.virtual('attendancePercentage').get(function () {
  if (this.attendanceSettings.totalClasses > 0) {
    return (this.attendanceSettings.conductedClasses / this.attendanceSettings.totalClasses) * 100;
  }
  return 0;
});

// Pre-save middleware to validate schedule times
courseSchema.pre('save', function (next) {
  for (const slot of this.schedule) {
    const startTime = new Date(`2000-01-01T${slot.startTime}:00`);
    const endTime = new Date(`2000-01-01T${slot.endTime}:00`);
    
    if (endTime <= startTime) {
      return next(new Error('End time must be after start time'));
    }
  }
  next();
});

// Static method to find courses by department
courseSchema.statics.findByDepartment = function (department: string) {
  return this.find({ department, isActive: true });
};

// Static method to find courses by faculty
courseSchema.statics.findByFaculty = function (facultyId: string) {
  return this.find({ faculty: facultyId, isActive: true });
};

// Static method to find courses by semester and year
courseSchema.statics.findBySemesterAndYear = function (semester: number, year: number) {
  return this.find({ semester, year, isActive: true });
};

export const Course = mongoose.model<ICourse>('Course', courseSchema);
