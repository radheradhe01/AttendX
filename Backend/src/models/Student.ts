import mongoose, { Document, Schema } from 'mongoose';

/**
 * Student Model - Extended profile for students
 * Contains academic information, attendance records, and performance data
 */

export interface IStudent extends Document {
  _id: string;
  user: mongoose.Types.ObjectId;
  studentId: string; // Unique student ID (e.g., "STU2024001")
  firstName: string;
  lastName: string;
  dateOfBirth: Date;
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
    admissionDate: Date;
    currentSemester: number;
    currentYear: number;
    department: string;
    course: string;
    rollNumber: string;
    batch: string;
  };
  attendance: {
    totalClasses: number;
    attendedClasses: number;
    attendancePercentage: number;
  };
  isActive: boolean;
  createdAt: Date;
  updatedAt: Date;
}

const studentSchema = new Schema<IStudent>({
  user: {
    type: Schema.Types.ObjectId,
    ref: 'User',
    required: [true, 'User reference is required'],
    unique: true,
  },
  studentId: {
    type: String,
    required: [true, 'Student ID is required'],
    unique: true,
    uppercase: true,
    trim: true,
  },
  firstName: {
    type: String,
    required: [true, 'First name is required'],
    trim: true,
    maxlength: [50, 'First name cannot exceed 50 characters'],
  },
  lastName: {
    type: String,
    required: [true, 'Last name is required'],
    trim: true,
    maxlength: [50, 'Last name cannot exceed 50 characters'],
  },
  dateOfBirth: {
    type: Date,
    required: [true, 'Date of birth is required'],
  },
  gender: {
    type: String,
    enum: ['male', 'female', 'other'],
    required: [true, 'Gender is required'],
  },
  phone: {
    type: String,
    required: [true, 'Phone number is required'],
    match: [/^\+?[\d\s\-\(\)]+$/, 'Please enter a valid phone number'],
  },
  address: {
    street: {
      type: String,
      required: [true, 'Street address is required'],
      trim: true,
    },
    city: {
      type: String,
      required: [true, 'City is required'],
      trim: true,
    },
    state: {
      type: String,
      required: [true, 'State is required'],
      trim: true,
    },
    zipCode: {
      type: String,
      required: [true, 'ZIP code is required'],
      trim: true,
    },
    country: {
      type: String,
      required: [true, 'Country is required'],
      trim: true,
      default: 'India',
    },
  },
  emergencyContact: {
    name: {
      type: String,
      required: [true, 'Emergency contact name is required'],
      trim: true,
    },
    relationship: {
      type: String,
      required: [true, 'Emergency contact relationship is required'],
      trim: true,
    },
    phone: {
      type: String,
      required: [true, 'Emergency contact phone is required'],
      match: [/^\+?[\d\s\-\(\)]+$/, 'Please enter a valid phone number'],
    },
    email: {
      type: String,
      trim: true,
      lowercase: true,
      match: [/^\w+([.-]?\w+)*@\w+([.-]?\w+)*(\.\w{2,3})+$/, 'Please enter a valid email'],
    },
  },
  academicInfo: {
    admissionDate: {
      type: Date,
      required: [true, 'Admission date is required'],
    },
    currentSemester: {
      type: Number,
      required: [true, 'Current semester is required'],
      min: [1, 'Semester must be at least 1'],
      max: [8, 'Semester cannot exceed 8'],
    },
    currentYear: {
      type: Number,
      required: [true, 'Current year is required'],
      min: [1, 'Year must be at least 1'],
      max: [4, 'Year cannot exceed 4'],
    },
    department: {
      type: String,
      required: [true, 'Department is required'],
      trim: true,
    },
    course: {
      type: String,
      required: [true, 'Course is required'],
      trim: true,
    },
    rollNumber: {
      type: String,
      required: [true, 'Roll number is required'],
      unique: true,
      trim: true,
      uppercase: true,
    },
    batch: {
      type: String,
      required: [true, 'Batch is required'],
      trim: true,
    },
  },
  attendance: {
    totalClasses: {
      type: Number,
      default: 0,
      min: [0, 'Total classes cannot be negative'],
    },
    attendedClasses: {
      type: Number,
      default: 0,
      min: [0, 'Attended classes cannot be negative'],
    },
    attendancePercentage: {
      type: Number,
      default: 0,
      min: [0, 'Attendance percentage cannot be negative'],
      max: [100, 'Attendance percentage cannot exceed 100'],
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

// Indexes for better query performance
studentSchema.index({ studentId: 1 });
studentSchema.index({ rollNumber: 1 });
studentSchema.index({ 'academicInfo.department': 1 });
studentSchema.index({ 'academicInfo.course': 1 });
studentSchema.index({ 'academicInfo.batch': 1 });
studentSchema.index({ isActive: 1 });

// Virtual for full name
studentSchema.virtual('fullName').get(function () {
  return `${this.firstName} ${this.lastName}`;
});

// Virtual for age calculation
studentSchema.virtual('age').get(function () {
  const today = new Date();
  const birthDate = new Date(this.dateOfBirth);
  let age = today.getFullYear() - birthDate.getFullYear();
  const monthDiff = today.getMonth() - birthDate.getMonth();
  
  if (monthDiff < 0 || (monthDiff === 0 && today.getDate() < birthDate.getDate())) {
    age--;
  }
  
  return age;
});

// Pre-save middleware to update attendance percentage
studentSchema.pre('save', function (next) {
  if (this.attendance.totalClasses > 0) {
    this.attendance.attendancePercentage = 
      (this.attendance.attendedClasses / this.attendance.totalClasses) * 100;
  }
  next();
});

// Static method to find student by roll number
studentSchema.statics.findByRollNumber = function (rollNumber: string) {
  return this.findOne({ 'academicInfo.rollNumber': rollNumber.toUpperCase() });
};

// Static method to find students by batch
studentSchema.statics.findByBatch = function (batch: string) {
  return this.find({ 'academicInfo.batch': batch, isActive: true });
};

export const Student = mongoose.model<IStudent>('Student', studentSchema);
