import mongoose, { Document, Schema } from 'mongoose';

/**
 * Faculty Model - Extended profile for faculty members
 * Contains professional information, assigned courses, and teaching records
 */

export interface IFaculty extends Document {
  _id: string;
  user: mongoose.Types.ObjectId;
  facultyId: string; // Unique faculty ID (e.g., "FAC2024001")
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
  professionalInfo: {
    employeeId: string;
    department: string;
    designation: string;
    qualification: string;
    specialization: string[];
    joiningDate: Date;
    experience: number; // in years
    salary: number;
  };
  assignedCourses: mongoose.Types.ObjectId[];
  isActive: boolean;
  createdAt: Date;
  updatedAt: Date;
}

const facultySchema = new Schema<IFaculty>({
  user: {
    type: Schema.Types.ObjectId,
    ref: 'User',
    required: [true, 'User reference is required'],
    unique: true,
  },
  facultyId: {
    type: String,
    required: [true, 'Faculty ID is required'],
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
  professionalInfo: {
    employeeId: {
      type: String,
      required: [true, 'Employee ID is required'],
      unique: true,
      trim: true,
      uppercase: true,
    },
    department: {
      type: String,
      required: [true, 'Department is required'],
      trim: true,
    },
    designation: {
      type: String,
      required: [true, 'Designation is required'],
      trim: true,
    },
    qualification: {
      type: String,
      required: [true, 'Qualification is required'],
      trim: true,
    },
    specialization: [{
      type: String,
      trim: true,
    }],
    joiningDate: {
      type: Date,
      required: [true, 'Joining date is required'],
    },
    experience: {
      type: Number,
      required: [true, 'Experience is required'],
      min: [0, 'Experience cannot be negative'],
    },
    salary: {
      type: Number,
      required: [true, 'Salary is required'],
      min: [0, 'Salary cannot be negative'],
    },
  },
  assignedCourses: [{
    type: Schema.Types.ObjectId,
    ref: 'Course',
  }],
  isActive: {
    type: Boolean,
    default: true,
  },
}, {
  timestamps: true,
  toJSON: { virtuals: true },
  toObject: { virtuals: true },
});

// Indexes for better query performance (facultyId and employeeId indexes are already created by unique: true)
facultySchema.index({ 'professionalInfo.department': 1 });
facultySchema.index({ 'professionalInfo.designation': 1 });
facultySchema.index({ assignedCourses: 1 });
facultySchema.index({ isActive: 1 });

// Virtual for full name
facultySchema.virtual('fullName').get(function () {
  return `${this.firstName} ${this.lastName}`;
});

// Virtual for age calculation
facultySchema.virtual('age').get(function () {
  const today = new Date();
  const birthDate = new Date(this.dateOfBirth);
  let age = today.getFullYear() - birthDate.getFullYear();
  const monthDiff = today.getMonth() - birthDate.getMonth();
  
  if (monthDiff < 0 || (monthDiff === 0 && today.getDate() < birthDate.getDate())) {
    age--;
  }
  
  return age;
});

// Virtual for years of service
facultySchema.virtual('yearsOfService').get(function () {
  const today = new Date();
  const joiningDate = new Date(this.professionalInfo.joiningDate);
  let years = today.getFullYear() - joiningDate.getFullYear();
  const monthDiff = today.getMonth() - joiningDate.getMonth();
  
  if (monthDiff < 0 || (monthDiff === 0 && today.getDate() < joiningDate.getDate())) {
    years--;
  }
  
  return years;
});

// Static method to find faculty by employee ID
facultySchema.statics.findByEmployeeId = function (employeeId: string) {
  return this.findOne({ 'professionalInfo.employeeId': employeeId.toUpperCase() });
};

// Static method to find faculty by department
facultySchema.statics.findByDepartment = function (department: string) {
  return this.find({ 'professionalInfo.department': department, isActive: true });
};

export const Faculty = mongoose.model<IFaculty>('Faculty', facultySchema);
