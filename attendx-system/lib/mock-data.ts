// Mock data for AttendX system

export interface User {
  id: string
  name: string
  email: string
  role: "admin" | "faculty" | "student"
  rollNo?: string
  department?: string
  createdAt: string
}

export interface Department {
  id: string
  name: string
  code: string
  hod: string
  totalStudents: number
  totalFaculty: number
}

export interface Class {
  id: string
  name: string
  code: string
  department: string
  facultyId: string
  facultyName: string
  schedule: string
  semester: string
  totalStudents: number
  studentIds: string[]
}

export interface AttendanceRecord {
  id: string
  studentId: string
  studentName: string
  rollNo: string
  classId: string
  className: string
  date: string
  time: string
  status: "Present" | "Absent" | "Late"
  verificationMethod: "Face + GPS" | "Face Only" | "GPS Only"
  location?: string
}

// Mock Users
export const mockUsers: User[] = [
  {
    id: "admin-1",
    name: "Admin User",
    email: "admin@attendx.edu",
    role: "admin",
    createdAt: "2024-01-01",
  },
  {
    id: "faculty-1",
    name: "Dr. Sarah Johnson",
    email: "sarah.johnson@attendx.edu",
    role: "faculty",
    department: "Computer Science",
    createdAt: "2024-01-15",
  },
  {
    id: "faculty-2",
    name: "Prof. Michael Chen",
    email: "michael.chen@attendx.edu",
    role: "faculty",
    department: "Electrical Engineering",
    createdAt: "2024-01-15",
  },
  {
    id: "faculty-3",
    name: "Dr. Emily Davis",
    email: "emily.davis@attendx.edu",
    role: "faculty",
    department: "Computer Science",
    createdAt: "2024-01-15",
  },
  {
    id: "student-1",
    name: "John Doe",
    email: "john.doe@attendx.edu",
    role: "student",
    rollNo: "2024CS001",
    department: "Computer Science",
    createdAt: "2024-02-01",
  },
  {
    id: "student-2",
    name: "Jane Smith",
    email: "jane.smith@attendx.edu",
    role: "student",
    rollNo: "2024CS002",
    department: "Computer Science",
    createdAt: "2024-02-01",
  },
  {
    id: "student-3",
    name: "Mike Johnson",
    email: "mike.johnson@attendx.edu",
    role: "student",
    rollNo: "2024EE001",
    department: "Electrical Engineering",
    createdAt: "2024-02-01",
  },
]

// Mock Departments
export const mockDepartments: Department[] = [
  {
    id: "dept-1",
    name: "Computer Science",
    code: "CS",
    hod: "Dr. Sarah Johnson",
    totalStudents: 120,
    totalFaculty: 8,
  },
  {
    id: "dept-2",
    name: "Electrical Engineering",
    code: "EE",
    hod: "Prof. Michael Chen",
    totalStudents: 80,
    totalFaculty: 6,
  },
  {
    id: "dept-3",
    name: "Mechanical Engineering",
    code: "ME",
    hod: "Dr. Robert Williams",
    totalStudents: 90,
    totalFaculty: 7,
  },
  {
    id: "dept-4",
    name: "Civil Engineering",
    code: "CE",
    hod: "Prof. Lisa Anderson",
    totalStudents: 70,
    totalFaculty: 5,
  },
]

// Mock Classes
export const mockClasses: Class[] = [
  {
    id: "class-1",
    name: "Computer Science 101",
    code: "CS101",
    department: "Computer Science",
    facultyId: "faculty-1",
    facultyName: "Dr. Sarah Johnson",
    schedule: "Mon, Wed, Fri - 9:00 AM",
    semester: "Spring 2024",
    totalStudents: 45,
    studentIds: ["student-1", "student-2"],
  },
  {
    id: "class-2",
    name: "Data Structures",
    code: "CS201",
    department: "Computer Science",
    facultyId: "faculty-1",
    facultyName: "Dr. Sarah Johnson",
    schedule: "Tue, Thu - 11:00 AM",
    semester: "Spring 2024",
    totalStudents: 40,
    studentIds: ["student-1", "student-2"],
  },
  {
    id: "class-3",
    name: "Database Systems",
    code: "CS301",
    department: "Computer Science",
    facultyId: "faculty-3",
    facultyName: "Dr. Emily Davis",
    schedule: "Mon, Wed - 2:00 PM",
    semester: "Spring 2024",
    totalStudents: 38,
    studentIds: ["student-1"],
  },
  {
    id: "class-4",
    name: "Circuit Analysis",
    code: "EE101",
    department: "Electrical Engineering",
    facultyId: "faculty-2",
    facultyName: "Prof. Michael Chen",
    schedule: "Tue, Thu - 10:00 AM",
    semester: "Spring 2024",
    totalStudents: 35,
    studentIds: ["student-3"],
  },
]

// Mock Attendance Records
export const mockAttendanceRecords: AttendanceRecord[] = [
  {
    id: "att-1",
    studentId: "student-1",
    studentName: "John Doe",
    rollNo: "2024CS001",
    classId: "class-1",
    className: "Computer Science 101",
    date: "2024-03-10",
    time: "09:15 AM",
    status: "Present",
    verificationMethod: "Face + GPS",
    location: "Building A, Room 101",
  },
  {
    id: "att-2",
    studentId: "student-2",
    studentName: "Jane Smith",
    rollNo: "2024CS002",
    classId: "class-1",
    className: "Computer Science 101",
    date: "2024-03-10",
    time: "09:12 AM",
    status: "Present",
    verificationMethod: "Face + GPS",
    location: "Building A, Room 101",
  },
  {
    id: "att-3",
    studentId: "student-3",
    studentName: "Mike Johnson",
    rollNo: "2024EE001",
    classId: "class-4",
    className: "Circuit Analysis",
    date: "2024-03-10",
    time: "10:18 AM",
    status: "Present",
    verificationMethod: "Face + GPS",
    location: "Building B, Room 205",
  },
]

// Helper functions
export function getUsersByRole(role: "admin" | "faculty" | "student") {
  return mockUsers.filter((user) => user.role === role)
}

export function getClassesByFaculty(facultyId: string) {
  return mockClasses.filter((cls) => cls.facultyId === facultyId)
}

export function getClassesByStudent(studentId: string) {
  return mockClasses.filter((cls) => cls.studentIds.includes(studentId))
}

export function getAttendanceByClass(classId: string) {
  return mockAttendanceRecords.filter((record) => record.classId === classId)
}

export function getAttendanceByStudent(studentId: string) {
  return mockAttendanceRecords.filter((record) => record.studentId === studentId)
}
