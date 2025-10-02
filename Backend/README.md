# AttendEase Backend - Smart College ERP System

A comprehensive, production-ready backend for a Smart College ERP system with integrated Attendance module, built with Node.js, Express, TypeScript, and MongoDB.

## 🚀 Features

### Core ERP Functionality
- **User Management**: Student, Faculty, and Admin roles with JWT authentication
- **Student Management**: Complete student profiles, enrollment, academic records
- **Faculty Management**: Faculty profiles, course assignments, qualifications
- **Course Management**: Course creation, enrollment, scheduling
- **Attendance System**: GPS-validated attendance with face recognition integration
- **Reports & Analytics**: Comprehensive reporting with CSV/PDF export
- **Fee Management**: Fee tracking and collection reports

### Advanced Features
- **GPS Geofencing**: Campus boundary validation for attendance
- **Face Recognition**: Integration with AI microservice for attendance verification
- **Role-Based Access Control**: Secure API endpoints based on user roles
- **Real-time Notifications**: Email alerts and system notifications
- **Data Export**: CSV and PDF report generation
- **Comprehensive Logging**: Winston-based logging system
- **Rate Limiting**: API protection against abuse
- **Security Headers**: Helmet.js security middleware

## 📁 Project Structure

```
Backend/
├── package.json                 # Dependencies and scripts
├── tsconfig.json               # TypeScript configuration
├── jest.config.js              # Jest testing configuration
├── .env                        # Environment variables (create this)
├── .gitignore                  # Git ignore rules
├── README.md                   # This file
│
├── src/                        # Source code
│   ├── config/                 # Configuration files
│   │   ├── db.ts              # MongoDB connection
│   │   ├── cloud.ts           # Cloud/AI service configs
│   │   └── logger.ts          # Winston logger setup
│   │
│   ├── middleware/             # Express middleware
│   │   ├── authMiddleware.ts   # JWT authentication & RBAC
│   │   ├── errorHandler.ts     # Global error handling
│   │   └── uploadMiddleware.ts # File upload handling
│   │
│   ├── models/                 # MongoDB schemas
│   │   ├── User.ts            # User model (base for all roles)
│   │   ├── Student.ts         # Student profile model
│   │   ├── Faculty.ts         # Faculty profile model
│   │   ├── Course.ts          # Course/subject model
│   │   ├── Attendance.ts      # Attendance records model
│   │   └── Report.ts          # Generated reports model
│   │
│   ├── services/               # Business logic
│   │   ├── faceService.ts     # Face recognition API integration
│   │   ├── gpsService.ts      # GPS geofencing validation
│   │   └── erpService.ts      # Core ERP business logic
│   │
│   ├── controllers/            # API request handlers
│   │   ├── authController.ts   # Authentication endpoints
│   │   ├── studentController.ts # Student management
│   │   ├── facultyController.ts # Faculty management
│   │   ├── courseController.ts  # Course management
│   │   ├── attendanceController.ts # Attendance management
│   │   └── reportController.ts   # Report generation
│   │
│   ├── routes/                 # API route definitions
│   │   ├── authRoutes.ts       # Authentication routes
│   │   ├── studentRoutes.ts    # Student routes
│   │   ├── facultyRoutes.ts    # Faculty routes
│   │   ├── courseRoutes.ts     # Course routes
│   │   ├── attendanceRoutes.ts # Attendance routes
│   │   └── reportRoutes.ts     # Report routes
│   │
│   ├── utils/                  # Utility functions
│   │   ├── jwtUtils.ts         # JWT token management
│   │   ├── validators.ts       # Input validation rules
│   │   └── emailUtils.ts       # Email service utilities
│   │
│   ├── app.ts                  # Express app configuration
│   └── server.ts               # Server entry point
│
└── tests/                      # Test files
    ├── setup.ts               # Test setup configuration
    └── simple.test.ts         # Integration tests
```

## 🛠️ Installation & Setup

### Prerequisites
- Node.js (v16 or higher)
- MongoDB (v4.4 or higher)
- npm or yarn

### 1. Install Dependencies
```bash
cd Backend
npm install
```

### 2. Environment Configuration
Create a `.env` file in the Backend directory:

```env
# Server Configuration
NODE_ENV=development
PORT=5000
HOST=localhost

# Database
MONGODB_URI=mongodb://localhost:27017/attendex_erp

# JWT Configuration
JWT_SECRET=your-super-secret-jwt-key-here
JWT_EXPIRE=7d
JWT_REFRESH_SECRET=your-refresh-secret-key-here
JWT_REFRESH_EXPIRE=30d

# Face Recognition Service
FACE_RECOGNITION_SERVICE_URL=http://localhost:8000/api/face-recognition
FACE_RECOGNITION_API_KEY=your-face-recognition-api-key

# Email Configuration (for notifications)
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your-email@gmail.com
SMTP_PASS=your-app-password

# GPS Configuration (Campus boundaries)
CAMPUS_LATITUDE=12.9716
CAMPUS_LONGITUDE=77.5946
CAMPUS_RADIUS=1000

# File Upload
MAX_FILE_SIZE=5242880
UPLOAD_PATH=./uploads
```

### 3. Build and Run
```bash
# Build TypeScript
npm run build

# Start production server
npm start

# Or run in development mode
npm run dev
```

### 4. Run Tests
```bash
npm test
```

## 🔑 API Keys & External Services

### Face Recognition Service
The system integrates with a Python-based face recognition microservice. You need to:

1. **Set up the face recognition service** (separate Python service)
2. **Get API key** from your face recognition provider
3. **Configure the service URL** in `.env` file

**Example Face Recognition Service Setup:**
```python
# This would be a separate Python service
# Endpoint: POST /api/face-recognition
# Request: { "image": "base64_encoded_image", "student_id": "STU001" }
# Response: { "verified": true, "confidence": 95.5 }
```

### Email Service
For email notifications, configure SMTP settings:

**Gmail Setup:**
1. Enable 2-factor authentication
2. Generate an App Password
3. Use the App Password in `SMTP_PASS`

**Other SMTP Providers:**
- Update `SMTP_HOST`, `SMTP_PORT`, `SMTP_USER`, `SMTP_PASS` accordingly

### GPS Configuration
Set your campus coordinates in the `.env` file:
- `CAMPUS_LATITUDE`: Your campus latitude
- `CAMPUS_LONGITUDE`: Your campus longitude  
- `CAMPUS_RADIUS`: Radius in meters for geofencing

## 📚 API Documentation

### Authentication Endpoints

#### Register User
```http
POST /api/auth/register
Content-Type: application/json

{
  "email": "user@example.com",
  "password": "SecurePass123",
  "role": "student"
}
```

#### Login
```http
POST /api/auth/login
Content-Type: application/json

{
  "email": "user@example.com",
  "password": "SecurePass123"
}
```

#### Get Profile
```http
GET /api/auth/me
Authorization: Bearer <jwt_token>
```

### Student Management

#### Create Student
```http
POST /api/students
Authorization: Bearer <admin_token>
Content-Type: application/json

{
  "email": "student@example.com",
  "studentId": "STU001",
  "firstName": "John",
  "lastName": "Doe",
  "phone": "+1234567890",
  "dateOfBirth": "2000-01-01",
  "gender": "male",
  "address": {
    "street": "123 Main St",
    "city": "City",
    "state": "State",
    "zipCode": "12345",
    "country": "Country"
  },
  "emergencyContact": {
    "name": "Jane Doe",
    "relationship": "Mother",
    "phone": "+1234567891"
  },
  "academicInfo": {
    "admissionDate": "2023-01-01",
    "currentSemester": 1,
    "currentYear": 1,
    "department": "Computer Science",
    "course": "B.Tech",
    "rollNumber": "CS001",
    "batch": "2023"
  }
}
```

### Attendance Management

#### Mark Attendance
```http
POST /api/attendance/mark
Authorization: Bearer <student_token>
Content-Type: application/json

{
  "studentId": "STU001",
  "courseId": "COURSE_ID",
  "gpsLocation": {
    "latitude": 12.9716,
    "longitude": 77.5946,
    "accuracy": 10
  },
  "faceImage": "base64_encoded_image_data",
  "deviceInfo": {
    "deviceId": "device123",
    "platform": "Android",
    "appVersion": "1.0.0",
    "ipAddress": "192.168.1.1"
  }
}
```

#### Get Student Attendance
```http
GET /api/attendance/student/:studentId?startDate=2023-01-01&endDate=2023-12-31
Authorization: Bearer <token>
```

### Course Management

#### Create Course
```http
POST /api/courses
Authorization: Bearer <admin_token>
Content-Type: application/json

{
  "courseCode": "CS101",
  "title": "Introduction to Programming",
  "description": "Basic programming concepts",
  "department": "Computer Science",
  "credits": 3,
  "faculty": "FACULTY_ID",
  "schedule": {
    "days": ["Monday", "Wednesday", "Friday"],
    "time": "10:00-11:00",
    "room": "Lab 101"
  }
}
```

### Report Generation

#### Generate Attendance Report
```http
POST /api/reports/generate
Authorization: Bearer <admin_token>
Content-Type: application/json

{
  "reportType": "attendance",
  "parameters": {
    "startDate": "2023-01-01",
    "endDate": "2023-12-31",
    "department": "Computer Science",
    "format": "pdf"
  }
}
```

## 🔒 Security Features

- **JWT Authentication**: Secure token-based authentication
- **Role-Based Access Control**: Different access levels for students, faculty, and admins
- **Password Hashing**: bcrypt for secure password storage
- **Rate Limiting**: API protection against abuse
- **Input Validation**: Comprehensive validation using express-validator
- **Security Headers**: Helmet.js for security headers
- **CORS Configuration**: Configurable cross-origin resource sharing

## 🧪 Testing

The project includes comprehensive testing:

```bash
# Run all tests
npm test

# Run tests in watch mode
npm run test:watch

# Run specific test file
npx jest tests/simple.test.ts
```

**Test Coverage:**
- Authentication flow (register, login, protected routes)
- Error handling
- Input validation
- Database operations

## 📊 Monitoring & Logging

- **Winston Logger**: Comprehensive logging system
- **Request Logging**: Morgan HTTP request logger
- **Error Tracking**: Detailed error logging with stack traces
- **Performance Monitoring**: Request timing and database query logging

## 🚀 Production Deployment

### Environment Setup
1. Set `NODE_ENV=production`
2. Use a production MongoDB instance
3. Configure proper JWT secrets
4. Set up SSL certificates
5. Configure reverse proxy (nginx)

### Docker Deployment (Optional)
```dockerfile
FROM node:16-alpine
WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production
COPY dist ./dist
EXPOSE 5000
CMD ["npm", "start"]
```

### Performance Optimization
- Enable compression middleware
- Use MongoDB connection pooling
- Implement caching for frequently accessed data
- Set up CDN for static assets

## 🔧 Development

### Available Scripts
```bash
npm run dev          # Start development server with hot reload
npm run build        # Build TypeScript to JavaScript
npm start           # Start production server
npm test            # Run tests
npm run lint        # Run ESLint
npm run lint:fix    # Fix ESLint errors
```

### Code Quality
- **TypeScript**: Full type safety
- **ESLint**: Code linting and formatting
- **Prettier**: Code formatting (configure as needed)
- **Jest**: Unit and integration testing

## 📝 API Response Format

All API responses follow a consistent format:

**Success Response:**
```json
{
  "success": true,
  "message": "Operation completed successfully",
  "data": { ... },
  "timestamp": "2023-01-01T00:00:00.000Z"
}
```

**Error Response:**
```json
{
  "success": false,
  "message": "Error description",
  "code": "ERROR_CODE",
  "details": { ... },
  "timestamp": "2023-01-01T00:00:00.000Z"
}
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## 📄 License

This project is licensed under the ISC License.

## 🆘 Support

For support and questions:
- Check the API documentation above
- Review the test files for usage examples
- Check the logs for detailed error information
- Ensure all environment variables are properly configured

---

**AttendEase Backend** - A comprehensive ERP solution for modern educational institutions.