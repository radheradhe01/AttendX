# Smart College ERP - Frontend

A modern, responsive Next.js 15 frontend for the Smart College ERP system with advanced attendance tracking, student management, and analytics.

## 🚀 Features

### Core Functionality
- **Role-based Authentication** - Student, Faculty, and Admin dashboards
- **Smart Attendance System** - GPS + Face recognition attendance marking
- **Student Management** - Comprehensive student profiles and academic tracking
- **Course Management** - Course creation, scheduling, and enrollment
- **Analytics & Reports** - Detailed reports with charts and data visualization
- **Real-time Updates** - Live data synchronization with backend APIs

### Technical Features
- **Next.js 15** with App Router
- **TypeScript** for type safety
- **Tailwind CSS** for styling
- **shadcn/ui** for modern UI components
- **Framer Motion** for smooth animations
- **React Query** for data fetching and caching
- **NextAuth.js v5** for authentication
- **React Hook Form + Zod** for form validation
- **Recharts** for data visualization
- **Responsive Design** - Works on all devices

## 📁 Project Structure

```
frontend/
├── app/                          # Next.js App Router
│   ├── (auth)/                   # Auth layout group
│   │   ├── layout.tsx           # Auth layout
│   │   └── auth/                # Auth pages
│   │       ├── login/page.tsx   # Login page
│   │       └── register/page.tsx # Registration page
│   ├── (dashboard)/             # Dashboard layout group
│   │   ├── layout.tsx           # Dashboard layout
│   │   ├── dashboard/           # Dashboard pages
│   │   │   ├── student/page.tsx # Student dashboard
│   │   │   ├── faculty/page.tsx # Faculty dashboard
│   │   │   └── admin/page.tsx   # Admin dashboard
│   │   ├── attendance/          # Attendance pages
│   │   ├── courses/             # Course management
│   │   └── reports/             # Reports & analytics
│   ├── api/                     # API routes
│   │   └── auth/[...nextauth]/  # NextAuth API
│   ├── layout.tsx               # Root layout
│   └── page.tsx                 # Landing page
├── components/                   # Reusable components
│   ├── ui/                      # shadcn/ui components
│   ├── common/                  # Common components
│   │   ├── Navbar.tsx           # Navigation bar
│   │   ├── Sidebar.tsx          # Sidebar navigation
│   │   └── Loader.tsx           # Loading components
│   ├── charts/                  # Chart components
│   │   ├── AttendanceChart.tsx  # Line chart for attendance
│   │   ├── AttendancePieChart.tsx # Pie chart for attendance
│   │   └── PerformanceChart.tsx # Bar chart for performance
│   ├── forms/                   # Form components
│   │   ├── LoginForm.tsx        # Login form
│   │   └── AttendanceForm.tsx   # Attendance marking form
│   └── animations/              # Animation components
│       ├── PageTransition.tsx   # Page transition animations
│       ├── FadeIn.tsx           # Fade in animation
│       └── StaggerContainer.tsx # Staggered animations
├── hooks/                       # Custom React hooks
│   ├── useAuth.ts               # Authentication hook
│   └── useAttendance.ts         # Attendance-related hooks
├── lib/                         # Utility libraries
│   ├── api.ts                   # API client with Axios
│   ├── auth.ts                  # NextAuth configuration
│   ├── validators.ts            # Zod validation schemas
│   └── utils.ts                 # General utilities
├── styles/                      # Global styles
│   └── globals.css              # Tailwind CSS configuration
└── public/                      # Static assets
```

## 🛠️ Installation & Setup

### Prerequisites
- Node.js 18+ 
- npm or yarn
- Backend API running on `http://localhost:5000`

### Installation

1. **Install dependencies:**
```bash
npm install
```

2. **Environment Setup:**
Create a `.env.local` file in the root directory:
```env
# NextAuth Configuration
NEXTAUTH_URL=http://localhost:3000
NEXTAUTH_SECRET=your-secret-key-here

# API Configuration
NEXT_PUBLIC_API_URL=http://localhost:5000/api

# Other environment variables
NEXT_PUBLIC_APP_NAME=Smart College ERP
NEXT_PUBLIC_APP_VERSION=1.0.0
```

3. **Run the development server:**
```bash
npm run dev
```

4. **Open your browser:**
Navigate to `http://localhost:3000`

## 🎨 UI Components

### shadcn/ui Components
The project uses shadcn/ui for consistent, accessible components:
- Button, Input, Card, Badge
- Table, Avatar, Dialog
- Form components with validation
- Toast notifications

### Custom Components
- **Navbar** - Responsive navigation with user menu
- **Sidebar** - Collapsible sidebar with role-based navigation
- **Charts** - Recharts-based data visualization
- **Forms** - React Hook Form with Zod validation
- **Animations** - Framer Motion animations

## 🔐 Authentication

### NextAuth.js v5 Setup
- JWT-based sessions
- Credentials provider
- Role-based access control
- Automatic token refresh

### User Roles
- **Student** - View courses, mark attendance, view grades
- **Faculty** - Manage courses, mark attendance, view reports
- **Admin** - Full system access, user management, analytics

## 📊 Data Management

### React Query Integration
- Automatic caching and background updates
- Optimistic updates
- Error handling and retry logic
- Loading states

### API Integration
- Axios-based HTTP client
- Automatic token attachment
- Request/response interceptors
- Error handling with toast notifications

## 🎭 Animations

### Framer Motion
- Page transitions
- Staggered animations
- Hover effects
- Loading animations

### GSAP (Optional)
- Page loading animations
- Scroll-triggered animations
- Complex timeline animations

## 📱 Responsive Design

### Mobile-First Approach
- Tailwind CSS responsive utilities
- Mobile-optimized navigation
- Touch-friendly interactions
- Progressive Web App features

### Breakpoints
- Mobile: < 768px
- Tablet: 768px - 1024px
- Desktop: > 1024px

## 🧪 Form Validation

### React Hook Form + Zod
- Type-safe form validation
- Real-time validation feedback
- Custom validation rules
- Error message handling

### Example Form Schemas
```typescript
// Login form
const loginSchema = z.object({
  email: z.string().email("Invalid email address"),
  password: z.string().min(6, "Password must be at least 6 characters"),
})

// Attendance form
const attendanceCheckInSchema = z.object({
  courseId: z.string().min(1, "Course is required"),
  latitude: z.number().min(-90).max(90, "Invalid latitude"),
  longitude: z.number().min(-180).max(180, "Invalid longitude"),
  faceImage: z.string().min(1, "Face image is required"),
})
```

## 📈 Charts & Analytics

### Recharts Integration
- Line charts for attendance trends
- Pie charts for attendance distribution
- Bar charts for performance comparison
- Responsive chart components

### Chart Types
- **AttendanceChart** - Daily attendance trends
- **AttendancePieChart** - Attendance distribution
- **PerformanceChart** - Academic performance metrics

## 🚀 Deployment

### Build for Production
```bash
npm run build
npm start
```

### Environment Variables for Production
```env
NEXTAUTH_URL=https://your-domain.com
NEXTAUTH_SECRET=your-production-secret
NEXT_PUBLIC_API_URL=https://your-api-domain.com/api
```

### Deployment Platforms
- **Vercel** (Recommended for Next.js)
- **Netlify**
- **AWS Amplify**
- **Docker** containerization

## 🔧 Development

### Available Scripts
```bash
npm run dev          # Start development server
npm run build        # Build for production
npm run start        # Start production server
npm run lint         # Run ESLint
```

### Code Quality
- TypeScript for type safety
- ESLint for code linting
- Prettier for code formatting
- Husky for git hooks

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🆘 Support

For support and questions:
- Create an issue in the repository
- Check the documentation
- Contact the development team

---

**Built with ❤️ using Next.js, TypeScript, and modern web technologies.**