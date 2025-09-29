"use client"

import { useQuery } from "@tanstack/react-query"
import { motion } from "framer-motion"
import { 
  Users, 
  BookOpen, 
  GraduationCap, 
  TrendingUp,
  BarChart3,
  Calendar,
  DollarSign,
  Activity
} from "lucide-react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { AttendanceChart, AttendancePieChart } from "@/components/charts"
import { FadeIn, StaggerContainer } from "@/components/animations"
import { PageLoader, Skeleton } from "@/components/common/Loader"
import Link from "next/link"

// Mock data for charts
const attendanceData = [
  { date: "2024-01-01", present: 45, absent: 5, late: 3, total: 53 },
  { date: "2024-01-02", present: 48, absent: 3, late: 2, total: 53 },
  { date: "2024-01-03", present: 42, absent: 8, late: 3, total: 53 },
  { date: "2024-01-04", present: 50, absent: 2, late: 1, total: 53 },
  { date: "2024-01-05", present: 46, absent: 4, late: 3, total: 53 },
]

const attendancePieData = [
  { name: "Present", value: 85, color: "#22c55e" },
  { name: "Absent", value: 10, color: "#ef4444" },
  { name: "Late", value: 5, color: "#f59e0b" },
]

export default function AdminDashboard() {
  const { data: stats, isLoading: statsLoading } = useQuery({
    queryKey: ["admin", "stats"],
    queryFn: async () => {
      // Mock API call - replace with actual API
      return {
        success: true,
        data: {
          totalStudents: 1250,
          totalFaculty: 85,
          totalCourses: 120,
          totalDepartments: 8,
          attendanceRate: 92.5,
          activeUsers: 1100,
          pendingApplications: 25,
          monthlyRevenue: 45000,
        }
      }
    },
  })

  if (statsLoading) {
    return <PageLoader message="Loading admin dashboard..." />
  }

  return (
    <div className="space-y-6">
      {/* Welcome Section */}
      <FadeIn>
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold">Admin Dashboard</h1>
            <p className="text-muted-foreground">
              Overview of your institution&apos;s performance and statistics.
            </p>
          </div>
          <Badge variant="secondary" className="text-sm">
            System Administrator
          </Badge>
        </div>
      </FadeIn>

      {/* Key Metrics */}
      <StaggerContainer className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <FadeIn delay={0.1}>
          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Total Students</CardTitle>
              <Users className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">
                {stats?.data?.totalStudents?.toLocaleString() || 0}
              </div>
              <p className="text-xs text-muted-foreground">
                +12% from last month
              </p>
            </CardContent>
          </Card>
        </FadeIn>

        <FadeIn delay={0.2}>
          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Total Faculty</CardTitle>
              <GraduationCap className="h-4 w-4 text-blue-500" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-blue-600">
                {stats?.data?.totalFaculty || 0}
              </div>
              <p className="text-xs text-muted-foreground">
                +3 new this month
              </p>
            </CardContent>
          </Card>
        </FadeIn>

        <FadeIn delay={0.3}>
          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Active Courses</CardTitle>
              <BookOpen className="h-4 w-4 text-green-500" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-green-600">
                {stats?.data?.totalCourses || 0}
              </div>
              <p className="text-xs text-muted-foreground">
                This semester
              </p>
            </CardContent>
          </Card>
        </FadeIn>

        <FadeIn delay={0.4}>
          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Attendance Rate</CardTitle>
              <TrendingUp className="h-4 w-4 text-purple-500" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-purple-600">
                {stats?.data?.attendanceRate || 0}%
              </div>
              <p className="text-xs text-muted-foreground">
                +2.5% from last month
              </p>
            </CardContent>
          </Card>
        </FadeIn>
      </StaggerContainer>

      {/* Charts Section */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <FadeIn delay={0.5}>
          <AttendanceChart 
            data={attendanceData}
            title="Weekly Attendance Trend"
            description="Daily attendance overview for the past week"
          />
        </FadeIn>

        <FadeIn delay={0.6}>
          <AttendancePieChart 
            data={attendancePieData}
            title="Overall Attendance Distribution"
            description="Current semester attendance breakdown"
          />
        </FadeIn>
      </div>

      {/* Management Cards */}
      <StaggerContainer className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        <FadeIn delay={0.7}>
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center space-x-2">
                <Users className="h-5 w-5" />
                <span>Student Management</span>
              </CardTitle>
              <CardDescription>
                Manage student records, enrollment, and academic progress
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex items-center justify-between">
                <span className="text-sm text-muted-foreground">Total Students</span>
                <span className="font-medium">{stats?.data?.totalStudents?.toLocaleString()}</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm text-muted-foreground">Active Users</span>
                <span className="font-medium">{stats?.data?.activeUsers?.toLocaleString()}</span>
              </div>
              <Link href="/students">
                <Button className="w-full">Manage Students</Button>
              </Link>
            </CardContent>
          </Card>
        </FadeIn>

        <FadeIn delay={0.8}>
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center space-x-2">
                <GraduationCap className="h-5 w-5" />
                <span>Faculty Management</span>
              </CardTitle>
              <CardDescription>
                Manage faculty profiles, course assignments, and schedules
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex items-center justify-between">
                <span className="text-sm text-muted-foreground">Total Faculty</span>
                <span className="font-medium">{stats?.data?.totalFaculty}</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm text-muted-foreground">Departments</span>
                <span className="font-medium">{stats?.data?.totalDepartments}</span>
              </div>
              <Link href="/faculty">
                <Button className="w-full">Manage Faculty</Button>
              </Link>
            </CardContent>
          </Card>
        </FadeIn>

        <FadeIn delay={0.9}>
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center space-x-2">
                <BookOpen className="h-5 w-5" />
                <span>Course Management</span>
              </CardTitle>
              <CardDescription>
                Create and manage courses, schedules, and curriculum
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex items-center justify-between">
                <span className="text-sm text-muted-foreground">Active Courses</span>
                <span className="font-medium">{stats?.data?.totalCourses}</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm text-muted-foreground">Departments</span>
                <span className="font-medium">{stats?.data?.totalDepartments}</span>
              </div>
              <Link href="/courses">
                <Button className="w-full">Manage Courses</Button>
              </Link>
            </CardContent>
          </Card>
        </FadeIn>
      </StaggerContainer>

      {/* Quick Actions */}
      <FadeIn delay={1.0}>
        <Card>
          <CardHeader>
            <CardTitle>Quick Actions</CardTitle>
            <CardDescription>
              Common administrative tasks and shortcuts
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <Link href="/reports">
                <Button variant="outline" className="w-full h-20 flex flex-col space-y-2">
                  <BarChart3 className="h-6 w-6" />
                  <span>Reports</span>
                </Button>
              </Link>
              
              <Link href="/attendance">
                <Button variant="outline" className="w-full h-20 flex flex-col space-y-2">
                  <Calendar className="h-6 w-6" />
                  <span>Attendance</span>
                </Button>
              </Link>
              
              <Link href="/settings">
                <Button variant="outline" className="w-full h-20 flex flex-col space-y-2">
                  <Activity className="h-6 w-6" />
                  <span>Settings</span>
                </Button>
              </Link>
              
              <Link href="/analytics">
                <Button variant="outline" className="w-full h-20 flex flex-col space-y-2">
                  <TrendingUp className="h-6 w-6" />
                  <span>Analytics</span>
                </Button>
              </Link>
            </div>
          </CardContent>
        </Card>
      </FadeIn>
    </div>
  )
}
