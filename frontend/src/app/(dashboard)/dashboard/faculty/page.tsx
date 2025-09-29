"use client"

import { useQuery } from "@tanstack/react-query"
import { motion } from "framer-motion"
import { 
  Users, 
  BookOpen, 
  Calendar, 
  TrendingUp,
  Clock,
  CheckCircle,
  XCircle
} from "lucide-react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { facultyApi } from "@/lib/api"
import { FadeIn, StaggerContainer } from "@/components/animations"
import { PageLoader, Skeleton } from "@/components/common/Loader"
import { formatDate } from "@/lib/utils"
import Link from "next/link"

export default function FacultyDashboard() {
  const { data: profile, isLoading: profileLoading } = useQuery({
    queryKey: ["faculty", "profile"],
    queryFn: () => facultyApi.getProfile(),
  })

  const { data: courses, isLoading: coursesLoading } = useQuery({
    queryKey: ["faculty", "courses"],
    queryFn: () => facultyApi.getCourses(),
  })

  if (profileLoading || coursesLoading) {
    return <PageLoader message="Loading dashboard..." />
  }

  const totalStudents = courses?.data?.reduce((acc, course) => acc + (course.enrolledStudents || 0), 0) || 0
  const totalClasses = courses?.data?.length || 0

  return (
    <div className="space-y-6">
      {/* Welcome Section */}
      <FadeIn>
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold">
              Welcome, Prof. {profile?.data?.lastName}!
            </h1>
            <p className="text-muted-foreground">
              Manage your courses and track student attendance.
            </p>
          </div>
          <Badge variant="secondary" className="text-sm">
            {profile?.data?.department} • {profile?.data?.designation}
          </Badge>
        </div>
      </FadeIn>

      {/* Stats Cards */}
      <StaggerContainer className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <FadeIn delay={0.1}>
          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Total Courses</CardTitle>
              <BookOpen className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">
                {totalClasses}
              </div>
              <p className="text-xs text-muted-foreground">
                This semester
              </p>
            </CardContent>
          </Card>
        </FadeIn>

        <FadeIn delay={0.2}>
          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Total Students</CardTitle>
              <Users className="h-4 w-4 text-blue-500" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-blue-600">
                {totalStudents}
              </div>
              <p className="text-xs text-muted-foreground">
                Across all courses
              </p>
            </CardContent>
          </Card>
        </FadeIn>

        <FadeIn delay={0.3}>
          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Today&apos;s Classes</CardTitle>
              <Calendar className="h-4 w-4 text-green-500" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-green-600">
              {courses?.data?.filter(course => {
                const today = new Date().toLocaleDateString('en-US', { weekday: 'long' }).toLowerCase()
                return course.schedule.day === today
              }).length || 0}
              </div>
              <p className="text-xs text-muted-foreground">
                Scheduled for today
              </p>
            </CardContent>
          </Card>
        </FadeIn>

        <FadeIn delay={0.4}>
          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Avg. Attendance</CardTitle>
              <TrendingUp className="h-4 w-4 text-purple-500" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-purple-600">
                85%
              </div>
              <p className="text-xs text-muted-foreground">
                This semester
              </p>
            </CardContent>
          </Card>
        </FadeIn>
      </StaggerContainer>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* My Courses */}
        <FadeIn delay={0.5}>
          <Card>
            <CardHeader>
              <CardTitle>My Courses</CardTitle>
              <CardDescription>
                Courses you&apos;re teaching this semester
              </CardDescription>
            </CardHeader>
            <CardContent>
              {coursesLoading ? (
                <div className="space-y-3">
                  {Array.from({ length: 3 }).map((_, i) => (
                    <Skeleton key={i} className="h-20 w-full" />
                  ))}
                </div>
              ) : courses?.data?.length ? (
                <div className="space-y-4">
                  {courses.data.map((course) => (
                    <div key={course.id} className="p-4 border rounded-lg">
                      <div className="flex items-center justify-between mb-2">
                        <h4 className="font-medium">{course.name}</h4>
                        <Badge variant="outline">{course.credits} credits</Badge>
                      </div>
                      <p className="text-sm text-muted-foreground mb-2">
                        {course.code} • {course.department}
                      </p>
                      <div className="flex items-center justify-between">
                        <p className="text-xs text-muted-foreground">
                          {course.schedule.day} {course.schedule.startTime} - {course.schedule.endTime}
                        </p>
                        <Link href={`/courses/${course.id}`}>
                          <Button variant="outline" size="sm">
                            View Details
                          </Button>
                        </Link>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <p className="text-muted-foreground text-center py-4">
                  No courses assigned
                </p>
              )}
            </CardContent>
          </Card>
        </FadeIn>

        {/* Quick Actions */}
        <FadeIn delay={0.6}>
          <Card>
            <CardHeader>
              <CardTitle>Quick Actions</CardTitle>
              <CardDescription>
                Common tasks and shortcuts
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <Link href="/attendance" className="block">
                <Button className="w-full justify-start" variant="outline">
                  <Calendar className="mr-2 h-4 w-4" />
                  Mark Attendance
                </Button>
              </Link>
              
              <Link href="/courses" className="block">
                <Button className="w-full justify-start" variant="outline">
                  <BookOpen className="mr-2 h-4 w-4" />
                  Manage Courses
                </Button>
              </Link>
              
              <Link href="/reports" className="block">
                <Button className="w-full justify-start" variant="outline">
                  <TrendingUp className="mr-2 h-4 w-4" />
                  View Reports
                </Button>
              </Link>
              
              <Link href="/students" className="block">
                <Button className="w-full justify-start" variant="outline">
                  <Users className="mr-2 h-4 w-4" />
                  Student Management
                </Button>
              </Link>
            </CardContent>
          </Card>
        </FadeIn>
      </div>

      {/* Recent Activity */}
      <FadeIn delay={0.7}>
        <Card>
          <CardHeader>
            <CardTitle>Recent Activity</CardTitle>
            <CardDescription>
              Your recent actions and updates
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div className="flex items-center space-x-4 p-3 border rounded-lg">
                <CheckCircle className="h-5 w-5 text-green-500" />
                <div className="flex-1">
                  <p className="text-sm font-medium">Attendance marked for Data Structures</p>
                  <p className="text-xs text-muted-foreground">2 hours ago</p>
                </div>
              </div>
              
              <div className="flex items-center space-x-4 p-3 border rounded-lg">
                <BookOpen className="h-5 w-5 text-blue-500" />
                <div className="flex-1">
                  <p className="text-sm font-medium">Course materials updated for Algorithms</p>
                  <p className="text-xs text-muted-foreground">1 day ago</p>
                </div>
              </div>
              
              <div className="flex items-center space-x-4 p-3 border rounded-lg">
                <Users className="h-5 w-5 text-purple-500" />
                <div className="flex-1">
                  <p className="text-sm font-medium">New student enrolled in Database Systems</p>
                  <p className="text-xs text-muted-foreground">2 days ago</p>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      </FadeIn>
    </div>
  )
}
