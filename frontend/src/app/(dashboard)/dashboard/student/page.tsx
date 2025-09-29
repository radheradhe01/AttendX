"use client"

import { useQuery } from "@tanstack/react-query"
import { motion } from "framer-motion"
import { 
  Calendar, 
  BookOpen, 
  TrendingUp, 
  Clock,
  CheckCircle,
  XCircle,
  AlertCircle
} from "lucide-react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { studentApi } from "@/lib/api"
import { useStudentAttendanceStats, useStudentAttendance } from "@/hooks/useAttendance"
import { FadeIn, StaggerContainer } from "@/components/animations"
import { PageLoader, Skeleton } from "@/components/common/Loader"
import { formatDate } from "@/lib/utils"

export default function StudentDashboard() {
  const { data: profile, isLoading: profileLoading } = useQuery({
    queryKey: ["student", "profile"],
    queryFn: () => studentApi.getProfile(),
  })

  const { data: courses, isLoading: coursesLoading } = useQuery({
    queryKey: ["student", "courses"],
    queryFn: () => studentApi.getCourses(),
  })

  const { data: attendanceStats, isLoading: statsLoading } = useStudentAttendanceStats()
  const { data: recentAttendance, isLoading: attendanceLoading } = useStudentAttendance({
    startDate: new Date(Date.now() - 7 * 24 * 60 * 60 * 1000).toISOString(),
  })

  if (profileLoading || coursesLoading || statsLoading) {
    return <PageLoader message="Loading dashboard..." />
  }

  const getStatusIcon = (status: string) => {
    switch (status) {
      case "present":
        return <CheckCircle className="h-4 w-4 text-green-500" />
      case "absent":
        return <XCircle className="h-4 w-4 text-red-500" />
      case "late":
        return <AlertCircle className="h-4 w-4 text-yellow-500" />
      default:
        return <Clock className="h-4 w-4 text-gray-500" />
    }
  }

  const getStatusBadge = (status: string) => {
    switch (status) {
      case "present":
        return <Badge variant="success">Present</Badge>
      case "absent":
        return <Badge variant="destructive">Absent</Badge>
      case "late":
        return <Badge variant="warning">Late</Badge>
      default:
        return <Badge variant="secondary">Unknown</Badge>
    }
  }

  return (
    <div className="space-y-6">
      {/* Welcome Section */}
      <FadeIn>
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold">
              Welcome back, {profile?.data?.firstName}!
            </h1>
            <p className="text-muted-foreground">
              Here&apos;s what&apos;s happening with your studies today.
            </p>
          </div>
          <Badge variant="secondary" className="text-sm">
            {profile?.data?.department} • Year {profile?.data?.year}
          </Badge>
        </div>
      </FadeIn>

      {/* Stats Cards */}
      <StaggerContainer className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <FadeIn delay={0.1}>
          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Total Classes</CardTitle>
              <Calendar className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">
                {attendanceStats?.data?.totalClasses || 0}
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
              <CardTitle className="text-sm font-medium">Present Classes</CardTitle>
              <CheckCircle className="h-4 w-4 text-green-500" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-green-600">
                {attendanceStats?.data?.presentClasses || 0}
              </div>
              <p className="text-xs text-muted-foreground">
                This semester
              </p>
            </CardContent>
          </Card>
        </FadeIn>

        <FadeIn delay={0.3}>
          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Attendance Rate</CardTitle>
              <TrendingUp className="h-4 w-4 text-blue-500" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-blue-600">
                {attendanceStats?.data?.attendancePercentage?.toFixed(1) || 0}%
              </div>
              <p className="text-xs text-muted-foreground">
                Overall performance
              </p>
            </CardContent>
          </Card>
        </FadeIn>

        <FadeIn delay={0.4}>
          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Enrolled Courses</CardTitle>
              <BookOpen className="h-4 w-4 text-purple-500" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-purple-600">
                {courses?.data?.length || 0}
              </div>
              <p className="text-xs text-muted-foreground">
                Current semester
              </p>
            </CardContent>
          </Card>
        </FadeIn>
      </StaggerContainer>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Recent Attendance */}
        <FadeIn delay={0.5}>
          <Card>
            <CardHeader>
              <CardTitle>Recent Attendance</CardTitle>
              <CardDescription>
                Your attendance for the last 7 days
              </CardDescription>
            </CardHeader>
            <CardContent>
              {attendanceLoading ? (
                <div className="space-y-3">
                  {Array.from({ length: 5 }).map((_, i) => (
                    <Skeleton key={i} className="h-4 w-full" />
                  ))}
                </div>
              ) : recentAttendance?.data?.length ? (
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Date</TableHead>
                      <TableHead>Course</TableHead>
                      <TableHead>Status</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {recentAttendance.data.slice(0, 5).map((attendance) => (
                      <TableRow key={attendance.id}>
                        <TableCell className="font-medium">
                          {formatDate(attendance.date)}
                        </TableCell>
                        <TableCell>{attendance.course?.name}</TableCell>
                        <TableCell>
                          <div className="flex items-center space-x-2">
                            {getStatusIcon(attendance.status)}
                            {getStatusBadge(attendance.status)}
                          </div>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              ) : (
                <p className="text-muted-foreground text-center py-4">
                  No recent attendance records
                </p>
              )}
            </CardContent>
          </Card>
        </FadeIn>

        {/* Enrolled Courses */}
        <FadeIn delay={0.6}>
          <Card>
            <CardHeader>
              <CardTitle>Enrolled Courses</CardTitle>
              <CardDescription>
                Your courses for this semester
              </CardDescription>
            </CardHeader>
            <CardContent>
              {coursesLoading ? (
                <div className="space-y-3">
                  {Array.from({ length: 3 }).map((_, i) => (
                    <Skeleton key={i} className="h-16 w-full" />
                  ))}
                </div>
              ) : courses?.data?.length ? (
                <div className="space-y-4">
                  {courses.data.map((course) => (
                    <div key={course.id} className="flex items-center justify-between p-3 border rounded-lg">
                      <div>
                        <h4 className="font-medium">{course.name}</h4>
                        <p className="text-sm text-muted-foreground">
                          {course.code} • {course.credits} credits
                        </p>
                        <p className="text-xs text-muted-foreground">
                          {course.schedule.day} {course.schedule.startTime} - {course.schedule.endTime}
                        </p>
                      </div>
                      <Badge variant="outline">
                        {course.department}
                      </Badge>
                    </div>
                  ))}
                </div>
              ) : (
                <p className="text-muted-foreground text-center py-4">
                  No courses enrolled
                </p>
              )}
            </CardContent>
          </Card>
        </FadeIn>
      </div>
    </div>
  )
}
