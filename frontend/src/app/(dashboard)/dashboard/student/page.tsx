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
  AlertCircle,
  User2
} from "lucide-react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar"
import { Button } from "@/components/ui/button"
import { studentApi, authApi } from "@/lib/api"
import { useAuth } from "@/hooks/useAuth"
import { FadeIn, StaggerContainer } from "@/components/animations"
import { PageLoader, Skeleton } from "@/components/common/Loader"
import { formatDate } from "@/lib/utils"

export default function StudentDashboard() {
  const { user, isAuthenticated, isLoading: authLoading } = useAuth()

  // Get user profile first to get the student ID
  const { data: userProfile, isLoading: userLoading } = useQuery({
    queryKey: ["auth", "me"],
    queryFn: () => authApi.me(),
    enabled: isAuthenticated,
  })

  const studentId = userProfile?.data?.profile?.id

  const { data: profile, isLoading: profileLoading } = useQuery({
    queryKey: ["student", "profile", studentId],
    queryFn: () => studentApi.getProfile(studentId || ""),
    enabled: !!studentId,
  })

  const { data: courses, isLoading: coursesLoading } = useQuery({
    queryKey: ["student", "courses", studentId],
    queryFn: () => studentApi.getCourses(studentId || ""),
    enabled: !!studentId,
  })

  const { data: attendanceHistory, isLoading: attendanceLoading } = useQuery({
    queryKey: ["student", "attendance", studentId],
    queryFn: () => studentApi.getAttendance(studentId || ""),
    enabled: !!studentId,
  })

  if (authLoading || userLoading || profileLoading || coursesLoading || attendanceLoading) {
    return <PageLoader message="Loading dashboard..." />
  }

  if (!isAuthenticated || user?.role !== "student") {
    return (
      <div className="flex items-center justify-center min-h-[calc(100vh-150px)]">
        <Card className="w-full max-w-md text-center">
          <CardHeader>
            <CardTitle>Access Denied</CardTitle>
            <CardDescription>You do not have permission to view this page.</CardDescription>
          </CardHeader>
          <CardContent>
            <Button onClick={() => window.location.href = "/auth/login"}>Go to Login</Button>
          </CardContent>
        </Card>
      </div>
    )
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
              <CardTitle className="text-sm font-medium">Profile</CardTitle>
              <User2 className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="flex items-center space-x-3">
                <Avatar className="h-12 w-12">
                  <AvatarImage src={profile?.data?.avatar || "/avatars/01.png"} alt={profile?.data?.firstName} />
                  <AvatarFallback>{profile?.data?.firstName?.charAt(0)}{profile?.data?.lastName?.charAt(0)}</AvatarFallback>
                </Avatar>
                <div>
                  <div className="text-lg font-bold">{profile?.data?.firstName} {profile?.data?.lastName}</div>
                  <p className="text-xs text-muted-foreground">{profile?.data?.email}</p>
                </div>
              </div>
            </CardContent>
          </Card>
        </FadeIn>

        <FadeIn delay={0.2}>
          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Recent Attendance</CardTitle>
              <CheckCircle className="h-4 w-4 text-green-500" />
            </CardHeader>
            <CardContent>
              {attendanceHistory?.data && attendanceHistory.data.length > 0 ? (
                <>
                  <div className="text-2xl font-bold">
                    {attendanceHistory.data[0].status === "present" ? "Present" : "Absent"}
                  </div>
                  <p className="text-xs text-muted-foreground">
                    Last updated: {formatDate(attendanceHistory.data[0].date)}
                  </p>
                </>
              ) : (
                <div className="text-muted-foreground">No recent attendance.</div>
              )}
            </CardContent>
          </Card>
        </FadeIn>

        <FadeIn delay={0.3}>
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

        <FadeIn delay={0.4}>
          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Department</CardTitle>
              <TrendingUp className="h-4 w-4 text-blue-500" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-blue-600">
                {profile?.data?.department || "N/A"}
              </div>
              <p className="text-xs text-muted-foreground">
                Year {profile?.data?.year || "N/A"}
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
                Your attendance history
              </CardDescription>
            </CardHeader>
            <CardContent>
              {attendanceLoading ? (
                <div className="space-y-3">
                  {Array.from({ length: 5 }).map((_, i) => (
                    <Skeleton key={i} className="h-4 w-full" />
                  ))}
                </div>
              ) : attendanceHistory?.data?.length ? (
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Date</TableHead>
                      <TableHead>Course</TableHead>
                      <TableHead>Status</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {attendanceHistory.data.slice(0, 5).map((attendance: any) => (
                      <TableRow key={attendance.id}>
                        <TableCell className="font-medium">
                          {formatDate(attendance.date)}
                        </TableCell>
                        <TableCell>{attendance.courseId}</TableCell>
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
                        <h4 className="font-medium">{course.courseName}</h4>
                        <p className="text-sm text-muted-foreground">
                          {course.courseCode} • {course.credits} credits
                        </p>
                        <p className="text-xs text-muted-foreground">
                          {course.schedule?.[0]?.dayOfWeek} {course.schedule?.[0]?.startTime} - {course.schedule?.[0]?.endTime}
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
