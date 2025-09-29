"use client"

import { useState } from "react"
import { useQuery } from "@tanstack/react-query"
import { motion } from "framer-motion"
import { 
  MapPin, 
  Camera, 
  Clock, 
  CheckCircle,
  AlertTriangle,
  BookOpen
} from "lucide-react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { AttendanceForm } from "@/components/forms/AttendanceForm"
import { studentApi, facultyApi } from "@/lib/api"
import { useAuth } from "@/hooks/useAuth"
import { FadeIn } from "@/components/animations/FadeIn"
import { StaggerContainer } from "@/components/animations/StaggerContainer"
import { PageLoader } from "@/components/common/Loader"

export default function AttendancePage() {
  const [selectedCourse, setSelectedCourse] = useState<string | null>(null)
  const [showForm, setShowForm] = useState(false)
  const { user } = useAuth()

  const { data: courses, isLoading: coursesLoading } = useQuery({
    queryKey: ["student", "courses"],
    queryFn: () => studentApi.getCourses(user?.id || ""),
    enabled: user?.role === "student" && !!user?.id,
  })

  const { data: facultyCourses, isLoading: facultyCoursesLoading } = useQuery({
    queryKey: ["faculty", "courses"],
    queryFn: () => facultyApi.getCourses(user?.id || ""),
    enabled: user?.role === "faculty" && !!user?.id,
  })

  if (coursesLoading || facultyCoursesLoading) {
    return <PageLoader message="Loading courses..." />
  }

  const currentCourses = user?.role === "student" ? courses?.data : facultyCourses?.data

  const handleMarkAttendance = (courseId: string) => {
    setSelectedCourse(courseId)
    setShowForm(true)
  }

  const handleAttendanceSuccess = () => {
    setShowForm(false)
    setSelectedCourse(null)
  }

  if (showForm && selectedCourse) {
    const course = currentCourses?.find(c => c.id === selectedCourse)
    return (
      <div className="max-w-4xl mx-auto">
        <AttendanceForm
          courseId={selectedCourse}
          courseName={course?.courseName || "Unknown Course"}
          onSuccess={handleAttendanceSuccess}
        />
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <FadeIn>
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold">Attendance</h1>
            <p className="text-muted-foreground">
              {user?.role === "student" 
                ? "Mark your attendance for today&apos;s classes"
                : "Manage student attendance for your courses"
              }
            </p>
          </div>
          <Badge variant="secondary">
            {new Date().toLocaleDateString()}
          </Badge>
        </div>
      </FadeIn>

      {/* Today's Classes */}
      <FadeIn delay={0.1}>
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <Clock className="h-5 w-5" />
              <span>Today&apos;s Classes</span>
            </CardTitle>
            <CardDescription>
              {user?.role === "student" 
                ? "Select a class to mark your attendance"
                : "Select a class to manage attendance"
              }
            </CardDescription>
          </CardHeader>
          <CardContent>
            {currentCourses?.length ? (
              <StaggerContainer className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {currentCourses.map((course, index) => {
                  const today = new Date().toLocaleDateString('en-US', { weekday: 'long' }).toLowerCase()
                  const firstSchedule = course.schedule?.[0] // Get first schedule item
                  const isToday = firstSchedule?.dayOfWeek?.toLowerCase() === today
                  const currentTime = new Date().toLocaleTimeString('en-US', { hour12: false, hour: '2-digit', minute: '2-digit' })
                  const isActive = isToday && currentTime >= firstSchedule?.startTime && currentTime <= firstSchedule?.endTime
                  
                  return (
                    <FadeIn key={course.id} delay={index * 0.1}>
                      <Card className={`cursor-pointer transition-all hover:shadow-md ${isActive ? 'ring-2 ring-primary' : ''}`}>
                        <CardContent className="p-6">
                          <div className="flex items-start justify-between mb-4">
                            <div>
                              <h3 className="font-semibold">{course.courseName}</h3>
                              <p className="text-sm text-muted-foreground">{course.courseCode}</p>
                            </div>
                            <Badge variant={isActive ? "default" : "secondary"}>
                              {isActive ? "Active" : "Scheduled"}
                            </Badge>
                          </div>
                          
                          <div className="space-y-2 mb-4">
                            <div className="flex items-center space-x-2 text-sm">
                              <Clock className="h-4 w-4 text-muted-foreground" />
                              <span>{firstSchedule?.startTime} - {firstSchedule?.endTime}</span>
                            </div>
                            <div className="flex items-center space-x-2 text-sm">
                              <BookOpen className="h-4 w-4 text-muted-foreground" />
                              <span>{firstSchedule?.room || "TBA"}</span>
                            </div>
                          </div>

                          {user?.role === "student" ? (
                            <Button 
                              className="w-full" 
                              onClick={() => handleMarkAttendance(course.id)}
                              disabled={!isToday}
                            >
                              {isToday ? "Mark Attendance" : "Not Today"}
                            </Button>
                          ) : (
                            <Button 
                              className="w-full" 
                              onClick={() => handleMarkAttendance(course.id)}
                              variant="outline"
                            >
                              Manage Attendance
                            </Button>
                          )}
                        </CardContent>
                      </Card>
                    </FadeIn>
                  )
                })}
              </StaggerContainer>
            ) : (
              <div className="text-center py-8">
                <BookOpen className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
                <h3 className="text-lg font-semibold mb-2">No Classes Today</h3>
                <p className="text-muted-foreground">
                  You don&apos;t have any classes scheduled for today.
                </p>
              </div>
            )}
          </CardContent>
        </Card>
      </FadeIn>

      {/* Instructions */}
      <FadeIn delay={0.2}>
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <AlertTriangle className="h-5 w-5" />
              <span>Attendance Instructions</span>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="space-y-4">
                <h4 className="font-semibold">For Students:</h4>
                <ul className="space-y-2 text-sm text-muted-foreground">
                  <li className="flex items-start space-x-2">
                    <MapPin className="h-4 w-4 mt-0.5 text-primary" />
                    <span>Ensure you&apos;re within the designated classroom area</span>
                  </li>
                  <li className="flex items-start space-x-2">
                    <Camera className="h-4 w-4 mt-0.5 text-primary" />
                    <span>Take a clear selfie for face verification</span>
                  </li>
                  <li className="flex items-start space-x-2">
                    <Clock className="h-4 w-4 mt-0.5 text-primary" />
                    <span>Mark attendance only during class hours</span>
                  </li>
                </ul>
              </div>
              
              <div className="space-y-4">
                <h4 className="font-semibold">For Faculty:</h4>
                <ul className="space-y-2 text-sm text-muted-foreground">
                  <li className="flex items-start space-x-2">
                    <CheckCircle className="h-4 w-4 mt-0.5 text-primary" />
                    <span>Review and approve student attendance</span>
                  </li>
                  <li className="flex items-start space-x-2">
                    <BookOpen className="h-4 w-4 mt-0.5 text-primary" />
                    <span>Manage attendance for all your courses</span>
                  </li>
                  <li className="flex items-start space-x-2">
                    <AlertTriangle className="h-4 w-4 mt-0.5 text-primary" />
                    <span>Mark absent students manually if needed</span>
                  </li>
                </ul>
              </div>
            </div>
          </CardContent>
        </Card>
      </FadeIn>
    </div>
  )
}
