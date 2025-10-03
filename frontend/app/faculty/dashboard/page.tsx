"use client"

import { useEffect, useState } from "react"
import { useRouter } from "next/navigation"
import { DashboardLayout } from "@/components/dashboard-layout"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Users, CheckCircle, XCircle, Clock, BookOpen } from "lucide-react"
import { getClassesByFaculty } from "@/lib/mock-data"

export default function FacultyDashboard() {
  const router = useRouter()
  const [userName, setUserName] = useState("")
  const [facultyId, setFacultyId] = useState("")
  const [classes, setClasses] = useState<any[]>([])

  useEffect(() => {
    const role = localStorage.getItem("userRole")
    const email = localStorage.getItem("userEmail")

    if (role !== "faculty") {
      router.push("/login")
      return
    }

    if (email) {
      setUserName(email.split("@")[0])
      const mockFacultyId = email.includes("sarah")
        ? "faculty-1"
        : email.includes("michael")
          ? "faculty-2"
          : "faculty-3"
      setFacultyId(mockFacultyId)

      const facultyClasses = getClassesByFaculty(mockFacultyId)
      setClasses(facultyClasses)
    }
  }, [router])

  const totalStudents = classes.reduce((sum, cls) => sum + cls.totalStudents, 0)
  const avgAttendance = 85.4

  const stats = [
    {
      title: "Total Classes",
      value: classes.length.toString(),
      icon: BookOpen,
      description: "Active courses",
      color: "text-blue-600",
      bgColor: "bg-blue-50",
    },
    {
      title: "Total Students",
      value: totalStudents.toString(),
      icon: Users,
      description: "Across all classes",
      color: "text-purple-600",
      bgColor: "bg-purple-50",
    },
    {
      title: "Avg. Attendance",
      value: `${avgAttendance}%`,
      icon: CheckCircle,
      description: "This week",
      color: "text-green-600",
      bgColor: "bg-green-50",
    },
    {
      title: "Pending Reviews",
      value: "12",
      icon: Clock,
      description: "Attendance requests",
      color: "text-orange-600",
      bgColor: "bg-orange-50",
    },
  ]

  return (
    <DashboardLayout role="faculty">
      <div className="space-y-6">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Welcome back, Prof. {userName}!</h1>
          <p className="text-muted-foreground mt-2">Here's an overview of your classes and student attendance.</p>
        </div>

        {/* Stats Grid */}
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
          {stats.map((stat) => {
            const Icon = stat.icon
            return (
              <Card key={stat.title}>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">{stat.title}</CardTitle>
                  <div className={`p-2 rounded-lg ${stat.bgColor}`}>
                    <Icon className={`h-4 w-4 ${stat.color}`} />
                  </div>
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">{stat.value}</div>
                  <p className="text-xs text-muted-foreground mt-1">{stat.description}</p>
                </CardContent>
              </Card>
            )
          })}
        </div>

        {/* Classes Overview */}
        <Card>
          <CardHeader>
            <CardTitle>My Classes</CardTitle>
            <CardDescription>Manage attendance for your courses</CardDescription>
          </CardHeader>
          <CardContent>
            {classes.length === 0 ? (
              <div className="text-center py-8 text-muted-foreground">
                <BookOpen className="h-12 w-12 mx-auto mb-3 opacity-50" />
                <p>No classes assigned yet</p>
              </div>
            ) : (
              <div className="space-y-4">
                {classes.map((classItem) => {
                  const present = Math.floor(classItem.totalStudents * 0.85)
                  const absent = classItem.totalStudents - present

                  return (
                    <div
                      key={classItem.id}
                      className="flex items-center justify-between p-4 border rounded-lg hover:bg-accent/50 transition-colors"
                    >
                      <div className="space-y-1">
                        <div className="flex items-center gap-3">
                          <div className="w-10 h-10 bg-primary/10 rounded-lg flex items-center justify-center">
                            <BookOpen className="h-5 w-5 text-primary" />
                          </div>
                          <div>
                            <h3 className="font-semibold">{classItem.name}</h3>
                            <p className="text-sm text-muted-foreground">{classItem.code}</p>
                          </div>
                        </div>
                        <div className="flex items-center gap-4 text-sm text-muted-foreground ml-13">
                          <span className="flex items-center gap-1">
                            <Users className="h-4 w-4" />
                            {classItem.totalStudents} students
                          </span>
                          <span className="flex items-center gap-1">
                            <Clock className="h-4 w-4" />
                            {classItem.schedule}
                          </span>
                        </div>
                      </div>
                      <div className="flex items-center gap-4">
                        <div className="text-right">
                          <div className="flex items-center gap-2 text-sm">
                            <CheckCircle className="h-4 w-4 text-green-600" />
                            <span className="font-medium">{present} Present</span>
                          </div>
                          <div className="flex items-center gap-2 text-sm text-muted-foreground">
                            <XCircle className="h-4 w-4 text-red-600" />
                            <span>{absent} Absent</span>
                          </div>
                        </div>
                        <Button onClick={() => router.push("/faculty/attendance")}>View Details</Button>
                      </div>
                    </div>
                  )
                })}
              </div>
            )}
          </CardContent>
        </Card>

        {/* Quick Actions */}
        <Card>
          <CardHeader>
            <CardTitle>Quick Actions</CardTitle>
            <CardDescription>Common tasks and shortcuts</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid gap-4 md:grid-cols-3">
              <Button variant="outline" className="h-auto py-4 flex flex-col items-center gap-2 bg-transparent">
                <Users className="h-6 w-6" />
                <span>Mark Attendance</span>
              </Button>
              <Button variant="outline" className="h-auto py-4 flex flex-col items-center gap-2 bg-transparent">
                <Clock className="h-6 w-6" />
                <span>Review Requests</span>
              </Button>
              <Button variant="outline" className="h-auto py-4 flex flex-col items-center gap-2 bg-transparent">
                <BookOpen className="h-6 w-6" />
                <span>Generate Report</span>
              </Button>
            </div>
          </CardContent>
        </Card>
      </div>
    </DashboardLayout>
  )
}
