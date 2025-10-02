"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Camera, MapPin, CheckCircle, Clock, Calendar } from "lucide-react"
import { AttendanceModal } from "@/components/attendance-modal"
import { DashboardLayout } from "@/components/dashboard-layout"

// Mock attendance data
const mockAttendanceRecords = [
  { id: 1, date: "2025-03-10", time: "09:15 AM", status: "Present", location: "Main Campus" },
  { id: 2, date: "2025-03-09", time: "09:12 AM", status: "Present", location: "Main Campus" },
  { id: 3, date: "2025-03-08", time: "09:18 AM", status: "Present", location: "Main Campus" },
  { id: 4, date: "2025-03-07", time: "09:20 AM", status: "Present", location: "Main Campus" },
  { id: 5, date: "2025-03-06", time: "09:10 AM", status: "Present", location: "Main Campus" },
]

export default function StudentDashboard() {
  const [isModalOpen, setIsModalOpen] = useState(false)
  const [userName, setUserName] = useState("Student")
  const [attendanceStats, setAttendanceStats] = useState({
    total: 45,
    present: 42,
    absent: 3,
    percentage: 93.3,
  })

  useEffect(() => {
    // Get user info from localStorage
    const email = localStorage.getItem("userEmail")
    if (email) {
      setUserName(email.split("@")[0])
    }
  }, [])

  return (
    <DashboardLayout role="student">
      <div className="space-y-6">
        {/* Welcome Section */}
        <div>
          <h1 className="text-3xl font-bold text-balance">Welcome back, {userName}!</h1>
          <p className="text-muted-foreground mt-1">Track your attendance and stay updated</p>
        </div>

        {/* Stats Cards */}
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Total Classes</CardTitle>
              <Calendar className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{attendanceStats.total}</div>
              <p className="text-xs text-muted-foreground">This semester</p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Present</CardTitle>
              <CheckCircle className="h-4 w-4 text-green-600" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-green-600">{attendanceStats.present}</div>
              <p className="text-xs text-muted-foreground">Classes attended</p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Absent</CardTitle>
              <Clock className="h-4 w-4 text-red-600" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-red-600">{attendanceStats.absent}</div>
              <p className="text-xs text-muted-foreground">Classes missed</p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Attendance Rate</CardTitle>
              <CheckCircle className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{attendanceStats.percentage}%</div>
              <p className="text-xs text-muted-foreground">Overall performance</p>
            </CardContent>
          </Card>
        </div>

        {/* Mark Attendance Section */}
        <Card className="border-2 border-primary/20 bg-gradient-to-br from-primary/5 to-transparent">
          <CardHeader>
            <CardTitle className="text-xl">Mark Your Attendance</CardTitle>
            <CardDescription>Use facial recognition and GPS verification to mark your attendance</CardDescription>
          </CardHeader>
          <CardContent>
            <Button onClick={() => setIsModalOpen(true)} size="lg" className="w-full md:w-auto">
              <Camera className="mr-2 h-5 w-5" />
              Mark Attendance Now
            </Button>
          </CardContent>
        </Card>

        {/* Recent Attendance Records */}
        <Card>
          <CardHeader>
            <CardTitle>Recent Attendance</CardTitle>
            <CardDescription>Your last 5 attendance records</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {mockAttendanceRecords.map((record) => (
                <div
                  key={record.id}
                  className="flex items-center justify-between p-4 rounded-lg border bg-card hover:bg-accent/50 transition-colors"
                >
                  <div className="flex items-center gap-4">
                    <div className="w-10 h-10 rounded-full bg-green-100 dark:bg-green-900/20 flex items-center justify-center">
                      <CheckCircle className="h-5 w-5 text-green-600" />
                    </div>
                    <div>
                      <p className="font-medium">{record.date}</p>
                      <p className="text-sm text-muted-foreground">{record.time}</p>
                    </div>
                  </div>
                  <div className="text-right">
                    <p className="font-medium text-green-600">{record.status}</p>
                    <p className="text-sm text-muted-foreground flex items-center gap-1">
                      <MapPin className="h-3 w-3" />
                      {record.location}
                    </p>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>

      <AttendanceModal isOpen={isModalOpen} onClose={() => setIsModalOpen(false)} />
    </DashboardLayout>
  )
}
