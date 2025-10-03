"use client"

import { useState } from "react"
import { DashboardLayout } from "@/components/dashboard-layout"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Users, CheckCircle, XCircle, TrendingUp, Calendar } from "lucide-react"

// Mock data
const recentAttendance = [
  {
    id: 1,
    name: "John Doe",
    rollNo: "2024CS001",
    time: "09:15 AM",
    status: "Present",
    department: "Computer Science",
  },
  {
    id: 2,
    name: "Jane Smith",
    rollNo: "2024CS002",
    time: "09:12 AM",
    status: "Present",
    department: "Computer Science",
  },
  {
    id: 3,
    name: "Mike Johnson",
    rollNo: "2024EE001",
    time: "09:18 AM",
    status: "Present",
    department: "Electrical Engineering",
  },
  {
    id: 4,
    name: "Sarah Williams",
    rollNo: "2024ME001",
    time: "09:20 AM",
    status: "Present",
    department: "Mechanical Engineering",
  },
  {
    id: 5,
    name: "Tom Brown",
    rollNo: "2024CS003",
    time: "09:10 AM",
    status: "Present",
    department: "Computer Science",
  },
]

const departmentStats = [
  { name: "Computer Science", total: 120, present: 115, absent: 5, percentage: 95.8 },
  { name: "Electrical Engineering", total: 80, present: 74, absent: 6, percentage: 92.5 },
  { name: "Mechanical Engineering", total: 90, present: 82, absent: 8, percentage: 91.1 },
  { name: "Civil Engineering", total: 70, present: 65, absent: 5, percentage: 92.9 },
]

export default function AdminDashboard() {
  const [stats] = useState({
    totalStudents: 360,
    presentToday: 336,
    absentToday: 24,
    averageAttendance: 93.3,
  })

  return (
    <DashboardLayout role="admin">
      <div className="space-y-6">
        {/* Header */}
        <div>
          <h1 className="text-3xl font-bold text-balance">Admin Dashboard</h1>
          <p className="text-muted-foreground mt-1">
            Overview of today's attendance - {new Date().toLocaleDateString()}
          </p>
        </div>

        {/* Stats Cards */}
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Total Students</CardTitle>
              <Users className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{stats.totalStudents}</div>
              <p className="text-xs text-muted-foreground">Across all departments</p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Present Today</CardTitle>
              <CheckCircle className="h-4 w-4 text-green-600" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-green-600">{stats.presentToday}</div>
              <p className="text-xs text-muted-foreground">Students marked present</p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Absent Today</CardTitle>
              <XCircle className="h-4 w-4 text-red-600" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-red-600">{stats.absentToday}</div>
              <p className="text-xs text-muted-foreground">Students absent</p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Average Attendance</CardTitle>
              <TrendingUp className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{stats.averageAttendance}%</div>
              <p className="text-xs text-muted-foreground">This semester</p>
            </CardContent>
          </Card>
        </div>

        {/* Department-wise Stats */}
        <Card>
          <CardHeader>
            <CardTitle>Department-wise Attendance</CardTitle>
            <CardDescription>Today's attendance breakdown by department</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {departmentStats.map((dept) => (
                <div key={dept.name} className="space-y-2">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      <div className="w-10 h-10 rounded-lg bg-primary/10 flex items-center justify-center">
                        <Calendar className="h-5 w-5 text-primary" />
                      </div>
                      <div>
                        <p className="font-medium">{dept.name}</p>
                        <p className="text-sm text-muted-foreground">
                          {dept.present}/{dept.total} students
                        </p>
                      </div>
                    </div>
                    <div className="text-right">
                      <p className="font-bold text-lg">{dept.percentage}%</p>
                      <p className="text-xs text-muted-foreground">{dept.absent} absent</p>
                    </div>
                  </div>
                  <div className="w-full bg-muted rounded-full h-2">
                    <div
                      className="bg-primary h-2 rounded-full transition-all"
                      style={{ width: `${dept.percentage}%` }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* Real-time Attendance Feed */}
        <Card>
          <CardHeader>
            <CardTitle>Real-time Attendance Feed</CardTitle>
            <CardDescription>Latest attendance records from students</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {recentAttendance.map((record) => (
                <div
                  key={record.id}
                  className="flex items-center justify-between p-4 rounded-lg border bg-card hover:bg-accent/50 transition-colors"
                >
                  <div className="flex items-center gap-4">
                    <div className="w-10 h-10 rounded-full bg-green-100 dark:bg-green-900/20 flex items-center justify-center">
                      <CheckCircle className="h-5 w-5 text-green-600" />
                    </div>
                    <div>
                      <p className="font-medium">{record.name}</p>
                      <p className="text-sm text-muted-foreground">
                        {record.rollNo} • {record.department}
                      </p>
                    </div>
                  </div>
                  <div className="text-right">
                    <p className="font-medium text-green-600">{record.status}</p>
                    <p className="text-sm text-muted-foreground">{record.time}</p>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>
    </DashboardLayout>
  )
}
