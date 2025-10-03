"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Camera, MapPin, CheckCircle, Clock, Calendar, AlertCircle } from "lucide-react"
import { AttendanceModal } from "@/components/attendance-modal"
import { DashboardLayout } from "@/components/dashboard-layout"
import { useAttendance, useAttendanceStats } from "@/hooks/use-attendance"
import { useToast } from "@/hooks/use-toast"

export default function StudentDashboard() {
  const [isModalOpen, setIsModalOpen] = useState(false)
  const [userName, setUserName] = useState("Student")
  const { toast } = useToast()

  // Get user info from localStorage
  useEffect(() => {
    const email = localStorage.getItem("userEmail")
    if (email) {
      setUserName(email.split("@")[0])
    }
  }, [])

  // Use the attendance hook to get real data
  const { records, loading, error, markAttendance, refetch } = useAttendance(userName)
  const stats = useAttendanceStats(records)

  // Handle successful attendance marking
  const handleAttendanceSuccess = () => {
    toast({
      title: "Attendance Marked Successfully!",
      description: "Your attendance has been recorded with face verification.",
    })
    refetch() // Refresh the data
  }

  // Handle attendance marking error
  const handleAttendanceError = (error: string) => {
    toast({
      title: "Attendance Failed",
      description: error,
      variant: "destructive",
    })
  }

  // Format date for display
  const formatDate = (timestamp: string) => {
    const date = new Date(timestamp)
    return date.toLocaleDateString('en-US', { 
      year: 'numeric', 
      month: 'short', 
      day: 'numeric' 
    })
  }

  // Format time for display
  const formatTime = (timestamp: string) => {
    const date = new Date(timestamp)
    return date.toLocaleTimeString('en-US', { 
      hour: '2-digit', 
      minute: '2-digit',
      hour12: true 
    })
  }

  // Get status color and icon
  const getStatusDisplay = (record: any) => {
    if (record.status === 'present') {
      return {
        color: 'text-green-600',
        bgColor: 'bg-green-100 dark:bg-green-900/20',
        icon: CheckCircle,
        text: 'Present'
      }
    } else {
      return {
        color: 'text-red-600',
        bgColor: 'bg-red-100 dark:bg-red-900/20',
        icon: AlertCircle,
        text: 'Absent'
      }
    }
  }

  if (error) {
    return (
      <DashboardLayout role="student">
        <div className="space-y-6">
          <div>
            <h1 className="text-3xl font-bold text-balance">Welcome back, {userName}!</h1>
            <p className="text-muted-foreground mt-1">Track your attendance and stay updated</p>
          </div>
          
          <Card className="border-red-200 bg-red-50 dark:bg-red-900/10">
            <CardContent className="pt-6">
              <div className="flex items-center gap-2 text-red-600">
                <AlertCircle className="h-5 w-5" />
                <p className="font-medium">Error loading attendance data</p>
              </div>
              <p className="text-sm text-red-600 mt-1">{error}</p>
              <Button 
                onClick={refetch} 
                variant="outline" 
                size="sm" 
                className="mt-3"
              >
                Retry
              </Button>
            </CardContent>
          </Card>
        </div>
      </DashboardLayout>
    )
  }

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
              <CardTitle className="text-sm font-medium">Total Records</CardTitle>
              <Calendar className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{loading ? "..." : stats.total}</div>
              <p className="text-xs text-muted-foreground">All time</p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Present</CardTitle>
              <CheckCircle className="h-4 w-4 text-green-600" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-green-600">{loading ? "..." : stats.present}</div>
              <p className="text-xs text-muted-foreground">Times marked present</p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Absent</CardTitle>
              <Clock className="h-4 w-4 text-red-600" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-red-600">{loading ? "..." : stats.absent}</div>
              <p className="text-xs text-muted-foreground">Times marked absent</p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Success Rate</CardTitle>
              <CheckCircle className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{loading ? "..." : `${stats.percentage}%`}</div>
              <p className="text-xs text-muted-foreground">Face verification success</p>
            </CardContent>
          </Card>
        </div>

        {/* Mark Attendance Section */}
        <Card className="border-2 border-primary/20 bg-gradient-to-br from-primary/5 to-transparent">
          <CardHeader>
            <CardTitle className="text-xl">Mark Your Attendance</CardTitle>
            <CardDescription>Use facial recognition to mark your attendance</CardDescription>
          </CardHeader>
          <CardContent>
            <Button 
              onClick={() => setIsModalOpen(true)} 
              size="lg" 
              className="w-full md:w-auto"
              disabled={loading}
            >
              <Camera className="mr-2 h-5 w-5" />
              {loading ? "Loading..." : "Mark Attendance Now"}
            </Button>
          </CardContent>
        </Card>

        {/* Recent Attendance Records */}
        <Card>
          <CardHeader>
            <CardTitle>Recent Attendance</CardTitle>
            <CardDescription>
              {loading ? "Loading records..." : `Your last ${Math.min(records.length, 5)} attendance records`}
            </CardDescription>
          </CardHeader>
          <CardContent>
            {loading ? (
              <div className="space-y-3">
                {[...Array(3)].map((_, i) => (
                  <div key={i} className="flex items-center justify-between p-4 rounded-lg border bg-muted animate-pulse">
                    <div className="flex items-center gap-4">
                      <div className="w-10 h-10 rounded-full bg-muted"></div>
                      <div className="space-y-2">
                        <div className="h-4 w-20 bg-muted rounded"></div>
                        <div className="h-3 w-16 bg-muted rounded"></div>
                      </div>
                    </div>
                    <div className="space-y-2">
                      <div className="h-4 w-16 bg-muted rounded"></div>
                      <div className="h-3 w-24 bg-muted rounded"></div>
                    </div>
                  </div>
                ))}
              </div>
            ) : records.length === 0 ? (
              <div className="text-center py-8 text-muted-foreground">
                <Calendar className="h-12 w-12 mx-auto mb-4 opacity-50" />
                <p>No attendance records found</p>
                <p className="text-sm">Mark your first attendance to get started!</p>
              </div>
            ) : (
              <div className="space-y-3">
                {records.slice(0, 5).map((record) => {
                  const statusDisplay = getStatusDisplay(record)
                  const StatusIcon = statusDisplay.icon
                  
                  return (
                    <div
                      key={record.id}
                      className="flex items-center justify-between p-4 rounded-lg border bg-card hover:bg-accent/50 transition-colors"
                    >
                      <div className="flex items-center gap-4">
                        <div className={`w-10 h-10 rounded-full ${statusDisplay.bgColor} flex items-center justify-center`}>
                          <StatusIcon className={`h-5 w-5 ${statusDisplay.color}`} />
                        </div>
                        <div>
                          <p className="font-medium">{formatDate(record.timestamp)}</p>
                          <p className="text-sm text-muted-foreground">{formatTime(record.timestamp)}</p>
                          {record.notes && (
                            <p className="text-xs text-muted-foreground mt-1">Note: {record.notes}</p>
                          )}
                        </div>
                      </div>
                      <div className="text-right">
                        <p className={`font-medium ${statusDisplay.color}`}>{statusDisplay.text}</p>
                        <p className="text-sm text-muted-foreground">
                          {record.face_verified ? (
                            <span className="flex items-center gap-1">
                              <CheckCircle className="h-3 w-3 text-green-600" />
                              Face verified
                            </span>
                          ) : (
                            <span className="flex items-center gap-1">
                              <AlertCircle className="h-3 w-3 text-red-600" />
                              Not verified
                            </span>
                          )}
                        </p>
                        {record.confidence_score && (
                          <p className="text-xs text-muted-foreground">
                            Confidence: {(record.confidence_score * 100).toFixed(1)}%
                          </p>
                        )}
                      </div>
                    </div>
                  )
                })}
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      <AttendanceModal 
        isOpen={isModalOpen} 
        onClose={() => setIsModalOpen(false)}
        onSuccess={handleAttendanceSuccess}
        onError={handleAttendanceError}
        personName={userName}
      />
    </DashboardLayout>
  )
}