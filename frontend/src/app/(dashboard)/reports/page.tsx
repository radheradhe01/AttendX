"use client"

import { useState } from "react"
import { useQuery } from "@tanstack/react-query"
import { motion } from "framer-motion"
import { 
  Download, 
  Calendar, 
  Users, 
  TrendingUp,
  FileText,
  BarChart3,
  Filter
} from "lucide-react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { AttendanceChart, AttendancePieChart, PerformanceChart } from "@/components/charts"
import { FadeIn, StaggerContainer } from "@/components/animations"
import { PageLoader } from "@/components/common/Loader"
import { reportApi } from "@/lib/api"

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

const performanceData = [
  { subject: "Data Structures", marks: 85, attendance: 92, grade: "A" },
  { subject: "Algorithms", marks: 78, attendance: 88, grade: "B+" },
  { subject: "Database Systems", marks: 92, attendance: 95, grade: "A+" },
  { subject: "Computer Networks", marks: 76, attendance: 82, grade: "B" },
]

export default function ReportsPage() {
  const [startDate, setStartDate] = useState("")
  const [endDate, setEndDate] = useState("")
  const [selectedReport, setSelectedReport] = useState("attendance")
  const [isGenerating, setIsGenerating] = useState(false)

  const { data: attendanceReport, isLoading: attendanceLoading } = useQuery({
    queryKey: ["reports", "attendance", startDate, endDate],
    queryFn: () => reportApi.getAttendanceReport({
      startDate: startDate || new Date(Date.now() - 30 * 24 * 60 * 60 * 1000).toISOString(),
      endDate: endDate || new Date().toISOString(),
    }),
    enabled: selectedReport === "attendance",
  })

  const { data: academicReport, isLoading: academicLoading } = useQuery({
    queryKey: ["reports", "academic", startDate, endDate],
    queryFn: () => reportApi.getAcademicReport({
      startDate: startDate || new Date(Date.now() - 30 * 24 * 60 * 60 * 1000).toISOString(),
      endDate: endDate || new Date().toISOString(),
    }),
    enabled: selectedReport === "academic",
  })

  const handleDownloadReport = async (format: "csv" | "pdf") => {
    setIsGenerating(true)
    try {
      const blob = await reportApi.downloadReport(selectedReport as "attendance" | "academic", format, {
        startDate: startDate || new Date(Date.now() - 30 * 24 * 60 * 60 * 1000).toISOString(),
        endDate: endDate || new Date().toISOString(),
      })
      
      const url = window.URL.createObjectURL(blob)
      const a = document.createElement("a")
      a.href = url
      a.download = `${selectedReport}-report.${format}`
      document.body.appendChild(a)
      a.click()
      window.URL.revokeObjectURL(url)
      document.body.removeChild(a)
    } catch (error) {
      console.error("Download failed:", error)
    } finally {
      setIsGenerating(false)
    }
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <FadeIn>
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold">Reports & Analytics</h1>
            <p className="text-muted-foreground">
              Generate and download comprehensive reports
            </p>
          </div>
          <div className="flex space-x-2">
            <Button variant="outline" onClick={() => handleDownloadReport("csv")} disabled={isGenerating}>
              <Download className="mr-2 h-4 w-4" />
              Export CSV
            </Button>
            <Button onClick={() => handleDownloadReport("pdf")} disabled={isGenerating}>
              <FileText className="mr-2 h-4 w-4" />
              Export PDF
            </Button>
          </div>
        </div>
      </FadeIn>

      {/* Filters */}
      <FadeIn delay={0.1}>
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <Filter className="h-5 w-5" />
              <span>Report Filters</span>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
              <div>
                <label className="text-sm font-medium mb-2 block">Report Type</label>
                <select
                  value={selectedReport}
                  onChange={(e) => setSelectedReport(e.target.value)}
                  className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
                >
                  <option value="attendance">Attendance Report</option>
                  <option value="academic">Academic Report</option>
                  <option value="financial">Financial Report</option>
                </select>
              </div>
              
              <div>
                <label className="text-sm font-medium mb-2 block">Start Date</label>
                <Input
                  type="date"
                  value={startDate}
                  onChange={(e) => setStartDate(e.target.value)}
                />
              </div>
              
              <div>
                <label className="text-sm font-medium mb-2 block">End Date</label>
                <Input
                  type="date"
                  value={endDate}
                  onChange={(e) => setEndDate(e.target.value)}
                />
              </div>
              
              <div className="flex items-end">
                <Button className="w-full">
                  <BarChart3 className="mr-2 h-4 w-4" />
                  Generate Report
                </Button>
              </div>
            </div>
          </CardContent>
        </Card>
      </FadeIn>

      {/* Charts Section */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <FadeIn delay={0.2}>
          <AttendanceChart 
            data={attendanceData}
            title="Attendance Trend"
            description="Daily attendance overview"
          />
        </FadeIn>

        <FadeIn delay={0.3}>
          <AttendancePieChart 
            data={attendancePieData}
            title="Attendance Distribution"
            description="Overall attendance breakdown"
          />
        </FadeIn>
      </div>

      {/* Performance Chart */}
      <FadeIn delay={0.4}>
        <PerformanceChart 
          data={performanceData}
          title="Academic Performance"
          description="Marks vs Attendance correlation"
        />
      </FadeIn>

      {/* Report Data Tables */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Attendance Report */}
        <FadeIn delay={0.5}>
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center space-x-2">
                <Users className="h-5 w-5" />
                <span>Attendance Summary</span>
              </CardTitle>
              <CardDescription>
                Student attendance statistics
              </CardDescription>
            </CardHeader>
            <CardContent>
              {attendanceLoading ? (
                <div className="space-y-3">
                  {Array.from({ length: 5 }).map((_, i) => (
                    <div key={i} className="h-4 bg-muted rounded animate-pulse" />
                  ))}
                </div>
              ) : attendanceReport?.data?.length ? (
                <div className="space-y-4">
                  {attendanceReport.data.slice(0, 5).map((report, index) => (
                    <div key={index} className="flex items-center justify-between p-3 border rounded-lg">
                      <div>
                        <h4 className="font-medium">{report.studentName}</h4>
                        <p className="text-sm text-muted-foreground">{report.courseName}</p>
                      </div>
                      <div className="text-right">
                        <div className="font-medium">{report.attendancePercentage.toFixed(1)}%</div>
                        <div className="text-sm text-muted-foreground">
                          {report.presentClasses}/{report.totalClasses}
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <p className="text-muted-foreground text-center py-4">
                  No attendance data available
                </p>
              )}
            </CardContent>
          </Card>
        </FadeIn>

        {/* Academic Report */}
        <FadeIn delay={0.6}>
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center space-x-2">
                <TrendingUp className="h-5 w-5" />
                <span>Academic Performance</span>
              </CardTitle>
              <CardDescription>
                Student academic statistics
              </CardDescription>
            </CardHeader>
            <CardContent>
              {academicLoading ? (
                <div className="space-y-3">
                  {Array.from({ length: 5 }).map((_, i) => (
                    <div key={i} className="h-4 bg-muted rounded animate-pulse" />
                  ))}
                </div>
              ) : academicReport?.data?.length ? (
                <div className="space-y-4">
                  {academicReport.data.slice(0, 5).map((report, index) => (
                    <div key={index} className="flex items-center justify-between p-3 border rounded-lg">
                      <div>
                        <h4 className="font-medium">{report.studentName}</h4>
                        <p className="text-sm text-muted-foreground">{report.courseName}</p>
                      </div>
                      <div className="text-right">
                        <div className="font-medium">{report.grade || "N/A"}</div>
                        <div className="text-sm text-muted-foreground">
                          {report.marks ? `${report.marks}%` : "No marks"}
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <p className="text-muted-foreground text-center py-4">
                  No academic data available
                </p>
              )}
            </CardContent>
          </Card>
        </FadeIn>
      </div>
    </div>
  )
}
