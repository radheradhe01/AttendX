"use client"

import { useEffect, useState } from "react"
import { useRouter } from "next/navigation"
import { DashboardLayout } from "@/components/dashboard-layout"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Avatar, AvatarFallback } from "@/components/ui/avatar"
import { CheckCircle, XCircle, Clock, Search, Download, Filter } from "lucide-react"

export default function FacultyAttendance() {
  const router = useRouter()
  const [searchQuery, setSearchQuery] = useState("")
  const [selectedClass, setSelectedClass] = useState("all")
  const [selectedStatus, setSelectedStatus] = useState("all")

  useEffect(() => {
    const role = localStorage.getItem("userRole")
    if (role !== "faculty") {
      router.push("/login")
    }
  }, [router])

  const attendanceRecords = [
    {
      id: 1,
      studentName: "John Doe",
      rollNo: "CS2021001",
      class: "CS101",
      date: "2025-03-10",
      time: "09:15 AM",
      status: "present",
      verificationMethod: "Face + GPS",
    },
    {
      id: 2,
      studentName: "Jane Smith",
      rollNo: "CS2021002",
      class: "CS101",
      date: "2025-03-10",
      time: "09:18 AM",
      status: "present",
      verificationMethod: "Face + GPS",
    },
    {
      id: 3,
      studentName: "Mike Johnson",
      rollNo: "CS2021003",
      class: "CS101",
      date: "2025-03-10",
      time: "-",
      status: "absent",
      verificationMethod: "-",
    },
    {
      id: 4,
      studentName: "Sarah Williams",
      rollNo: "CS2021004",
      class: "CS201",
      date: "2025-03-10",
      time: "11:20 AM",
      status: "present",
      verificationMethod: "Face + GPS",
    },
    {
      id: 5,
      studentName: "David Brown",
      rollNo: "CS2021005",
      class: "CS201",
      date: "2025-03-10",
      time: "-",
      status: "absent",
      verificationMethod: "-",
    },
    {
      id: 6,
      studentName: "Emily Davis",
      rollNo: "CS2021006",
      class: "CS301",
      date: "2025-03-10",
      time: "02:15 PM",
      status: "present",
      verificationMethod: "Face + GPS",
    },
  ]

  const filteredRecords = attendanceRecords.filter((record) => {
    const matchesSearch =
      record.studentName.toLowerCase().includes(searchQuery.toLowerCase()) ||
      record.rollNo.toLowerCase().includes(searchQuery.toLowerCase())
    const matchesClass = selectedClass === "all" || record.class === selectedClass
    const matchesStatus = selectedStatus === "all" || record.status === selectedStatus
    return matchesSearch && matchesClass && matchesStatus
  })

  const getStatusIcon = (status: string) => {
    switch (status) {
      case "present":
        return <CheckCircle className="h-5 w-5 text-green-600" />
      case "absent":
        return <XCircle className="h-5 w-5 text-red-600" />
      case "late":
        return <Clock className="h-5 w-5 text-orange-600" />
      default:
        return null
    }
  }

  const getStatusBadge = (status: string) => {
    const styles = {
      present: "bg-green-100 text-green-700 border-green-200",
      absent: "bg-red-100 text-red-700 border-red-200",
      late: "bg-orange-100 text-orange-700 border-orange-200",
    }
    return (
      <span className={`px-3 py-1 rounded-full text-xs font-medium border ${styles[status as keyof typeof styles]}`}>
        {status.charAt(0).toUpperCase() + status.slice(1)}
      </span>
    )
  }

  return (
    <DashboardLayout role="faculty">
      <div className="space-y-6">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Class Attendance</h1>
          <p className="text-muted-foreground mt-2">View and manage student attendance for your classes</p>
        </div>

        {/* Filters */}
        <Card>
          <CardHeader>
            <CardTitle>Filters</CardTitle>
            <CardDescription>Search and filter attendance records</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid gap-4 md:grid-cols-4">
              <div className="space-y-2">
                <Label htmlFor="search">Search Student</Label>
                <div className="relative">
                  <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                  <Input
                    id="search"
                    placeholder="Name or Roll No"
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    className="pl-9"
                  />
                </div>
              </div>

              <div className="space-y-2">
                <Label htmlFor="class">Class</Label>
                <Select value={selectedClass} onValueChange={setSelectedClass}>
                  <SelectTrigger id="class">
                    <SelectValue placeholder="Select class" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="all">All Classes</SelectItem>
                    <SelectItem value="CS101">CS101 - Computer Science 101</SelectItem>
                    <SelectItem value="CS201">CS201 - Data Structures</SelectItem>
                    <SelectItem value="CS301">CS301 - Database Systems</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-2">
                <Label htmlFor="status">Status</Label>
                <Select value={selectedStatus} onValueChange={setSelectedStatus}>
                  <SelectTrigger id="status">
                    <SelectValue placeholder="Select status" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="all">All Status</SelectItem>
                    <SelectItem value="present">Present</SelectItem>
                    <SelectItem value="absent">Absent</SelectItem>
                    <SelectItem value="late">Late</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-2">
                <Label>&nbsp;</Label>
                <Button className="w-full bg-transparent" variant="outline">
                  <Download className="mr-2 h-4 w-4" />
                  Export
                </Button>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Attendance Table */}
        <Card>
          <CardHeader>
            <CardTitle>Attendance Records</CardTitle>
            <CardDescription>
              Showing {filteredRecords.length} of {attendanceRecords.length} records
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {filteredRecords.map((record) => (
                <div
                  key={record.id}
                  className="flex items-center justify-between p-4 border rounded-lg hover:bg-accent/50 transition-colors"
                >
                  <div className="flex items-center gap-4">
                    <Avatar className="h-12 w-12">
                      <AvatarFallback className="bg-primary/10 text-primary font-semibold">
                        {record.studentName
                          .split(" ")
                          .map((n) => n[0])
                          .join("")}
                      </AvatarFallback>
                    </Avatar>
                    <div>
                      <h3 className="font-semibold">{record.studentName}</h3>
                      <p className="text-sm text-muted-foreground">{record.rollNo}</p>
                    </div>
                  </div>

                  <div className="flex items-center gap-6">
                    <div className="text-center">
                      <p className="text-sm font-medium">{record.class}</p>
                      <p className="text-xs text-muted-foreground">Class</p>
                    </div>

                    <div className="text-center">
                      <p className="text-sm font-medium">{record.date}</p>
                      <p className="text-xs text-muted-foreground">{record.time}</p>
                    </div>

                    <div className="text-center min-w-[120px]">
                      <p className="text-sm font-medium">{record.verificationMethod}</p>
                      <p className="text-xs text-muted-foreground">Verification</p>
                    </div>

                    <div className="flex items-center gap-2">
                      {getStatusIcon(record.status)}
                      {getStatusBadge(record.status)}
                    </div>
                  </div>
                </div>
              ))}

              {filteredRecords.length === 0 && (
                <div className="text-center py-12">
                  <Filter className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
                  <h3 className="text-lg font-semibold mb-2">No records found</h3>
                  <p className="text-muted-foreground">Try adjusting your filters</p>
                </div>
              )}
            </div>
          </CardContent>
        </Card>
      </div>
    </DashboardLayout>
  )
}
