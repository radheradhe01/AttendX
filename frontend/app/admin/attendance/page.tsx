"use client"

import { useState } from "react"
import { DashboardLayout } from "@/components/dashboard-layout"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Button } from "@/components/ui/button"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Search, Download, Filter, CheckCircle, XCircle, MapPin } from "lucide-react"

// Mock data
const attendanceData = [
  {
    id: 1,
    name: "John Doe",
    rollNo: "2024CS001",
    department: "Computer Science",
    date: "2025-03-10",
    time: "09:15 AM",
    status: "Present",
    location: "Main Campus",
  },
  {
    id: 2,
    name: "Jane Smith",
    rollNo: "2024CS002",
    department: "Computer Science",
    date: "2025-03-10",
    time: "09:12 AM",
    status: "Present",
    location: "Main Campus",
  },
  {
    id: 3,
    name: "Mike Johnson",
    rollNo: "2024EE001",
    department: "Electrical Engineering",
    date: "2025-03-10",
    time: "09:18 AM",
    status: "Present",
    location: "Main Campus",
  },
  {
    id: 4,
    name: "Sarah Williams",
    rollNo: "2024ME001",
    department: "Mechanical Engineering",
    date: "2025-03-10",
    time: "—",
    status: "Absent",
    location: "—",
  },
  {
    id: 5,
    name: "Tom Brown",
    rollNo: "2024CS003",
    department: "Computer Science",
    date: "2025-03-10",
    time: "09:10 AM",
    status: "Present",
    location: "Main Campus",
  },
  {
    id: 6,
    name: "Emily Davis",
    rollNo: "2024EE002",
    department: "Electrical Engineering",
    date: "2025-03-10",
    time: "09:14 AM",
    status: "Present",
    location: "Main Campus",
  },
  {
    id: 7,
    name: "David Wilson",
    rollNo: "2024ME002",
    department: "Mechanical Engineering",
    date: "2025-03-10",
    time: "09:16 AM",
    status: "Present",
    location: "Main Campus",
  },
  {
    id: 8,
    name: "Lisa Anderson",
    rollNo: "2024CE001",
    department: "Civil Engineering",
    date: "2025-03-10",
    time: "—",
    status: "Absent",
    location: "—",
  },
]

export default function AdminAttendancePage() {
  const [searchQuery, setSearchQuery] = useState("")
  const [departmentFilter, setDepartmentFilter] = useState("all")
  const [statusFilter, setStatusFilter] = useState("all")

  const filteredData = attendanceData.filter((record) => {
    const matchesSearch =
      record.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      record.rollNo.toLowerCase().includes(searchQuery.toLowerCase())
    const matchesDepartment = departmentFilter === "all" || record.department === departmentFilter
    const matchesStatus = statusFilter === "all" || record.status.toLowerCase() === statusFilter.toLowerCase()

    return matchesSearch && matchesDepartment && matchesStatus
  })

  return (
    <DashboardLayout role="admin">
      <div className="space-y-6">
        <div>
          <h1 className="text-3xl font-bold text-balance">Attendance Logs</h1>
          <p className="text-muted-foreground mt-1">View and manage student attendance records</p>
        </div>

        <Card>
          <CardHeader>
            <CardTitle>All Attendance Records</CardTitle>
            <CardDescription>Search, filter, and export attendance data</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="flex flex-col gap-4 mb-6">
              <div className="flex flex-col md:flex-row gap-4">
                <div className="relative flex-1">
                  <Search className="absolute left-3 top-3 h-4 w-4 text-muted-foreground" />
                  <Input
                    placeholder="Search by name or roll number..."
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    className="pl-10"
                  />
                </div>
                <Button variant="outline">
                  <Download className="mr-2 h-4 w-4" />
                  Export CSV
                </Button>
              </div>

              <div className="flex flex-col md:flex-row gap-4">
                <Select value={departmentFilter} onValueChange={setDepartmentFilter}>
                  <SelectTrigger className="w-full md:w-[200px]">
                    <Filter className="mr-2 h-4 w-4" />
                    <SelectValue placeholder="Department" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="all">All Departments</SelectItem>
                    <SelectItem value="Computer Science">Computer Science</SelectItem>
                    <SelectItem value="Electrical Engineering">Electrical Engineering</SelectItem>
                    <SelectItem value="Mechanical Engineering">Mechanical Engineering</SelectItem>
                    <SelectItem value="Civil Engineering">Civil Engineering</SelectItem>
                  </SelectContent>
                </Select>

                <Select value={statusFilter} onValueChange={setStatusFilter}>
                  <SelectTrigger className="w-full md:w-[200px]">
                    <Filter className="mr-2 h-4 w-4" />
                    <SelectValue placeholder="Status" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="all">All Status</SelectItem>
                    <SelectItem value="present">Present</SelectItem>
                    <SelectItem value="absent">Absent</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </div>

            <div className="rounded-lg border">
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead className="bg-muted/50">
                    <tr>
                      <th className="px-4 py-3 text-left text-sm font-medium">Name</th>
                      <th className="px-4 py-3 text-left text-sm font-medium">Roll No</th>
                      <th className="px-4 py-3 text-left text-sm font-medium">Department</th>
                      <th className="px-4 py-3 text-left text-sm font-medium">Date</th>
                      <th className="px-4 py-3 text-left text-sm font-medium">Time</th>
                      <th className="px-4 py-3 text-left text-sm font-medium">Status</th>
                      <th className="px-4 py-3 text-left text-sm font-medium">Location</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y">
                    {filteredData.map((record) => (
                      <tr key={record.id} className="hover:bg-muted/50 transition-colors">
                        <td className="px-4 py-3 text-sm font-medium">{record.name}</td>
                        <td className="px-4 py-3 text-sm">{record.rollNo}</td>
                        <td className="px-4 py-3 text-sm text-muted-foreground">{record.department}</td>
                        <td className="px-4 py-3 text-sm">{record.date}</td>
                        <td className="px-4 py-3 text-sm">{record.time}</td>
                        <td className="px-4 py-3 text-sm">
                          <div className="flex items-center gap-2">
                            {record.status === "Present" ? (
                              <>
                                <CheckCircle className="h-4 w-4 text-green-600" />
                                <span className="text-green-600 font-medium">Present</span>
                              </>
                            ) : (
                              <>
                                <XCircle className="h-4 w-4 text-red-600" />
                                <span className="text-red-600 font-medium">Absent</span>
                              </>
                            )}
                          </div>
                        </td>
                        <td className="px-4 py-3 text-sm">
                          {record.location !== "—" ? (
                            <div className="flex items-center gap-1 text-muted-foreground">
                              <MapPin className="h-3 w-3" />
                              {record.location}
                            </div>
                          ) : (
                            <span className="text-muted-foreground">—</span>
                          )}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            {filteredData.length === 0 && (
              <div className="text-center py-8 text-muted-foreground">No records found matching your filters.</div>
            )}
          </CardContent>
        </Card>
      </div>
    </DashboardLayout>
  )
}
