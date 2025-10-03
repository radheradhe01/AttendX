"use client"

import { useState } from "react"
import { DashboardLayout } from "@/components/dashboard-layout"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Button } from "@/components/ui/button"
import { Search, Download, CheckCircle, XCircle, MapPin } from "lucide-react"

// Mock data
const attendanceData = [
  {
    id: 1,
    date: "2025-03-10",
    time: "09:15 AM",
    status: "Present",
    location: "Main Campus",
    subject: "Data Structures",
  },
  { id: 2, date: "2025-03-09", time: "09:12 AM", status: "Present", location: "Main Campus", subject: "Algorithms" },
  {
    id: 3,
    date: "2025-03-08",
    time: "09:18 AM",
    status: "Present",
    location: "Main Campus",
    subject: "Database Systems",
  },
  {
    id: 4,
    date: "2025-03-07",
    time: "09:20 AM",
    status: "Present",
    location: "Main Campus",
    subject: "Web Development",
  },
  {
    id: 5,
    date: "2025-03-06",
    time: "09:10 AM",
    status: "Present",
    location: "Main Campus",
    subject: "Data Structures",
  },
  { id: 6, date: "2025-03-05", time: "—", status: "Absent", location: "—", subject: "Algorithms" },
  {
    id: 7,
    date: "2025-03-04",
    time: "09:14 AM",
    status: "Present",
    location: "Main Campus",
    subject: "Database Systems",
  },
  {
    id: 8,
    date: "2025-03-03",
    time: "09:16 AM",
    status: "Present",
    location: "Main Campus",
    subject: "Web Development",
  },
]

export default function StudentAttendancePage() {
  const [searchQuery, setSearchQuery] = useState("")

  const filteredData = attendanceData.filter(
    (record) =>
      record.date.toLowerCase().includes(searchQuery.toLowerCase()) ||
      record.subject.toLowerCase().includes(searchQuery.toLowerCase()) ||
      record.status.toLowerCase().includes(searchQuery.toLowerCase()),
  )

  return (
    <DashboardLayout role="student">
      <div className="space-y-6">
        <div>
          <h1 className="text-3xl font-bold text-balance">Attendance History</h1>
          <p className="text-muted-foreground mt-1">View your complete attendance records</p>
        </div>

        <Card>
          <CardHeader>
            <CardTitle>All Attendance Records</CardTitle>
            <CardDescription>Search and filter your attendance history</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="flex flex-col md:flex-row gap-4 mb-6">
              <div className="relative flex-1">
                <Search className="absolute left-3 top-3 h-4 w-4 text-muted-foreground" />
                <Input
                  placeholder="Search by date, subject, or status..."
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

            <div className="rounded-lg border">
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead className="bg-muted/50">
                    <tr>
                      <th className="px-4 py-3 text-left text-sm font-medium">Date</th>
                      <th className="px-4 py-3 text-left text-sm font-medium">Subject</th>
                      <th className="px-4 py-3 text-left text-sm font-medium">Time</th>
                      <th className="px-4 py-3 text-left text-sm font-medium">Status</th>
                      <th className="px-4 py-3 text-left text-sm font-medium">Location</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y">
                    {filteredData.map((record) => (
                      <tr key={record.id} className="hover:bg-muted/50 transition-colors">
                        <td className="px-4 py-3 text-sm">{record.date}</td>
                        <td className="px-4 py-3 text-sm font-medium">{record.subject}</td>
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
          </CardContent>
        </Card>
      </div>
    </DashboardLayout>
  )
}
