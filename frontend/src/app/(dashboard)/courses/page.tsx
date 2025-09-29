"use client"

import { useState } from "react"
import { useQuery } from "@tanstack/react-query"
import { motion } from "framer-motion"
import { 
  BookOpen, 
  Clock, 
  Users, 
  MapPin,
  Plus,
  Search,
  Filter
} from "lucide-react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { studentApi } from "@/lib/api"
import { useAuth } from "@/hooks/useAuth"
import { FadeIn, StaggerContainer } from "@/components/animations"
import { PageLoader } from "@/components/common/Loader"
import { formatDate } from "@/lib/utils"

export default function CoursesPage() {
  const [searchTerm, setSearchTerm] = useState("")
  const [selectedDepartment, setSelectedDepartment] = useState("all")
  const { user } = useAuth()

  const { data: courses, isLoading } = useQuery({
    queryKey: ["courses"],
    queryFn: () => studentApi.getCourses(), // Replace with courseApi.getAll()
  })

  if (isLoading) {
    return <PageLoader message="Loading courses..." />
  }

  const filteredCourses = courses?.data?.filter(course => {
    const matchesSearch = course.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         course.code.toLowerCase().includes(searchTerm.toLowerCase())
    const matchesDepartment = selectedDepartment === "all" || course.department === selectedDepartment
    return matchesSearch && matchesDepartment
  }) || []

  const departments = [...new Set(courses?.data?.map(course => course.department) || [])]

  return (
    <div className="space-y-6">
      {/* Header */}
      <FadeIn>
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold">Courses</h1>
            <p className="text-muted-foreground">
              Browse and manage all available courses
            </p>
          </div>
          {user?.role === "admin" && (
            <Button>
              <Plus className="mr-2 h-4 w-4" />
              Add Course
            </Button>
          )}
        </div>
      </FadeIn>

      {/* Filters */}
      <FadeIn delay={0.1}>
        <Card>
          <CardContent className="p-6">
            <div className="flex flex-col md:flex-row gap-4">
              <div className="flex-1">
                <div className="relative">
                  <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                  <Input
                    placeholder="Search courses..."
                    value={searchTerm}
                    onChange={(e) => setSearchTerm(e.target.value)}
                    className="pl-10"
                  />
                </div>
              </div>
              <div className="flex items-center space-x-2">
                <Filter className="h-4 w-4 text-muted-foreground" />
                <select
                  value={selectedDepartment}
                  onChange={(e) => setSelectedDepartment(e.target.value)}
                  className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
                >
                  <option value="all">All Departments</option>
                  {departments.map(dept => (
                    <option key={dept} value={dept}>{dept}</option>
                  ))}
                </select>
              </div>
            </div>
          </CardContent>
        </Card>
      </FadeIn>

      {/* Courses Grid */}
      <StaggerContainer className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {filteredCourses.map((course, index) => (
          <FadeIn key={course.id} delay={index * 0.1}>
            <Card className="h-full">
              <CardHeader>
                <div className="flex items-start justify-between">
                  <div>
                    <CardTitle className="text-lg">{course.name}</CardTitle>
                    <CardDescription>{course.code}</CardDescription>
                  </div>
                  <Badge variant="outline">{course.credits} credits</Badge>
                </div>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <div className="flex items-center space-x-2 text-sm">
                    <BookOpen className="h-4 w-4 text-muted-foreground" />
                    <span>{course.department}</span>
                  </div>
                  <div className="flex items-center space-x-2 text-sm">
                    <Clock className="h-4 w-4 text-muted-foreground" />
                    <span>{course.schedule.day} {course.schedule.startTime} - {course.schedule.endTime}</span>
                  </div>
                  <div className="flex items-center space-x-2 text-sm">
                    <MapPin className="h-4 w-4 text-muted-foreground" />
                    <span>{course.schedule.room || "TBA"}</span>
                  </div>
                  {course.faculty && (
                    <div className="flex items-center space-x-2 text-sm">
                      <Users className="h-4 w-4 text-muted-foreground" />
                      <span>Prof. {course.faculty.lastName}</span>
                    </div>
                  )}
                </div>
                
                {course.description && (
                  <p className="text-sm text-muted-foreground line-clamp-2">
                    {course.description}
                  </p>
                )}

                <div className="flex space-x-2">
                  <Button variant="outline" size="sm" className="flex-1">
                    View Details
                  </Button>
                  {user?.role === "student" && (
                    <Button size="sm" className="flex-1">
                      Enroll
                    </Button>
                  )}
                </div>
              </CardContent>
            </Card>
          </FadeIn>
        ))}
      </StaggerContainer>

      {filteredCourses.length === 0 && (
        <FadeIn>
          <Card>
            <CardContent className="text-center py-12">
              <BookOpen className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
              <h3 className="text-lg font-semibold mb-2">No courses found</h3>
              <p className="text-muted-foreground">
                {searchTerm || selectedDepartment !== "all" 
                  ? "Try adjusting your search criteria"
                  : "No courses are available at the moment"
                }
              </p>
            </CardContent>
          </Card>
        </FadeIn>
      )}
    </div>
  )
}
