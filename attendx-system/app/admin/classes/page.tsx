"use client"

import { useState } from "react"
import { DashboardLayout } from "@/components/dashboard-layout"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { BookOpen, Plus, Users, Clock, Edit, Trash2, UserPlus } from "lucide-react"
import { mockClasses, mockDepartments, getUsersByRole, type Class } from "@/lib/mock-data"

export default function ClassesPage() {
  const [classes, setClasses] = useState<Class[]>(mockClasses)
  const [isAddDialogOpen, setIsAddDialogOpen] = useState(false)
  const [isAssignStudentsOpen, setIsAssignStudentsOpen] = useState(false)
  const [selectedClass, setSelectedClass] = useState<Class | null>(null)
  const [newClass, setNewClass] = useState({
    name: "",
    code: "",
    department: "",
    facultyId: "",
    schedule: "",
    semester: "Spring 2024",
  })

  const facultyMembers = getUsersByRole("faculty")
  const students = getUsersByRole("student")

  const handleAddClass = () => {
    const faculty = facultyMembers.find((f) => f.id === newClass.facultyId)
    const cls: Class = {
      id: `class-${Date.now()}`,
      name: newClass.name,
      code: newClass.code,
      department: newClass.department,
      facultyId: newClass.facultyId,
      facultyName: faculty?.name || "",
      schedule: newClass.schedule,
      semester: newClass.semester,
      totalStudents: 0,
      studentIds: [],
    }
    setClasses([...classes, cls])
    setIsAddDialogOpen(false)
    setNewClass({
      name: "",
      code: "",
      department: "",
      facultyId: "",
      schedule: "",
      semester: "Spring 2024",
    })
  }

  const handleDeleteClass = (classId: string) => {
    if (confirm("Are you sure you want to delete this class?")) {
      setClasses(classes.filter((c) => c.id !== classId))
    }
  }

  const handleAssignStudent = (studentId: string) => {
    if (selectedClass) {
      const updatedClasses = classes.map((cls) => {
        if (cls.id === selectedClass.id) {
          return {
            ...cls,
            studentIds: [...cls.studentIds, studentId],
            totalStudents: cls.totalStudents + 1,
          }
        }
        return cls
      })
      setClasses(updatedClasses)
    }
  }

  return (
    <DashboardLayout role="admin">
      <div className="space-y-6">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-balance">Class Management</h1>
            <p className="text-muted-foreground mt-1">Create classes and assign faculty</p>
          </div>
          <Dialog open={isAddDialogOpen} onOpenChange={setIsAddDialogOpen}>
            <DialogTrigger asChild>
              <Button>
                <Plus className="h-4 w-4 mr-2" />
                Add Class
              </Button>
            </DialogTrigger>
            <DialogContent>
              <DialogHeader>
                <DialogTitle>Create New Class</DialogTitle>
                <DialogDescription>Add a new class and assign a faculty member</DialogDescription>
              </DialogHeader>
              <div className="space-y-4 py-4">
                <div className="space-y-2">
                  <Label htmlFor="class-name">Class Name</Label>
                  <Input
                    id="class-name"
                    placeholder="Computer Science 101"
                    value={newClass.name}
                    onChange={(e) => setNewClass({ ...newClass, name: e.target.value })}
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="class-code">Class Code</Label>
                  <Input
                    id="class-code"
                    placeholder="CS101"
                    value={newClass.code}
                    onChange={(e) => setNewClass({ ...newClass, code: e.target.value })}
                  />
                </div>
                <div className="space-y-2">
                  <Label>Department</Label>
                  <Select
                    value={newClass.department}
                    onValueChange={(value) => setNewClass({ ...newClass, department: value })}
                  >
                    <SelectTrigger>
                      <SelectValue placeholder="Select department" />
                    </SelectTrigger>
                    <SelectContent>
                      {mockDepartments.map((dept) => (
                        <SelectItem key={dept.id} value={dept.name}>
                          {dept.name}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
                <div className="space-y-2">
                  <Label>Assign Faculty</Label>
                  <Select
                    value={newClass.facultyId}
                    onValueChange={(value) => setNewClass({ ...newClass, facultyId: value })}
                  >
                    <SelectTrigger>
                      <SelectValue placeholder="Select faculty" />
                    </SelectTrigger>
                    <SelectContent>
                      {facultyMembers.map((faculty) => (
                        <SelectItem key={faculty.id} value={faculty.id}>
                          {faculty.name} - {faculty.department}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
                <div className="space-y-2">
                  <Label htmlFor="schedule">Schedule</Label>
                  <Input
                    id="schedule"
                    placeholder="Mon, Wed, Fri - 9:00 AM"
                    value={newClass.schedule}
                    onChange={(e) => setNewClass({ ...newClass, schedule: e.target.value })}
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="semester">Semester</Label>
                  <Input
                    id="semester"
                    placeholder="Spring 2024"
                    value={newClass.semester}
                    onChange={(e) => setNewClass({ ...newClass, semester: e.target.value })}
                  />
                </div>
                <Button onClick={handleAddClass} className="w-full">
                  Create Class
                </Button>
              </div>
            </DialogContent>
          </Dialog>
        </div>

        {/* Classes List */}
        <div className="space-y-4">
          {classes.map((cls) => (
            <Card key={cls.id}>
              <CardHeader>
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-4">
                    <div className="w-12 h-12 bg-primary/10 rounded-lg flex items-center justify-center">
                      <BookOpen className="h-6 w-6 text-primary" />
                    </div>
                    <div>
                      <CardTitle>{cls.name}</CardTitle>
                      <CardDescription>
                        {cls.code} • {cls.department}
                      </CardDescription>
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => {
                        setSelectedClass(cls)
                        setIsAssignStudentsOpen(true)
                      }}
                    >
                      <UserPlus className="h-4 w-4 mr-2" />
                      Assign Students
                    </Button>
                    <Button variant="ghost" size="sm">
                      <Edit className="h-4 w-4" />
                    </Button>
                    <Button variant="ghost" size="sm" onClick={() => handleDeleteClass(cls.id)}>
                      <Trash2 className="h-4 w-4 text-red-600" />
                    </Button>
                  </div>
                </div>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <div className="space-y-1">
                    <p className="text-sm text-muted-foreground">Faculty</p>
                    <p className="font-medium">{cls.facultyName}</p>
                  </div>
                  <div className="space-y-1">
                    <div className="flex items-center gap-2 text-muted-foreground">
                      <Clock className="h-4 w-4" />
                      <span className="text-sm">Schedule</span>
                    </div>
                    <p className="font-medium text-sm">{cls.schedule}</p>
                  </div>
                  <div className="space-y-1">
                    <div className="flex items-center gap-2 text-muted-foreground">
                      <Users className="h-4 w-4" />
                      <span className="text-sm">Students</span>
                    </div>
                    <p className="font-medium">{cls.totalStudents}</p>
                  </div>
                  <div className="space-y-1">
                    <p className="text-sm text-muted-foreground">Semester</p>
                    <p className="font-medium">{cls.semester}</p>
                  </div>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>

        {/* Assign Students Dialog */}
        <Dialog open={isAssignStudentsOpen} onOpenChange={setIsAssignStudentsOpen}>
          <DialogContent>
            <DialogHeader>
              <DialogTitle>Assign Students to {selectedClass?.name}</DialogTitle>
              <DialogDescription>Select students to add to this class</DialogDescription>
            </DialogHeader>
            <div className="space-y-3 py-4 max-h-[400px] overflow-y-auto">
              {students
                .filter((student) => !selectedClass?.studentIds.includes(student.id))
                .map((student) => (
                  <div
                    key={student.id}
                    className="flex items-center justify-between p-3 border rounded-lg hover:bg-accent/50"
                  >
                    <div>
                      <p className="font-medium">{student.name}</p>
                      <p className="text-sm text-muted-foreground">
                        {student.rollNo} • {student.department}
                      </p>
                    </div>
                    <Button size="sm" onClick={() => handleAssignStudent(student.id)}>
                      Assign
                    </Button>
                  </div>
                ))}
            </div>
          </DialogContent>
        </Dialog>
      </div>
    </DashboardLayout>
  )
}
