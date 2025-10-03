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
import { Building2, Plus, Users, GraduationCap, Edit, Trash2 } from "lucide-react"
import { mockDepartments, type Department } from "@/lib/mock-data"

export default function DepartmentsPage() {
  const [departments, setDepartments] = useState<Department[]>(mockDepartments)
  const [isAddDialogOpen, setIsAddDialogOpen] = useState(false)
  const [newDept, setNewDept] = useState({
    name: "",
    code: "",
    hod: "",
  })

  const handleAddDepartment = () => {
    const dept: Department = {
      id: `dept-${Date.now()}`,
      name: newDept.name,
      code: newDept.code,
      hod: newDept.hod,
      totalStudents: 0,
      totalFaculty: 0,
    }
    setDepartments([...departments, dept])
    setIsAddDialogOpen(false)
    setNewDept({ name: "", code: "", hod: "" })
  }

  const handleDeleteDepartment = (deptId: string) => {
    if (confirm("Are you sure you want to delete this department?")) {
      setDepartments(departments.filter((d) => d.id !== deptId))
    }
  }

  return (
    <DashboardLayout role="admin">
      <div className="space-y-6">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-balance">Department Management</h1>
            <p className="text-muted-foreground mt-1">Create and manage academic departments</p>
          </div>
          <Dialog open={isAddDialogOpen} onOpenChange={setIsAddDialogOpen}>
            <DialogTrigger asChild>
              <Button>
                <Plus className="h-4 w-4 mr-2" />
                Add Department
              </Button>
            </DialogTrigger>
            <DialogContent>
              <DialogHeader>
                <DialogTitle>Create New Department</DialogTitle>
                <DialogDescription>Add a new academic department to the system</DialogDescription>
              </DialogHeader>
              <div className="space-y-4 py-4">
                <div className="space-y-2">
                  <Label htmlFor="dept-name">Department Name</Label>
                  <Input
                    id="dept-name"
                    placeholder="Computer Science"
                    value={newDept.name}
                    onChange={(e) => setNewDept({ ...newDept, name: e.target.value })}
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="dept-code">Department Code</Label>
                  <Input
                    id="dept-code"
                    placeholder="CS"
                    value={newDept.code}
                    onChange={(e) => setNewDept({ ...newDept, code: e.target.value })}
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="dept-hod">Head of Department</Label>
                  <Input
                    id="dept-hod"
                    placeholder="Dr. John Smith"
                    value={newDept.hod}
                    onChange={(e) => setNewDept({ ...newDept, hod: e.target.value })}
                  />
                </div>
                <Button onClick={handleAddDepartment} className="w-full">
                  Create Department
                </Button>
              </div>
            </DialogContent>
          </Dialog>
        </div>

        {/* Departments Grid */}
        <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
          {departments.map((dept) => (
            <Card key={dept.id} className="hover:shadow-lg transition-shadow">
              <CardHeader>
                <div className="flex items-start justify-between">
                  <div className="flex items-center gap-3">
                    <div className="w-12 h-12 bg-primary/10 rounded-lg flex items-center justify-center">
                      <Building2 className="h-6 w-6 text-primary" />
                    </div>
                    <div>
                      <CardTitle className="text-lg">{dept.name}</CardTitle>
                      <CardDescription>{dept.code}</CardDescription>
                    </div>
                  </div>
                  <div className="flex gap-1">
                    <Button variant="ghost" size="sm">
                      <Edit className="h-4 w-4" />
                    </Button>
                    <Button variant="ghost" size="sm" onClick={() => handleDeleteDepartment(dept.id)}>
                      <Trash2 className="h-4 w-4 text-red-600" />
                    </Button>
                  </div>
                </div>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <p className="text-sm text-muted-foreground">Head of Department</p>
                  <p className="font-medium">{dept.hod}</p>
                </div>
                <div className="grid grid-cols-2 gap-4">
                  <div className="space-y-1">
                    <div className="flex items-center gap-2 text-muted-foreground">
                      <Users className="h-4 w-4" />
                      <span className="text-sm">Students</span>
                    </div>
                    <p className="text-2xl font-bold">{dept.totalStudents}</p>
                  </div>
                  <div className="space-y-1">
                    <div className="flex items-center gap-2 text-muted-foreground">
                      <GraduationCap className="h-4 w-4" />
                      <span className="text-sm">Faculty</span>
                    </div>
                    <p className="text-2xl font-bold">{dept.totalFaculty}</p>
                  </div>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      </div>
    </DashboardLayout>
  )
}
