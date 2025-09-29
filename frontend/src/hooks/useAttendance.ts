import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query"
import { attendanceApi, studentApi, facultyApi } from "@/lib/api"
import { toast } from "react-hot-toast"

export function useAttendanceHistory(params?: {
  courseId?: string
  studentId?: string
  startDate?: string
  endDate?: string
}) {
  return useQuery({
    queryKey: ["attendance", "history", params],
    queryFn: () => attendanceApi.getHistory(params),
    staleTime: 5 * 60 * 1000, // 5 minutes
  })
}

export function useAttendanceStats(params?: {
  courseId?: string
  studentId?: string
}) {
  return useQuery({
    queryKey: ["attendance", "stats", params],
    queryFn: () => attendanceApi.getStats(params),
    staleTime: 5 * 60 * 1000, // 5 minutes
  })
}

export function useStudentAttendance(params?: {
  courseId?: string
  startDate?: string
  endDate?: string
}) {
  return useQuery({
    queryKey: ["student", "attendance", params],
    queryFn: () => studentApi.getAttendance(params),
    staleTime: 5 * 60 * 1000, // 5 minutes
  })
}

export function useStudentAttendanceStats() {
  return useQuery({
    queryKey: ["student", "attendance", "stats"],
    queryFn: () => studentApi.getAttendanceStats(),
    staleTime: 5 * 60 * 1000, // 5 minutes
  })
}

export function useCourseAttendance(courseId: string, params?: {
  startDate?: string
  endDate?: string
}) {
  return useQuery({
    queryKey: ["faculty", "course", courseId, "attendance", params],
    queryFn: () => facultyApi.getCourseAttendance(courseId, params),
    enabled: !!courseId,
    staleTime: 5 * 60 * 1000, // 5 minutes
  })
}

export function useAttendanceCheckIn() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: attendanceApi.checkIn,
    onSuccess: () => {
      toast.success("Attendance marked successfully!")
      // Invalidate related queries
      queryClient.invalidateQueries({ queryKey: ["attendance"] })
      queryClient.invalidateQueries({ queryKey: ["student", "attendance"] })
    },
    onError: (error: unknown) => {
      const errorMessage = error instanceof Error ? error.message : "Failed to mark attendance"
      toast.error(errorMessage)
    },
  })
}

export function useMarkAttendance() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: facultyApi.markAttendance,
    onSuccess: () => {
      toast.success("Attendance marked successfully!")
      // Invalidate related queries
      queryClient.invalidateQueries({ queryKey: ["attendance"] })
      queryClient.invalidateQueries({ queryKey: ["faculty", "course"] })
    },
    onError: (error: unknown) => {
      const errorMessage = error instanceof Error ? error.message : "Failed to mark attendance"
      toast.error(errorMessage)
    },
  })
}
