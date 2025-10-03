"use client"

import { useState, useEffect, useCallback } from 'react'
import { AttendanceApi, AttendanceRecord, AttendanceResponse, ApiError } from '@/lib/api'

export function useAttendance(personName?: string) {
  const [records, setRecords] = useState<AttendanceRecord[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const fetchRecords = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const data = personName 
        ? await AttendanceApi.getPersonAttendanceRecords(personName)
        : await AttendanceApi.getAttendanceRecords()
      setRecords(data)
    } catch (err) {
      const errorMessage = err instanceof ApiError ? err.message : 'Failed to fetch attendance records'
      setError(errorMessage)
      console.error('Error fetching records:', err)
    } finally {
      setLoading(false)
    }
  }, [personName])

  const markAttendance = useCallback(async (
    imageFile: File,
    notes?: string
  ): Promise<AttendanceResponse | null> => {
    if (!personName) {
      console.error('Person name is required')
      return null
    }

    setLoading(true)
    setError(null)
    try {
      const response = await AttendanceApi.markAttendance(personName, imageFile, notes)
      
      // Refresh records after successful attendance marking
      await fetchRecords()
      
      return response
    } catch (err) {
      const errorMessage = err instanceof ApiError ? err.message : 'Failed to mark attendance'
      setError(errorMessage)
      console.error('Error marking attendance:', err)
      return null
    } finally {
      setLoading(false)
    }
  }, [personName, fetchRecords])

  useEffect(() => {
    fetchRecords()
  }, [fetchRecords])

  return {
    records,
    loading,
    error,
    markAttendance,
    refetch: fetchRecords,
  }
}

export function useAttendanceStats(records: AttendanceRecord[]) {
  const stats = {
    total: records.length,
    present: records.filter(r => r.status === 'present').length,
    absent: records.filter(r => r.status === 'absent').length,
    percentage: 0,
  }
  
  stats.percentage = stats.total > 0 ? Math.round((stats.present / stats.total) * 100) : 0
  
  return stats
}
