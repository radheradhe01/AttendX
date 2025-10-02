"use client"

import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend } from "recharts"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"

interface PerformanceData {
  subject: string
  marks: number
  attendance: number
  grade: string
}

interface PerformanceChartProps {
  data: PerformanceData[]
  title?: string
  description?: string
  className?: string
}

export function PerformanceChart({ 
  data, 
  title = "Academic Performance", 
  description = "Marks vs Attendance correlation",
  className 
}: PerformanceChartProps) {
  return (
    <Card className={className}>
      <CardHeader>
        <CardTitle>{title}</CardTitle>
        <CardDescription>{description}</CardDescription>
      </CardHeader>
      <CardContent>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={data}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis 
              dataKey="subject" 
              tick={{ fontSize: 12 }}
              angle={-45}
              textAnchor="end"
              height={80}
            />
            <YAxis yAxisId="left" orientation="left" tick={{ fontSize: 12 }} />
            <YAxis yAxisId="right" orientation="right" tick={{ fontSize: 12 }} />
            <Tooltip />
            <Legend />
            <Bar 
              yAxisId="left"
              dataKey="marks" 
              fill="#3b82f6" 
              name="Marks"
              radius={[4, 4, 0, 0]}
            />
            <Bar 
              yAxisId="right"
              dataKey="attendance" 
              fill="#10b981" 
              name="Attendance %"
              radius={[4, 4, 0, 0]}
            />
          </BarChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  )
}
