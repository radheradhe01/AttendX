"use client"

import { useEffect } from "react"
import { useRouter } from "next/navigation"
import { useAuth } from "@/hooks/useAuth"
import { PageLoader } from "@/components/common/Loader"

export default function DashboardPage() {
  const { user, isLoading } = useAuth()
  const router = useRouter()

  useEffect(() => {
    if (!isLoading && user) {
      // Redirect to role-specific dashboard
      if (user.role === "admin") {
        router.push("/dashboard/admin")
      } else if (user.role === "faculty") {
        router.push("/dashboard/faculty")
      } else if (user.role === "student") {
        router.push("/dashboard/student")
      } else {
        router.push("/dashboard/student") // Default fallback
      }
    } else if (!isLoading && !user) {
      // Redirect to login if not authenticated
      router.push("/auth/login")
    }
  }, [user, isLoading, router])

  if (isLoading) {
    return <PageLoader message="Loading dashboard..." />
  }

  return <PageLoader message="Redirecting to your dashboard..." />
}
