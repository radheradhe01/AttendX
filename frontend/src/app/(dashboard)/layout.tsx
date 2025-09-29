"use client"

import { useEffect } from "react"
import { useRouter } from "next/navigation"
import { useAuth } from "@/hooks/useAuth"
import { Navbar } from "@/components/common/Navbar"
import { Sidebar } from "@/components/common/Sidebar"
import { PageLoader } from "@/components/common/Loader"

export default function DashboardLayout({
  children,
}: {
  children: React.ReactNode
}) {
  const { isAuthenticated, isLoading, requireAuth } = useAuth()
  const router = useRouter()

  requireAuth("/auth/login")

  if (isLoading) {
    return <PageLoader message="Loading dashboard..." />
  }

  if (!isAuthenticated) {
    return null
  }

  return (
    <div className="min-h-screen bg-background">
      <Navbar />
      <div className="flex">
        <Sidebar />
        <main className="flex-1 ml-64 p-6">
          {children}
        </main>
      </div>
    </div>
  )
}
