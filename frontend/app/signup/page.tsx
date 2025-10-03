"use client"

import { useEffect } from "react"
import { useRouter } from "next/navigation"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { UserCircle, AlertCircle } from "lucide-react"
import Link from "next/link"

export default function SignupPage() {
  const router = useRouter()

  useEffect(() => {
    const timer = setTimeout(() => {
      router.push("/login")
    }, 3000)

    return () => clearTimeout(timer)
  }, [router])

  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-blue-50 to-slate-100 dark:from-slate-900 dark:to-slate-800 p-4">
      <Card className="w-full max-w-md shadow-xl">
        <CardHeader className="space-y-1 text-center">
          <div className="flex justify-center mb-4">
            <div className="w-16 h-16 bg-orange-100 dark:bg-orange-900/20 rounded-2xl flex items-center justify-center">
              <AlertCircle className="w-10 h-10 text-orange-600" />
            </div>
          </div>
          <CardTitle className="text-3xl font-bold text-balance">Registration Restricted</CardTitle>
          <CardDescription className="text-base">Only administrators can register new users</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="p-4 bg-muted rounded-lg space-y-2">
            <p className="text-sm text-muted-foreground">
              AttendX uses a secure registration system where only administrators can create new student and faculty
              accounts.
            </p>
            <p className="text-sm text-muted-foreground">
              If you need an account, please contact your institution's administrator.
            </p>
          </div>

          <div className="space-y-2">
            <p className="text-sm font-medium">Already have an account?</p>
            <Link href="/login">
              <Button className="w-full">
                <UserCircle className="mr-2 h-4 w-4" />
                Go to Login
              </Button>
            </Link>
          </div>

          <p className="text-xs text-center text-muted-foreground">Redirecting to login page in 3 seconds...</p>
        </CardContent>
      </Card>
    </div>
  )
}
