import { useSession, signIn, signOut } from "next-auth/react"
import { useRouter } from "next/navigation"
import { toast } from "react-hot-toast"

export function useAuth() {
  const { data: session, status } = useSession()
  const router = useRouter()

  const isLoading = status === "loading"
  const isAuthenticated = !!session
  const user = session?.user

  const login = async (email: string, password: string) => {
    try {
      const result = await signIn("credentials", {
        email,
        password,
        redirect: false,
      })

      if (result?.error) {
        toast.error("Invalid credentials")
        return false
      }

      toast.success("Login successful!")
      return true
    } catch (error) {
      toast.error("Login failed")
      return false
    }
  }

  const logout = async () => {
    try {
      await signOut({ redirect: false })
      toast.success("Logged out successfully")
      router.push("/auth/login")
    } catch (error) {
      toast.error("Logout failed")
    }
  }

  const requireAuth = (redirectTo = "/auth/login") => {
    if (!isLoading && !isAuthenticated) {
      router.push(redirectTo)
    }
  }

  const requireRole = (allowedRoles: string[], redirectTo = "/dashboard") => {
    if (!isLoading && isAuthenticated && user?.role) {
      if (!allowedRoles.includes(user.role)) {
        router.push(redirectTo)
      }
    }
  }

  return {
    user,
    session,
    isLoading,
    isAuthenticated,
    login,
    logout,
    requireAuth,
    requireRole,
  }
}
