"use client"

import { LoginForm } from "@/components/forms/LoginForm"
import { FadeIn } from "@/components/animations/FadeIn"

export default function LoginPage() {
  return (
    <FadeIn>
      <LoginForm />
    </FadeIn>
  )
}
