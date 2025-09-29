"use client"

import Link from "next/link"
import { motion } from "framer-motion"
import { 
  GraduationCap, 
  Users, 
  BookOpen, 
  BarChart3, 
  Shield, 
  Smartphone,
  ArrowRight,
  CheckCircle,
  Calendar
} from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { FadeIn } from "@/components/animations/FadeIn"
import {StaggerContainer} from "@/components/animations/StaggerContainer"
const features = [
  {
    icon: Calendar,
    title: "Smart Attendance",
    description: "GPS-based location tracking with face recognition for accurate attendance marking",
  },
  {
    icon: Users,
    title: "Student Management",
    description: "Comprehensive student profiles, course enrollment, and academic tracking",
  },
  {
    icon: BookOpen,
    title: "Course Management",
    description: "Easy course creation, scheduling, and faculty assignment",
  },
  {
    icon: BarChart3,
    title: "Analytics & Reports",
    description: "Detailed reports and analytics for attendance, grades, and performance",
  },
  {
    icon: Shield,
    title: "Role-based Access",
    description: "Secure access control for students, faculty, and administrators",
  },
  {
    icon: Smartphone,
    title: "Mobile Responsive",
    description: "Fully responsive design that works on all devices",
  },
]

const stats = [
  { label: "Students", value: "10,000+" },
  { label: "Faculty", value: "500+" },
  { label: "Courses", value: "1,000+" },
  { label: "Attendance Rate", value: "95%" },
]

export default function Home() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 dark:from-gray-900 dark:to-gray-800">
      {/* Header */}
      <header className="container mx-auto px-4 py-6">
        <nav className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <GraduationCap className="h-8 w-8 text-primary" />
            <span className="text-2xl font-bold">Smart College ERP</span>
          </div>
          <div className="flex items-center space-x-4">
            <Link href="/auth/login">
              <Button variant="ghost">Login</Button>
            </Link>
            <Link href="/auth/register">
              <Button>Get Started</Button>
            </Link>
          </div>
        </nav>
      </header>

      {/* Hero Section */}
      <section className="container mx-auto px-4 py-20">
        <StaggerContainer className="text-center space-y-8">
          <FadeIn delay={0.1}>
            <Badge variant="secondary" className="mb-4">
              🎓 Modern Education Management
            </Badge>
          </FadeIn>
          
          <FadeIn delay={0.2}>
            <h1 className="text-5xl md:text-6xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
              Smart College ERP
            </h1>
          </FadeIn>
          
          <FadeIn delay={0.3}>
            <p className="text-xl text-muted-foreground max-w-3xl mx-auto">
              Comprehensive ERP solution for educational institutions with advanced attendance tracking, 
              student management, and analytics powered by modern technology.
            </p>
          </FadeIn>
          
          <FadeIn delay={0.4}>
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <Link href="/auth/register">
                <Button size="lg" className="w-full sm:w-auto">
                  Get Started
                  <ArrowRight className="ml-2 h-4 w-4" />
                </Button>
              </Link>
              <Link href="/auth/login">
                <Button variant="outline" size="lg" className="w-full sm:w-auto">
                  Sign In
                </Button>
              </Link>
            </div>
          </FadeIn>
        </StaggerContainer>
      </section>

      {/* Stats Section */}
      <section className="container mx-auto px-4 py-16">
        <StaggerContainer className="grid grid-cols-2 md:grid-cols-4 gap-8">
          {stats.map((stat, index) => (
            <FadeIn key={stat.label} delay={index * 0.1}>
              <Card className="text-center">
                <CardContent className="pt-6">
                  <div className="text-3xl font-bold text-primary">{stat.value}</div>
                  <div className="text-sm text-muted-foreground">{stat.label}</div>
                </CardContent>
              </Card>
            </FadeIn>
          ))}
        </StaggerContainer>
      </section>

      {/* Features Section */}
      <section className="container mx-auto px-4 py-20">
        <FadeIn>
          <div className="text-center mb-16">
            <h2 className="text-3xl md:text-4xl font-bold mb-4">
              Powerful Features
            </h2>
            <p className="text-xl text-muted-foreground">
              Everything you need to manage your educational institution
            </p>
          </div>
        </FadeIn>

        <StaggerContainer className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
          {features.map((feature, index) => (
            <FadeIn key={feature.title} delay={index * 0.1}>
              <Card className="h-full">
                <CardHeader>
                  <div className="h-12 w-12 rounded-lg bg-primary/10 flex items-center justify-center mb-4">
                    <feature.icon className="h-6 w-6 text-primary" />
                  </div>
                  <CardTitle>{feature.title}</CardTitle>
                </CardHeader>
                <CardContent>
                  <CardDescription>{feature.description}</CardDescription>
                </CardContent>
              </Card>
            </FadeIn>
          ))}
        </StaggerContainer>
      </section>

      {/* CTA Section */}
      <section className="container mx-auto px-4 py-20">
        <FadeIn>
          <Card className="text-center bg-gradient-to-r from-blue-600 to-purple-600 text-white border-0">
            <CardContent className="pt-12 pb-12">
              <h2 className="text-3xl md:text-4xl font-bold mb-4">
                Ready to Transform Your Institution?
              </h2>
              <p className="text-xl mb-8 opacity-90">
                Join thousands of educational institutions already using Smart College ERP
              </p>
              <Link href="/auth/register">
                <Button size="lg" variant="secondary">
                  Start Free Trial
                  <ArrowRight className="ml-2 h-4 w-4" />
                </Button>
              </Link>
            </CardContent>
          </Card>
        </FadeIn>
      </section>

      {/* Footer */}
      <footer className="container mx-auto px-4 py-12 border-t">
        <div className="flex flex-col md:flex-row justify-between items-center">
          <div className="flex items-center space-x-2 mb-4 md:mb-0">
            <GraduationCap className="h-6 w-6 text-primary" />
            <span className="text-lg font-semibold">Smart College ERP</span>
          </div>
          <p className="text-sm text-muted-foreground">
            © 2024 Smart College ERP. All rights reserved.
          </p>
        </div>
      </footer>
    </div>
  )
}
