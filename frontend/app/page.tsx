"use client"

import { useState } from 'react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { CheckCircle, Shield, Zap, Users, Mail, MapPin } from 'lucide-react'
import Link from 'next/link'

export default function LandingPage() {
  const [email, setEmail] = useState('')
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [message, setMessage] = useState('')

  const handleSubscribe = async (e: React.FormEvent) => {
    e.preventDefault()
    setIsSubmitting(true)
    setMessage('')

    try {
      const response = await fetch('/api/subscribe', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ email }),
      })

      const data = await response.json()

      if (response.ok) {
        setMessage('Thank you for subscribing!')
        setEmail('')
      } else {
        setMessage(data.error || 'Something went wrong')
      }
    } catch (error) {
      setMessage('Failed to subscribe. Please try again.')
    } finally {
      setIsSubmitting(false)
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      {/* Hero Section */}
      <section className="container mx-auto px-4 py-20">
        <div className="text-center max-w-4xl mx-auto">
          <h1 className="font-dosis text-6xl md:text-7xl font-bold text-gray-900 mb-6">
            AttendX
          </h1>
          <p className="font-manrope text-xl md:text-2xl text-gray-600 mb-8 max-w-2xl mx-auto">
            Smart attendance system powered by facial recognition and GPS verification
          </p>
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <Button size="lg" className="text-lg px-8 py-4">
              <Link href="/login">
              Get Started
              </Link>
            </Button>
            <Button size="lg" variant="outline" className="text-lg px-8 py-4">
              Learn More
            </Button>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="container mx-auto px-4 py-20">
        <div className="text-center mb-16">
          <h2 className="font-dosis text-4xl md:text-5xl font-bold text-gray-900 mb-4">
            Why Choose AttendX?
          </h2>
          <p className="font-manrope text-xl text-gray-600 max-w-2xl mx-auto">
            Advanced technology meets user-friendly design for the perfect attendance solution
          </p>
        </div>

        <div className="grid md:grid-cols-3 gap-8 max-w-6xl mx-auto">
          <Card className="text-center p-8 hover:shadow-lg transition-shadow">
            <CardHeader>
              <div className="w-16 h-16 bg-blue-100 rounded-full flex items-center justify-center mx-auto mb-4">
                <Shield className="h-8 w-8 text-blue-600" />
              </div>
              <CardTitle className="font-dosis text-2xl font-semibold">
                Secure & Accurate
              </CardTitle>
            </CardHeader>
            <CardContent>
              <CardDescription className="font-manrope text-lg">
                Dual verification system using facial recognition and GPS location ensures maximum security and accuracy in attendance tracking.
              </CardDescription>
            </CardContent>
          </Card>

          <Card className="text-center p-8 hover:shadow-lg transition-shadow">
            <CardHeader>
              <div className="w-16 h-16 bg-green-100 rounded-full flex items-center justify-center mx-auto mb-4">
                <Zap className="h-8 w-8 text-green-600" />
              </div>
              <CardTitle className="font-dosis text-2xl font-semibold">
                Lightning Fast
              </CardTitle>
            </CardHeader>
            <CardContent>
              <CardDescription className="font-manrope text-lg">
                Mark attendance in seconds with our optimized face recognition technology. No more waiting in long queues or manual processes.
              </CardDescription>
            </CardContent>
          </Card>

          <Card className="text-center p-8 hover:shadow-lg transition-shadow">
            <CardHeader>
              <div className="w-16 h-16 bg-purple-100 rounded-full flex items-center justify-center mx-auto mb-4">
                <Users className="h-8 w-8 text-purple-600" />
              </div>
              <CardTitle className="font-dosis text-2xl font-semibold">
                Easy Management
              </CardTitle>
            </CardHeader>
            <CardContent>
              <CardDescription className="font-manrope text-lg">
                Comprehensive dashboard for students, faculty, and administrators with real-time analytics and detailed reporting.
              </CardDescription>
            </CardContent>
          </Card>
        </div>
      </section>

      {/* Stats Section */}
      <section className="bg-gray-900 text-white py-20">
        <div className="container mx-auto px-4">
          <div className="grid md:grid-cols-4 gap-8 text-center">
            <div>
              <h3 className="font-dosis text-4xl font-bold mb-2">1000+</h3>
              <p className="font-manrope text-lg text-gray-300">Students</p>
            </div>
            <div>
              <h3 className="font-dosis text-4xl font-bold mb-2">99.9%</h3>
              <p className="font-manrope text-lg text-gray-300">Accuracy</p>
            </div>
            <div>
              <h3 className="font-dosis text-4xl font-bold mb-2">50+</h3>
              <p className="font-manrope text-lg text-gray-300">Institutions</p>
            </div>
            <div>
              <h3 className="font-dosis text-4xl font-bold mb-2">24/7</h3>
              <p className="font-manrope text-lg text-gray-300">Support</p>
            </div>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="bg-white py-16">
        <div className="container mx-auto px-4">
          <div className="grid md:grid-cols-2 gap-12 max-w-4xl mx-auto">
            <div>
              <h3 className="font-dosis text-3xl font-bold text-gray-900 mb-4">
                Stay Updated
              </h3>
              <p className="font-manrope text-lg text-gray-600 mb-6">
                Get the latest updates about AttendX features and improvements delivered to your inbox.
              </p>
              <form onSubmit={handleSubscribe} className="flex gap-2">
                <Input
                  type="email"
                  placeholder="Enter your email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  className="flex-1"
                  required
                />
                <Button type="submit" disabled={isSubmitting}>
                  {isSubmitting ? 'Subscribing...' : 'Subscribe'}
                </Button>
              </form>
              {message && (
                <p className={`mt-2 font-manrope ${message.includes('Thank you') ? 'text-green-600' : 'text-red-600'}`}>
                  {message}
                </p>
              )}
            </div>
            
            <div>
              <h4 className="font-dosis text-xl font-semibold text-gray-900 mb-4">
                Contact Info
              </h4>
              <div className="space-y-3 font-manrope text-gray-600">
                <div className="flex items-center gap-2">
                  <Mail className="h-5 w-5" />
                  <span>contact@attendx.com</span>
                </div>
                <div className="flex items-center gap-2">
                  <MapPin className="h-5 w-5" />
                  <span>San Francisco, CA</span>
                </div>
              </div>
            </div>
          </div>
          
          <div className="border-t border-gray-200 mt-12 pt-8 text-center">
            <p className="font-manrope text-gray-500">
              © 2024 AttendX. All rights reserved. Built with Next.js and Tailwind CSS.
            </p>
          </div>
        </div>
      </footer>
    </div>
  )
}