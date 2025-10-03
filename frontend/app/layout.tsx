import type React from "react"
import { Dosis, Manrope } from "next/font/google"
import "./globals.css"

const dosis = Dosis({
  subsets: ["latin"],
  variable: "--font-dosis",
  weight: ["400", "500", "600", "700"],
})

const manrope = Manrope({
  subsets: ["latin"],
  variable: "--font-manrope",
  weight: ["400", "500", "600", "700"],
})

export const metadata = {
  title: "AttendX - Smart Attendance System",
  description: "Dual verification attendance system using facial recognition and GPS",
    generator: 'v0.app'
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" className={`${dosis.variable} ${manrope.variable}`}>
      <body className="font-manrope antialiased">{children}</body>
    </html>
  )
}
