"use client"

import { useState, useRef, useEffect } from "react"
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from "@/components/ui/dialog"
import { Button } from "@/components/ui/button"
import { Camera, MapPin, CheckCircle, Loader2, X } from "lucide-react"
import { Card } from "@/components/ui/card"

interface AttendanceModalProps {
  isOpen: boolean
  onClose: () => void
}

export function AttendanceModal({ isOpen, onClose }: AttendanceModalProps) {
  const [step, setStep] = useState<"initial" | "camera" | "processing" | "success">("initial")
  const [location, setLocation] = useState<{ lat: number; lng: number } | null>(null)
  const [locationName, setLocationName] = useState<string>("")
  const [stream, setStream] = useState<MediaStream | null>(null)
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)

  // Get GPS location
  useEffect(() => {
    if (isOpen && step === "initial") {
      if (navigator.geolocation) {
        navigator.geolocation.getCurrentPosition(
          (position) => {
            setLocation({
              lat: position.coords.latitude,
              lng: position.coords.longitude,
            })
            // Mock location name
            setLocationName("Main Campus Building A")
          },
          (error) => {
            console.error("Error getting location:", error)
            setLocationName("Location unavailable")
          },
        )
      }
    }
  }, [isOpen, step])

  // Start camera
  const startCamera = async () => {
    try {
      const mediaStream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "user" },
        audio: false,
      })
      setStream(mediaStream)
      if (videoRef.current) {
        videoRef.current.srcObject = mediaStream
      }
      setStep("camera")
    } catch (error) {
      console.error("Error accessing camera:", error)
      alert("Unable to access camera. Please check permissions.")
    }
  }

  // Stop camera
  const stopCamera = () => {
    if (stream) {
      stream.getTracks().forEach((track) => track.stop())
      setStream(null)
    }
  }

  // Capture photo
  const capturePhoto = () => {
    if (videoRef.current && canvasRef.current) {
      const video = videoRef.current
      const canvas = canvasRef.current
      canvas.width = video.videoWidth
      canvas.height = video.videoHeight
      const ctx = canvas.getContext("2d")
      if (ctx) {
        ctx.drawImage(video, 0, 0)
      }

      // Simulate processing
      setStep("processing")
      stopCamera()

      setTimeout(() => {
        setStep("success")
      }, 2000)
    }
  }

  // Reset modal
  const handleClose = () => {
    stopCamera()
    setStep("initial")
    setLocation(null)
    setLocationName("")
    onClose()
  }

  return (
    <Dialog open={isOpen} onOpenChange={handleClose}>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle>Mark Attendance</DialogTitle>
          <DialogDescription>Complete facial recognition and GPS verification</DialogDescription>
        </DialogHeader>

        <div className="space-y-4">
          {/* Initial Step - Show Location */}
          {step === "initial" && (
            <>
              <Card className="p-4 bg-accent/50">
                <div className="flex items-start gap-3">
                  <MapPin className="h-5 w-5 text-primary mt-0.5" />
                  <div className="flex-1">
                    <p className="font-medium">Location Detected</p>
                    <p className="text-sm text-muted-foreground">{locationName || "Detecting location..."}</p>
                    {location && (
                      <p className="text-xs text-muted-foreground mt-1">
                        Lat: {location.lat.toFixed(6)}, Lng: {location.lng.toFixed(6)}
                      </p>
                    )}
                  </div>
                  <CheckCircle className="h-5 w-5 text-green-600" />
                </div>
              </Card>

              <div className="space-y-2">
                <p className="text-sm text-muted-foreground">
                  Next, we'll verify your identity using facial recognition.
                </p>
                <Button onClick={startCamera} className="w-full" disabled={!location}>
                  <Camera className="mr-2 h-4 w-4" />
                  Start Camera
                </Button>
              </div>
            </>
          )}

          {/* Camera Step */}
          {step === "camera" && (
            <>
              <div className="relative aspect-video bg-black rounded-lg overflow-hidden">
                <video ref={videoRef} autoPlay playsInline className="w-full h-full object-cover" />
                <div className="absolute inset-0 border-2 border-primary/50 rounded-lg pointer-events-none">
                  <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-48 h-64 border-2 border-primary rounded-lg" />
                </div>
              </div>
              <canvas ref={canvasRef} className="hidden" />

              <div className="flex gap-2">
                <Button onClick={capturePhoto} className="flex-1">
                  <Camera className="mr-2 h-4 w-4" />
                  Capture Photo
                </Button>
                <Button onClick={handleClose} variant="outline">
                  <X className="h-4 w-4" />
                </Button>
              </div>
            </>
          )}

          {/* Processing Step */}
          {step === "processing" && (
            <div className="py-8 text-center space-y-4">
              <Loader2 className="h-12 w-12 animate-spin text-primary mx-auto" />
              <div>
                <p className="font-medium">Verifying your identity...</p>
                <p className="text-sm text-muted-foreground">Please wait while we process your attendance</p>
              </div>
            </div>
          )}

          {/* Success Step */}
          {step === "success" && (
            <div className="py-8 text-center space-y-4">
              <div className="w-16 h-16 bg-green-100 dark:bg-green-900/20 rounded-full flex items-center justify-center mx-auto">
                <CheckCircle className="h-10 w-10 text-green-600" />
              </div>
              <div>
                <p className="text-xl font-bold text-green-600">Attendance Marked!</p>
                <p className="text-sm text-muted-foreground mt-1">Your attendance has been successfully recorded</p>
              </div>
              <Card className="p-4 bg-accent/50 text-left">
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Date:</span>
                    <span className="font-medium">{new Date().toLocaleDateString()}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Time:</span>
                    <span className="font-medium">{new Date().toLocaleTimeString()}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Location:</span>
                    <span className="font-medium">{locationName}</span>
                  </div>
                </div>
              </Card>
              <Button onClick={handleClose} className="w-full">
                Done
              </Button>
            </div>
          )}
        </div>
      </DialogContent>
    </Dialog>
  )
}
