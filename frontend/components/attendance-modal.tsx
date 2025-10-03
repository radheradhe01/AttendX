"use client"

import { useState, useRef, useEffect } from "react"
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from "@/components/ui/dialog"
import { Button } from "@/components/ui/button"
import { Camera, MapPin, CheckCircle, Loader2, X, AlertCircle } from "lucide-react"
import { Card } from "@/components/ui/card"
import { Textarea } from "@/components/ui/textarea"
import { Label } from "@/components/ui/label"
import { AttendanceApi } from "@/lib/api"

interface AttendanceModalProps {
  isOpen: boolean
  onClose: () => void
  onSuccess?: () => void
  onError?: (error: string) => void
  personName: string
}

export function AttendanceModal({ 
  isOpen, 
  onClose, 
  onSuccess, 
  onError, 
  personName 
}: AttendanceModalProps) {
  const [step, setStep] = useState<"initial" | "camera" | "processing" | "success" | "error">("initial")
  const [location, setLocation] = useState<{ lat: number; lng: number } | null>(null)
  const [locationName, setLocationName] = useState<string>("")
  const [stream, setStream] = useState<MediaStream | null>(null)
  const [notes, setNotes] = useState<string>("")
  const [errorMessage, setErrorMessage] = useState<string>("")
  const [attendanceResult, setAttendanceResult] = useState<any>(null)
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
            // Mock location name - in real app, you'd reverse geocode this
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
      setErrorMessage("Unable to access camera. Please check permissions.")
      setStep("error")
    }
  }

  // Stop camera
  const stopCamera = () => {
    if (stream) {
      stream.getTracks().forEach((track) => track.stop())
      setStream(null)
    }
  }

  // Capture photo and mark attendance
  const capturePhoto = async () => {
    if (videoRef.current && canvasRef.current) {
      const video = videoRef.current
      const canvas = canvasRef.current
      canvas.width = video.videoWidth
      canvas.height = video.videoHeight
      const ctx = canvas.getContext("2d")
      if (ctx) {
        ctx.drawImage(video, 0, 0)
      }

      // Convert canvas to blob
      const blob = await new Promise<Blob>((resolve) => {
        canvas.toBlob((blob) => {
          if (blob) resolve(blob)
        }, 'image/jpeg', 0.8)
      })

      // Create file from blob
      const file = new File([blob], 'attendance-photo.jpg', { type: 'image/jpeg' })

      // Start processing
      setStep("processing")
      stopCamera()

      try {
        // Call the API to mark attendance
        const result = await AttendanceApi.markAttendance(
          personName,
          file,
          notes.trim() || undefined
        )

        setAttendanceResult(result)
        setStep("success")
        
        // Call success callback
        if (onSuccess) {
          onSuccess()
        }
      } catch (error) {
        console.error("Error marking attendance:", error)
        const errorMsg = error instanceof Error ? error.message : "Failed to mark attendance"
        setErrorMessage(errorMsg)
        setStep("error")
        
        // Call error callback
        if (onError) {
          onError(errorMsg)
        }
      }
    }
  }

  // Reset modal
  const handleClose = () => {
    stopCamera()
    setStep("initial")
    setLocation(null)
    setLocationName("")
    setNotes("")
    setErrorMessage("")
    setAttendanceResult(null)
    onClose()
  }

  return (
    <Dialog open={isOpen} onOpenChange={handleClose}>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle>Mark Attendance</DialogTitle>
          <DialogDescription>Complete facial recognition verification</DialogDescription>
        </DialogHeader>

        <div className="space-y-4">
          {/* Initial Step - Show Location and Notes */}
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

              <div className="space-y-3">
                <div>
                  <Label htmlFor="notes">Notes (Optional)</Label>
                  <Textarea
                    id="notes"
                    placeholder="Add any notes about your attendance..."
                    value={notes}
                    onChange={(e) => setNotes(e.target.value)}
                    className="mt-1"
                    rows={2}
                  />
                </div>
                
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
                  Capture & Mark Attendance
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
          {step === "success" && attendanceResult && (
            <div className="py-8 text-center space-y-4">
              <div className="w-16 h-16 bg-green-100 dark:bg-green-900/20 rounded-full flex items-center justify-center mx-auto">
                <CheckCircle className="h-10 w-10 text-green-600" />
              </div>
              <div>
                <p className="text-xl font-bold text-green-600">Attendance Marked!</p>
                <p className="text-sm text-muted-foreground mt-1">
                  {attendanceResult.message}
                </p>
              </div>
              <Card className="p-4 bg-accent/50 text-left">
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Status:</span>
                    <span className={`font-medium ${attendanceResult.status === 'present' ? 'text-green-600' : 'text-red-600'}`}>
                      {attendanceResult.status === 'present' ? 'Present' : 'Absent'}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Face Verified:</span>
                    <span className={`font-medium ${attendanceResult.face_verified ? 'text-green-600' : 'text-red-600'}`}>
                      {attendanceResult.face_verified ? 'Yes' : 'No'}
                    </span>
                  </div>
                  {attendanceResult.confidence_score && (
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">Confidence:</span>
                      <span className="font-medium">
                        {(attendanceResult.confidence_score * 100).toFixed(1)}%
                      </span>
                    </div>
                  )}
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Time:</span>
                    <span className="font-medium">
                      {new Date(attendanceResult.timestamp).toLocaleString()}
                    </span>
                  </div>
                </div>
              </Card>
              <Button onClick={handleClose} className="w-full">
                Done
              </Button>
            </div>
          )}

          {/* Error Step */}
          {step === "error" && (
            <div className="py-8 text-center space-y-4">
              <div className="w-16 h-16 bg-red-100 dark:bg-red-900/20 rounded-full flex items-center justify-center mx-auto">
                <AlertCircle className="h-10 w-10 text-red-600" />
              </div>
              <div>
                <p className="text-xl font-bold text-red-600">Attendance Failed</p>
                <p className="text-sm text-muted-foreground mt-1">{errorMessage}</p>
              </div>
              <div className="flex gap-2">
                <Button onClick={() => setStep("initial")} variant="outline" className="flex-1">
                  Try Again
                </Button>
                <Button onClick={handleClose} className="flex-1">
                  Close
                </Button>
              </div>
            </div>
          )}
        </div>
      </DialogContent>
    </Dialog>
  )
}
