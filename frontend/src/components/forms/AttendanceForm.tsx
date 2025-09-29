"use client"

import { useState, useRef } from "react"
import { useForm } from "react-hook-form"
import { zodResolver } from "@hookform/resolvers/zod"
import { motion } from "framer-motion"
import { Camera, MapPin, Loader2, CheckCircle } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { attendanceCheckInSchema, type AttendanceCheckInFormData } from "@/lib/validators"
import { useAttendanceCheckIn } from "@/hooks/useAttendance"
import { toast } from "react-hot-toast"
import { cn } from "@/lib/utils"

interface AttendanceFormProps {
  courseId: string
  courseName: string
  onSuccess?: () => void
}

export function AttendanceForm({ courseId, courseName, onSuccess }: AttendanceFormProps) {
  const [capturedImage, setCapturedImage] = useState<string | null>(null)
  const [location, setLocation] = useState<{ lat: number; lng: number } | null>(null)
  const [isCapturing, setIsCapturing] = useState(false)
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const streamRef = useRef<MediaStream | null>(null)

  const attendanceMutation = useAttendanceCheckIn()

  const {
    handleSubmit,
    setValue,
    formState: { errors },
  } = useForm<AttendanceCheckInFormData>({
    resolver: zodResolver(attendanceCheckInSchema),
    defaultValues: {
      courseId,
    },
  })

  const getCurrentLocation = () => {
    if (!navigator.geolocation) {
      toast.error("Geolocation is not supported by this browser")
      return
    }

    navigator.geolocation.getCurrentPosition(
      (position) => {
        const { latitude, longitude } = position.coords
        setLocation({ lat: latitude, lng: longitude })
        setValue("latitude", latitude)
        setValue("longitude", longitude)
        toast.success("Location captured successfully!")
      },
      (error) => {
        toast.error("Failed to get location: " + error.message)
      },
      {
        enableHighAccuracy: true,
        timeout: 10000,
        maximumAge: 60000,
      }
    )
  }

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "user" },
      })
      streamRef.current = stream
      if (videoRef.current) {
        videoRef.current.srcObject = stream
      }
      setIsCapturing(true)
    } catch (error) {
      toast.error("Failed to access camera")
    }
  }

  const capturePhoto = () => {
    if (videoRef.current && canvasRef.current) {
      const canvas = canvasRef.current
      const video = videoRef.current
      const context = canvas.getContext("2d")

      if (context) {
        canvas.width = video.videoWidth
        canvas.height = video.videoHeight
        context.drawImage(video, 0, 0)

        const imageData = canvas.toDataURL("image/jpeg")
        setCapturedImage(imageData)
        setValue("faceImage", imageData)
        stopCamera()
        toast.success("Photo captured successfully!")
      }
    }
  }

  const stopCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop())
      streamRef.current = null
    }
    setIsCapturing(false)
  }

  const onSubmit = async (data: AttendanceCheckInFormData) => {
    if (!location) {
      toast.error("Please capture your location first")
      return
    }

    if (!capturedImage) {
      toast.error("Please capture your photo first")
      return
    }

    attendanceMutation.mutate(data, {
      onSuccess: () => {
        onSuccess?.()
      },
    })
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
    >
      <Card className="w-full max-w-2xl mx-auto">
        <CardHeader>
          <CardTitle className="text-2xl">Mark Attendance</CardTitle>
          <CardDescription>
            Course: <Badge variant="secondary">{courseName}</Badge>
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          <form onSubmit={handleSubmit(onSubmit)} className="space-y-6">
            {/* Location Section */}
            <div className="space-y-4">
              <div className="flex items-center space-x-2">
                <MapPin className="h-5 w-5" />
                <h3 className="text-lg font-semibold">Location Verification</h3>
              </div>
              
              <div className="flex items-center space-x-4">
                <Button
                  type="button"
                  onClick={getCurrentLocation}
                  variant={location ? "outline" : "default"}
                  className="flex items-center space-x-2"
                >
                  {location ? (
                    <>
                      <CheckCircle className="h-4 w-4 text-green-500" />
                      <span>Location Captured</span>
                    </>
                  ) : (
                    <>
                      <MapPin className="h-4 w-4" />
                      <span>Capture Location</span>
                    </>
                  )}
                </Button>
                
                {location && (
                  <div className="text-sm text-muted-foreground">
                    Lat: {location.lat.toFixed(6)}, Lng: {location.lng.toFixed(6)}
                  </div>
                )}
              </div>
            </div>

            {/* Face Capture Section */}
            <div className="space-y-4">
              <div className="flex items-center space-x-2">
                <Camera className="h-5 w-5" />
                <h3 className="text-lg font-semibold">Face Verification</h3>
              </div>

              {!capturedImage && !isCapturing && (
                <Button
                  type="button"
                  onClick={startCamera}
                  className="w-full"
                >
                  <Camera className="mr-2 h-4 w-4" />
                  Start Camera
                </Button>
              )}

              {isCapturing && (
                <div className="space-y-4">
                  <div className="relative">
                    <video
                      ref={videoRef}
                      autoPlay
                      playsInline
                      className="w-full h-64 object-cover rounded-lg border"
                    />
                    <canvas ref={canvasRef} className="hidden" />
                  </div>
                  <div className="flex space-x-2">
                    <Button
                      type="button"
                      onClick={capturePhoto}
                      className="flex-1"
                    >
                      <Camera className="mr-2 h-4 w-4" />
                      Capture Photo
                    </Button>
                    <Button
                      type="button"
                      onClick={stopCamera}
                      variant="outline"
                    >
                      Cancel
                    </Button>
                  </div>
                </div>
              )}

              {capturedImage && (
                <div className="space-y-4">
                  <div className="relative">
                    <img
                      src={capturedImage}
                      alt="Captured face"
                      className="w-full h-64 object-cover rounded-lg border"
                    />
                    <div className="absolute top-2 right-2">
                      <Badge variant="success" className="flex items-center space-x-1">
                        <CheckCircle className="h-3 w-3" />
                        <span>Captured</span>
                      </Badge>
                    </div>
                  </div>
                  <Button
                    type="button"
                    onClick={() => {
                      setCapturedImage(null)
                      setValue("faceImage", "")
                    }}
                    variant="outline"
                    className="w-full"
                  >
                    Retake Photo
                  </Button>
                </div>
              )}
            </div>

            {/* Submit Button */}
            <Button
              type="submit"
              className="w-full"
              disabled={!location || !capturedImage || attendanceMutation.isPending}
            >
              {attendanceMutation.isPending ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Marking Attendance...
                </>
              ) : (
                "Mark Attendance"
              )}
            </Button>
          </form>
        </CardContent>
      </Card>
    </motion.div>
  )
}
