"use client"

import { useState, useRef, useEffect } from 'react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Camera, CheckCircle, AlertCircle } from 'lucide-react'

export function CameraTest() {
  const [stream, setStream] = useState<MediaStream | null>(null)
  const [error, setError] = useState<string>('')
  const [isLoading, setIsLoading] = useState(false)
  const [captureResult, setCaptureResult] = useState<string>('')
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)

  const startCamera = async () => {
    setIsLoading(true)
    setError('')

    try {
      // Check if getUserMedia is supported
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        throw new Error("Camera access not supported in this browser")
      }

      // Request camera permission
      const mediaStream = await navigator.mediaDevices.getUserMedia({
        video: { 
          facingMode: "user",
          width: { ideal: 640 },
          height: { ideal: 480 }
        },
        audio: false,
      })
      
      setStream(mediaStream)
      
      if (videoRef.current) {
        videoRef.current.srcObject = mediaStream
      }
      
    } catch (err) {
      console.error("Camera error:", err)
      
      let errorMessage = "Camera access failed. "
      
      if (err instanceof Error) {
        if (err.name === "NotAllowedError") {
          errorMessage += "Please allow camera permissions and try again."
        } else if (err.name === "NotFoundError") {
          errorMessage += "No camera found. Please connect a camera."
        } else if (err.name === "NotSupportedError") {
          errorMessage += "Camera access not supported. Please use HTTPS or localhost."
        } else {
          errorMessage += err.message
        }
      }
      
      setError(errorMessage)
    } finally {
      setIsLoading(false)
    }
  }

  const stopCamera = () => {
    if (stream) {
      stream.getTracks().forEach((track) => track.stop())
      setStream(null)
    }
  }

  const testCapture = () => {
    if (!videoRef.current || !canvasRef.current) {
      setCaptureResult("Error: Camera or canvas not ready")
      return
    }

    const video = videoRef.current
    const canvas = canvasRef.current

    if (video.videoWidth === 0 || video.videoHeight === 0) {
      setCaptureResult("Error: Video not ready")
      return
    }

    try {
      canvas.width = video.videoWidth
      canvas.height = video.videoHeight
      
      const ctx = canvas.getContext("2d")
      if (!ctx) {
        setCaptureResult("Error: Could not get canvas context")
        return
      }

      ctx.drawImage(video, 0, 0, canvas.width, canvas.height)

      canvas.toBlob((blob) => {
        if (blob) {
          setCaptureResult(`✅ Image captured successfully! Size: ${blob.size} bytes`)
        } else {
          setCaptureResult("❌ Failed to create image blob")
        }
      }, 'image/jpeg', 0.8)
    } catch (error) {
      setCaptureResult(`❌ Capture error: ${error instanceof Error ? error.message : 'Unknown error'}`)
    }
  }

  useEffect(() => {
    return () => {
      stopCamera()
    }
  }, [])

  return (
    <Card className="w-full max-w-md">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Camera className="h-5 w-5" />
          Camera Test
        </CardTitle>
        <CardDescription>
          Test camera access for face recognition
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="relative aspect-video bg-black rounded-lg overflow-hidden">
          <video 
            ref={videoRef} 
            autoPlay 
            playsInline 
            muted
            className="w-full h-full object-cover" 
          />
          {!stream && !isLoading && (
            <div className="absolute inset-0 flex items-center justify-center bg-black/50">
              <div className="text-center text-white">
                <Camera className="h-12 w-12 mx-auto mb-2 opacity-50" />
                <p className="text-sm">No camera feed</p>
              </div>
            </div>
          )}
          {isLoading && (
            <div className="absolute inset-0 flex items-center justify-center bg-black/50">
              <div className="text-center text-white">
                <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-white mx-auto mb-2"></div>
                <p className="text-sm">Loading camera...</p>
              </div>
            </div>
          )}
        </div>

        {error && (
          <div className="flex items-center gap-2 p-3 bg-red-50 border border-red-200 rounded-md">
            <AlertCircle className="h-4 w-4 text-red-600" />
            <p className="text-sm text-red-600">{error}</p>
          </div>
        )}

        {stream && (
          <div className="flex items-center gap-2 p-3 bg-green-50 border border-green-200 rounded-md">
            <CheckCircle className="h-4 w-4 text-green-600" />
            <p className="text-sm text-green-600">Camera is working!</p>
          </div>
        )}

        <div className="flex gap-2">
          <Button 
            onClick={startCamera} 
            disabled={isLoading || !!stream}
            className="flex-1"
          >
            <Camera className="mr-2 h-4 w-4" />
            {isLoading ? 'Starting...' : 'Start Camera'}
          </Button>
          {stream && (
            <Button onClick={stopCamera} variant="outline">
              Stop Camera
            </Button>
          )}
        </div>

        <div className="text-xs text-muted-foreground space-y-1">
          <p><strong>Requirements:</strong></p>
          <p>• Use HTTPS or localhost</p>
          <p>• Allow camera permissions</p>
          <p>• Have a working camera</p>
        </div>
      </CardContent>
    </Card>
  )
}
