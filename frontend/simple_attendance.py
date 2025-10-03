"""
Simple FastAPI server for attendance marking with personal face verification.
Compares uploaded image with the person's own reference image.
"""

import io
import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, Dict
from collections import namedtuple

import face_recognition
import numpy as np
from PIL import Image
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Create FastAPI app
app = FastAPI(
    title="AttendX - Personal Face Verification",
    description="Simple attendance system that verifies against personal reference images",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Data models
class AttendanceResponse(BaseModel):
    message: str
    attendance_id: str
    status: str
    face_verified: bool
    confidence_score: Optional[float]
    timestamp: str
    person_name: Optional[str] = None

class AttendanceRecordResponse(BaseModel):
    id: str
    person_name: str
    timestamp: str
    face_verified: bool
    confidence_score: Optional[float]
    status: str
    notes: Optional[str]
    reference_image_path: str
    attendance_image_path: str

class AttendanceRecordsResponse(BaseModel):
    count: int
    records: list[AttendanceRecordResponse]

class ReferenceImageResponse(BaseModel):
    person_name: str
    image_path: str

class ReferenceImagesResponse(BaseModel):
    count: int
    images: list[ReferenceImageResponse]

class AddReferenceImageResponse(BaseModel):
    message: str
    filename: str
    path: str
    person_name: str

# Global variables
ATTENDANCE_FILE = "attendance_data.json"
REFERENCE_IMAGES_FOLDER = "reference_images"
ATTENDANCE_IMAGES_FOLDER = "attendance_images"

# Create AttendanceRecord namedtuple
AttendanceRecord = namedtuple('AttendanceRecord', [
    'id', 'person_name', 'timestamp', 'face_verified', 'confidence_score', 
    'status', 'notes', 'reference_image_path', 'attendance_image_path'
])

def load_attendance_data() -> list:
    """Load attendance data from JSON file."""
    try:
        if os.path.exists(ATTENDANCE_FILE):
            with open(ATTENDANCE_FILE, 'r') as f:
                data = json.load(f)
                records = []
                for record_data in data:
                    record = AttendanceRecord(
                        id=record_data['id'],
                        person_name=record_data['person_name'],
                        timestamp=record_data['timestamp'],
                        face_verified=record_data['face_verified'],
                        confidence_score=record_data['confidence_score'],
                        status=record_data['status'],
                        notes=record_data['notes'],
                        reference_image_path=record_data['reference_image_path'],
                        attendance_image_path=record_data['attendance_image_path']
                    )
                    records.append(record)
                return records
        return []
    except Exception as e:
        print(f"Warning: Could not load attendance data: {e}")
        return []

def save_attendance_data(records: list):
    """Save attendance data to JSON file."""
    try:
        data = []
        for record in records:
            # Ensure all values are JSON serializable
            record_dict = {
                'id': str(record.id),
                'person_name': str(record.person_name),
                'timestamp': str(record.timestamp),
                'face_verified': bool(record.face_verified),
                'confidence_score': float(record.confidence_score) if record.confidence_score is not None else None,
                'status': str(record.status),
                'notes': str(record.notes) if record.notes is not None else None,
                'reference_image_path': str(record.reference_image_path),
                'attendance_image_path': str(record.attendance_image_path)
            }
            data.append(record_dict)
        
        # Write directly to file
        with open(ATTENDANCE_FILE, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Successfully saved {len(records)} attendance records")
        
    except Exception as e:
        print(f"Error saving attendance data: {e}")
        import traceback
        traceback.print_exc()

def get_reference_images() -> Dict[str, str]:
    """Get all reference images from the reference_images folder."""
    reference_images = {}
    ref_path = Path(REFERENCE_IMAGES_FOLDER)
    
    if not ref_path.exists():
        print(f"Warning: Reference images folder '{REFERENCE_IMAGES_FOLDER}' not found")
        return reference_images
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    
    for image_file in ref_path.iterdir():
        if image_file.suffix.lower() in image_extensions:
            # Use filename (without extension) as person name
            person_name = image_file.stem
            reference_images[person_name.lower()] = str(image_file)
            print(f"Found reference image: {person_name} -> {image_file}")
    
    return reference_images

def extract_face_encoding(image_data: bytes) -> Optional[np.ndarray]:
    """Extract face encoding from image data."""
    try:
        image = face_recognition.load_image_file(io.BytesIO(image_data))
        face_locations = face_recognition.face_locations(image)
        
        if not face_locations:
            return None
        
        face_encodings = face_recognition.face_encodings(image, face_locations)
        return face_encodings[0] if face_encodings else None
    
    except Exception as e:
        print(f"Error extracting face encoding: {e}")
        return None

def compare_faces(encoding1: np.ndarray, encoding2: np.ndarray, threshold: float = 0.4) -> Tuple[bool, float]:
    """Compare two face encodings and return match result and confidence."""
    try:
        # Calculate distance between encodings
        distance = face_recognition.face_distance([encoding1], encoding2)[0]
        confidence = 1 - distance
        
        # Check if faces match based on threshold
        is_match = confidence >= threshold
        
        return is_match, confidence
    
    except Exception as e:
        print(f"Error comparing faces: {e}")
        return False, 0.0

def verify_person_identity(uploaded_image_data: bytes, person_name: str) -> Tuple[bool, float, str]:
    """
    Verify if the uploaded image matches the person's reference image.
    
    Returns:
        (is_verified, confidence_score, message)
    """
    try:
        # Get reference images
        reference_images = get_reference_images()
        
        # Find the person's reference image
        person_key = person_name.lower()
        if person_key not in reference_images:
            return False, 0.0, f"No reference image found for {person_name}"
        
        reference_image_path = reference_images[person_key]
        
        # Extract encoding from uploaded image
        uploaded_encoding = extract_face_encoding(uploaded_image_data)
        if uploaded_encoding is None:
            return False, 0.0, "No face detected in uploaded image"
        
        # Load and extract encoding from reference image
        with open(reference_image_path, 'rb') as f:
            reference_image_data = f.read()
        
        reference_encoding = extract_face_encoding(reference_image_data)
        if reference_encoding is None:
            return False, 0.0, f"No face detected in reference image for {person_name}"
        
        # Compare faces
        is_match, confidence = compare_faces(uploaded_encoding, reference_encoding, threshold=0.4)
        
        if is_match:
            message = f"Identity verified for {person_name}"
        else:
            message = f"Identity verification failed for {person_name}. Confidence: {confidence:.2f}"
        
        return is_match, confidence, message
    
    except Exception as e:
        return False, 0.0, f"Error during verification: {str(e)}"

def save_attendance_image(image_data: bytes, person_name: str) -> str:
    """Save the attendance image and return the file path."""
    try:
        # Create attendance images folder if it doesn't exist
        Path(ATTENDANCE_IMAGES_FOLDER).mkdir(exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{person_name}_{timestamp}.jpg"
        file_path = Path(ATTENDANCE_IMAGES_FOLDER) / filename
        
        # Save image
        with open(file_path, 'wb') as f:
            f.write(image_data)
        
        return str(file_path)
    
    except Exception as e:
        print(f"Error saving attendance image: {e}")
        return ""

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "AttendX - Personal Face Verification API",
        "version": "1.0.0",
        "endpoint": "/mark-attendance",
        "docs": "/docs",
        "reference_images_folder": REFERENCE_IMAGES_FOLDER
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    reference_images = get_reference_images()
    return {
        "status": "healthy",
        "message": "Personal face verification service is running",
        "timestamp": datetime.utcnow().isoformat(),
        "reference_images_count": len(reference_images)
    }

@app.get("/reference-images", response_model=ReferenceImagesResponse)
async def get_reference_images_list():
    """Get list of available reference images."""
    reference_images = get_reference_images()
    
    return ReferenceImagesResponse(
        count=len(reference_images),
        images=[
            ReferenceImageResponse(
                person_name=person_name,
                image_path=image_path
            )
            for person_name, image_path in reference_images.items()
        ]
    )

@app.post("/mark-attendance", response_model=AttendanceResponse)
async def mark_attendance(
    person_name: str = Form(..., description="Name of the person marking attendance"),
    image: UploadFile = File(..., description="Face image file"),
    notes: Optional[str] = Form(None, description="Optional notes")
):
    """
    Mark attendance by verifying the person's identity against their reference image.
    
    This endpoint:
    1. Takes the person's name and their photo
    2. Finds their reference image in the reference_images folder
    3. Compares the uploaded photo with their reference image
    4. Creates attendance record if verification passes
    
    Returns attendance result with verification status.
    """
    try:
        # Read image data
        image_data = await image.read()
        
        # Verify person's identity
        is_verified, confidence, message = verify_person_identity(image_data, person_name)
        
        # Determine status
        if is_verified:
            status = "present"
            final_message = f"Attendance marked successfully for {person_name}"
        else:
            status = "absent"
            final_message = message
        
        # Save attendance image
        attendance_image_path = save_attendance_image(image_data, person_name)
        
        # Load existing attendance records
        attendance_records = load_attendance_data()
        
        # Create new attendance record
        attendance_id = f"att_{len(attendance_records) + 1}"
        
        new_record = AttendanceRecord(
            id=attendance_id,
            person_name=person_name,
            timestamp=datetime.utcnow().isoformat(),
            face_verified=is_verified,
            confidence_score=confidence,
            status=status,
            notes=notes,
            reference_image_path=get_reference_images().get(person_name.lower(), ""),
            attendance_image_path=attendance_image_path
        )
        
        # Save attendance record
        attendance_records.append(new_record)
        print(f"About to save {len(attendance_records)} records")
        print(f"New record: {new_record}")
        save_attendance_data(attendance_records)
        
        return AttendanceResponse(
            message=final_message,
            attendance_id=attendance_id,
            status=status,
            face_verified=is_verified,
            confidence_score=confidence,
            timestamp=new_record.timestamp,
            person_name=person_name
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/attendance-records", response_model=AttendanceRecordsResponse)
async def get_attendance_records():
    """Get all attendance records."""
    records = load_attendance_data()
    
    return AttendanceRecordsResponse(
        count=len(records),
        records=[
            AttendanceRecordResponse(
                id=record.id,
                person_name=record.person_name,
                timestamp=record.timestamp,
                face_verified=record.face_verified,
                confidence_score=record.confidence_score,
                status=record.status,
                notes=record.notes,
                reference_image_path=record.reference_image_path,
                attendance_image_path=record.attendance_image_path
            )
            for record in records
        ]
    )

@app.post("/add-reference-image", response_model=AddReferenceImageResponse)
async def add_reference_image(
    person_name: str = Form(..., description="Name of the person"),
    image: UploadFile = File(..., description="Reference face image file")
):
    """Add a new reference image for a person."""
    try:
        # Create reference images folder if it doesn't exist
        Path(REFERENCE_IMAGES_FOLDER).mkdir(exist_ok=True)
        
        # Generate filename
        safe_name = person_name.replace(" ", "_").lower()
        file_extension = Path(image.filename).suffix.lower()
        filename = f"{safe_name}{file_extension}"
        file_path = Path(REFERENCE_IMAGES_FOLDER) / filename
        
        # Save the uploaded file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
        
        return AddReferenceImageResponse(
            message=f"Reference image for '{person_name}' added successfully",
            filename=filename,
            path=str(file_path),
            person_name=person_name
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save reference image: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    
    # Create necessary folders
    Path(REFERENCE_IMAGES_FOLDER).mkdir(exist_ok=True)
    Path(ATTENDANCE_IMAGES_FOLDER).mkdir(exist_ok=True)
    
    print("Starting AttendX Personal Face Verification Server...")
    print(f"Reference images folder: {REFERENCE_IMAGES_FOLDER}")
    print(f"Attendance images folder: {ATTENDANCE_IMAGES_FOLDER}")
    print("Add reference face images to the 'reference_images' folder")
    print("Access API docs at: http://localhost:8000/docs")
    
    uvicorn.run(
        "simple_attendance:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
