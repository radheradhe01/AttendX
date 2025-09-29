"""
Attendance management router.
"""

from datetime import datetime
from typing import List, Optional
import time

from fastapi import APIRouter, HTTPException, status, UploadFile, File, Form

from app.models.attendance import AttendanceResponse, AttendanceCreate
from app.models.base import Location
from app.services.attendance_service import AttendanceService

router = APIRouter()


@router.post("/mark")
async def mark_attendance(
    latitude: float = Form(..., ge=-90, le=90),
    longitude: float = Form(..., ge=-180, le=180),
    image: UploadFile = File(...),
    notes: Optional[str] = Form(None)
):
    """Mark attendance with face and location verification."""
    # Use default test user for now
    user_id = "test_user_123"
    organization_id = "test_org_123"

    # Read image data
    image_data = await image.read()

    # Create location object
    location = Location(
        latitude=latitude,
        longitude=longitude,
        address=None  # Will be set by location service if needed
    )

    # Get device info from headers (simplified)
    device_info = {
        "user_agent": "Unknown",  # Would get from request headers
        "ip_address": "Unknown",   # Would get from request
        "platform": "Web"
    }

    # Mark attendance
    attendance, message = await AttendanceService.mark_attendance(
        user_id=user_id,
        organization_id=organization_id,
        location=location,
        image_data=image_data,
        device_info=device_info,
        notes=notes
    )

    if not attendance:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=message
        )

    return {
        "message": message,
        "attendance_id": str(attendance.id),
        "status": attendance.status,
        "face_verified": attendance.face_verified,
        "location_verified": attendance.location_verified,
        "confidence_score": attendance.confidence_score,
        "timestamp": attendance.timestamp
    }


@router.get("/records", response_model=List[AttendanceResponse])
async def get_attendance_records(
    user_id: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit: int = 50
):
    """Get attendance records."""
    # Use default test user if not provided
    if not user_id:
        user_id = "test_user_123"

    attendance_records = await AttendanceService.get_user_attendance(
        user_id,
        start_date,
        end_date,
        limit
    )

    return [
        AttendanceResponse(
            id=str(record.id),
            user_id=record.user_id,
            organization_id=record.organization_id,
            timestamp=record.timestamp,
            location=record.location,
            location_verified=record.location_verified,
            location_distance_km=record.location_distance_km,
            approved_location_id=record.approved_location_id,
            face_verified=record.face_verified,
            confidence_score=record.confidence_score,
            device_info=record.device_info,
            status=record.status,
            notes=record.notes,
            approved_by=record.approved_by,
            approved_at=record.approved_at,
            work_session_id=record.work_session_id,
            work_hours_logged=record.work_hours_logged,
            created_at=record.created_at.isoformat(),
            updated_at=record.updated_at.isoformat()
        )
        for record in attendance_records
    ]


@router.get("/today")
async def get_today_attendance():
    """Get today's attendance summary."""
    # Use default test user
    user_id = "test_user_123"

    # Get today's date
    today = datetime.now().date().isoformat()

    attendance_records = await AttendanceService.get_user_attendance(
        user_id,
        start_date=today,
        end_date=today
    )

    return {
        "date": today,
        "records": len(attendance_records),
        "latest_record": {
            "timestamp": attendance_records[0].timestamp if attendance_records else None,
            "status": attendance_records[0].status if attendance_records else None,
            "face_verified": attendance_records[0].face_verified if attendance_records else None,
            "location_verified": attendance_records[0].location_verified if attendance_records else None
        }
    }


@router.get("/summary")
async def get_attendance_summary(
    days: int = 30
):
    """Get attendance summary for current user."""
    # Use default test user
    user_id = "test_user_123"

    summary = await AttendanceService.get_attendance_summary(user_id, days)

    return {
        "user_id": user_id,
        "period_days": days,
        "summary": summary
    }


@router.post("/{attendance_id}/approve")
async def approve_attendance(
    attendance_id: str
):
    """Approve attendance record (admin/manager only)."""
    # Use default test user as approver
    approver_id = "test_user_123"

    success = await AttendanceService.approve_attendance(attendance_id, approver_id)

    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Attendance record not found"
        )

    return {"message": "Attendance approved successfully"}


@router.post("/process-images")
async def process_images():
    """Process all images in the images folder and mark attendance."""
    results = await AttendanceService.process_images_from_folder()

    if "error" in results:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=results["error"]
        )

    return {
        "message": f"Processed {results['processed']} images, {results['matched']} matched, {results['errors']} errors",
        "results": results
    }


@router.get("/unprocessed-images")
async def get_unprocessed_images():
    """Get list of unprocessed images in the images folder."""
    unprocessed = await AttendanceService.get_unprocessed_images()

    return {
        "count": len(unprocessed),
        "images": unprocessed
    }


@router.post("/reload-faces")
async def reload_reference_faces():
    """Reload reference faces from the images folder."""
    from services.face_recognition_service import FaceRecognitionService

    faces = FaceRecognitionService.reload_reference_faces()

    return {
        "message": f"Reloaded {len(faces)} reference faces",
        "faces": list(faces.keys())
    }


@router.get("/reference-faces")
async def get_reference_faces():
    """Get list of currently loaded reference faces."""
    from services.face_recognition_service import FaceRecognitionService

    faces = FaceRecognitionService.get_reference_faces()

    return {
        "count": len(faces),
        "faces": [
            {
                "id": user_data["id"],
                "name": user_data["name"],
                "image_path": user_data.get("image_path", "unknown")
            }
            for user_data in faces.values()
        ]
    }


@router.get("/all")
async def get_all_attendance():
    """Get all attendance records."""
    # Create instance to access records
    service = AttendanceService()

    records = service.get_all_attendance()

    return {
        "count": len(records),
        "records": [
            {
                "id": record.id,
                "user_id": record.user_id,
                "organization_id": record.organization_id,
                "timestamp": record.timestamp,
                "location": {
                    "latitude": record.location.latitude,
                    "longitude": record.location.longitude,
                    "address": getattr(record.location, 'address', '')
                },
                "location_verified": record.location_verified,
                "location_distance_km": record.location_distance_km,
                "approved_location_id": record.approved_location_id,
                "face_verified": record.face_verified,
                "confidence_score": record.confidence_score,
                "device_info": record.device_info,
                "status": record.status,
                "notes": record.notes,
                "work_session_id": record.work_session_id,
                "work_hours_logged": record.work_hours_logged,
                "approved_by": record.approved_by,
                "approved_at": record.approved_at
            }
            for record in records
        ]
    }


@router.post("/add-face")
async def add_reference_face(
    name: str = Form(..., description="Name for the reference face"),
    image: UploadFile = File(...),
):
    """Add a new reference face to the images folder."""
    from pathlib import Path
    import shutil

    # Validate file type
    allowed_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    file_extension = Path(image.filename).suffix.lower()

    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}"
        )

    # Create images folder if it doesn't exist
    images_folder = Path("app/images")
    images_folder.mkdir(exist_ok=True)

    # Create filename with name prefix
    safe_name = name.replace(" ", "_").lower()
    filename = f"{safe_name}_{int(time.time())}{file_extension}"
    file_path = images_folder / filename

    try:
        # Save the uploaded file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)

        # Reload reference faces to include the new one
        from services.face_recognition_service import FaceRecognitionService
        FaceRecognitionService.reload_reference_faces()

        return {
            "message": f"Reference face '{name}' added successfully",
            "filename": filename,
            "path": str(file_path),
            "user_id": safe_name
        }

    except Exception as e:
        # Clean up file if there was an error
        if file_path.exists():
            file_path.unlink()

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save face: {str(e)}"
        )


@router.delete("/remove-face/{user_id}")
async def remove_reference_face(user_id: str):
    """Remove a reference face from the images folder."""
    from pathlib import Path
    import json

    images_folder = Path("app/images")

    if not images_folder.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Images folder not found"
        )

    # Find the image file for this user
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    found_file = None

    for image_file in images_folder.iterdir():
        if image_file.suffix.lower() not in image_extensions:
            continue

        # Check if filename starts with user_id
        if image_file.stem.lower().startswith(user_id.lower()):
            found_file = image_file
            break

    if not found_file:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No reference face found for user ID: {user_id}"
        )

    try:
        # Remove the file
        found_file.unlink()

        # Update processed images list to remove any references
        processed_file = Path("processed_images.json")
        if processed_file.exists():
            try:
                with open(processed_file, 'r') as f:
                    processed_images = set(json.load(f))

                # Remove any references to this file
                processed_images = {img for img in processed_images if user_id not in img}

                with open(processed_file, 'w') as f:
                    json.dump(list(processed_images), f)
            except:
                pass  # Ignore errors with processed images file

        # Reload reference faces
        from services.face_recognition_service import FaceRecognitionService
        FaceRecognitionService.reload_reference_faces()

        return {
            "message": f"Reference face for '{user_id}' removed successfully",
            "removed_file": str(found_file)
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to remove face: {str(e)}"
        )

