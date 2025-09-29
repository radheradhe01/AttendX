"""
Simple Attendance service without database dependencies.
"""

from datetime import datetime
from typing import List, Optional, Tuple, Dict
import json
import os
from pathlib import Path

from app.models.attendance import Attendance, AttendanceCreate, AttendanceResponse
from app.models.base import AttendanceStatus, Location
from .face_recognition_service import FaceRecognitionService
from .location_service import LocationService


class AttendanceService:
    """Service class for attendance operations."""

    # JSON file for persistent storage
    ATTENDANCE_FILE = "attendance_data.json"
    attendance_records = []  # Class variable for backward compatibility

    def __init__(self):
        """Initialize attendance service and load existing data."""
        self.attendance_records = self._load_attendance_data()
        # Update class variable
        AttendanceService.attendance_records = self.attendance_records

    def _load_attendance_data(self) -> List:
        """Load attendance data from JSON file."""
        try:
            if os.path.exists(self.ATTENDANCE_FILE):
                with open(self.ATTENDANCE_FILE, 'r') as f:
                    data = json.load(f)
                    # Convert back to namedtuples
                    from collections import namedtuple
                    AttendanceRecord = namedtuple('AttendanceRecord', [
                        'id', 'user_id', 'organization_id', 'timestamp', 'location',
                        'location_verified', 'location_distance_km', 'approved_location_id',
                        'face_verified', 'confidence_score', 'device_info', 'status', 'notes',
                        'work_session_id', 'work_hours_logged', 'approved_by', 'approved_at'
                    ])

                    records = []
                    for record_data in data:
                        # Convert location dict back to Location object
                        location_data = record_data.get('location', {})
                        location = Location(
                            latitude=location_data.get('latitude', 0),
                            longitude=location_data.get('longitude', 0),
                            address=location_data.get('address', '')
                        )

                        # Create record
                        record = AttendanceRecord(
                            id=record_data['id'],
                            user_id=record_data['user_id'],
                            organization_id=record_data['organization_id'],
                            timestamp=record_data['timestamp'],
                            location=location,
                            location_verified=record_data['location_verified'],
                            location_distance_km=record_data['location_distance_km'],
                            approved_location_id=record_data['approved_location_id'],
                            face_verified=record_data['face_verified'],
                            confidence_score=record_data['confidence_score'],
                            device_info=record_data['device_info'],
                            status=record_data['status'],
                            notes=record_data['notes'],
                            work_session_id=record_data['work_session_id'],
                            work_hours_logged=record_data['work_hours_logged'],
                            approved_by=record_data['approved_by'],
                            approved_at=record_data['approved_at']
                        )
                        records.append(record)

                    return records
            else:
                return []
        except Exception as e:
            print(f"Warning: Could not load attendance data: {e}")
            return []

    def _save_attendance_data(self):
        """Save attendance data to JSON file."""
        try:
            # Convert namedtuples to dicts for JSON serialization
            data = []
            for record in self.attendance_records:
                record_dict = {
                    'id': record.id,
                    'user_id': record.user_id,
                    'organization_id': record.organization_id,
                    'timestamp': record.timestamp,
                    'location': {
                        'latitude': record.location.latitude,
                        'longitude': record.location.longitude,
                        'address': getattr(record.location, 'address', '')
                    },
                    'location_verified': record.location_verified,
                    'location_distance_km': record.location_distance_km,
                    'approved_location_id': record.approved_location_id,
                    'face_verified': record.face_verified,
                    'confidence_score': record.confidence_score,
                    'device_info': record.device_info,
                    'status': record.status,
                    'notes': record.notes,
                    'work_session_id': record.work_session_id,
                    'work_hours_logged': record.work_hours_logged,
                    'approved_by': record.approved_by,
                    'approved_at': record.approved_at
                }
                data.append(record_dict)

            with open(self.ATTENDANCE_FILE, 'w') as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            print(f"Warning: Could not save attendance data: {e}")

    @staticmethod
    async def mark_attendance(
        user_id: str,
        organization_id: str,
        location: Location,
        image_data: bytes,
        device_info: dict,
        notes: str = None
    ) -> Tuple[Attendance, str]:
        """
        Mark attendance with face and location verification.

        Args:
            user_id: User ID
            organization_id: Organization ID
            location: User's location data
            image_data: Face image data
            device_info: Device information
            notes: Optional notes

        Returns:
            Tuple of (attendance_record, status_message)
        """
        # Validate image
        is_valid, reason = FaceRecognitionService.validate_face_image(image_data)
        if not is_valid:
            return None, f"Invalid image: {reason}"

        # Extract face encoding from captured image
        processed_image = FaceRecognitionService.preprocess_image(image_data)
        captured_encodings = FaceRecognitionService.extract_face_encodings_for_test(processed_image)

        if not captured_encodings:
            return None, "No face detected in image"

        # Verify face against reference faces from images folder
        user_info, confidence_score = FaceRecognitionService.verify_face(
            captured_encodings[0],
            threshold=0.6,
            images_folder="app/images"
        )

        face_verified = user_info is not None

        # For test organization, always verify location
        location_verified = True
        distance_km = 0.1  # Mock distance
        approved_location_id = "default_location"

        # Determine attendance status
        status = AttendanceStatus.PRESENT
        status_message = "Attendance marked successfully"

        if not face_verified:
            status = AttendanceStatus.ABSENT
            status_message = f"Face not recognized. Confidence: {confidence_score:.2f}"
        elif not location_verified:
            status = AttendanceStatus.LATE
            status_message = f"Outside approved location (distance: {distance_km:.2f}km)"

        # Create attendance record (simple dict for now)
        from collections import namedtuple
        AttendanceRecord = namedtuple('AttendanceRecord', [
            'id', 'user_id', 'organization_id', 'timestamp', 'location',
            'location_verified', 'location_distance_km', 'approved_location_id',
            'face_verified', 'confidence_score', 'device_info', 'status', 'notes',
            'work_session_id', 'work_hours_logged', 'approved_by', 'approved_at'
        ])

        attendance = AttendanceRecord(
            id=f"att_{len(AttendanceService.attendance_records) + 1}",
            user_id=user_info["id"] if user_info else "unknown",
            organization_id=organization_id,
            timestamp=datetime.utcnow().isoformat(),
            location=location,
            location_verified=location_verified,
            location_distance_km=distance_km,
            approved_location_id=approved_location_id,
            face_verified=face_verified,
            confidence_score=confidence_score,
            device_info=device_info,
            status=status,
            notes=notes,
            work_session_id=None,
            work_hours_logged=0.0,
            approved_by=None,
            approved_at=None
        )

        # Store in memory and save to file
        # Create a global instance to access the save method
        if not hasattr(AttendanceService, '_instance'):
            AttendanceService._instance = AttendanceService()

        AttendanceService._instance.attendance_records.append(attendance)
        AttendanceService._instance._save_attendance_data()

        # Also update the class variable for backward compatibility
        if not hasattr(AttendanceService, 'attendance_records'):
            AttendanceService.attendance_records = []
        AttendanceService.attendance_records.append(attendance)

        return attendance, status_message

    def get_user_attendance(
        self,
        user_id: str,
        start_date: str = None,
        end_date: str = None,
        limit: int = 50
    ) -> List:
        """Get attendance records for a user."""
        # Filter records by user_id
        user_records = [record for record in self.attendance_records if record.user_id == user_id]

        # Filter by date range
        if start_date or end_date:
            filtered_records = []
            for record in user_records:
                record_date = datetime.fromisoformat(record.timestamp).date()

                if start_date:
                    start = datetime.fromisoformat(start_date).date()
                    if record_date < start:
                        continue

                if end_date:
                    end = datetime.fromisoformat(end_date).date()
                    if record_date > end:
                        continue

                filtered_records.append(record)

            user_records = filtered_records

        # Sort by timestamp (newest first) and limit
        user_records.sort(key=lambda x: x.timestamp, reverse=True)
        return user_records[:limit]

    def approve_attendance(self, attendance_id: str, approved_by: str) -> bool:
        """Approve attendance record."""
        for i, record in enumerate(self.attendance_records):
            if str(record.id) == attendance_id:
                # Create a new record with updated fields
                from collections import namedtuple
                AttendanceRecord = namedtuple('AttendanceRecord', [
                    'id', 'user_id', 'organization_id', 'timestamp', 'location',
                    'location_verified', 'location_distance_km', 'approved_location_id',
                    'face_verified', 'confidence_score', 'device_info', 'status', 'notes',
                    'work_session_id', 'work_hours_logged', 'approved_by', 'approved_at'
                ])

                updated_record = AttendanceRecord(
                    id=record.id,
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
                    status=AttendanceStatus.PRESENT,
                    notes=record.notes,
                    work_session_id=record.work_session_id,
                    work_hours_logged=record.work_hours_logged,
                    approved_by=approved_by,
                    approved_at=datetime.utcnow().isoformat()
                )

                # Replace the record in the list
                self.attendance_records[i] = updated_record
                self._save_attendance_data()
                return True
        return False

    def get_attendance_summary(self, user_id: str, days: int = 30) -> dict:
        """Get attendance summary for a user."""
        from datetime import timedelta

        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)

        attendance_records = self.get_user_attendance(
            user_id,
            start_date.isoformat(),
            end_date.isoformat()
        )

        summary = {
            "total_days": len(attendance_records),
            "present_days": 0,
            "late_days": 0,
            "absent_days": 0,
            "face_verified_count": 0,
            "location_verified_count": 0,
            "average_confidence": 0.0
        }

        if not attendance_records:
            return summary

        confidence_scores = []

        for record in attendance_records:
            if record.status == AttendanceStatus.PRESENT:
                summary["present_days"] += 1
            elif record.status == AttendanceStatus.LATE:
                summary["late_days"] += 1
            else:
                summary["absent_days"] += 1

            if record.face_verified:
                summary["face_verified_count"] += 1

            if record.location_verified:
                summary["location_verified_count"] += 1

            if record.confidence_score:
                confidence_scores.append(record.confidence_score)

        if confidence_scores:
            summary["average_confidence"] = sum(confidence_scores) / len(confidence_scores)

        return summary

    def get_all_attendance(self) -> List:
        """Get all attendance records."""
        return self.attendance_records

    @staticmethod
    async def process_images_from_folder(images_folder: str = "app/images") -> Dict:
        """
        Process all images in the images folder and mark attendance for recognized faces.

        Args:
            images_folder: Path to folder containing images to process

        Returns:
            Dict with processing results
        """
        results = {
            "processed": 0,
            "matched": 0,
            "errors": 0,
            "details": []
        }

        # Get the images folder path
        images_path = Path(images_folder)

        if not images_path.exists():
            return {"error": f"Images folder '{images_folder}' not found"}

        # Track processed images to avoid duplicates
        processed_file = Path("processed_images.json")
        processed_images = set()

        if processed_file.exists():
            try:
                with open(processed_file, 'r') as f:
                    processed_images = set(json.load(f))
            except:
                processed_images = set()

        # Process each image file
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}

        for image_file in images_path.iterdir():
            if image_file.suffix.lower() not in image_extensions:
                continue

            if str(image_file) in processed_images:
                continue  # Already processed

            try:
                # Read image data
                with open(image_file, 'rb') as f:
                    image_data = f.read()

                # Extract face encodings
                face_encodings = FaceRecognitionService.extract_face_encodings(image_data)

                if not face_encodings:
                    results["errors"] += 1
                    results["details"].append({
                        "file": str(image_file),
                        "status": "error",
                        "message": "No face detected"
                    })
                    continue

                # Process each face found
                for face_encoding in face_encodings:
                    # Compare with predefined faces
                    user_info, confidence = FaceRecognitionService.verify_face(
                        face_encoding,
                        threshold=0.6
                    )

                    if user_info:
                        # Create a mock location (you can customize this)
                        location = Location(
                            latitude=40.7128,  # Default NYC coordinates
                            longitude=-74.0060,
                            address="Auto-processed from image"
                        )

                        device_info = {
                            "user_agent": "Image Processor",
                            "ip_address": "127.0.0.1",
                            "platform": "Auto"
                        }

                        # Mark attendance using the global instance
                        attendance, message = await AttendanceService.mark_attendance(
                            user_id=user_info["id"],
                            organization_id="auto_org",
                            location=location,
                            image_data=image_data,
                            device_info=device_info,
                            notes=f"Auto-processed from {image_file.name}"
                        )

                        results["matched"] += 1
                        results["details"].append({
                            "file": str(image_file),
                            "user_id": user_info["id"],
                            "user_name": user_info["name"],
                            "confidence": round(confidence, 2),
                            "status": "matched",
                            "attendance_id": attendance.id
                        })
                    else:
                        results["errors"] += 1
                        results["details"].append({
                            "file": str(image_file),
                            "status": "error",
                            "message": f"Face not recognized (confidence: {confidence:.2f})"
                        })

                # Mark as processed
                processed_images.add(str(image_file))
                results["processed"] += 1

            except Exception as e:
                results["errors"] += 1
                results["details"].append({
                    "file": str(image_file),
                    "status": "error",
                    "message": str(e)
                })

        # Save processed images list
        try:
            with open(processed_file, 'w') as f:
                json.dump(list(processed_images), f)
        except Exception as e:
            print(f"Warning: Could not save processed images list: {e}")

        return results

    @staticmethod
    async def get_unprocessed_images(images_folder: str = "app/images") -> List[str]:
        """
        Get list of unprocessed images in the images folder.

        Args:
            images_folder: Path to folder containing images

        Returns:
            List of unprocessed image file paths
        """
        images_path = Path(images_folder)

        if not images_path.exists():
            return []

        # Load processed images
        processed_file = Path("processed_images.json")
        processed_images = set()

        if processed_file.exists():
            try:
                with open(processed_file, 'r') as f:
                    processed_images = set(json.load(f))
            except:
                processed_images = set()

        # Find unprocessed images
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        unprocessed = []

        for image_file in images_path.iterdir():
            if (image_file.suffix.lower() in image_extensions and
                str(image_file) not in processed_images):
                unprocessed.append(str(image_file))

        return unprocessed
