"""
Attendance router for marking attendance and viewing records
"""

from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
import base64
import os
import uuid

from app.database import get_db
from app.schemas.user import (
    AttendanceCreate, AttendanceResponse, AttendanceRecord, 
    AttendanceType, AttendanceLocation, AttendanceSession
)
from app.models.user import User, AttendanceRecord as AttendanceRecordModel, Organization, AttendanceSession as AttendanceSessionModel
from app.services.face_recognition_service import FaceRecognitionService
from app.services.location_service import LocationService
from app.routers.auth import get_current_user
from loguru import logger

router = APIRouter()
face_service = FaceRecognitionService()
location_service = LocationService()


def save_uploaded_image(image_data: str, user_id: int) -> str:
    """Save uploaded image and return file path"""
    try:
        # Create uploads directory if it doesn't exist
        upload_dir = "uploads/attendance"
        os.makedirs(upload_dir, exist_ok=True)
        
        # Generate unique filename
        filename = f"attendance_{user_id}_{uuid.uuid4().hex}.jpg"
        file_path = os.path.join(upload_dir, filename)
        
        # Decode and save image
        image_bytes = base64.b64decode(image_data)
        with open(file_path, "wb") as f:
            f.write(image_bytes)
        
        return file_path
        
    except Exception as e:
        logger.error(f"Error saving image: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to save image"
        )


@router.post("/mark", response_model=AttendanceResponse)
async def mark_attendance(
    attendance_data: AttendanceCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Mark attendance with face recognition and location verification"""
    try:
        # Get user's organization
        organization = db.query(Organization).filter(
            Organization.id == current_user.organization_id
        ).first()
        
        if not organization:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Organization not found"
            )
        
        # Verify location
        location_validation = location_service.validate_location(
            attendance_data.location.latitude,
            attendance_data.location.longitude,
            organization.latitude,
            organization.longitude,
            organization.radius,
            attendance_data.location.accuracy
        )
        
        if not location_validation.get("is_valid", False):
            return AttendanceResponse(
                success=False,
                message=f"Location not within allowed area. Distance: {location_validation.get('distance', 0):.2f}m, Allowed: {organization.radius}m",
                confidence_score=0.0
            )
        
        # Verify face recognition if image provided
        user_id = None
        confidence_score = 0.0
        
        if attendance_data.image_data:
            user_id, confidence_score, is_verified = face_service.verify_attendance(
                attendance_data.image_data
            )
            
            if not is_verified or user_id != current_user.id:
                return AttendanceResponse(
                    success=False,
                    message="Face recognition failed. Please ensure your face is clearly visible.",
                    confidence_score=confidence_score
                )
        else:
            # If no image provided, we'll still allow attendance but mark as unverified
            user_id = current_user.id
            confidence_score = 0.0
        
        # Save captured image if provided
        image_path = None
        if attendance_data.image_data:
            image_path = save_uploaded_image(attendance_data.image_data, current_user.id)
        
        # Create attendance record
        attendance_record = AttendanceRecordModel(
            user_id=current_user.id,
            organization_id=current_user.organization_id,
            attendance_type=attendance_data.attendance_type,
            latitude=attendance_data.location.latitude,
            longitude=attendance_data.location.longitude,
            address=location_validation.get("address"),
            accuracy=attendance_data.location.accuracy,
            face_confidence=confidence_score,
            is_verified=user_id == current_user.id and confidence_score > 0.5,
            captured_image_path=image_path
        )
        
        db.add(attendance_record)
        db.commit()
        db.refresh(attendance_record)
        
        # Handle attendance session logic
        await handle_attendance_session(
            db, current_user, attendance_data.attendance_type, 
            attendance_data.location.latitude, attendance_data.location.longitude
        )
        
        logger.info(f"Attendance marked successfully for user {current_user.username}")
        
        return AttendanceResponse(
            success=True,
            message="Attendance marked successfully",
            attendance_record=attendance_record,
            confidence_score=confidence_score
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error marking attendance: {e}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to mark attendance"
        )


async def handle_attendance_session(
    db: Session, 
    user: User, 
    attendance_type: AttendanceType,
    latitude: float,
    longitude: float
):
    """Handle attendance session logic for work hours tracking"""
    try:
        # Get active session for user
        active_session = db.query(AttendanceSessionModel).filter(
            AttendanceSessionModel.user_id == user.id,
            AttendanceSessionModel.is_active == True
        ).first()
        
        if attendance_type == AttendanceType.CHECK_IN:
            if active_session:
                # User already has an active session
                logger.warning(f"User {user.username} already has an active session")
                return
            
            # Create new session
            new_session = AttendanceSessionModel(
                user_id=user.id,
                organization_id=user.organization_id,
                check_in_time=datetime.utcnow(),
                check_in_latitude=latitude,
                check_in_longitude=longitude,
                is_active=True,
                status="active"
            )
            db.add(new_session)
            
        elif attendance_type == AttendanceType.CHECK_OUT:
            if not active_session:
                # No active session to check out from
                logger.warning(f"User {user.username} has no active session to check out")
                return
            
            # Calculate total hours
            total_hours = (datetime.utcnow() - active_session.check_in_time).total_seconds() / 3600
            
            # Update session
            active_session.check_out_time = datetime.utcnow()
            active_session.check_out_latitude = latitude
            active_session.check_out_longitude = longitude
            active_session.total_hours = total_hours
            active_session.is_active = False
            active_session.status = "completed"
        
        db.commit()
        
    except Exception as e:
        logger.error(f"Error handling attendance session: {e}")
        db.rollback()


@router.get("/records", response_model=List[AttendanceRecord])
async def get_attendance_records(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    attendance_type: Optional[AttendanceType] = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get attendance records for current user"""
    try:
        query = db.query(AttendanceRecordModel).filter(
            AttendanceRecordModel.user_id == current_user.id
        )
        
        if start_date:
            query = query.filter(AttendanceRecordModel.timestamp >= start_date)
        
        if end_date:
            query = query.filter(AttendanceRecordModel.timestamp <= end_date)
        
        if attendance_type:
            query = query.filter(AttendanceRecordModel.attendance_type == attendance_type)
        
        records = query.order_by(AttendanceRecordModel.timestamp.desc()).all()
        
        return records
        
    except Exception as e:
        logger.error(f"Error getting attendance records: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get attendance records"
        )


@router.get("/sessions", response_model=List[AttendanceSession])
async def get_attendance_sessions(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get attendance sessions for current user"""
    try:
        query = db.query(AttendanceSessionModel).filter(
            AttendanceSessionModel.user_id == current_user.id
        )
        
        if start_date:
            query = query.filter(AttendanceSessionModel.check_in_time >= start_date)
        
        if end_date:
            query = query.filter(AttendanceSessionModel.check_in_time <= end_date)
        
        sessions = query.order_by(AttendanceSessionModel.check_in_time.desc()).all()
        
        return sessions
        
    except Exception as e:
        logger.error(f"Error getting attendance sessions: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get attendance sessions"
        )


@router.get("/current-session", response_model=Optional[AttendanceSession])
async def get_current_session(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get current active attendance session"""
    try:
        session = db.query(AttendanceSessionModel).filter(
            AttendanceSessionModel.user_id == current_user.id,
            AttendanceSessionModel.is_active == True
        ).first()
        
        return session
        
    except Exception as e:
        logger.error(f"Error getting current session: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get current session"
        )


@router.get("/stats", response_model=Dict[str, Any])
async def get_attendance_stats(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get attendance statistics for current user"""
    try:
        # Set default date range if not provided
        if not start_date:
            start_date = datetime.utcnow().replace(day=1)  # First day of current month
        if not end_date:
            end_date = datetime.utcnow()
        
        # Get attendance records
        records = db.query(AttendanceRecordModel).filter(
            AttendanceRecordModel.user_id == current_user.id,
            AttendanceRecordModel.timestamp >= start_date,
            AttendanceRecordModel.timestamp <= end_date
        ).all()
        
        # Get sessions
        sessions = db.query(AttendanceSessionModel).filter(
            AttendanceSessionModel.user_id == current_user.id,
            AttendanceSessionModel.check_in_time >= start_date,
            AttendanceSessionModel.check_in_time <= end_date
        ).all()
        
        # Calculate statistics
        total_attendance = len(records)
        check_ins = len([r for r in records if r.attendance_type == AttendanceType.CHECK_IN])
        check_outs = len([r for r in records if r.attendance_type == AttendanceType.CHECK_OUT])
        
        total_hours = sum(s.total_hours or 0 for s in sessions if s.total_hours)
        completed_sessions = len([s for s in sessions if s.status == "completed"])
        
        average_hours = total_hours / completed_sessions if completed_sessions > 0 else 0
        
        # Calculate attendance rate (assuming 5 days per week)
        days_in_period = (end_date - start_date).days
        expected_attendance = (days_in_period // 7) * 5 * 2  # 2 per day (check-in, check-out)
        attendance_rate = (check_ins / expected_attendance * 100) if expected_attendance > 0 else 0
        
        return {
            "total_attendance": total_attendance,
            "check_ins": check_ins,
            "check_outs": check_outs,
            "total_hours": round(total_hours, 2),
            "average_hours_per_day": round(average_hours, 2),
            "attendance_rate": round(attendance_rate, 2),
            "completed_sessions": completed_sessions,
            "period": {
                "start_date": start_date,
                "end_date": end_date
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting attendance stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get attendance statistics"
        )
