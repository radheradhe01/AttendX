"""
Admin router for system administration and reporting
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy import func, desc
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any

from app.database import get_db
from app.schemas.user import User, AttendanceRecord, AttendanceSession, AttendanceStats
from app.models.user import (
    User as UserModel, 
    AttendanceRecord as AttendanceRecordModel,
    AttendanceSession as AttendanceSessionModel,
    Organization as OrganizationModel
)
from app.routers.auth import get_current_user
from app.services.face_recognition_service import FaceRecognitionService
from loguru import logger

router = APIRouter()
face_service = FaceRecognitionService()


def require_admin(current_user: UserModel = Depends(get_current_user)):
    """Require admin role"""
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    return current_user


@router.get("/stats/overview")
async def get_system_overview(
    current_user: UserModel = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """Get system overview statistics"""
    try:
        # Get counts
        total_organizations = db.query(OrganizationModel).count()
        total_users = db.query(UserModel).count()
        active_users = db.query(UserModel).filter(UserModel.is_active == True).count()
        total_attendance_records = db.query(AttendanceRecordModel).count()
        
        # Get today's attendance
        today = datetime.utcnow().date()
        today_attendance = db.query(AttendanceRecordModel).filter(
            func.date(AttendanceRecordModel.timestamp) == today
        ).count()
        
        # Get face recognition stats
        face_stats = face_service.get_stats()
        
        return {
            "total_organizations": total_organizations,
            "total_users": total_users,
            "active_users": active_users,
            "total_attendance_records": total_attendance_records,
            "today_attendance": today_attendance,
            "face_recognition": face_stats
        }
        
    except Exception as e:
        logger.error(f"Error getting system overview: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get system overview"
        )


@router.get("/stats/attendance")
async def get_attendance_statistics(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    organization_id: Optional[int] = None,
    current_user: UserModel = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """Get attendance statistics"""
    try:
        # Set default date range if not provided
        if not start_date:
            start_date = datetime.utcnow() - timedelta(days=30)
        if not end_date:
            end_date = datetime.utcnow()
        
        # Build query
        query = db.query(AttendanceRecordModel).filter(
            AttendanceRecordModel.timestamp >= start_date,
            AttendanceRecordModel.timestamp <= end_date
        )
        
        if organization_id:
            query = query.filter(AttendanceRecordModel.organization_id == organization_id)
        
        records = query.all()
        
        # Calculate statistics
        total_records = len(records)
        check_ins = len([r for r in records if r.attendance_type == "check_in"])
        check_outs = len([r for r in records if r.attendance_type == "check_out"])
        verified_records = len([r for r in records if r.is_verified])
        
        # Average confidence score
        confidence_scores = [r.face_confidence for r in records if r.face_confidence is not None]
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
        
        # Daily breakdown
        daily_stats = {}
        for record in records:
            date_key = record.timestamp.date().isoformat()
            if date_key not in daily_stats:
                daily_stats[date_key] = {"check_ins": 0, "check_outs": 0}
            daily_stats[date_key][record.attendance_type] += 1
        
        return {
            "period": {
                "start_date": start_date,
                "end_date": end_date
            },
            "total_records": total_records,
            "check_ins": check_ins,
            "check_outs": check_outs,
            "verified_records": verified_records,
            "verification_rate": (verified_records / total_records * 100) if total_records > 0 else 0,
            "average_confidence": round(avg_confidence, 3),
            "daily_breakdown": daily_stats
        }
        
    except Exception as e:
        logger.error(f"Error getting attendance statistics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get attendance statistics"
        )


@router.get("/users/attendance-summary")
async def get_users_attendance_summary(
    organization_id: Optional[int] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    current_user: UserModel = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """Get attendance summary for all users"""
    try:
        # Set default date range if not provided
        if not start_date:
            start_date = datetime.utcnow() - timedelta(days=30)
        if not end_date:
            end_date = datetime.utcnow()
        
        # Build user query
        user_query = db.query(UserModel)
        if organization_id:
            user_query = user_query.filter(UserModel.organization_id == organization_id)
        
        users = user_query.all()
        
        user_summaries = []
        
        for user in users:
            # Get user's attendance records
            records = db.query(AttendanceRecordModel).filter(
                AttendanceRecordModel.user_id == user.id,
                AttendanceRecordModel.timestamp >= start_date,
                AttendanceRecordModel.timestamp <= end_date
            ).all()
            
            # Get user's sessions
            sessions = db.query(AttendanceSessionModel).filter(
                AttendanceSessionModel.user_id == user.id,
                AttendanceSessionModel.check_in_time >= start_date,
                AttendanceSessionModel.check_in_time <= end_date
            ).all()
            
            # Calculate stats
            total_attendance = len(records)
            check_ins = len([r for r in records if r.attendance_type == "check_in"])
            check_outs = len([r for r in records if r.attendance_type == "check_out"])
            total_hours = sum(s.total_hours or 0 for s in sessions if s.total_hours)
            completed_sessions = len([s for s in sessions if s.status == "completed"])
            
            average_hours = total_hours / completed_sessions if completed_sessions > 0 else 0
            
            # Calculate attendance rate (assuming 5 days per week)
            days_in_period = (end_date - start_date).days
            expected_attendance = (days_in_period // 7) * 5 * 2  # 2 per day
            attendance_rate = (check_ins / expected_attendance * 100) if expected_attendance > 0 else 0
            
            user_summaries.append({
                "user": {
                    "id": user.id,
                    "username": user.username,
                    "full_name": user.full_name,
                    "email": user.email,
                    "role": user.role,
                    "organization_id": user.organization_id
                },
                "stats": {
                    "total_attendance": total_attendance,
                    "check_ins": check_ins,
                    "check_outs": check_outs,
                    "total_hours": round(total_hours, 2),
                    "average_hours_per_day": round(average_hours, 2),
                    "attendance_rate": round(attendance_rate, 2),
                    "completed_sessions": completed_sessions
                }
            })
        
        return {
            "period": {
                "start_date": start_date,
                "end_date": end_date
            },
            "user_summaries": user_summaries
        }
        
    except Exception as e:
        logger.error(f"Error getting users attendance summary: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get users attendance summary"
        )


@router.get("/organizations/stats")
async def get_organizations_statistics(
    current_user: UserModel = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """Get statistics for all organizations"""
    try:
        organizations = db.query(OrganizationModel).all()
        
        org_stats = []
        
        for org in organizations:
            # Get user count
            user_count = db.query(UserModel).filter(
                UserModel.organization_id == org.id
            ).count()
            
            # Get active user count
            active_user_count = db.query(UserModel).filter(
                UserModel.organization_id == org.id,
                UserModel.is_active == True
            ).count()
            
            # Get attendance records count
            attendance_count = db.query(AttendanceRecordModel).filter(
                AttendanceRecordModel.organization_id == org.id
            ).count()
            
            # Get today's attendance
            today = datetime.utcnow().date()
            today_attendance = db.query(AttendanceRecordModel).filter(
                AttendanceRecordModel.organization_id == org.id,
                func.date(AttendanceRecordModel.timestamp) == today
            ).count()
            
            org_stats.append({
                "organization": {
                    "id": org.id,
                    "name": org.name,
                    "description": org.description,
                    "is_active": org.is_active
                },
                "stats": {
                    "total_users": user_count,
                    "active_users": active_user_count,
                    "total_attendance_records": attendance_count,
                    "today_attendance": today_attendance
                }
            })
        
        return {"organizations": org_stats}
        
    except Exception as e:
        logger.error(f"Error getting organizations statistics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get organizations statistics"
        )


@router.get("/face-recognition/stats")
async def get_face_recognition_stats(
    current_user: UserModel = Depends(require_admin)
):
    """Get face recognition service statistics"""
    try:
        stats = face_service.get_stats()
        return stats
        
    except Exception as e:
        logger.error(f"Error getting face recognition stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get face recognition stats"
        )


@router.post("/face-recognition/update-tolerance")
async def update_face_recognition_tolerance(
    tolerance: float,
    current_user: UserModel = Depends(require_admin)
):
    """Update face recognition tolerance"""
    try:
        if not 0.0 <= tolerance <= 1.0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Tolerance must be between 0.0 and 1.0"
            )
        
        face_service.update_tolerance(tolerance)
        
        return {
            "message": "Face recognition tolerance updated successfully",
            "new_tolerance": tolerance
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating face recognition tolerance: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update face recognition tolerance"
        )


@router.get("/reports/export")
async def export_attendance_report(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    organization_id: Optional[int] = None,
    format: str = "json",
    current_user: UserModel = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """Export attendance report"""
    try:
        # Set default date range if not provided
        if not start_date:
            start_date = datetime.utcnow() - timedelta(days=30)
        if not end_date:
            end_date = datetime.utcnow()
        
        # Build query
        query = db.query(AttendanceRecordModel).filter(
            AttendanceRecordModel.timestamp >= start_date,
            AttendanceRecordModel.timestamp <= end_date
        )
        
        if organization_id:
            query = query.filter(AttendanceRecordModel.organization_id == organization_id)
        
        records = query.order_by(AttendanceRecordModel.timestamp.desc()).all()
        
        # Format data for export
        export_data = []
        for record in records:
            export_data.append({
                "id": record.id,
                "user_id": record.user_id,
                "organization_id": record.organization_id,
                "attendance_type": record.attendance_type,
                "timestamp": record.timestamp.isoformat(),
                "latitude": record.latitude,
                "longitude": record.longitude,
                "address": record.address,
                "accuracy": record.accuracy,
                "face_confidence": record.face_confidence,
                "is_verified": record.is_verified,
                "verification_notes": record.verification_notes
            })
        
        return {
            "export_data": export_data,
            "metadata": {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "organization_id": organization_id,
                "total_records": len(export_data),
                "exported_at": datetime.utcnow().isoformat()
            }
        }
        
    except Exception as e:
        logger.error(f"Error exporting attendance report: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to export attendance report"
        )
