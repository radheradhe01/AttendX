"""
Reports and analytics router.
"""

from fastapi import APIRouter

router = APIRouter()


@router.get("/attendance")
async def get_attendance_report():
    """Get attendance report."""
    return {"message": "Attendance report - TODO: Implement"}


@router.get("/work-hours")
async def get_work_hours_report():
    """Get work hours report."""
    return {"message": "Work hours report - TODO: Implement"}

