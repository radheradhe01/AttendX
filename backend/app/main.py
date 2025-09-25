"""
AttendX - Main FastAPI Application
AI-Powered Attendance Management System
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import os
from loguru import logger

from app.database import engine, Base
from app.routers import auth, users, attendance, organizations, admin
from app.services.face_recognition_service import FaceRecognitionService


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    # Startup
    logger.info("Starting AttendX application...")
    
    # Create database tables
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables created successfully")
    
    # Initialize face recognition service
    try:
        face_service = FaceRecognitionService()
        await face_service.initialize()
        logger.info("Face recognition service initialized")
    except Exception as e:
        logger.error(f"Failed to initialize face recognition service: {e}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down AttendX application...")


# Create FastAPI application
app = FastAPI(
    title="AttendX API",
    description="AI-Powered Attendance Management System with Face Recognition and Geolocation",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],  # Frontend URLs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth.router, prefix="/api/auth", tags=["Authentication"])
app.include_router(users.router, prefix="/api/users", tags=["Users"])
app.include_router(attendance.router, prefix="/api/attendance", tags=["Attendance"])
app.include_router(organizations.router, prefix="/api/organizations", tags=["Organizations"])
app.include_router(admin.router, prefix="/api/admin", tags=["Admin"])

# Static files for uploaded images
if not os.path.exists("uploads"):
    os.makedirs("uploads")
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Welcome to AttendX API",
        "version": "1.0.0",
        "docs": "/docs",
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "AttendX API"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
