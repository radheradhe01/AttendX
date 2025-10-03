#!/usr/bin/env python3
"""
Setup script for the simple attendance system.
"""

import os
import subprocess
import sys
import shutil
from pathlib import Path

def install_requirements():
    """Install required packages."""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing requirements: {e}")
        return False

def create_folders():
    """Create necessary folders."""
    folders = ["reference_images", "attendance_images"]
    
    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder)
            print(f"✅ Created '{folder}' folder")
        else:
            print(f"✅ '{folder}' folder already exists")

def copy_reference_image():
    """Copy the reference image from backend to reference_images folder."""
    source_path = "/Users/bhaveshvarma/projects/miniP-attendX/AttendX/back/AttendX/app/images/bhaveshvarma.jpg"
    dest_path = "reference_images/bhaveshvarma.jpg"
    
    if os.path.exists(source_path):
        try:
            shutil.copy2(source_path, dest_path)
            print(f"✅ Copied reference image: {dest_path}")
        except Exception as e:
            print(f"❌ Error copying reference image: {e}")
    else:
        print(f"⚠️  Source image not found: {source_path}")
        print("   Please manually add reference images to the 'reference_images' folder")

def main():
    print("🚀 Setting up AttendX Simple Personal Face Verification...")
    print("=" * 60)
    
    if install_requirements():
        create_folders()
        copy_reference_image()
        
        print("\n" + "=" * 60)
        print("📋 Setup complete!")
        print("\n📁 Folder structure:")
        print("   reference_images/  - Put one reference image per person here")
        print("   attendance_images/ - Attendance photos will be saved here")
        print("\n🏃 Run: python3 simple_attendance.py")
        print("📖 API docs: http://localhost:8000/docs")
        print("🎯 Main endpoint: POST http://localhost:8000/mark-attendance")
        print("\n💡 Usage:")
        print("   1. Add reference images to 'reference_images/' folder")
        print("   2. When marking attendance, provide person_name and their photo")
        print("   3. System compares the photo with their reference image")
    else:
        print("❌ Setup failed!")

if __name__ == "__main__":
    main()
