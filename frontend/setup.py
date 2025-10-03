# setup.py - Run this to set up the environment
import os
import subprocess
import sys

def install_requirements():
    """Install required packages."""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing requirements: {e}")
        return False
    return True

def create_images_folder():
    """Create images folder for reference faces."""
    if not os.path.exists("images"):
        os.makedirs("images")
        print("✅ Created 'images' folder for reference faces")
    else:
        print("✅ 'images' folder already exists")

def main():
    print("🚀 Setting up AttendX Simple Attendance Server...")
    
    if install_requirements():
        create_images_folder()
        print("\n📋 Setup complete!")
        print("\n📁 Add reference face images to the 'images' folder")
        print("🏃 Run: python main.py")
        print("📖 API docs: http://localhost:8000/docs")
        print("🎯 Main endpoint: POST http://localhost:8000/mark-attendance")
    else:
        print("❌ Setup failed!")

if __name__ == "__main__":
    main()
