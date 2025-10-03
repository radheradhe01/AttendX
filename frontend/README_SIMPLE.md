# AttendX Simple Personal Face Verification System

A simplified FastAPI server for attendance marking that verifies each person against their own reference image.

## How It Works

1. **Reference Images**: Each person has their own reference image stored in the `reference_images/` folder
2. **Attendance Marking**: When someone marks attendance, they provide their name and a photo
3. **Verification**: The system compares their photo with their own reference image
4. **Result**: Attendance is marked as "present" if verification passes, "absent" if it fails

## Setup

### 1. Install Dependencies
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install packages
pip install -r requirements.txt
```

### 2. Add Reference Images
Add reference face images to the `reference_images/` folder:
```
reference_images/
├── john_doe.jpg
├── jane_smith.png
└── sonakshi.jpg
```

**Important**: The filename (without extension) becomes the person's name in the system.

### 3. Start Server
```bash
python3 simple_attendance.py
```

## API Endpoints

### 1. **POST /mark-attendance** - Mark Attendance
Mark attendance by verifying against personal reference image.

**Request:**
```bash
curl -X POST "http://localhost:8000/mark-attendance" \
  -F "person_name=sonakshi" \
  -F "image=@photo.jpg" \
  -F "notes=Regular attendance"
```

**Response:**
```json
{
  "message": "Attendance marked successfully for sonakshi",
  "attendance_id": "att_1",
  "status": "present",
  "face_verified": true,
  "confidence_score": 0.85,
  "timestamp": "2024-01-15T10:30:00.000Z",
  "person_name": "sonakshi"
}
```

### 2. **GET /attendance-records** - Get All Attendance Records
Retrieve all attendance records.

**Response:**
```json
{
  "count": 2,
  "records": [
    {
      "id": "att_1",
      "person_name": "sonakshi",
      "timestamp": "2024-01-15T10:30:00.000Z",
      "face_verified": true,
      "confidence_score": 0.85,
      "status": "present",
      "notes": "Regular attendance",
      "reference_image_path": "reference_images/sonakshi.jpg",
      "attendance_image_path": "attendance_images/sonakshi_20240115_103000.jpg"
    }
  ]
}
```

### 3. **GET /reference-images** - Get Reference Images List
List all available reference images.

**Response:**
```json
{
  "count": 2,
  "images": [
    {
      "person_name": "sonakshi",
      "image_path": "reference_images/sonakshi.jpg"
    },
    {
      "person_name": "suraj",
      "image_path": "reference_images/suraj.png"
    }
  ]
}
```

### 4. **POST /add-reference-image** - Add Reference Image
Add a new reference image for a person.

**Request:**
```bash
curl -X POST "http://localhost:8000/add-reference-image" \
  -F "person_name=john_doe" \
  -F "image=@reference_photo.jpg"
```

### 5. **GET /health** - Health Check
Check if the service is running and get reference images count.

## File Structure

```
app/
├── simple_attendance.py      # Main FastAPI server
├── requirements.txt          # Python dependencies
├── reference_images/         # Reference face images (one per person)
│   ├── sonakshi.jpg
│   └── suraj.png
├── attendance_images/        # Attendance photos (auto-created)
│   └── sonakshi_20240115_103000.jpg
└── attendance_data.json      # Attendance records (auto-created)
```

## Configuration

- **Face Recognition Threshold**: 0.4 (configurable in code)
- **Server Port**: 8000
- **CORS**: Enabled for all origins (development mode)

## Usage Examples

### Mark Attendance
```bash
# Mark attendance for sonakshi
curl -X POST "http://localhost:8000/mark-attendance" \
  -F "person_name=sonakshi" \
  -F "image=@/path/to/photo.jpg"
```

### Check Attendance Records
```bash
curl -X GET "http://localhost:8000/attendance-records"
```

### Add New Person
```bash
# Add reference image for new person
curl -X POST "http://localhost:8000/add-reference-image" \
  -F "person_name=new_person" \
  -F "image=@/path/to/reference.jpg"
```

## Notes

- Each person must have exactly one reference image
- Reference images should be clear, single-face photos
- The system compares the uploaded photo with the person's own reference image
- Attendance photos are automatically saved for record keeping
- Face recognition confidence threshold is set to 0.4 (adjustable)

## API Documentation

Access the interactive API documentation at: http://localhost:8000/docs
