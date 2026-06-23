# Face Attendance System

## Project Overview
A Python Flask web application that uses InsightFace for face detection and recognition to automate attendance logging. Data is stored in Supabase (PostgreSQL).

## Tech Stack
- Python 3.10+
- Flask (web framework)
- InsightFace (face detection & recognition)
- Supabase Python Client (database)
- OpenCV (image processing & webcam)
- HTML/CSS/JS (frontend)

## Project Structure
```
face-attendance-system/
├── app.py                 # Main Flask application
├── face_utils.py          # Face detection & recognition module
├── supabase_utils.py      # Supabase database operations
├── requirements.txt       # Python dependencies
├── .env.example           # Environment variables template
├── README.md
├── templates/             # HTML templates
│   ├── base.html
│   ├── index.html
│   ├── register.html
│   ├── mark_attendance.html
│   └── attendance_report.html
└── static/
    ├── css/
    │   └── style.css
    └── js/
        └── main.js
```

## Database Schema (Supabase)

### students table
- `id` (bigint, primary key)
- `student_id` (text, unique) - Student ID provided by user
- `name` (text) - Student name
- `face_embedding` (jsonb) - 512-d face embedding from InsightFace
- `created_at` (timestamptz)

### attendance table
- `id` (bigint, primary key)
- `student_id` (text) - References students.student_id
- `name` (text) - Student name at time of attendance
- `timestamp` (timestamptz) - When attendance was marked
- `confidence` (float) - Face matching confidence score

## Setup Instructions
1. Clone the project
2. Install dependencies: `pip install -r requirements.txt`
3. Copy `.env.example` to `.env` and fill in Supabase credentials
4. Run: `python app.py`
5. Open http://localhost:5000
