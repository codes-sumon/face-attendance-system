# Face Attendance System

An automated attendance tracking system using **InsightFace** for AI-powered face recognition and **Supabase** (PostgreSQL) for data storage.

## Features

- **Student Registration** — Register with Student ID, Name, and a face photo
- **Face Detection & Recognition** — Powered by InsightFace's ArcFace model
- **Attendance Marking** — Via webcam capture or image upload
- **Attendance Reports** — View, filter, and analyze attendance records
- **Real-time API** — REST endpoints for integration with other systems

## Tech Stack

| Component  | Technology                                |
| ---------- | ----------------------------------------- |
| Backend    | Python, Flask                             |
| Face AI    | InsightFace (buffalo_l / ArcFace)         |
| Database   | Supabase (PostgreSQL)                     |
| Frontend   | HTML, CSS, JavaScript (vanilla)           |
| Image Proc | OpenCV, NumPy, Pillow                     |

## Prerequisites

- Python 3.10+
- Supabase account (free tier works)
- Webcam (optional, for live capture)

## Setup

### 1. Clone & Install

```bash
cd face-attendance-system
pip install -r requirements.txt
```

### 2. Supabase Setup

Create a Supabase project and run the following SQL in the **SQL Editor**:

```sql
-- Students table
CREATE TABLE students (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    student_id TEXT UNIQUE NOT NULL,
    name TEXT NOT NULL,
    face_embedding JSONB NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Attendance table
CREATE TABLE attendance (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    student_id TEXT NOT NULL REFERENCES students(student_id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    confidence FLOAT NOT NULL
);

-- Indexes for performance
CREATE INDEX idx_student_id ON students(student_id);
CREATE INDEX idx_attendance_timestamp ON attendance(timestamp DESC);
CREATE INDEX idx_attendance_student ON attendance(student_id);
```

### 3. Configure Environment

```bash
cp .env.example .env
```

Edit `.env` with your Supabase credentials:

```
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-supabase-service-role-key
FLASK_SECRET_KEY=your-random-secret-key
```

### 4. Run

```bash
python app.py
```

Open **http://localhost:5000** in your browser.

## Usage

1. **Register Students** — Go to `/register`, enter ID & name, upload or capture a face photo
2. **Mark Attendance** — Go to `/mark-attendance`, capture/upload a face to auto-identify and log
3. **View Reports** — Go to `/attendance` to see all records, filter by student

## Project Structure

```
face-attendance-system/
├── app.py                  # Flask application (routes, API)
├── face_utils.py           # InsightFace detection & recognition
├── supabase_utils.py       # Supabase CRUD operations
├── requirements.txt        # Python dependencies
├── .env.example            # Environment template
├── README.md
├── schema.sql              # Database schema reference
├── templates/
│   ├── base.html           # Layout template
│   ├── index.html          # Home page
│   ├── register.html       # Student registration
│   ├── mark_attendance.html
│   └── attendance_report.html
└── static/
    ├── css/style.css
    └── js/
        ├── main.js         # General UI logic
        └── camera.js       # Webcam capture module
```

## API Endpoints

| Method | Endpoint               | Description                    |
| ------ | ---------------------- | ------------------------------ |
| GET    | `/api/students`        | List all registered students   |
| DELETE | `/api/students/<id>`   | Delete a student                |
| GET    | `/api/attendance/today`| Today's attendance records     |
| POST   | `/api/check-face`      | Check face against database    |

## License

MIT
