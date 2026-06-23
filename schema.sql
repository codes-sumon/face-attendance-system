-- Face Attendance System — Supabase Database Schema
-- Run this in your Supabase SQL Editor before using the application.
--
-- Migration for existing databases:
--   ALTER TABLE students ADD COLUMN face_image TEXT DEFAULT NULL;
--   ALTER TABLE attendance ALTER COLUMN name DROP NOT NULL;

-- ─── Students Table ───
CREATE TABLE IF NOT EXISTS students (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    student_id TEXT UNIQUE NOT NULL,
    name TEXT NOT NULL,
    face_embedding JSONB NOT NULL,
    face_image TEXT DEFAULT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- ─── Attendance Table ───
-- Note: student name is resolved via JOIN with students table.
-- The name column is kept for backward compatibility but is no longer written to.
CREATE TABLE IF NOT EXISTS attendance (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    student_id TEXT NOT NULL REFERENCES students(student_id) ON DELETE CASCADE,
    name TEXT,
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    confidence FLOAT NOT NULL
);

-- ─── Indexes ───
CREATE INDEX IF NOT EXISTS idx_student_id ON students(student_id);
CREATE INDEX IF NOT EXISTS idx_attendance_timestamp ON attendance(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_attendance_student ON attendance(student_id);

-- ─── Row Level Security (optional, enable if using anon key) ───
-- ALTER TABLE students ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE attendance ENABLE ROW LEVEL SECURITY;
--
-- CREATE POLICY "Enable all for authenticated users" ON students
--     FOR ALL USING (auth.role() = 'authenticated');
-- CREATE POLICY "Enable all for authenticated users" ON attendance
--     FOR ALL USING (auth.role() = 'authenticated');
