"""
Supabase database operations for the Face Attendance System.
"""
import os
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()

# ─── Supabase Client (singleton) ───
_supabase_client: Optional[Client] = None


def get_supabase() -> Client:
    """Get the Supabase client (singleton)."""
    global _supabase_client
    if _supabase_client is None:
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_KEY")
        if not url or not key:
            raise ValueError(
                "SUPABASE_URL and SUPABASE_KEY must be set in .env file"
            )
        _supabase_client = create_client(url, key)
    return _supabase_client


# ─── Student Operations ───


def register_student(
    student_id: str, name: str, face_embedding: List[float], face_image: Optional[str] = None
) -> Dict[str, Any]:
    """
    Register a new student with their face embedding.

    Args:
        student_id: Unique student identifier.
        name: Student's full name.
        face_embedding: 512-d face embedding vector.
        face_image: Optional base64-encoded JPEG of the face.

    Returns:
        The inserted student record.

    Raises:
        Exception if student_id already exists.
    """
    supabase = get_supabase()
    data = {
        "student_id": student_id,
        "name": name,
        "face_embedding": face_embedding,
    }
    if face_image is not None:
        data["face_image"] = face_image
    result = supabase.table("students").insert(data).execute()
    if not result.data:
        raise Exception("Failed to register student. Possibly duplicate student_id.")
    return result.data[0]


def get_all_students() -> List[Dict[str, Any]]:
    """Get all registered students."""
    supabase = get_supabase()
    result = supabase.table("students").select("*").order("created_at", desc=True).execute()
    return result.data or []


def get_student_by_id(student_id: str) -> Optional[Dict[str, Any]]:
    """Get a single student by their student_id."""
    supabase = get_supabase()
    result = (
        supabase.table("students")
        .select("*")
        .eq("student_id", student_id)
        .execute()
    )
    if result.data:
        return result.data[0]
    return None


def get_all_face_embeddings() -> List[Dict[str, Any]]:
    """
    Get all student face embeddings for matching.

    Returns:
        List of dicts with keys: student_id, name, face_embedding.
    """
    supabase = get_supabase()
    result = (
        supabase.table("students")
        .select("student_id, name, face_embedding")
        .execute()
    )
    return result.data or []


def delete_student(student_id: str) -> bool:
    """Delete a student record by student_id."""
    supabase = get_supabase()
    result = (
        supabase.table("students")
        .delete()
        .eq("student_id", student_id)
        .execute()
    )
    return len(result.data) > 0


def update_student(
    student_id: str, name: str, face_embedding: Optional[List[float]] = None,
    face_image: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    Update a student's name and optionally re-register their face embedding.

    Args:
        student_id: The student's unique ID (immutable).
        name: Updated full name.
        face_embedding: Optional new 512-d face embedding vector.
        face_image: Optional base64-encoded JPEG of the face.

    Returns:
        The updated student record, or None if not found.
    """
    supabase = get_supabase()
    data: Dict[str, Any] = {"name": name}
    if face_embedding is not None:
        data["face_embedding"] = face_embedding
    if face_image is not None:
        data["face_image"] = face_image
    result = (
        supabase.table("students")
        .update(data)
        .eq("student_id", student_id)
        .execute()
    )
    if result.data:
        return result.data[0]
    return None


# ─── Attendance Operations ───


def mark_attendance(
    student_id: str, confidence: float
) -> Dict[str, Any]:
    """
    Log an attendance record. The student name is resolved via the students table.

    Args:
        student_id: Student's unique ID.
        confidence: Face matching confidence score.

    Returns:
        The inserted attendance record.
    """
    supabase = get_supabase()

    # Resolve the student's current name from the students table
    student = get_student_by_id(student_id)
    name = student["name"] if student else "Unknown"

    data = {
        "student_id": student_id,
        "name": name,
        "confidence": confidence,
    }
    result = supabase.table("attendance").insert(data).execute()
    return result.data[0]


def get_attendance_records(
    limit: int = 100, student_id: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Get attendance records, optionally filtered by student_id.
    Student name is resolved from the students table.

    Args:
        limit: Maximum number of records to return.
        student_id: Optional filter by student ID.

    Returns:
        List of attendance records with name populated from students table.
    """
    supabase = get_supabase()
    query = supabase.table("attendance").select("*").order("timestamp", desc=True).limit(limit)
    if student_id:
        query = query.eq("student_id", student_id)
    result = query.execute()

    records = result.data or []

    # Build a student_id → name map from the students table
    students = get_all_students()
    student_names = {s["student_id"]: s["name"] for s in students}

    # Override the stored name with the current name from students table
    for r in records:
        r["name"] = student_names.get(r["student_id"], "Unknown")

    return records


def get_last_attendance(student_id: str) -> Optional[Dict[str, Any]]:
    """
    Get the most recent attendance record for a student.

    Args:
        student_id: The student's unique ID.

    Returns:
        The latest attendance record, or None if none found.
    """
    supabase = get_supabase()
    result = (
        supabase.table("attendance")
        .select("*")
        .eq("student_id", student_id)
        .order("timestamp", desc=True)
        .limit(1)
        .execute()
    )
    if result.data:
        return result.data[0]
    return None


def get_attendance_summary() -> List[Dict[str, Any]]:
    """
    Get a summary of total attendance per student.

    Returns:
        List of dicts with keys: student_id, name, total.
    """
    # Fetch all records and group locally (avoids complex PostgREST aggregation)
    records = get_attendance_records(limit=10000)
    # Build a student name map for fallback
    students = get_all_students()
    student_names = {s["student_id"]: s["name"] for s in students}
    summary: Dict[str, dict] = {}
    for r in records:
        key = r["student_id"]
        if key not in summary:
            summary[key] = {
                "student_id": r["student_id"],
                "name": student_names.get(key, r.get("name", "Unknown")),
                "total": 0,
            }
        summary[key]["total"] += 1
    return list(summary.values())
