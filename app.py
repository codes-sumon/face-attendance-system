"""
Face Attendance System - Main Flask Application

Uses InsightFace for face detection/recognition and Supabase for storage.
"""
import os
import base64
import logging
import time
from datetime import datetime
from typing import Optional

from flask import (
    Flask,
    render_template,
    request,
    jsonify,
    redirect,
    url_for,
    flash,
)
from dotenv import load_dotenv
import numpy as np

from face_utils import (
    extract_embedding,
    find_best_match,
    load_image_from_bytes,
    image_to_base64_jpeg,
    detect_faces,
    detect_faces_fast,
)
from supabase_utils import (
    register_student,
    get_all_students,
    get_student_by_id,
    get_all_face_embeddings,
    mark_attendance,
    get_attendance_records,
    get_attendance_summary,
    delete_student,
    update_student,
    get_last_attendance,
)

load_dotenv()

# ─── App Configuration ───
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret-key-change-in-production")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Face matching threshold (cosine similarity, 0.0 - 1.0)
FACE_MATCH_THRESHOLD = float(os.getenv("FACE_MATCH_THRESHOLD", "0.4"))

# ─── Live Recognition Configuration ───
CAMERA_ID = os.getenv("CAMERA_ID", "CAM01")
ATTENDANCE_COOLDOWN = 60  # seconds — skip duplicate logs within this window
EMBEDDING_CACHE_TTL = 30  # refresh stored embeddings every N seconds

# ─── In-Memory State for Live Recognition ───
_last_attendance_time: dict[str, float] = {}  # student_id -> timestamp
_embedded_cache: list[dict] = []
_embedded_cache_time: float = 0


# ─── Routes ───


@app.route("/")
def index():
    """Home page."""
    return render_template("index.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    """Student registration page."""
    if request.method == "POST":
        student_id = request.form.get("student_id", "").strip()
        name = request.form.get("name", "").strip()

        if not student_id or not name:
            flash("Student ID and Name are required.", "danger")
            return render_template("register.html")

        # Check for existing student
        existing = get_student_by_id(student_id)
        if existing:
            flash(f"Student ID '{student_id}' is already registered.", "warning")
            return render_template("register.html")

        # Get face image from form
        image_data = _get_image_from_request()
        if image_data is None:
            flash("Please provide a face image (upload or webcam capture).", "danger")
            return render_template("register.html")

        # Extract face embedding
        embedding = extract_embedding(image_data)
        if embedding is None:
            flash("No face detected in the image. Please try again with a clearer photo.", "danger")
            return render_template("register.html")

        # Check if this face already belongs to another registered student
        stored = get_all_face_embeddings()
        if stored:
            match = find_best_match(embedding, stored, threshold=FACE_MATCH_THRESHOLD)
            if match:
                matched_id, matched_name, confidence = match
                flash(
                    f"This face already belongs to '{matched_name}' (ID: {matched_id}) "
                    f"with {confidence:.1%} confidence. A person can only register once.",
                    "warning",
                )
                return render_template("register.html")

        # Convert image to base64 for storage
        face_image_b64 = image_to_base64_jpeg(image_data)

        # Register in Supabase
        try:
            register_student(student_id, name, embedding, face_image=face_image_b64)
            flash(f"Student '{name}' registered successfully!", "success")
            return redirect(url_for("index"))
        except Exception as e:
            logger.error(f"Registration failed: {e}")
            flash(f"Registration failed: {str(e)}", "danger")
            return render_template("register.html")

    return render_template("register.html")


@app.route("/edit-student/<student_id>", methods=["GET", "POST"])
def edit_student(student_id: str):
    """Edit student name and optionally re-capture face."""
    student = get_student_by_id(student_id)
    if not student:
        flash("Student not found.", "danger")
        return redirect(url_for("attendance_report"))

    if request.method == "POST":
        name = request.form.get("name", "").strip()
        if not name:
            flash("Name is required.", "danger")
            return render_template("edit_student.html", student=student)

        # Check if a new face image was provided
        image_data = _get_image_from_request()
        embedding = None
        face_image_b64 = None
        if image_data is not None:
            embedding = extract_embedding(image_data)
            if embedding is None:
                flash("No face detected in the new image. Keeping existing face data.", "warning")
            else:
                face_image_b64 = image_to_base64_jpeg(image_data)

        try:
            update_student(student_id, name, face_embedding=embedding, face_image=face_image_b64)
            flash(f"Student '{name}' updated successfully!", "success")
            return redirect(url_for("attendance_report"))
        except Exception as e:
            logger.error(f"Update failed: {e}")
            flash(f"Update failed: {str(e)}", "danger")
            return render_template("edit_student.html", student=student)

    return render_template("edit_student.html", student=student)


@app.route("/mark-attendance", methods=["GET", "POST"])
def mark_attendance_view():
    """Mark attendance page."""
    if request.method == "POST":
        # Get image from request
        image_data = _get_image_from_request()
        if image_data is None:
            flash("Please provide a face image (upload or webcam capture).", "danger")
            return render_template("mark_attendance.html")

        # Detect faces and extract embedding
        embedding = extract_embedding(image_data)
        if embedding is None:
            flash("No face detected. Please try again.", "danger")
            return render_template("mark_attendance.html")

        # Compare with all stored embeddings
        stored = get_all_face_embeddings()
        if not stored:
            flash("No registered students found. Please register first.", "warning")
            return render_template("mark_attendance.html")

        match = find_best_match(embedding, stored, threshold=FACE_MATCH_THRESHOLD)

        if match is None:
            flash("No matching student found. The face may not be registered, or the image quality is low.", "danger")
            return render_template("mark_attendance.html")

        student_id, name, confidence = match

        # Mark attendance
        try:
            mark_attendance(student_id, round(confidence, 4))
            flash(
                f"Attendance marked for {name} (ID: {student_id}) "
                f"with {confidence:.1%} confidence!",
                "success",
            )
            return redirect(url_for("attendance_report"))
        except Exception as e:
            logger.error(f"Attendance marking failed: {e}")
            flash(f"Failed to mark attendance: {str(e)}", "danger")
            return render_template("mark_attendance.html")

    return render_template("mark_attendance.html")


@app.route("/attendance")
def attendance_report():
    """Attendance report page."""
    student_id_filter = request.args.get("student_id", "")
    records = get_attendance_records(limit=200, student_id=student_id_filter or None)
    students = get_all_students()
    summary = get_attendance_summary()

    # Build a student_id → name map from the students table (single source of truth)
    student_names = {s["student_id"]: s["name"] for s in students}

    # Override every record's name with the current name from the students table
    for r in records:
        r["name"] = student_names.get(r["student_id"], "Unknown")
    for s in summary:
        s["name"] = student_names.get(s["student_id"], "Unknown")

    return render_template(
        "attendance_report.html",
        records=records,
        students=students,
        summary=summary,
        filter_id=student_id_filter,
    )


# ─── Live Continuous Recognition ───


@app.route("/live-recognition")
def live_recognition():
    """Real-time continuous face recognition page."""
    return render_template("live_recognition.html", camera_id=CAMERA_ID)


def _get_cached_embeddings() -> list[dict]:
    """Get face embeddings with local caching to avoid DB calls on every scan."""
    global _embedded_cache, _embedded_cache_time
    now = time.time()
    if now - _embedded_cache_time > EMBEDDING_CACHE_TTL:
        _embedded_cache = get_all_face_embeddings()
        _embedded_cache_time = now
        logger.info(f"Refreshed embedding cache: {len(_embedded_cache)} students")
    return _embedded_cache


def _check_cooldown(student_id: str) -> bool:
    """
    Check if the student is still within the attendance cooldown window.
    Returns True if the student CAN be logged (cooldown expired).
    """
    global _last_attendance_time
    now = time.time()
    last = _last_attendance_time.get(student_id, 0)
    if now - last >= ATTENDANCE_COOLDOWN:
        _last_attendance_time[student_id] = now
        return True
    return False


# (server-side MJPEG stream and detection pipeline removed in favor of
#  client-side camera capture — see static/js/live.js)


# ─── API Routes ───


@app.route("/api/students")
def api_get_students():
    """API: Get all registered students."""
    students = get_all_students()
    return jsonify(students)


@app.route("/api/students/<student_id>", methods=["DELETE"])
def api_delete_student(student_id: str):
    """API: Delete a student registration."""
    success = delete_student(student_id)
    if success:
        return jsonify({"message": "Student deleted successfully"}), 200
    return jsonify({"error": "Student not found"}), 404


@app.route("/api/students/<student_id>", methods=["PUT"])
def api_update_student(student_id: str):
    """API: Update a student's name."""
    data = request.get_json(silent=True)
    if not data or not data.get("name", "").strip():
        return jsonify({"error": "Name is required"}), 400

    name = data["name"].strip()
    result = update_student(student_id, name)
    if result:
        return jsonify({"message": "Student updated successfully", "student": result}), 200
    return jsonify({"error": "Student not found"}), 404


@app.route("/api/attendance/today")
def api_today_attendance():
    """API: Get today's attendance records."""
    records = get_attendance_records(limit=500)
    today = datetime.now().strftime("%Y-%m-%d")
    today_records = [
        r
        for r in records
        if r.get("timestamp", "").startswith(today)
    ]
    return jsonify(today_records)


@app.route("/api/check-face", methods=["POST"])
def api_check_face():
    """
    API: Check if a face image matches any registered student.
    Used for real-time webcam detection.
    """
    image_data = _get_image_from_request()
    if image_data is None:
        return jsonify({"error": "No image provided"}), 400

    embedding = extract_embedding(image_data)
    if embedding is None:
        return jsonify({"error": "No face detected", "detected": False}), 200

    stored = get_all_face_embeddings()
    if not stored:
        return jsonify({"error": "No students registered", "detected": True, "matched": False}), 200

    match = find_best_match(embedding, stored, threshold=FACE_MATCH_THRESHOLD)
    if match:
        student_id, name, confidence = match
        # Auto-mark attendance via API
        try:
            mark_attendance(student_id, round(confidence, 4))
        except Exception as e:
            logger.error(f"Auto attendance failed: {e}")
        return jsonify({
            "detected": True,
            "matched": True,
            "student_id": student_id,
            "name": name,
            "confidence": round(confidence, 4),
        }), 200

    return jsonify({"detected": True, "matched": False, "message": "No matching student"}), 200


@app.route("/api/live-detect", methods=["POST"])
def api_live_detect():
    """
    API: Detect all faces in an image and match each against registered students.
    Designed for client-side live recognition — returns bounding boxes, match
    results, and auto-marks attendance for recognized faces.
    """
    image_data = _get_image_from_request()
    if image_data is None:
        return jsonify({"error": "No image provided"}), 400

    # Detect ALL faces in the image (not just the highest-confidence one)
    faces = detect_faces(image_data)
    if not faces:
        return jsonify({"detected": False, "faces": [], "message": "No faces detected"}), 200

    stored = get_all_face_embeddings()
    h, w = image_data.shape[:2]
    results = []

    for face in faces:
        bbox = face["bbox"]  # [x1, y1, x2, y2]
        # Normalize bbox to 0-1 range for responsive canvas drawing
        norm_bbox = [
            round(bbox[0] / w, 4),
            round(bbox[1] / h, 4),
            round(bbox[2] / w, 4),
            round(bbox[3] / h, 4),
        ]
        entry = {
            "bbox": norm_bbox,
            "bbox_raw": [round(v, 2) for v in bbox],
            "det_score": round(float(face["det_score"]), 4),
            "matched": False,
        }

        if stored:
            match = find_best_match(face["embedding"], stored, threshold=FACE_MATCH_THRESHOLD)
            if match:
                student_id, name, confidence = match
                entry["matched"] = True
                entry["student_id"] = student_id
                entry["name"] = name
                entry["confidence"] = round(confidence, 4)

                # Auto-mark attendance with cooldown check
                if _check_cooldown(student_id):
                    try:
                        mark_attendance(student_id, round(confidence, 4))
                        logger.info(f"[LIVE] Attendance logged: {name} ({student_id}) at {confidence:.1%}")
                    except Exception as e:
                        logger.error(f"[LIVE] Failed to log attendance: {e}")

        results.append(entry)

    return jsonify({
        "detected": True,
        "face_count": len(results),
        "faces": results,
    }), 200


# ─── Helpers ───


def _get_image_from_request():
    """
    Extract an image from the HTTP request.
    Supports: file upload, webcam base64 data, or URL reference.
    """
    # Case 1: File upload
    if "image" in request.files:
        file = request.files["image"]
        if file and file.filename:
            return load_image_from_bytes(file.read())

    # Case 2: Webcam base64 data
    if "image_data" in request.form:
        data_url = request.form["image_data"]
        if data_url and "," in data_url:
            base64_str = data_url.split(",")[1]
            try:
                image_bytes = base64.b64decode(base64_str)
                return load_image_from_bytes(image_bytes)
            except Exception:
                pass

    # Case 3: Raw base64 in JSON body
    if request.is_json:
        body = request.get_json(silent=True) or {}
        raw = body.get("image") or body.get("image_data")
        if raw:
            if "," in raw:
                raw = raw.split(",")[1]
            try:
                image_bytes = base64.b64decode(raw)
                return load_image_from_bytes(image_bytes)
            except Exception:
                pass

    return None


# ─── Entry Point ───

if __name__ == "__main__":
    port = int(os.getenv("FLASK_PORT", 5000))
    debug = os.getenv("FLASK_DEBUG", "True").lower() == "true"

    # Warm up InsightFace models BEFORE starting the server
    logger.info("⏳ Warming up InsightFace models...")
    try:
        # Load dummy black image to trigger model initialization
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        detect_faces(dummy)
        # Also warm up the fast model (320x320 for live feed)
        dummy_small = np.zeros((320, 320, 3), dtype=np.uint8)
        detect_faces_fast(dummy_small)
        logger.info("✅ Both InsightFace models warmed up successfully")
    except Exception as e:
        logger.warning(f"⚠️ InsightFace warm-up failed (may affect live feed): {e}")

    # Check for SSL certificate files
    cert_file = os.getenv("SSL_CERT", "cert.pem")
    key_file = os.getenv("SSL_KEY", "key.pem")
    use_ssl = os.path.exists(cert_file) and os.path.exists(key_file)

    if use_ssl:
        ssl_context = (cert_file, key_file)
        protocol = "https"
        logger.info(
            f"🔒 SSL enabled — using {cert_file} / {key_file}"
        )
    else:
        ssl_context = None
        protocol = "http"
        logger.warning(
            "⚠️  SSL not enabled. Camera access requires localhost or HTTPS.\n"
            "   Run 'python gencert.py' to generate self-signed certificates, then restart."
        )

    logger.info(
        f"🌐 Starting Face Attendance System on "
        f"{protocol}://0.0.0.0:{port} (debug={debug})"
    )
    logger.info(
        f"   📱 Open {protocol}://localhost:{port} in your browser "
        f"(camera requires localhost or HTTPS)"
    )
    # use_reloader=False is CRITICAL: the reloader creates a subprocess that
    # competes for the webcam resource, causing video_feed to fail silently.
    app.run(host="0.0.0.0", port=port, debug=debug, use_reloader=False, ssl_context=ssl_context)
