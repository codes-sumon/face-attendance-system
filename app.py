"""
Face Attendance System - Main Flask Application

Uses InsightFace for face detection/recognition and Supabase for storage.
"""
import os
import base64
import logging
import time
import threading
import cv2
from io import BytesIO
from datetime import datetime
import platform
from typing import Optional

from flask import (
    Flask,
    render_template,
    request,
    jsonify,
    redirect,
    url_for,
    flash,
    Response,
)
from dotenv import load_dotenv
from PIL import Image
import numpy as np

from face_utils import (
    extract_embedding,
    find_best_match,
    load_image_from_bytes,
    detect_faces,
    detect_faces_fast,
    compare_embeddings,
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
FRAME_SKIP = 3            # run detection on every (FRAME_SKIP+1)th frame
EMBEDDING_CACHE_TTL = 30  # refresh stored embeddings every N seconds

# ─── In-Memory State for Live Recognition ───
_last_attendance_time: dict[str, float] = {}  # student_id -> timestamp
_embedded_cache: list[dict] = []
_embedded_cache_time: float = 0
_stream_stats: dict = {
    "fps": 0,
    "total_faces": 0,
    "recognized": 0,
    "unknown": 0,
    "status": "idle",
}
_stream_lock = threading.Lock()

# ─── Threaded Detection Pipeline ───
# Decouples face detection from video streaming so the feed never blocks.
_detection_frame = None               # frame waiting to be processed (BGR)
_detection_frame_lock = threading.Lock()
_detection_results: list[dict] = []   # latest face detections
_detection_results_lock = threading.Lock()
_detection_busy = False               # whether the worker is busy
_detection_known_embeddings: list[dict] = []  # snapshot of embeddings for matching

# ─── Global Webcam (warmed up at startup) ───
_global_camera = None
_global_camera_lock = threading.Lock()


def _open_webcam():
    """Open webcam with minimal probing (max 2 attempts)."""
    camera_id_str = os.getenv("CAMERA_DEVICE_ID", "0")
    try:
        camera_id = int(camera_id_str)
    except ValueError:
        camera_id = 0

    backend = cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_ANY

    # Only try the configured device and one fallback — avoids multi-second
    # timeouts on non-existent DSHOW devices.
    for dev_id in [camera_id, 0]:
        c = cv2.VideoCapture(dev_id, backend)
        if c.isOpened():
            c.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            c.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            ret_test, _ = c.read()
            if ret_test:
                logger.info(f"[LIVE] Webcam opened on device {dev_id} with backend {backend}")
                return c
            c.release()
        else:
            c.release()
    return None


def _warmup_camera():
    """Pre-open the webcam at startup so the first connection is instant."""
    global _global_camera
    try:
        cam = _open_webcam()
        if cam is not None:
            with _global_camera_lock:
                _global_camera = cam
            logger.info("✅ Webcam warmed up and ready")
        else:
            logger.warning("⚠️  No webcam available at startup (will retry on first request)")
    except Exception as e:
        logger.warning(f"⚠️  Webcam warm-up failed: {e}")


def _detection_worker():
    """
    Background worker that continuously runs face detection + recognition.
    Feeds on frames from _detection_frame and stores results in _detection_results.
    """
    global _detection_busy, _detection_frame, _detection_results
    logger.info("[DETECT] Detection worker started")
    while True:
        frame = None
        with _detection_frame_lock:
            if _detection_frame is not None:
                frame = _detection_frame
                _detection_frame = None

        if frame is not None:
            _detection_busy = True
            try:
                # — Detection —
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                faces = detect_faces_fast(rgb)

                # — Recognition against known embeddings —
                with _detection_results_lock:
                    stored = list(_detection_known_embeddings) if _detection_known_embeddings else _get_cached_embeddings()

                for face in faces:
                    match = find_best_match(face["embedding"], stored, threshold=FACE_MATCH_THRESHOLD)
                    face["match"] = match  # (student_id, name, confidence) or None

                with _detection_results_lock:
                    _detection_results.clear()
                    _detection_results.extend(faces)
            except Exception as e:
                logger.error(f"[DETECT] Error: {e}")
            finally:
                _detection_busy = False
        else:
            time.sleep(0.005)


# Start the detection daemon thread at module level (skip on Vercel)
_on_vercel = os.environ.get("VERCEL") == "1"
if not _on_vercel:
    _detection_thread = threading.Thread(target=_detection_worker, daemon=True)
    _detection_thread.start()


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

        # Register in Supabase
        try:
            register_student(student_id, name, embedding)
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
        if image_data is not None:
            embedding = extract_embedding(image_data)
            if embedding is None:
                flash("No face detected in the new image. Keeping existing face data.", "warning")

        try:
            update_student(student_id, name, face_embedding=embedding)
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
            mark_attendance(student_id, name, round(confidence, 4))
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
    """Get face embeddings with local caching to avoid DB calls every frame."""
    global _embedded_cache, _embedded_cache_time, _detection_known_embeddings
    now = time.time()
    if now - _embedded_cache_time > EMBEDDING_CACHE_TTL:
        _embedded_cache = get_all_face_embeddings()
        _embedded_cache_time = now
        # Also sync the detection worker's known embeddings snapshot
        with _detection_results_lock:
            _detection_known_embeddings = _embedded_cache
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


def _draw_detections(frame: np.ndarray) -> np.ndarray:
    """
    Draw bounding boxes and labels from the latest detection results.
    Does NOT run detection — uses the background thread's results.
    """
    global _stream_stats

    # Grab the latest detection results
    with _detection_results_lock:
        faces = list(_detection_results)

    total_faces = len(faces)
    recognized_count = 0
    unknown_count = 0

    for face in faces:
        bbox = face["bbox"]
        x1, y1, x2, y2 = map(int, bbox[:4])
        match = face.get("match")  # (student_id, name, confidence) or None

        if match:
            student_id, name, confidence = match
            recognized_count += 1

            # — Green box for recognized —
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)

            label = f"{name} | ID: {student_id}"
            conf_text = f"{confidence:.1%} Match"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            (cw, ch), _ = cv2.getTextSize(conf_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            label_w = max(lw, cw)
            cv2.rectangle(
                frame,
                (x1, y1 - 52),
                (x1 + label_w + 12, y1),
                (0, 100, 0),
                -1,
            )
            cv2.putText(
                frame, label,
                (x1 + 6, y1 - 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2,
            )
            cv2.putText(
                frame, conf_text,
                (x1 + 6, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 255, 200), 1,
            )

            # — Auto-log attendance with cooldown —
            if _check_cooldown(student_id):
                try:
                    mark_attendance(student_id, name, round(confidence, 4))
                    logger.info(
                        f"[LIVE] Attendance logged: {name} ({student_id}) "
                        f"at {confidence:.1%}"
                    )
                except Exception as e:
                    logger.error(f"[LIVE] Failed to log attendance: {e}")

        else:
            unknown_count += 1

            # — Red box for unknown —
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 2)

            label = "Unknown Person"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(
                frame,
                (x1, y1 - 30),
                (x1 + lw + 12, y1),
                (0, 0, 100),
                -1,
            )
            cv2.putText(
                frame, label,
                (x1 + 6, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 200, 200), 2,
            )

    # Update stream stats
    with _stream_lock:
        _stream_stats["total_faces"] = total_faces
        _stream_stats["recognized"] = recognized_count
        _stream_stats["unknown"] = unknown_count

    return frame


def _generate_feed():
    """Generator for MJPEG video stream with threaded detection."""
    global _global_camera, _detection_frame, _detection_busy

    # Use the pre-warmed global camera if available, otherwise open a new one
    cap = None
    with _global_camera_lock:
        if _global_camera is not None and _global_camera.isOpened():
            cap = _global_camera
            # Detach from global so this generator owns it exclusively
            _global_camera = None
            logger.info("[LIVE] Using pre-warmed webcam")

    if cap is None:
        cap = _open_webcam()

    if cap is None:
        logger.error("[LIVE] Cannot open any webcam device")
        with _stream_lock:
            _stream_stats["status"] = "error: webcam unavailable"
        error_frame = _create_error_frame("Webcam unavailable. Check device connection.")
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + error_frame + b"\r\n"
        )
        return

    # Refresh embedding cache when a new connection starts
    embeddings = _get_cached_embeddings()
    with _detection_results_lock:
        _detection_known_embeddings = embeddings

    with _stream_lock:
        _stream_stats["status"] = "running"

    frame_count = 0
    fps_start = time.time()
    fps_frames = 0
    consecutive_failures = 0
    max_failures = 30

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                consecutive_failures += 1
                if consecutive_failures > max_failures:
                    logger.error("[LIVE] Too many consecutive frame read failures")
                    break
                time.sleep(0.03)
                continue
            consecutive_failures = 0

            frame_count += 1
            fps_frames += 1

            # Calculate FPS periodically
            elapsed = time.time() - fps_start
            if elapsed >= 2.0:
                with _stream_lock:
                    _stream_stats["fps"] = round(fps_frames / elapsed, 1)
                fps_frames = 0
                fps_start = time.time()

            # Resize for consistent performance
            h, w = frame.shape[:2]
            if w > 640:
                scale = 640 / w
                new_w, new_h = 640, int(h * scale)
                frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

            # Feed the detection thread every Nth frame (non-blocking)
            if frame_count % (FRAME_SKIP + 1) == 0:
                with _detection_frame_lock:
                    if _detection_frame is None and not _detection_busy:
                        _detection_frame = frame.copy()

            # Draw the latest available detection results (non-blocking)
            frame = _draw_detections(frame)

            # Encode as JPEG at lower quality for faster streaming
            ret_jpeg, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 55])
            if not ret_jpeg:
                continue

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
            )

    except GeneratorExit:
        pass
    except Exception as e:
        logger.error(f"[LIVE] Feed error: {e}")
    finally:
        cap.release()
        logger.info("[LIVE] Webcam released")
        with _stream_lock:
            _stream_stats["status"] = "stopped"


def _create_error_frame(message: str) -> bytes:
    """Create a JPEG frame with an error message for display in the video feed."""
    img = np.ones((480, 640, 3), dtype=np.uint8) * 30  # dark gray
    # Split message into lines
    lines = message.split(".")
    y = 220
    for line in lines:
        line = line.strip()
        if not line:
            continue
        cv2.putText(
            img, line + ("." if not line.endswith(".") else ""),
            (50, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2,
        )
        y += 40
    _, buffer = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return buffer.tobytes()


@app.route("/video_feed/live")
def video_feed_live():
    """MJPEG video stream endpoint for live recognition."""
    return Response(
        _generate_feed(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/api/live-stats")
def api_live_stats():
    """API: Get current live recognition stats."""
    with _stream_lock:
        stats = dict(_stream_stats)
    # Add current time
    stats["current_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return jsonify(stats)


# ─── API Routes ───
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
            mark_attendance(student_id, name, round(confidence, 4))
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
        from face_utils import detect_faces, detect_faces_fast
        detect_faces(dummy)
        # Also warm up the fast model (320x320 for live feed)
        dummy_small = np.zeros((320, 320, 3), dtype=np.uint8)
        detect_faces_fast(dummy_small)
        logger.info("✅ Both InsightFace models warmed up successfully")
    except Exception as e:
        logger.warning(f"⚠️ InsightFace warm-up failed (may affect live feed): {e}")

    # Pre-open webcam so the first user doesn't wait
    _warmup_camera()

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
            "⚠️  SSL not enabled. Webcam will be BLOCKED by the browser "
            "on non-localhost connections.\n"
            "   Run 'python gencert.py' to generate self-signed certificates, then restart."
        )

    logger.info(
        f"🌐 Starting Face Attendance System on "
        f"{protocol}://0.0.0.0:{port} (debug={debug})"
    )
    # use_reloader=False is CRITICAL: the reloader creates a subprocess that
    # competes for the webcam resource, causing video_feed to fail silently.
    app.run(host="0.0.0.0", port=port, debug=debug, use_reloader=False, ssl_context=ssl_context)
