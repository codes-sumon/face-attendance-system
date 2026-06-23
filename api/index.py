"""
Vercel serverless entry point for Face Attendance System.
Exposes the Flask app as a WSGI handler for Vercel.

Live recognition uses the client's device camera (MediaDevices API) and
sends frames to the server for processing — no server-side webcam needed.
"""
import sys
import os
import logging

# Add the project root to the path so imports from app.py resolve
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# Signal to app that we're running on Vercel (disables webcam/threaded features)
os.environ["VERCEL"] = "1"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── Import the Flask app ───
# Vercel's Python runtime auto-detects the module-level `app` variable as the
# WSGI handler. The import is wrapped in a try/except so Vercel can detect
# the function even if heavy ML dependencies fail to load.
try:
    from app import app
    logger.info("✅ Face Attendance app loaded successfully")
except Exception as e:
    logger.error(f"⚠️ Failed to load app: {e}")
    # Fallback: create a minimal Flask app so Vercel can still route health checks
    from flask import Flask, jsonify
    app = Flask(__name__)

    @app.route("/")
    @app.route("/api/health")
    def health():
        return jsonify({
            "status": "degraded",
            "error": str(e),
            "message": "Face recognition unavailable. Check server logs.",
        })

    @app.route("/<path:_>")
    def catch_all(_):
        return jsonify({
            "status": "error",
            "message": "Face recognition module failed to load. Please check server configuration.",
        }), 503
