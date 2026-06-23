"""
Vercel serverless entry point for Face Attendance System.
Exposes the Flask app as a WSGI handler for Vercel.

Note: The live webcam recognition feature requires a physical camera
and won't work in Vercel's serverless environment. Use the app
locally for that feature. Registration, attendance (with photo upload),
and reports work normally.
"""
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Disable webcam/threaded features on Vercel
os.environ["VERCEL"] = "1"

from app import app

# Vercel expects a WSGI-compatible app as the module-level `app` variable
# Flask's `app` object IS a WSGI app, so this works directly.
