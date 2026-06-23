/**
 * Face Attendance System — Webcam Camera Module
 *
 * Provides webcam capture functionality used in register.html
 * and mark_attendance.html.
 */
(function () {
    "use strict";

    // ─── DOM refs ───
    const video = document.getElementById("webcam-video");
    const canvas = document.getElementById("webcam-canvas");
    const overlay = document.getElementById("webcam-overlay");
    const startBtn = document.getElementById("btn-start-cam");
    const captureBtn = document.getElementById("btn-capture");
    const retakeBtn = document.getElementById("btn-retake");
    const hiddenInput = document.getElementById("image_data");

    if (!video) return; // not on a page with webcam

    let stream = null;

    // ─── Check browser camera support ───
    function checkCameraSupport() {
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            let hint = "";
            if (window.location.hostname === "0.0.0.0") {
                hint = "Open http://localhost:5000 instead of http://0.0.0.0:5000.";
            } else if (window.location.protocol === "file:") {
                hint = "Open via the Flask server (http://localhost:5000), not as a local file.";
            } else {
                hint = "Your browser does not support camera access. Try Chrome, Firefox, or Edge.";
            }
            throw new Error(hint);
        }
    }

    // ─── Start camera ───
    async function startCamera() {
        try {
            checkCameraSupport();
            stream = await navigator.mediaDevices.getUserMedia({
                video: { facingMode: "user", width: { ideal: 640 }, height: { ideal: 480 } },
                audio: false,
            });
            video.srcObject = stream;
            await video.play();
            overlay.classList.add("hidden");
            startBtn.style.display = "none";
            captureBtn.style.display = "inline-flex";
            if (retakeBtn) retakeBtn.style.display = "none";
        } catch (err) {
            console.error("Camera error:", err);
            let msg = "Camera access denied or not available. Please use the upload option.";
            if (err.name === "NotReadableError") {
                msg = "Camera is in use by another app (Zoom, Teams, another browser tab). "
                    + "Close other camera apps and try again, or use the upload option.";
            } else if (err.name === "NotAllowedError") {
                msg = "Camera permission was denied. Allow camera access in your browser "
                    + "settings, or use the upload option.";
            } else if (err.name === "NotFoundError") {
                msg = "No camera found. Connect a webcam or use the upload option.";
            } else if (err.message) {
                msg = err.message + " Please use the upload option.";
            }
            alert(msg);
        }
    }

    // ─── Capture frame ───
    function captureFrame() {
        canvas.width = video.videoWidth || 640;
        canvas.height = video.videoHeight || 480;
        const ctx = canvas.getContext("2d");
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        const dataUrl = canvas.toDataURL("image/jpeg", 0.92);
        hiddenInput.value = dataUrl;
        video.style.display = "none";
        canvas.style.display = "block";
        captureBtn.style.display = "none";
        if (retakeBtn) retakeBtn.style.display = "inline-flex";

        // Show preview status
        const statusEl = document.getElementById("face-status");
        if (statusEl) {
            const statusText = document.getElementById("face-status-text");
            if (statusText) {
                statusEl.style.display = "block";
                statusEl.className = "face-status";
                statusText.textContent = "Image captured. You can submit or retake.";
            }
        }
    }

    // ─── Retake ───
    function retake() {
        hiddenInput.value = "";
        video.style.display = "block";
        canvas.style.display = "none";
        captureBtn.style.display = "inline-flex";
        retakeBtn.style.display = "none";
    }

    // ─── Stop camera (cleanup) ───
    function stopCamera() {
        if (stream) {
            stream.getTracks().forEach((track) => track.stop());
            stream = null;
        }
    }

    // ─── Event listeners ───
    startBtn.addEventListener("click", startCamera);
    captureBtn.addEventListener("click", captureFrame);
    if (retakeBtn) retakeBtn.addEventListener("click", retake);

    // Cleanup on page unload
    window.addEventListener("beforeunload", stopCamera);
})();
