/**
 * Face Attendance System — Live Client-Side Recognition
 *
 * Uses the user's device camera (MediaDevices API) for real-time face
 * detection and recognition. Frames are sent to the server periodically
 * for processing, and results are drawn on a canvas overlay.
 */
(function () {
    "use strict";

    // ─── DOM refs ───
    const video = document.getElementById("live-video");
    const canvas = document.getElementById("live-canvas");
    const overlay = document.getElementById("video-overlay");
    const overlayText = document.getElementById("overlay-text");

    // Stats elements
    const statFps = document.getElementById("stat-fps");
    const statFaces = document.getElementById("stat-faces");
    const statRecognized = document.getElementById("stat-recognized");
    const statUnknown = document.getElementById("stat-unknown");
    const infoDatetime = document.getElementById("info-datetime");
    const infoStatus = document.getElementById("info-status");
    const statusBadge = document.getElementById("status-badge");
    const logList = document.getElementById("log-list");

    // ─── Configuration ───
    const SCAN_INTERVAL = 1500;      // ms between frame captures
    const LOG_POLL_INTERVAL = 5000;  // ms between recent-log refreshes
    const MAX_CANVAS_W = 640;        // max video/canvas width
    const BOX_LINE_WIDTH = 3;

    // ─── State ───
    let stream = null;
    let scanTimer = null;
    let logTimer = null;
    let lastFrameTime = 0;
    let frameCount = 0;
    let fpsValue = 0;
    let isProcessing = false;

    // ─── Check browser support ───
    function checkCameraSupport() {
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            let hint = "";
            if (window.location.hostname === "0.0.0.0") {
                hint = "Open http://localhost:5000 instead of http://0.0.0.0:5000. "
                       + "The browser blocks camera access on 0.0.0.0.";
            } else if (window.location.protocol === "file:") {
                hint = "Open the page via the Flask server (http://localhost:5000), not as a local file.";
            } else if (window.location.protocol !== "https:" && window.location.hostname !== "localhost") {
                hint = "Camera access requires HTTPS or localhost. "
                       + "Use https:// or access via localhost.";
            } else {
                hint = "Your browser does not support camera access (getUserMedia). "
                       + "Try a modern browser like Chrome, Firefox, or Edge.";
            }
            throw new Error(hint);
        }
    }

    // ─── Start Camera ───
    async function startCamera() {
        try {
            overlayText.textContent = "Requesting camera access...";
            checkCameraSupport();
            stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    facingMode: "user",
                    width: { ideal: 640 },
                    height: { ideal: 480 },
                },
                audio: false,
            });

            video.srcObject = stream;
            await video.play();

            // Wait until video is actually playing with dimensions
            await new Promise((resolve) => {
                const check = () => {
                    if (video.videoWidth > 0 && video.videoHeight > 0) {
                        resolve();
                    } else {
                        requestAnimationFrame(check);
                    }
                };
                check();
            });

            // Size canvas to match video (with max width constraint)
            resizeCanvas();
            hideOverlay();
            setStatus("running", "Running", "live-badge-active", "● LIVE");

            // Start the scan loop
            scanTimer = setInterval(scanFrame, SCAN_INTERVAL);
            // Start log polling
            logTimer = setInterval(fetchRecentLogs, LOG_POLL_INTERVAL);

            // FPS counter
            setInterval(updateFps, 2000);

            // Initial fetches
            fetchRecentLogs();
            updateDateTime();

            logger("[LIVE] Camera started successfully");
        } catch (err) {
            logger("[LIVE] Camera error: " + err.message);
            showError(err.message || "Camera access denied or unavailable.");
            setStatus("error", "Camera Unavailable", "live-badge-error", "● ERROR");
        }
    }

    // ─── Canvas sizing ───
    function resizeCanvas() {
        const vw = video.videoWidth || 640;
        const vh = video.videoHeight || 480;
        const scale = vw > MAX_CANVAS_W ? MAX_CANVAS_W / vw : 1;
        const dw = Math.floor(vw * scale);
        const dh = Math.floor(vh * scale);

        video.style.width = dw + "px";
        video.style.height = dh + "px";
        canvas.width = dw;
        canvas.height = dh;
        canvas.style.width = dw + "px";
        canvas.style.height = dh + "px";
    }

    // ─── Overlay helpers ───
    function hideOverlay() {
        overlay.style.display = "none";
    }

    function showError(msg) {
        overlay.innerHTML = `
            <div class="live-error-icon">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="48" height="48">
                    <circle cx="12" cy="12" r="10"/>
                    <line x1="15" y1="9" x2="9" y2="15"/>
                    <line x1="9" y1="9" x2="15" y2="15"/>
                </svg>
            </div>
            <p>${msg}</p>
        `;
        overlay.style.display = "flex";
    }

    // ─── Status helpers ───
    function setStatus(id, text, badgeClass, badgeText) {
        if (infoStatus) {
            infoStatus.textContent = text;
            infoStatus.style.color =
                id === "running" ? "var(--success)" :
                id === "error" ? "var(--danger)" :
                "var(--gray-400)";
        }
        if (statusBadge) {
            statusBadge.className = "live-badge " + badgeClass;
            statusBadge.textContent = badgeText;
        }
    }

    // ─── Frame capture & server scan ───
    async function scanFrame() {
        if (isProcessing) return;
        if (!video.videoWidth) return;

        isProcessing = true;

        try {
            // Capture frame to an offscreen canvas at the canvas dimensions
            const captureCanvas = document.createElement("canvas");
            captureCanvas.width = canvas.width;
            captureCanvas.height = canvas.height;
            const ctx = captureCanvas.getContext("2d");
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

            const dataUrl = captureCanvas.toDataURL("image/jpeg", 0.7);

            // Send to server for detection
            const response = await fetch("/api/live-detect", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ image: dataUrl }),
            });

            const result = await response.json();
            frameCount++;

            // Draw results on overlay canvas
            drawResults(result);

            // Update stats
            updateStats(result);
        } catch (err) {
            logger("[LIVE] Scan error: " + err.message);
        } finally {
            isProcessing = false;
        }
    }

    // ─── Draw bounding boxes on canvas ───
    function drawResults(result) {
        const ctx = canvas.getContext("2d");
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        if (!result.detected || !result.faces) return;

        const cw = canvas.width;
        const ch = canvas.height;

        for (const face of result.faces) {
            const [nx1, ny1, nx2, ny2] = face.bbox;
            const x1 = Math.round(nx1 * cw);
            const y1 = Math.round(ny1 * ch);
            const x2 = Math.round(nx2 * cw);
            const y2 = Math.round(ny2 * ch);

            if (face.matched) {
                // ── Green box for recognized ──
                ctx.strokeStyle = "rgba(0, 220, 80, 0.9)";
                ctx.lineWidth = BOX_LINE_WIDTH;
                ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

                // Label background
                const label = `${face.name} (${(face.confidence * 100).toFixed(1)}%)`;
                ctx.font = "bold 14px system-ui, -apple-system, sans-serif";
                const metrics = ctx.measureText(label);
                const lh = 22;
                const lw = metrics.width + 14;

                ctx.fillStyle = "rgba(0, 120, 40, 0.85)";
                const labelY = y1 - lh - 6;
                roundRect(ctx, x1, labelY, lw, lh, 4);
                ctx.fill();

                // Label text
                ctx.fillStyle = "#fff";
                ctx.fillText(label, x1 + 7, labelY + 16);

                // ID text
                ctx.font = "11px system-ui, -apple-system, sans-serif";
                ctx.fillStyle = "rgba(200, 255, 200, 0.8)";
                ctx.fillText(`ID: ${face.student_id}`, x1 + 7, labelY + lh + 14);
            } else {
                // ── Red box for unknown ──
                ctx.strokeStyle = "rgba(220, 50, 50, 0.9)";
                ctx.lineWidth = BOX_LINE_WIDTH;
                ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

                // Label background
                const label = "Unknown";
                ctx.font = "bold 14px system-ui, -apple-system, sans-serif";
                const metrics = ctx.measureText(label);
                const lh = 22;
                const lw = metrics.width + 14;

                ctx.fillStyle = "rgba(140, 30, 30, 0.85)";
                const labelY = y1 - lh - 6;
                roundRect(ctx, x1, labelY, lw, lh, 4);
                ctx.fill();

                // Label text
                ctx.fillStyle = "#ffc0c0";
                ctx.fillText(label, x1 + 7, labelY + 16);
            }
        }
    }

    // ─── Rounded rectangle helper (polyfill for roundRect) ───
    function roundRect(ctx, x, y, w, h, r) {
        if (typeof ctx.roundRect === "function") {
            ctx.roundRect(x, y, w, h, r);
            return;
        }
        // Manual implementation for older browsers
        ctx.beginPath();
        ctx.moveTo(x + r, y);
        ctx.lineTo(x + w - r, y);
        ctx.arcTo(x + w, y, x + w, y + r, r);
        ctx.lineTo(x + w, y + h - r);
        ctx.arcTo(x + w, y + h, x + w - r, y + h, r);
        ctx.lineTo(x + r, y + h);
        ctx.arcTo(x, y + h, x, y + h - r, r);
        ctx.lineTo(x, y + r);
        ctx.arcTo(x, y, x + r, y, r);
        ctx.closePath();
    }

    // ─── Update stats ───
    function updateStats(result) {
        const faces = result.faces || [];
        let recognized = 0;
        let unknown = 0;

        for (const f of faces) {
            if (f.matched) recognized++;
            else unknown++;
        }

        statFaces.textContent = faces.length;
        statRecognized.textContent = recognized;
        statUnknown.textContent = unknown;
    }

    // ─── FPS counter ───
    function updateFps() {
        const now = performance.now();
        if (lastFrameTime > 0) {
            const elapsed = (now - lastFrameTime) / 1000;
            fpsValue = Math.round(frameCount / elapsed);
        } else {
            fpsValue = 0;
        }
        statFps.textContent = fpsValue;
        lastFrameTime = now;
        frameCount = 0;
    }

    // ─── DateTime ───
    function updateDateTime() {
        const now = new Date();
        infoDatetime.textContent = now.toISOString().replace("T", " ").substring(0, 19);
    }
    setInterval(updateDateTime, 1000);

    // ─── Recent Logs ───
    async function fetchRecentLogs() {
        try {
            const res = await fetch("/api/attendance/today");
            const records = await res.json();

            if (!records || records.length === 0) {
                logList.innerHTML = '<div class="live-log-empty">No logs today.</div>';
                return;
            }

            const recent = records.slice(0, 10);
            logList.innerHTML = recent
                .map(
                    (r) =>
                        `<div class="live-log-entry">
                            <div class="live-log-name">${escHtml(r.name)}</div>
                            <div class="live-log-id">${escHtml(r.student_id)}</div>
                            <div class="live-log-time">${(r.timestamp || "").replace("T", " ").substring(0, 19)}</div>
                        </div>`
                )
                .join("");
        } catch (e) {
            console.error("[LIVE] Logs poll error:", e);
        }
    }

    // ─── Utility ───
    function logger(msg) {
        console.log(msg);
    }

    function escHtml(str) {
        const div = document.createElement("div");
        div.textContent = str;
        return div.innerHTML;
    }

    // ─── Cleanup ───
    function stopCamera() {
        if (scanTimer) clearInterval(scanTimer);
        if (logTimer) clearInterval(logTimer);
        if (stream) {
            stream.getTracks().forEach((t) => t.stop());
            stream = null;
        }
        setStatus("stopped", "Stopped", "live-badge-inactive", "● OFFLINE");
    }

    window.addEventListener("beforeunload", stopCamera);

    // ─── Init ───
    document.addEventListener("DOMContentLoaded", startCamera);
})();
