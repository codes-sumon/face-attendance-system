/**
 * Face Attendance System — Main JavaScript
 */

// ─── Flash message auto-dismiss ───
document.addEventListener("DOMContentLoaded", () => {
    document.querySelectorAll(".alert").forEach((alert) => {
        setTimeout(() => {
            alert.style.transition = "opacity 0.5s ease";
            alert.style.opacity = "0";
            setTimeout(() => alert.remove(), 500);
        }, 6000);
    });
});

// ─── Tab switching ───
document.addEventListener("DOMContentLoaded", () => {
    document.querySelectorAll(".tab-bar").forEach((bar) => {
        const tabs = bar.querySelectorAll(".tab");
        tabs.forEach((tab) => {
            tab.addEventListener("click", () => {
                const tabName = tab.dataset.tab;
                const parent = bar.closest(".form-group");

                // Deactivate all tabs in this group
                tabs.forEach((t) => t.classList.remove("active"));
                tab.classList.add("active");

                // Show corresponding content
                parent.querySelectorAll(".tab-content").forEach((tc) => {
                    tc.classList.remove("active");
                });
                const target = parent.querySelector(`#tab-${tabName}`);
                if (target) target.classList.add("active");

                // Reset hidden input
                const hidden = parent.querySelector("#image_data");
                if (hidden) hidden.value = "";
            });
        });
    });
});

// ─── Check face in uploaded image ───
async function checkFaceInImage(dataUrl) {
    const statusEl = document.getElementById("face-status");
    const statusText = document.getElementById("face-status-text");
    if (!statusEl) return;

    statusEl.style.display = "block";
    statusEl.className = "face-status";
    statusText.textContent = "Checking image for faces...";

    try {
        const response = await fetch("/api/check-face", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ image: dataUrl }),
        });
        const result = await response.json();

        if (result.error) {
            statusEl.classList.add("error");
            statusText.textContent = result.error;
        } else if (result.matched) {
            statusEl.classList.add("success");
            statusText.textContent = `✅ Match found: ${result.name} (${(result.confidence * 100).toFixed(1)}%)`;
        } else {
            statusEl.classList.add("error");
            statusText.textContent = "⚠️ No matching student found. Register first!";
        }
    } catch (e) {
        statusEl.classList.add("error");
        statusText.textContent = "Error checking face. Server may be offline.";
    }
}

// ─── Delete Student Confirmation ───
function confirmDelete(studentId, studentName) {
    const modal = document.getElementById("delete-modal");
    if (!modal) return;

    document.getElementById("delete-student-id").textContent = studentId;
    document.getElementById("delete-student-name").textContent = studentName;
    modal.classList.add("show");
}

function closeDeleteModal() {
    const modal = document.getElementById("delete-modal");
    if (modal) modal.classList.remove("show");
}

async function executeDelete() {
    const studentId = document.getElementById("delete-student-id").textContent;
    const modal = document.getElementById("delete-modal");

    try {
        const response = await fetch(`/api/students/${encodeURIComponent(studentId)}`, {
            method: "DELETE",
        });

        if (response.ok) {
            modal.classList.remove("show");
            window.location.reload();
        } else {
            const err = await response.json();
            alert("Delete failed: " + (err.error || "Unknown error"));
        }
    } catch (e) {
        alert("Network error. Could not delete student.");
    }
}

// Close modal on backdrop click
document.addEventListener("DOMContentLoaded", () => {
    const modal = document.getElementById("delete-modal");
    if (modal) {
        modal.addEventListener("click", (e) => {
            if (e.target === modal) closeDeleteModal();
        });
    }
});
