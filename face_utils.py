"""
Face detection and recognition utilities using InsightFace.
"""
import os
import pickle
import numpy as np
import cv2
import insightface
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model
from typing import List, Optional, Tuple


# ─── Global Face Analysis App (singleton) ───
_face_app = None
_face_model = None


def get_face_analysis() -> FaceAnalysis:
    """Initialize and return the InsightFace FaceAnalysis app (singleton)."""
    global _face_app
    if _face_app is None:
        _face_app = FaceAnalysis(
            name="buffalo_l",
            root=os.path.join(os.path.expanduser("~"), ".insightface"),
            providers=["CPUExecutionProvider"],
        )
        _face_app.prepare(ctx_id=0, det_thresh=0.5, det_size=(640, 640))
    return _face_app


# ─── Fast Face Analysis (for live recognition — smaller det_size = ~4x faster) ───
_face_app_fast = None


def get_face_analysis_fast() -> FaceAnalysis:
    """Initialize a lightweight FaceAnalysis app with reduced detection size."""
    global _face_app_fast
    if _face_app_fast is None:
        _face_app_fast = FaceAnalysis(
            name="buffalo_l",
            root=os.path.join(os.path.expanduser("~"), ".insightface"),
            providers=["CPUExecutionProvider"],
        )
        _face_app_fast.prepare(ctx_id=0, det_thresh=0.5, det_size=(320, 320))
    return _face_app_fast


def detect_faces_fast(image: np.ndarray) -> List[dict]:
    """
    Fast face detection using 320×320 input size.
    Same return format as detect_faces().
    ~4x faster on CPU with minimal accuracy loss for live scenarios.
    """
    app = get_face_analysis_fast()

    # Ensure RGB format
    if image.shape[2] == 3:
        if image[0, 0, 2] > image[0, 0, 0]:  # rough BGR heuristic
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            rgb = image
    else:
        rgb = image

    faces = app.get(rgb)
    results = []
    for face in faces:
        results.append(
            {
                "bbox": face.bbox.tolist(),
                "landmark": face.landmark.tolist() if face.landmark is not None else None,
                "embedding": face.normed_embedding.tolist(),
                "det_score": float(face.det_score),
            }
        )
    return results


def get_feature_model():
    """Get the ArcFace feature extraction model."""
    global _face_model
    if _face_model is None:
        model_path = os.path.join(
            os.path.expanduser("~"),
            ".insightface",
            "models",
            "buffalo_l",
            "w600k_r50.onnx",
        )
        if os.path.exists(model_path):
            _face_model = get_model(model_path)
            _face_model.prepare(ctx_id=0)
    return _face_model


def detect_faces(image: np.ndarray) -> List[dict]:
    """
    Detect faces in an image.

    Args:
        image: numpy array (BGR format from OpenCV or RGB from Pillow).

    Returns:
        List of dicts with keys: 'bbox', 'landmark', 'embedding', 'det_score'.
        Returns empty list if no faces found.
    """
    app = get_face_analysis()

    # Ensure RGB format
    if image.shape[2] == 3:
        # If BGR (from OpenCV), convert to RGB
        if image[0, 0, 2] > image[0, 0, 0]:  # rough heuristic
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            rgb = image
    else:
        rgb = image

    faces = app.get(rgb)
    results = []
    for face in faces:
        results.append(
            {
                "bbox": face.bbox.tolist(),
                "landmark": face.landmark.tolist() if face.landmark is not None else None,
                "embedding": face.normed_embedding.tolist(),
                "det_score": float(face.det_score),
            }
        )
    return results


def extract_embedding(image: np.ndarray) -> Optional[List[float]]:
    """
    Extract face embedding from an image containing a single face.

    Args:
        image: numpy array (RGB or BGR).

    Returns:
        512-d embedding as a list, or None if no face detected.
    """
    faces = detect_faces(image)
    if not faces:
        return None
    # Return the embedding of the highest-confidence face
    best = max(faces, key=lambda f: f["det_score"])
    return best["embedding"]


def compare_embeddings(
    embedding1: List[float],
    embedding2: List[float],
    threshold: float = 0.5,
) -> Tuple[bool, float]:
    """
    Compare two face embeddings using cosine similarity.

    Args:
        embedding1: First face embedding.
        embedding2: Second face embedding.
        threshold: Similarity threshold (0.0 - 1.0). Higher = stricter.

    Returns:
        (is_match, similarity_score)
    """
    e1 = np.array(embedding1, dtype=np.float32)
    e2 = np.array(embedding2, dtype=np.float32)
    e1 = e1 / np.linalg.norm(e1)
    e2 = e2 / np.linalg.norm(e2)
    similarity = float(np.dot(e1, e2))
    return similarity >= threshold, similarity


def find_best_match(
    query_embedding: List[float],
    stored_embeddings: List[dict],
    threshold: float = 0.5,
) -> Optional[Tuple[str, str, float]]:
    """
    Find the best matching student from a list of stored embeddings.

    Args:
        query_embedding: Face embedding to search for.
        stored_embeddings: List of dicts with keys 'student_id', 'name', 'face_embedding'.
        threshold: Minimum similarity threshold.

    Returns:
        (student_id, name, confidence) of best match, or None if no match.
    """
    best_match = None
    best_score = 0.0

    for entry in stored_embeddings:
        is_match, score = compare_embeddings(
            query_embedding, entry["face_embedding"], threshold=0.0  # no threshold here
        )
        if score > best_score:
            best_score = score
            best_match = (entry["student_id"], entry["name"], score)

    if best_match and best_score >= threshold:
        return best_match
    return None


def load_image_from_path(file_path: str) -> Optional[np.ndarray]:
    """Load an image from disk using OpenCV."""
    image = cv2.imread(file_path)
    if image is None:
        return None
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def load_image_from_bytes(file_bytes: bytes) -> Optional[np.ndarray]:
    """Load an image from raw bytes (e.g., uploaded file)."""
    nparr = np.frombuffer(file_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        return None
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def crop_face(image: np.ndarray, bbox: List[float], margin: float = 0.3) -> np.ndarray:
    """
    Crop the face region from an image with a margin.

    Args:
        image: numpy array (RGB).
        bbox: [x1, y1, x2, y2] bounding box.
        margin: Fractional margin to add around the face.

    Returns:
        Cropped face image.
    """
    h, w = image.shape[:2]
    x1, y1, x2, y2 = map(int, bbox)
    face_w, face_h = x2 - x1, y2 - y1
    mx, my = int(face_w * margin), int(face_h * margin)
    x1 = max(0, x1 - mx)
    y1 = max(0, y1 - my)
    x2 = min(w, x2 + mx)
    y2 = min(h, y2 + my)
    return image[y1:y2, x1:x2]
