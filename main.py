import json
import sqlite3
import uuid
import traceback
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile

from image_processing import (
    detect_and_align_card,
    preprocess_for_ocr,
)
from ocr import extract_text, extract_fields


# --------------------------------------------------
# Configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

UPLOAD_DIR = BASE_DIR / "uploads"
PROCESSED_DIR = BASE_DIR / "processed_faces"
FACES_DIR = PROCESSED_DIR / "faces"
DATABASE_PATH = BASE_DIR / "records.db"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
FACES_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_TYPES = {"image/jpeg", "image/png"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

app = FastAPI(
    title="Document AI Backend",
    description="ID Card Scanner API",
    version="1.0.0",
)


# --------------------------------------------------
# Database
# --------------------------------------------------

def get_connection():
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_database():
    with get_connection() as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                result_json TEXT NOT NULL
            )
        """)


def save_record(result):
    with get_connection() as connection:
        cursor = connection.execute(
            "INSERT INTO records (result_json) VALUES (?)",
            (json.dumps(result),),
        )
        return cursor.lastrowid


def read_record(record_id):
    with get_connection() as connection:
        row = connection.execute(
            "SELECT id, result_json FROM records WHERE id = ?",
            (record_id,),
        ).fetchone()

    if row is None:
        return None

    result = json.loads(row["result_json"])
    result["id"] = row["id"]
    return result


@app.on_event("startup")
def startup():
    initialize_database()


# --------------------------------------------------
# Face Thumbnail Extraction
# --------------------------------------------------

def extract_face_thumbnail(image, output_path):
    """Detect the largest face and save a 160x160 thumbnail."""

    cascade_path = (
        Path(cv2.data.haarcascades)
        / "haarcascade_frontalface_default.xml"
    )

    detector = cv2.CascadeClassifier(str(cascade_path))

    if detector.empty():
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
    )

    if len(faces) == 0:
        return None

    x, y, w, h = max(
        faces,
        key=lambda face: face[2] * face[3],
    )

    # Add a small margin around the face.
    margin_x = int(w * 0.15)
    margin_y = int(h * 0.15)

    x1 = max(0, x - margin_x)
    y1 = max(0, y - margin_y)
    x2 = min(image.shape[1], x + w + margin_x)
    y2 = min(image.shape[0], y + h + margin_y)

    face = image[y1:y2, x1:x2]

    if face.size == 0:
        return None

    face = cv2.resize(face, (160, 160))

    if cv2.imwrite(str(output_path), face):
        return str(
            output_path.relative_to(BASE_DIR)
        ).replace("\\", "/")

    return None


# --------------------------------------------------
# Home Endpoint
# --------------------------------------------------

@app.get("/")
def home():
    return {
        "message": "Document AI Backend is running",
        "docs": "/docs",
    }


# --------------------------------------------------
# POST: Process ID Card
# --------------------------------------------------

@app.post("/api/v1/process-id")
async def process_id(file: UploadFile = File(...)):

    # 1. Validate the declared content type.
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=415,
            detail="Only JPEG and PNG images are supported.",
        )

    # 2. Read and validate the upload size.
    contents = await file.read()

    if not contents:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty.",
        )

    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="Maximum upload size is 10 MB.",
        )

    # 3. Decode the uploaded image.
    encoded = np.frombuffer(contents, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)

    if image is None or image.size == 0:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is not a readable image.",
        )

    height, width = image.shape[:2]

    if max(height, width) > 10000:
        raise HTTPException(
            status_code=400,
            detail="Image dimensions exceed the allowed limit.",
        )

    if height < 2 or width < 2:
        raise HTTPException(
            status_code=400,
            detail="Invalid image dimensions.",
        )

    # 4. Save the original image using a safe unique filename.
    file_id = uuid.uuid4().hex
    upload_path = UPLOAD_DIR / f"{file_id}.jpg"

    if not cv2.imwrite(str(upload_path), image):
        raise HTTPException(
            status_code=500,
            detail="Could not save uploaded image.",
        )

    try:
        # 5. Detect and align the card.
        aligned_image, card_detected = detect_and_align_card(
            image
        )

        if aligned_image is None or aligned_image.size == 0:
            raise ValueError("Card processing returned an empty image.")

        # 6. Save the processed image.
        processed_path = PROCESSED_DIR / f"{file_id}.png"

        if not cv2.imwrite(str(processed_path), aligned_image):
            raise RuntimeError("Could not save processed image.")

        # 7. Preprocess and run OCR.
        ocr_image = preprocess_for_ocr(aligned_image)
        text_lines = extract_text(ocr_image)
        fields = extract_fields(text_lines)

        # 8. Extract the face thumbnail when possible.
        face_path = extract_face_thumbnail(
            aligned_image,
            FACES_DIR / f"{file_id}.jpg",
        )

        # 9. Build the structured response.
        result = {
            "extracted_fields": fields,
            "image_metadata": {
                "filename": file.filename or "uploaded-image",
                "content_type": file.content_type,
                "width": width,
                "height": height,
                "file_size_bytes": len(contents),
                "document_detected": bool(card_detected),
            },
            "processed_image_path": str(
                processed_path.relative_to(BASE_DIR)
            ).replace("\\", "/"),
            "face_thumbnail_path": face_path,
        }

        # 10. Save the result in SQLite.
        record_id = save_record(result)

        return {
            "id": record_id,
            **result,
        }

    except HTTPException:
        raise

    except Exception as exc:
        # Print the real exception in the terminal for debugging.
        print("\n--- ID CARD PROCESSING ERROR ---")
        traceback.print_exc()
        print("Error details:", repr(exc))
        print("--- END ERROR ---\n")

        raise HTTPException(
            status_code=500,
            detail="Image processing failed. Check server logs.",
        )


# --------------------------------------------------
# GET: Retrieve a Previously Processed Record
# --------------------------------------------------

@app.get("/api/v1/records/{record_id}")
def get_record(record_id: int):

    if record_id <= 0:
        raise HTTPException(
            status_code=400,
            detail="Record ID must be positive.",
        )

    record = read_record(record_id)

    if record is None:
        raise HTTPException(
            status_code=404,
            detail="Record not found.",
        )

    return record