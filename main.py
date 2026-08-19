import json
import os
import tempfile
import threading
import uuid
from pathlib import Path
from typing import Any

# Force official PaddleOCR models to be downloaded from Hugging Face.
os.environ.setdefault("PADDLE_PDX_MODEL_SOURCE", "huggingface")

from fastapi import FastAPI, File, HTTPException, Query, UploadFile,Request
from fastapi.concurrency import run_in_threadpool
from paddleocr import PaddleOCR

APP_NAME = "Lightweight Handwritten OCR API"
UPLOAD_DIR = Path(os.getenv("OCR_UPLOAD_DIR", "uploads"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

MAX_FILE_SIZE_MB = int(os.getenv("OCR_MAX_FILE_SIZE_MB", "15"))
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
OCR_CPU_THREADS = int(os.getenv("OCR_CPU_THREADS", "4"))

ALLOWED_EXTENSIONS = {
    ".pdf",
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
}

DETECTION_MODEL = "PP-OCRv5_mobile_det"
RECOGNITION_MODEL = "PP-OCRv5_mobile_rec"
MODEL_SOURCE = "huggingface"
INFERENCE_ENGINE = "onnxruntime"

app = FastAPI(
    title=APP_NAME,
    version="1.0.0",
    description=(
        "Upload handwritten PDFs or images and extract text using "
        "PP-OCRv5 mobile models downloaded from Hugging Face."
    ),
)

_ocr_instance: PaddleOCR | None = None
_model_init_lock = threading.Lock()
_inference_lock = threading.Lock()


def get_ocr() -> PaddleOCR:
    """Lazy-load OCR so the API can start before model download finishes."""
    global _ocr_instance
    if _ocr_instance is None:
        with _model_init_lock:
            if _ocr_instance is None:
                _ocr_instance = PaddleOCR(
                    text_detection_model_name=DETECTION_MODEL,
                    text_recognition_model_name=RECOGNITION_MODEL,
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    use_textline_orientation=False,
                    engine=INFERENCE_ENGINE,
                    engine_config={
                        "device_type": "cpu",
                        "providers": ["CPUExecutionProvider"],
                        "intra_op_num_threads": OCR_CPU_THREADS,
                        "inter_op_num_threads": 1,
                        "execution_mode": "sequential",
                    },
                )
    return _ocr_instance


def validate_file(upload: UploadFile) -> str:
    if not upload.filename:
        raise HTTPException(status_code=400, detail="File name is missing.")

    suffix = Path(upload.filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        allowed = ", ".join(sorted(ALLOWED_EXTENSIONS))
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Allowed: {allowed}",
        )
    return suffix


async def save_upload(upload: UploadFile, destination: Path) -> int:
    """Stream upload to disk and enforce a maximum file size."""
    total = 0
    chunk_size = 1024 * 1024

    try:
        with destination.open("wb") as output:
            while True:
                chunk = await upload.read(chunk_size)
                if not chunk:
                    break
                total += len(chunk)
                if total > MAX_FILE_SIZE_BYTES:
                    output.close()
                    destination.unlink(missing_ok=True)
                    raise HTTPException(
                        status_code=413,
                        detail=f"File is larger than {MAX_FILE_SIZE_MB} MB.",
                    )
                output.write(chunk)
    finally:
        await upload.close()

    if total == 0:
        destination.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    return total


def result_to_dict(result: Any) -> dict[str, Any]:
    """
    Convert PaddleOCR result to a normal Python dict.

    Newer PaddleOCR versions expose result.json. The save_to_json fallback
    keeps this API tolerant of versions where that attribute is unavailable.
    """
    try:
        value = result.json
        if callable(value):
            value = value()
        if isinstance(value, str):
            value = json.loads(value)
        if isinstance(value, dict):
            return value
    except Exception:
        pass

    tmp_path: Path | None = None
    try:
        fd, name = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        tmp_path = Path(name)
        result.save_to_json(str(tmp_path))
        return json.loads(tmp_path.read_text(encoding="utf-8"))
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)


def normalize_page(data: dict[str, Any], fallback_page_index: int) -> dict[str, Any]:
    payload = data.get("res", data)

    texts = payload.get("rec_texts") or []
    scores = payload.get("rec_scores") or []
    boxes = payload.get("rec_boxes") or []

    lines: list[dict[str, Any]] = []
    clean_texts: list[str] = []

    for index, raw_text in enumerate(texts):
        text = str(raw_text).strip()
        if not text:
            continue

        score = None
        if index < len(scores):
            try:
                score = round(float(scores[index]), 4)
            except (TypeError, ValueError):
                score = None

        bbox = None
        if index < len(boxes):
            raw_box = boxes[index]
            try:
                bbox = [int(v) for v in raw_box]
            except Exception:
                bbox = raw_box

        clean_texts.append(text)
        lines.append(
            {
                "text": text,
                "confidence": score,
                "bbox": bbox,
            }
        )

    page_index = payload.get("page_index")
    if page_index is None:
        page_index = fallback_page_index

    return {
        "page": int(page_index) + 1,
        "text": "\n".join(clean_texts),
        "line_count": len(lines),
        "lines": lines,
    }


def run_ocr(path: Path, min_confidence: float) -> dict[str, Any]:
    ocr = get_ocr()

    # Serializing inference is safer for a single lightweight CPU model.
    with _inference_lock:
        results = ocr.predict(
            str(path),
            text_rec_score_thresh=min_confidence,
        )

    pages: list[dict[str, Any]] = []
    for index, result in enumerate(results):
        pages.append(normalize_page(result_to_dict(result), index))

    combined_text = "\n\n".join(page["text"] for page in pages if page["text"])

    return {
        "filename": path.name,
        "model": {
            "detector": DETECTION_MODEL,
            "recognizer": RECOGNITION_MODEL,
            "source": MODEL_SOURCE,
            "engine": INFERENCE_ENGINE,
        },
        "page_count": len(pages),
        "line_count": sum(page["line_count"] for page in pages),
        "text": combined_text,
        "pages": pages,
    }


def find_uploaded_file(file_id: str) -> Path:
    if not file_id or any(ch not in "0123456789abcdef-" for ch in file_id.lower()):
        raise HTTPException(status_code=400, detail="Invalid file_id.")

    matches = list(UPLOAD_DIR.glob(f"{file_id}.*"))
    if not matches:
        raise HTTPException(status_code=404, detail="Uploaded file not found.")
    return matches[0]


@app.get("/")
def root() -> dict[str, Any]:
    return {
        "service": APP_NAME,
        "status": "ok",
        "docs": "/docs",
        "direct_ocr": "POST /ocr",
        "upload": "POST /upload",
        "ocr_uploaded": "POST /ocr/{file_id}",
    }


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "healthy",
        "model_loaded": _ocr_instance is not None,
    }


@app.get("/model-info")
def model_info() -> dict[str, Any]:
    return {
        "detector": DETECTION_MODEL,
        "recognizer": RECOGNITION_MODEL,
        "model_source": MODEL_SOURCE,
        "engine": INFERENCE_ENGINE,
        "device": "cpu",
        "cpu_threads": OCR_CPU_THREADS,
        "model_loaded": _ocr_instance is not None,
    }


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)) -> dict[str, Any]:
    """Upload and store a PDF/image. OCR can be run later using its file_id."""
    suffix = validate_file(file)
    file_id = str(uuid.uuid4())
    destination = UPLOAD_DIR / f"{file_id}{suffix}"
    size = await save_upload(file, destination)

    return {
        "message": "File uploaded successfully.",
        "file_id": file_id,
        "original_filename": file.filename,
        "stored_filename": destination.name,
        "size_bytes": size,
        "ocr_endpoint": f"/ocr/{file_id}",
    }


@app.post("/ocr")
async def ocr_direct(
    file: UploadFile = File(...),
    min_confidence: float = Query(0.20, ge=0.0, le=1.0),
) -> dict[str, Any]:
    """Upload a PDF/image and immediately return OCR text."""
    suffix = validate_file(file)

    with tempfile.TemporaryDirectory(prefix="ocr_") as tmp_dir:
        destination = Path(tmp_dir) / f"input{suffix}"
        await save_upload(file, destination)

        try:
            result = await run_in_threadpool(run_ocr, destination, min_confidence)
            result["filename"] = file.filename
            return result
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"OCR failed: {exc}") from exc


@app.post("/extract-pdf")
async def extract_pdf_raw(
    request: Request,
    min_confidence: float = Query(0.20, ge=0.0, le=1.0),
) -> dict[str, Any]:

    """
    ServiceNow-compatible OCR endpoint.

    Receives raw PDF bytes:

    Content-Type: application/pdf
    Body: raw PDF attachment
    """

    content_type = request.headers.get(
        "content-type",
        ""
    ).lower()

    # Allow PDF
    if "application/pdf" in content_type:
        suffix = ".pdf"

    # Optional image support
    elif "image/png" in content_type:
        suffix = ".png"

    elif "image/jpeg" in content_type:
        suffix = ".jpg"

    else:
        raise HTTPException(
            status_code=415,
            detail=(
                "Unsupported Content-Type. "
                "Expected application/pdf, "
                "image/png, or image/jpeg."
            ),
        )

    # Get raw attachment bytes
    body = await request.body()

    if not body:
        raise HTTPException(
            status_code=400,
            detail="Request body is empty."
        )

    # Protect the lightweight service
    if len(body) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds {MAX_FILE_SIZE_MB} MB."
        )

    with tempfile.TemporaryDirectory(
        prefix="servicenow_ocr_"
    ) as tmp_dir:

        file_path = (
            Path(tmp_dir) /
            f"servicenow_attachment{suffix}"
        )

        file_path.write_bytes(body)

        try:

            result = await run_in_threadpool(
                run_ocr,
                file_path,
                min_confidence
            )

            extracted_text = (
                result.get("text") or ""
            )

            return {
                "success": True,
                "text": extracted_text,
                "characters_extracted": len(
                    extracted_text
                ),
                "page_count": result.get(
                    "page_count",
                    0
                ),
                "line_count": result.get(
                    "line_count",
                    0
                ),
                "pages": result.get(
                    "pages",
                    []
                )
            }

        except Exception as exc:

            raise HTTPException(
                status_code=500,
                detail=f"OCR failed: {exc}"
            ) from exc

@app.post("/ocr/{file_id}")
async def ocr_uploaded(
    file_id: str,
    min_confidence: float = Query(0.20, ge=0.0, le=1.0),
) -> dict[str, Any]:
    """Run OCR on a file previously uploaded using POST /upload."""
    path = find_uploaded_file(file_id)
    try:
        result = await run_in_threadpool(run_ocr, path, min_confidence)
        result["file_id"] = file_id
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"OCR failed: {exc}") from exc


@app.delete("/upload/{file_id}")
def delete_uploaded_file(file_id: str) -> dict[str, Any]:
    """Delete a previously uploaded file."""
    path = find_uploaded_file(file_id)
    path.unlink(missing_ok=True)
    return {"message": "File deleted.", "file_id": file_id}
