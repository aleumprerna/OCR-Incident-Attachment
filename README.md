# Lightweight PaddleOCR FastAPI

A small CPU OCR API for handwritten PDFs and images.

## Model

- Detector: `PP-OCRv5_mobile_det`
- Recognizer: `PP-OCRv5_mobile_rec`
- Download source: Hugging Face
- Runtime: ONNX Runtime CPU
- No LLM/API key is required.

The first OCR request downloads and caches the model files. Later requests reuse the local cache.

## 1. Create environment

Recommended: Python 3.10, 3.11, or 3.12.

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

## 2. Install

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## 3. Start API

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

Open Swagger UI:

```text
http://127.0.0.1:8000/docs
```

## Endpoints

### GET /health

Checks whether the API is running.

```bash
curl http://127.0.0.1:8000/health
```

### GET /model-info

Shows detector, recognizer, model source, runtime, and load status.

```bash
curl http://127.0.0.1:8000/model-info
```

### POST /ocr

Recommended endpoint. Upload a PDF or image and immediately return OCR text.

```bash
curl -X POST "http://127.0.0.1:8000/ocr?min_confidence=0.20" \
  -F "file=@handwritten.pdf"
```

Example response:

```json
{
  "filename": "handwritten.pdf",
  "page_count": 1,
  "line_count": 3,
  "text": "This is handwritten text\nSecond line\nThird line",
  "pages": [
    {
      "page": 1,
      "text": "This is handwritten text\nSecond line\nThird line",
      "line_count": 3,
      "lines": [
        {
          "text": "This is handwritten text",
          "confidence": 0.91,
          "bbox": [20, 30, 550, 85]
        }
      ]
    }
  ]
}
```

### POST /upload

Stores a PDF/image and returns a `file_id`.

```bash
curl -X POST "http://127.0.0.1:8000/upload" \
  -F "file=@handwritten.pdf"
```

### POST /ocr/{file_id}

Runs OCR on a previously uploaded file.

```bash
curl -X POST "http://127.0.0.1:8000/ocr/YOUR_FILE_ID?min_confidence=0.20"
```

### DELETE /upload/{file_id}

Deletes a stored upload.

```bash
curl -X DELETE "http://127.0.0.1:8000/upload/YOUR_FILE_ID"
```

## Test with Python

Direct upload + OCR:

```bash
python test_api.py --file handwritten.pdf
```

Upload first, OCR second, then delete:

```bash
python test_api.py --file handwritten.pdf --mode two-step
```

## Useful environment variables

```bash
# Upload limit, default 15 MB
export OCR_MAX_FILE_SIZE_MB=20

# CPU threads, default 4
export OCR_CPU_THREADS=4

# Uploaded-file directory, default ./uploads
export OCR_UPLOAD_DIR=uploads
```

Windows PowerShell example:

```powershell
$env:OCR_CPU_THREADS="4"
```

## Notes for handwriting

- Use a clear scan/photo at roughly 200-300 DPI when possible.
- Keep the page upright.
- Avoid strong shadows and blur.
- `min_confidence=0.20` is intentionally low for handwriting. Increase it to `0.40` or `0.50` if you want fewer low-confidence lines.
