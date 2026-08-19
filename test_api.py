import argparse
import json
import mimetypes
from pathlib import Path

import requests


def pretty(response: requests.Response) -> None:
    print(f"HTTP {response.status_code}")
    try:
        print(json.dumps(response.json(), indent=2, ensure_ascii=False))
    except ValueError:
        print(response.text)


def test_health(base_url: str) -> None:
    print("\n=== GET /health ===")
    response = requests.get(f"{base_url}/health", timeout=30)
    pretty(response)
    response.raise_for_status()


def direct_ocr(base_url: str, file_path: Path, min_confidence: float) -> None:
    print("\n=== POST /ocr ===")
    mime = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
    with file_path.open("rb") as handle:
        response = requests.post(
            f"{base_url}/ocr",
            params={"min_confidence": min_confidence},
            files={"file": (file_path.name, handle, mime)},
            timeout=600,
        )
    pretty(response)
    response.raise_for_status()


def upload_then_ocr(base_url: str, file_path: Path, min_confidence: float) -> None:
    print("\n=== POST /upload ===")
    mime = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
    with file_path.open("rb") as handle:
        upload_response = requests.post(
            f"{base_url}/upload",
            files={"file": (file_path.name, handle, mime)},
            timeout=120,
        )
    pretty(upload_response)
    upload_response.raise_for_status()

    file_id = upload_response.json()["file_id"]

    print(f"\n=== POST /ocr/{file_id} ===")
    ocr_response = requests.post(
        f"{base_url}/ocr/{file_id}",
        params={"min_confidence": min_confidence},
        timeout=600,
    )
    pretty(ocr_response)
    ocr_response.raise_for_status()

    print(f"\n=== DELETE /upload/{file_id} ===")
    delete_response = requests.delete(
        f"{base_url}/upload/{file_id}",
        timeout=30,
    )
    pretty(delete_response)


def main() -> None:
    parser = argparse.ArgumentParser(description="Test the FastAPI PaddleOCR service")
    parser.add_argument("--file", required=True, help="Path to PDF or image")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument(
        "--mode",
        choices=["direct", "two-step"],
        default="direct",
        help="direct = POST /ocr, two-step = POST /upload then POST /ocr/{file_id}",
    )
    parser.add_argument("--min-confidence", type=float, default=0.20)
    args = parser.parse_args()

    file_path = Path(args.file)
    if not file_path.exists():
        raise SystemExit(f"File not found: {file_path}")

    base_url = args.base_url.rstrip("/")
    test_health(base_url)

    if args.mode == "direct":
        direct_ocr(base_url, file_path, args.min_confidence)
    else:
        upload_then_ocr(base_url, file_path, args.min_confidence)


if __name__ == "__main__":
    main()
