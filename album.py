from __future__ import annotations

import logging
import os
import socket
from datetime import datetime
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_from_directory
from werkzeug.utils import secure_filename

from duplicate_detector import DuplicateDetectorService, IMAGE_EXTENSIONS

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = BASE_DIR / "uploads"
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
app.config["MAX_CONTENT_LENGTH"] = 1024 * 1024 * 1024
app.config["ALLOWED_EXTENSIONS"] = {"png", "jpg", "jpeg", "gif", "webp", "bmp", "tif", "tiff", "mp4", "mov"}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(BASE_DIR / "app.log"), logging.StreamHandler()],
)
app.logger.setLevel(logging.INFO)

detector = DuplicateDetectorService(
    upload_root=UPLOAD_FOLDER,
    threshold=int(os.getenv("DUPLICATE_THRESHOLD", "6")),
    aspect_tolerance=float(os.getenv("DUPLICATE_ASPECT_TOLERANCE", "0.04")),
    batch_size=int(os.getenv("DUPLICATE_BATCH_SIZE", "128")),
)


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in app.config["ALLOWED_EXTENSIONS"]


def get_dir_name() -> str:
    return datetime.now().strftime("%Y%m%d")


def create_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def unique_destination(directory: Path, filename: str) -> Path:
    destination = directory / filename
    if not destination.exists():
        return destination
    stem = destination.stem
    suffix = destination.suffix
    counter = 1
    while True:
        candidate = directory / f"{stem}_{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


@app.before_request
def ensure_detector_started() -> None:
    # Idempotent fallback for `flask run` and WSGI imports.
    detector.start(initial_scan=True)


@app.route("/")
def index():
    return render_template("index.html", status=detector.status())


@app.route("/upload", methods=["GET", "POST"])
def upload():
    if request.method == "GET":
        return render_template("index.html", status=detector.status())

    uploaded_files = request.files.getlist("file[]")
    date_folder = get_dir_name()
    target_directory = UPLOAD_FOLDER / date_folder
    create_dir(target_directory)

    filenames: list[str] = []
    image_paths: list[Path] = []
    rejected: list[str] = []
    for uploaded_file in uploaded_files:
        original_name = uploaded_file.filename or ""
        if not uploaded_file or not allowed_file(original_name):
            if original_name:
                rejected.append(original_name)
            continue
        safe_name = secure_filename(original_name)
        if not safe_name:
            rejected.append(original_name)
            continue
        destination = unique_destination(target_directory, safe_name)
        uploaded_file.save(destination)
        relative = destination.relative_to(UPLOAD_FOLDER).as_posix()
        filenames.append(relative)
        if destination.suffix.lower() in IMAGE_EXTENSIONS:
            image_paths.append(destination)
        app.logger.info("File saved to %s", destination)

    detector.enqueue_files(image_paths)
    return render_template(
        "upload.html",
        filenames=filenames,
        rejected=rejected,
        image_count=len(image_paths),
        status=detector.status(),
    )


@app.route("/uploads/<path:filename>")
def uploaded_file(filename: str):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


@app.route("/duplicates")
def duplicates_page():
    return render_template("duplicates.html", duplicates=detector.duplicates(), status=detector.status())


@app.route("/api/detection/status")
def detection_status():
    return jsonify(detector.status())


@app.route("/api/detection/duplicates")
def detection_duplicates():
    return jsonify(detector.duplicates())


@app.route("/api/detection/rescan", methods=["POST"])
def detection_rescan():
    detector.enqueue_full_scan()
    return jsonify({"accepted": True, "status": detector.status()}), 202


def get_local_ip() -> str:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("8.8.8.8", 80))
        return sock.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        sock.close()


if __name__ == "__main__":
    detector.start(initial_scan=True)
    host_ip = get_local_ip()
    print(f"Local address: http://127.0.0.1:8080")
    print(f"LAN address:   http://{host_ip}:8080")
    print(f"Duplicate backend: {detector.backend.name}")
    app.run(host="0.0.0.0", port=8080, debug=False, use_reloader=False)
