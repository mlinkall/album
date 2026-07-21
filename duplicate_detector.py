from __future__ import annotations

import hashlib
import logging
import math
import os
import queue
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

LOGGER = logging.getLogger(__name__)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".tif", ".tiff"}
HASH_IMAGE_SIZE = 32
HASH_BLOCK_SIZE = 8
HASH_BITS = HASH_BLOCK_SIZE * HASH_BLOCK_SIZE


def _dct_matrix(size: int = HASH_IMAGE_SIZE) -> np.ndarray:
    matrix = np.empty((size, size), dtype=np.float32)
    factor = math.pi / (2.0 * size)
    for k in range(size):
        scale = math.sqrt(1.0 / size) if k == 0 else math.sqrt(2.0 / size)
        for n in range(size):
            matrix[k, n] = scale * math.cos((2 * n + 1) * k * factor)
    return matrix


DCT_MATRIX = _dct_matrix()


@dataclass(slots=True)
class ImageFingerprint:
    path: str
    mtime_ns: int
    size_bytes: int
    sha256: str
    phash: str
    width: int
    height: int
    duplicate_of: str | None = None
    distance: int | None = None
    match_type: str | None = None
    backend: str = "cpu"
    hash_ms: float = 0.0
    scanned_at: float = 0.0

    @property
    def aspect_ratio(self) -> float:
        return self.width / self.height if self.height else 0.0


@dataclass(slots=True)
class FingerprintStats:
    files_requested: int = 0
    files_processed: int = 0
    files_failed: int = 0
    decode_sha_seconds: float = 0.0
    phash_seconds: float = 0.0


class HashBackend:
    name = "base"

    def compute_phash_bits(self, images: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def nearest_against(
        self,
        query_bits: np.ndarray,
        query_ratios: np.ndarray,
        reference_bits: np.ndarray,
        reference_ratios: np.ndarray,
        aspect_tolerance: float,
        chunk_size: int = 256,
    ) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def nearest_previous(
        self,
        bits: np.ndarray,
        ratios: np.ndarray,
        aspect_tolerance: float,
        chunk_size: int = 256,
    ) -> tuple[np.ndarray, np.ndarray]:
        if len(bits) == 0:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int16)
        result_index = np.full(len(bits), -1, dtype=np.int64)
        result_distance = np.full(len(bits), 127, dtype=np.int16)
        for start in range(1, len(bits), chunk_size):
            end = min(start + chunk_size, len(bits))
            indices, distances = self.nearest_against(
                bits[start:end],
                ratios[start:end],
                bits[:end],
                ratios[:end],
                aspect_tolerance,
                chunk_size=chunk_size,
            )
            row_indices = np.arange(start, end)
            for local_row, global_row in enumerate(row_indices):
                candidate = int(indices[local_row])
                if candidate >= global_row:
                    candidate = -1
                if candidate >= 0:
                    result_index[global_row] = candidate
                    result_distance[global_row] = int(distances[local_row])
                    continue
                # The unrestricted nearest result may have pointed to itself or
                # a later row. Re-run this row against only previous records.
                fallback_index, fallback_distance = self.nearest_against(
                    bits[global_row : global_row + 1],
                    ratios[global_row : global_row + 1],
                    bits[:global_row],
                    ratios[:global_row],
                    aspect_tolerance,
                    chunk_size=chunk_size,
                )
                if int(fallback_index[0]) >= 0:
                    result_index[global_row] = int(fallback_index[0])
                    result_distance[global_row] = int(fallback_distance[0])
        return result_index, result_distance


class CPUHashBackend(HashBackend):
    name = "cpu"

    def __init__(self) -> None:
        self._dct = DCT_MATRIX
        self._dct_t = DCT_MATRIX.T

    def compute_phash_bits(self, images: np.ndarray) -> np.ndarray:
        if len(images) == 0:
            return np.empty((0, HASH_BITS), dtype=np.uint8)
        transformed = self._dct[None, :, :] @ images @ self._dct_t[None, :, :]
        low_frequency = transformed[:, :HASH_BLOCK_SIZE, :HASH_BLOCK_SIZE].reshape(len(images), -1)
        medians = np.median(low_frequency[:, 1:], axis=1, keepdims=True)
        bits = (low_frequency > medians).astype(np.uint8)
        bits[:, 0] = 0
        return bits

    def nearest_against(
        self,
        query_bits: np.ndarray,
        query_ratios: np.ndarray,
        reference_bits: np.ndarray,
        reference_ratios: np.ndarray,
        aspect_tolerance: float,
        chunk_size: int = 256,
    ) -> tuple[np.ndarray, np.ndarray]:
        if len(query_bits) == 0:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int16)
        if len(reference_bits) == 0:
            return np.full(len(query_bits), -1, dtype=np.int64), np.full(len(query_bits), 127, dtype=np.int16)

        refs = reference_bits.astype(np.float32, copy=False)
        ref_sums = refs.sum(axis=1)
        safe_ref_ratios = np.maximum(reference_ratios.astype(np.float64), 1e-12)
        best_indices = np.full(len(query_bits), -1, dtype=np.int64)
        best_distances = np.full(len(query_bits), 127, dtype=np.int16)

        for start in range(0, len(query_bits), chunk_size):
            end = min(start + chunk_size, len(query_bits))
            queries = query_bits[start:end].astype(np.float32, copy=False)
            distances = queries.sum(axis=1, keepdims=True) + ref_sums[None, :] - 2.0 * (queries @ refs.T)
            safe_query_ratios = np.maximum(query_ratios[start:end].astype(np.float64), 1e-12)
            ratio_delta = np.abs(np.log(safe_query_ratios[:, None] / safe_ref_ratios[None, :]))
            distances[ratio_delta > aspect_tolerance] = 999.0
            indices = np.argmin(distances, axis=1)
            values = distances[np.arange(end - start), indices]
            valid = values < 999.0
            best_indices[start:end][valid] = indices[valid]
            best_distances[start:end][valid] = np.rint(values[valid]).astype(np.int16)
        return best_indices, best_distances

    def nearest_previous(
        self,
        bits: np.ndarray,
        ratios: np.ndarray,
        aspect_tolerance: float,
        chunk_size: int = 256,
    ) -> tuple[np.ndarray, np.ndarray]:
        count = len(bits)
        best_indices = np.full(count, -1, dtype=np.int64)
        best_distances = np.full(count, 127, dtype=np.int16)
        if count < 2:
            return best_indices, best_distances

        refs = bits.astype(np.float32, copy=False)
        ref_sums = refs.sum(axis=1)
        safe_ratios = np.maximum(ratios.astype(np.float64), 1e-12)
        columns = np.arange(count)[None, :]
        for start in range(1, count, chunk_size):
            end = min(start + chunk_size, count)
            queries = refs[start:end]
            distances = queries.sum(axis=1, keepdims=True) + ref_sums[None, :] - 2.0 * (queries @ refs.T)
            rows = np.arange(start, end)[:, None]
            distances[columns >= rows] = 999.0
            ratio_delta = np.abs(np.log(safe_ratios[start:end, None] / safe_ratios[None, :]))
            distances[ratio_delta > aspect_tolerance] = 999.0
            indices = np.argmin(distances, axis=1)
            values = distances[np.arange(end - start), indices]
            valid = values < 999.0
            best_indices[start:end][valid] = indices[valid]
            best_distances[start:end][valid] = np.rint(values[valid]).astype(np.int16)
        return best_indices, best_distances



def build_backend() -> HashBackend:
    """Create the only supported backend: NumPy CPU processing."""
    return CPUHashBackend()


def bits_to_hex(bits: np.ndarray) -> str:
    packed = np.packbits(bits.astype(np.uint8))
    return packed.tobytes().hex()


def hex_to_bits(value: str) -> np.ndarray:
    raw = bytes.fromhex(value)
    return np.unpackbits(np.frombuffer(raw, dtype=np.uint8))[:HASH_BITS].astype(np.uint8)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def load_hash_input(path: Path, include_sha256: bool = True) -> tuple[np.ndarray, int, int, str]:
    with Image.open(path) as image:
        image.seek(0)
        image = ImageOps.exif_transpose(image)
        width, height = image.size
        gray = image.convert("L").resize((HASH_IMAGE_SIZE, HASH_IMAGE_SIZE), Image.Resampling.LANCZOS)
        array = np.asarray(gray, dtype=np.float32) / 255.0
    sha = sha256_file(path) if include_sha256 else ""
    return array, width, height, sha


def fingerprint_paths(
    paths: Sequence[Path],
    root: Path,
    backend: HashBackend,
    batch_size: int = 128,
    include_sha256: bool = True,
) -> tuple[list[ImageFingerprint], FingerprintStats]:
    stats = FingerprintStats(files_requested=len(paths))
    records: list[ImageFingerprint] = []

    for start in range(0, len(paths), batch_size):
        batch_paths = paths[start : start + batch_size]
        arrays: list[np.ndarray] = []
        metadata: list[tuple[Path, os.stat_result, int, int, str]] = []
        decode_started = time.perf_counter()
        for path in batch_paths:
            try:
                stat = path.stat()
                array, width, height, sha = load_hash_input(path, include_sha256=include_sha256)
                arrays.append(array)
                metadata.append((path, stat, width, height, sha))
            except (OSError, UnidentifiedImageError, ValueError) as exc:
                stats.files_failed += 1
                LOGGER.warning("Skipping unreadable image %s: %s", path, exc)
        stats.decode_sha_seconds += time.perf_counter() - decode_started
        if not arrays:
            continue

        image_batch = np.stack(arrays, axis=0).astype(np.float32, copy=False)
        hash_started = time.perf_counter()
        bit_rows = backend.compute_phash_bits(image_batch)
        elapsed = time.perf_counter() - hash_started
        stats.phash_seconds += elapsed
        per_image_ms = elapsed * 1000.0 / len(metadata)
        now = time.time()

        for bit_row, (path, stat, width, height, sha) in zip(bit_rows, metadata):
            relative = path.relative_to(root).as_posix()
            records.append(
                ImageFingerprint(
                    path=relative,
                    mtime_ns=stat.st_mtime_ns,
                    size_bytes=stat.st_size,
                    sha256=sha,
                    phash=bits_to_hex(bit_row),
                    width=width,
                    height=height,
                    backend=backend.name,
                    hash_ms=per_image_ms,
                    scanned_at=now,
                )
            )
            stats.files_processed += 1
    return records, stats


def _record_arrays(records: Sequence[ImageFingerprint]) -> tuple[np.ndarray, np.ndarray]:
    if not records:
        return np.empty((0, HASH_BITS), dtype=np.uint8), np.empty(0, dtype=np.float32)
    bits = np.stack([hex_to_bits(record.phash) for record in records], axis=0)
    ratios = np.asarray([record.aspect_ratio for record in records], dtype=np.float32)
    return bits, ratios


def classify_records(
    records: Sequence[ImageFingerprint],
    backend: HashBackend,
    threshold: int = 6,
    aspect_tolerance: float = 0.04,
) -> list[ImageFingerprint]:
    """Classify a full ordered collection and return new record objects."""
    if not records:
        return []
    bits, ratios = _record_arrays(records)
    near_indices, near_distances = backend.nearest_previous(bits, ratios, aspect_tolerance)
    output: list[ImageFingerprint] = []
    exact_map: dict[str, str] = {}
    roots: dict[str, str] = {}

    for index, original in enumerate(records):
        record = replace(original, duplicate_of=None, distance=None, match_type=None)
        exact_root = exact_map.get(record.sha256) if record.sha256 else None
        if exact_root is not None:
            record.duplicate_of = exact_root
            record.distance = 0
            record.match_type = "exact"
        else:
            candidate_index = int(near_indices[index])
            candidate_distance = int(near_distances[index])
            if candidate_index >= 0 and candidate_distance <= threshold:
                candidate_path = records[candidate_index].path
                record.duplicate_of = roots.get(candidate_path, candidate_path)
                record.distance = candidate_distance
                record.match_type = "similar"

        root = record.duplicate_of or record.path
        roots[record.path] = root
        if record.sha256 and record.sha256 not in exact_map:
            exact_map[record.sha256] = root
        output.append(record)
    return output


def classify_new_records(
    existing: Sequence[ImageFingerprint],
    new_records: Sequence[ImageFingerprint],
    backend: HashBackend,
    threshold: int = 6,
    aspect_tolerance: float = 0.04,
) -> list[ImageFingerprint]:
    """Classify only newly uploaded records against the existing index and each other."""
    if not new_records:
        return []
    existing_bits, existing_ratios = _record_arrays(existing)
    new_bits, new_ratios = _record_arrays(new_records)
    against_indices, against_distances = backend.nearest_against(
        new_bits, new_ratios, existing_bits, existing_ratios, aspect_tolerance
    )
    previous_indices, previous_distances = backend.nearest_previous(new_bits, new_ratios, aspect_tolerance)

    roots = {record.path: record.duplicate_of or record.path for record in existing}
    exact_map: dict[str, str] = {}
    for record in existing:
        if record.sha256:
            exact_map.setdefault(record.sha256, roots[record.path])

    output: list[ImageFingerprint] = []
    for index, original in enumerate(new_records):
        record = replace(original, duplicate_of=None, distance=None, match_type=None)
        exact_root = exact_map.get(record.sha256) if record.sha256 else None
        if exact_root is not None:
            record.duplicate_of = exact_root
            record.distance = 0
            record.match_type = "exact"
        else:
            candidates: list[tuple[int, str]] = []
            existing_index = int(against_indices[index])
            existing_distance = int(against_distances[index])
            if existing_index >= 0:
                existing_path = existing[existing_index].path
                candidates.append((existing_distance, roots.get(existing_path, existing_path)))

            new_index = int(previous_indices[index])
            new_distance = int(previous_distances[index])
            if new_index >= 0:
                previous_path = new_records[new_index].path
                candidates.append((new_distance, roots.get(previous_path, previous_path)))

            if candidates:
                distance, root = min(candidates, key=lambda item: item[0])
                if distance <= threshold:
                    record.duplicate_of = root
                    record.distance = distance
                    record.match_type = "similar"

        root = record.duplicate_of or record.path
        roots[record.path] = root
        if record.sha256:
            exact_map.setdefault(record.sha256, root)
        output.append(record)
    return output



class DuplicateDetectorService:
    """Background detector with a process-local, in-memory fingerprint collection.

    Fingerprints are intentionally not persisted. Every application restart performs
    a full background scan; uploads are incrementally compared for the lifetime of
    the running process.
    """

    def __init__(
        self,
        upload_root: Path,
        threshold: int = 6,
        aspect_tolerance: float = 0.04,
        batch_size: int = 128,
    ) -> None:
        self.upload_root = upload_root.resolve()
        self.upload_root.mkdir(parents=True, exist_ok=True)
        self.backend = build_backend()
        self.threshold = threshold
        self.aspect_tolerance = aspect_tolerance
        self.batch_size = batch_size
        self._records: dict[str, ImageFingerprint] = {}
        self._records_lock = threading.RLock()
        self._queue: queue.Queue[tuple[str, tuple[Path, ...]]] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._start_lock = threading.Lock()
        self._status_lock = threading.Lock()
        self._status: dict[str, object] = {
            "state": "not_started",
            "backend": self.backend.name,
            "storage": "memory",
            "queued_tasks": 0,
            "files_found": 0,
            "duplicates": 0,
            "last_error": None,
        }

    def start(self, initial_scan: bool = True) -> None:
        with self._start_lock:
            if self._thread and self._thread.is_alive():
                return
            self._thread = threading.Thread(target=self._worker, name="duplicate-detector", daemon=True)
            self._thread.start()
            self._set_status(state="idle")
            if initial_scan:
                self.enqueue_full_scan()

    def enqueue_full_scan(self) -> None:
        self._queue.put(("full", ()))
        self._set_status(queued_tasks=self._queue.qsize())

    def enqueue_files(self, paths: Sequence[Path]) -> None:
        image_paths = tuple(path.resolve() for path in paths if path.suffix.lower() in IMAGE_EXTENSIONS)
        if not image_paths:
            return
        self._queue.put(("files", image_paths))
        self._set_status(queued_tasks=self._queue.qsize())

    def status(self) -> dict[str, object]:
        with self._status_lock:
            data = dict(self._status)
        data["queued_tasks"] = self._queue.qsize()
        return data

    def duplicates(self) -> list[dict[str, object]]:
        records = self._record_snapshot()
        rows = [
            {
                "path": record.path,
                "duplicate_of": record.duplicate_of,
                "distance": record.distance,
                "match_type": record.match_type,
                "backend": record.backend,
                "scanned_at": record.scanned_at,
            }
            for record in records
            if record.duplicate_of is not None
        ]
        return sorted(
            rows,
            key=lambda row: (
                str(row["duplicate_of"]),
                str(row["match_type"]),
                int(row["distance"] or 0),
                str(row["path"]),
            ),
        )

    def _record_snapshot(self) -> list[ImageFingerprint]:
        with self._records_lock:
            return sorted(self._records.values(), key=lambda record: record.path)

    def _replace_records(self, records: Sequence[ImageFingerprint]) -> None:
        with self._records_lock:
            self._records = {record.path: record for record in records}

    def _upsert_records(self, records: Sequence[ImageFingerprint]) -> None:
        with self._records_lock:
            for record in records:
                self._records[record.path] = record

    def _set_status(self, **changes: object) -> None:
        with self._status_lock:
            self._status.update(changes)

    def _worker(self) -> None:
        while True:
            task_type, paths = self._queue.get()
            try:
                if task_type == "full":
                    self._scan_all()
                else:
                    combined = list(paths)
                    time.sleep(0.15)
                    while len(combined) < self.batch_size:
                        try:
                            next_type, next_paths = self._queue.get_nowait()
                        except queue.Empty:
                            break
                        if next_type == "full":
                            self._queue.task_done()
                            self._queue.put((next_type, next_paths))
                            break
                        combined.extend(next_paths)
                        self._queue.task_done()
                    self._scan_files(combined)
            except Exception as exc:  
                LOGGER.exception("Duplicate detector task failed")
                self._set_status(state="error", last_error=str(exc), finished_at=time.time())
            finally:
                self._queue.task_done()
                if self._queue.empty() and self.status().get("state") != "error":
                    self._set_status(state="idle", queued_tasks=0)

    def _image_paths(self) -> list[Path]:
        return sorted(
            path
            for path in self.upload_root.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )

    def _scan_all(self) -> None:
        started = time.perf_counter()
        self._set_status(state="scanning_startup", started_at=time.time(), last_error=None)
        paths = self._image_paths()

        # No persistent cache: every restart/rescan recomputes every fingerprint.
        records, stats = fingerprint_paths(
            paths, self.upload_root, self.backend, self.batch_size, include_sha256=True
        )
        match_started = time.perf_counter()
        classified = classify_records(
            records, self.backend, threshold=self.threshold, aspect_tolerance=self.aspect_tolerance
        )
        match_seconds = time.perf_counter() - match_started
        self._replace_records(classified)

        duplicates = sum(record.duplicate_of is not None for record in classified)
        elapsed = time.perf_counter() - started
        self._set_status(
            state="idle",
            backend=self.backend.name,
            storage="memory",
            files_found=len(paths),
            files_hashed=stats.files_processed,
            files_failed=stats.files_failed,
            duplicates=duplicates,
            exact_duplicates=sum(record.match_type == "exact" for record in classified),
            similar_duplicates=sum(record.match_type == "similar" for record in classified),
            decode_sha_seconds=round(stats.decode_sha_seconds, 4),
            phash_seconds=round(stats.phash_seconds, 4),
            matching_seconds=round(match_seconds, 4),
            elapsed_seconds=round(elapsed, 4),
            finished_at=time.time(),
        )
        LOGGER.info("Full duplicate scan finished: %s files, %s duplicates, %.3fs", len(paths), duplicates, elapsed)

    def _scan_files(self, paths: Sequence[Path]) -> None:
        started = time.perf_counter()
        unique_paths = sorted({path for path in paths if path.exists() and path.is_file()})
        self._set_status(
            state="scanning_uploads",
            started_at=time.time(),
            current_batch=len(unique_paths),
            last_error=None,
        )
        relative_paths = {path.relative_to(self.upload_root).as_posix() for path in unique_paths}
        existing = [record for record in self._record_snapshot() if record.path not in relative_paths]
        new_records, stats = fingerprint_paths(
            unique_paths, self.upload_root, self.backend, self.batch_size, include_sha256=True
        )
        match_started = time.perf_counter()
        classified = classify_new_records(
            existing,
            new_records,
            self.backend,
            threshold=self.threshold,
            aspect_tolerance=self.aspect_tolerance,
        )
        match_seconds = time.perf_counter() - match_started
        self._upsert_records(classified)

        all_records = self._record_snapshot()
        elapsed = time.perf_counter() - started
        self._set_status(
            state="idle",
            backend=self.backend.name,
            storage="memory",
            current_batch=0,
            files_found=len(all_records),
            duplicates=sum(record.duplicate_of is not None for record in all_records),
            exact_duplicates=sum(record.match_type == "exact" for record in all_records),
            similar_duplicates=sum(record.match_type == "similar" for record in all_records),
            last_batch_processed=stats.files_processed,
            last_batch_duplicates=sum(record.duplicate_of is not None for record in classified),
            files_failed=stats.files_failed,
            decode_sha_seconds=round(stats.decode_sha_seconds, 4),
            phash_seconds=round(stats.phash_seconds, 4),
            matching_seconds=round(match_seconds, 4),
            elapsed_seconds=round(elapsed, 4),
            finished_at=time.time(),
        )
        LOGGER.info("Upload duplicate scan finished: %s images in %.3fs", len(classified), elapsed)
