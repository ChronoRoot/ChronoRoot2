"""Per-video plate scale (pixel_size) registry under a project folder.

Keyed by absolute ImagePath: each video folder is a distinct imaging geometry.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from collections import defaultdict

from .fileUtilities import get_latest_result_dir
from .metadata_schema import (
    apply_load_aliases,
    dump_json,
    load_json,
    video_image_dir,
)

REGISTRY_FILENAME = "calibrations.json"
SCHEMA_VERSION = 1
DEFAULT_PIXEL_SIZE = 0.04

SOURCE_EXISTING = "existing"
SOURCE_REGISTRY = "registry"
SOURCE_MANUAL_PROJECT = "manual_project"
SOURCE_MANUAL = "manual"
SOURCE_QR = "qr"
SOURCE_DEFAULT = "default"
SOURCE_MISMATCH = "mismatch"

STATUS_OK_MANUAL = "ok_manual"
STATUS_OK_QR = "ok_qr"
STATUS_OK_EXISTING = "ok_existing"
STATUS_MISSING = "missing"
STATUS_DEFAULT = "default_fallback"
STATUS_MISMATCH = "mismatch"


def normalize_image_path(path):
    if not path:
        return ""
    return os.path.abspath(os.path.expanduser(str(path)))


def registry_path(project_dir):
    return os.path.join(project_dir, REGISTRY_FILENAME)


def empty_registry():
    return {"schema_version": SCHEMA_VERSION, "by_image_path": {}}


def load_registry(project_dir):
    path = registry_path(project_dir)
    if not os.path.isfile(path):
        return empty_registry()
    try:
        data = load_json(path)
    except (OSError, ValueError):
        return empty_registry()
    if not isinstance(data, dict):
        return empty_registry()
    by_path = data.get("by_image_path")
    if not isinstance(by_path, dict):
        by_path = {}
    return {
        "schema_version": int(data.get("schema_version") or SCHEMA_VERSION),
        "by_image_path": by_path,
    }


def save_registry(project_dir, registry):
    payload = {
        "schema_version": SCHEMA_VERSION,
        "by_image_path": dict((registry or {}).get("by_image_path") or {}),
    }
    dump_json(registry_path(project_dir), payload)


def get_calibration(project_dir, image_path):
    key = normalize_image_path(image_path)
    if not key:
        return None
    entry = load_registry(project_dir).get("by_image_path", {}).get(key)
    return dict(entry) if isinstance(entry, dict) else None


def compute_pixel_size(known_distance, pixel_distance):
    known = float(known_distance)
    pixels = float(pixel_distance)
    if known <= 0 or pixels <= 0:
        raise ValueError("knownDistance and pixelDistance must be positive")
    return known / pixels


def upsert_calibration(
    project_dir,
    image_path,
    *,
    pixel_size,
    source,
    known_distance=None,
    pixel_distance=None,
):
    key = normalize_image_path(image_path)
    if not key or not project_dir:
        return None
    registry = load_registry(project_dir)
    entry = {
        "pixel_size": float(pixel_size),
        "source": str(source),
        "updated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
    }
    if known_distance is not None and str(known_distance).strip() != "":
        entry["knownDistance"] = float(known_distance)
    if pixel_distance is not None and str(pixel_distance).strip() != "":
        entry["pixelDistance"] = int(float(pixel_distance))
    registry.setdefault("by_image_path", {})[key] = entry
    save_registry(project_dir, registry)
    return entry


def scale_matches_fields(entry, known_distance, pixel_distance, tol=1e-6):
    """True if registry entry matches the given manual known/pixel fields."""
    if not entry:
        return False
    try:
        known = float(known_distance)
        pixels = int(float(pixel_distance))
    except (TypeError, ValueError):
        return False
    if "knownDistance" in entry and "pixelDistance" in entry:
        try:
            return (
                abs(float(entry["knownDistance"]) - known) <= tol
                and int(float(entry["pixelDistance"])) == pixels
            )
        except (TypeError, ValueError):
            return False
    if "pixel_size" not in entry:
        return False
    try:
        expected = known / float(pixels)
        return abs(float(entry["pixel_size"]) - expected) <= max(tol, abs(expected) * 1e-6)
    except (TypeError, ValueError, ZeroDivisionError):
        return False


def manual_scale_warning(project_dir, image_path, known_distance, pixel_distance):
    """
    Return a warning string if manual plate scale is missing or mismatched
    for this video, else None.
    """
    key = normalize_image_path(image_path)
    if not key:
        return (
            "No video folder is set, so plate scale cannot be checked against "
            "a saved per-video estimate."
        )
    entry = get_calibration(project_dir, key)
    if entry is None:
        return (
            f"No saved plate scale for this video:\n{key}\n\n"
            "Camera-to-plate distance can change between videos. "
            "Use the calibration helper (or enter known mm / pixels) for this "
            "video before analyzing, or continue anyway."
        )
    if not scale_matches_fields(entry, known_distance, pixel_distance):
        return (
            f"The known/pixel values do not match the saved plate scale for:\n{key}\n\n"
            "Continue only if you intentionally changed the estimate for this video."
        )
    return None


def apply_scale_to_plants(
    plant_paths,
    pixel_size,
    *,
    source,
    known_distance=None,
    pixel_distance=None,
):
    """Write pixel_size (+ source) into each plant's latest Results metadata."""
    updated = 0
    for plant_path in plant_paths:
        result_dir = get_latest_result_dir(plant_path)
        if result_dir is None:
            continue
        meta_path = os.path.join(result_dir, "metadata.json")
        if not os.path.isfile(meta_path):
            continue
        try:
            meta = apply_load_aliases(load_json(meta_path))
        except (OSError, ValueError):
            continue
        meta["pixel_size"] = float(pixel_size)
        meta["calibration_source"] = str(source)
        if known_distance is not None and str(known_distance).strip() != "":
            meta["knownDistance"] = float(known_distance)
        if pixel_distance is not None and str(pixel_distance).strip() != "":
            meta["pixelDistance"] = int(float(pixel_distance))
        dump_json(meta_path, meta)
        updated += 1
    return updated


def _load_latest_metadata(plant_path):
    result_dir = get_latest_result_dir(plant_path)
    if result_dir is None:
        return None, None
    meta_path = os.path.join(result_dir, "metadata.json")
    if not os.path.isfile(meta_path):
        return result_dir, None
    try:
        return result_dir, apply_load_aliases(load_json(meta_path))
    except (OSError, ValueError):
        return result_dir, None


def plate_key_from_meta(meta, plant_path):
    """Same ImagePath + rpi + cam share plate scale across Experiments."""
    image_path = video_image_dir(meta) if meta else ""
    if meta and image_path:
        rpi = str(meta.get("rpi", ""))
        cam = str(meta.get("cam", ""))
        if not cam:
            cam = os.path.basename(os.path.dirname(plant_path))
        if not rpi:
            rpi = os.path.basename(os.path.dirname(os.path.dirname(plant_path)))
        return (normalize_image_path(image_path), rpi, cam)
    cam_dir = os.path.dirname(plant_path)
    return (os.path.abspath(cam_dir),)


def collect_plate_groups(analysis_root, *, require_raw_csv=False):
    """
    Walk Analysis/*/*/*/* and group plant slots by shared video/plate key.

    Returns dict: plate_key -> list of plant_path
    """
    import pathlib

    groups = defaultdict(list)
    if not analysis_root or not os.path.isdir(analysis_root):
        return groups

    plant_slots = sorted(
        str(p) for p in pathlib.Path(analysis_root).glob("*/*/*/*") if p.is_dir()
    )
    for plant_path in plant_slots:
        result_dir, meta = _load_latest_metadata(plant_path)
        if result_dir is None:
            continue
        if require_raw_csv:
            raw = os.path.join(result_dir, "Results_raw.csv")
            if not os.path.isfile(raw):
                continue
        key = plate_key_from_meta(meta, plant_path)
        groups[key].append(plant_path)
    return groups


def _status_from_sources(sources, pixel_sizes, registry_entry):
    unique_sizes = {round(float(v), 8) for v in pixel_sizes if v is not None}
    if len(unique_sizes) > 1:
        return STATUS_MISMATCH
    if SOURCE_DEFAULT in sources:
        return STATUS_DEFAULT
    if SOURCE_MISMATCH in sources:
        return STATUS_MISMATCH
    if not unique_sizes and not registry_entry:
        return STATUS_MISSING
    if not unique_sizes:
        # Registry-only (not yet written to plants)
        reg_src = (registry_entry or {}).get("source")
        if reg_src == SOURCE_QR:
            return STATUS_OK_QR
        if reg_src in (SOURCE_MANUAL, SOURCE_MANUAL_PROJECT):
            return STATUS_OK_MANUAL
        return STATUS_MISSING if not registry_entry else STATUS_OK_EXISTING
    if SOURCE_QR in sources or (registry_entry or {}).get("source") == SOURCE_QR:
        return STATUS_OK_QR
    if SOURCE_MANUAL in sources or SOURCE_MANUAL_PROJECT in sources:
        return STATUS_OK_MANUAL
    if (registry_entry or {}).get("source") == SOURCE_MANUAL:
        return STATUS_OK_MANUAL
    return STATUS_OK_EXISTING


def list_plate_statuses(project_dir):
    """
    Return a list of plate status dicts for the Plates & scale dialog.

    Each item: image_path, rpi, cam, plant_paths, plant_count, pixel_size,
    source, status, registry_entry
    """
    analysis_root = os.path.join(project_dir, "Analysis")
    groups = collect_plate_groups(analysis_root, require_raw_csv=False)
    registry = load_registry(project_dir)
    rows = []
    for plate_key in sorted(groups.keys(), key=lambda k: str(k)):
        plant_paths = groups[plate_key]
        if len(plate_key) >= 3:
            image_path, rpi, cam = plate_key[0], plate_key[1], plate_key[2]
        else:
            image_path, rpi, cam = plate_key[0], "", ""
        registry_entry = registry.get("by_image_path", {}).get(normalize_image_path(image_path))
        pixel_sizes = []
        sources = []
        for plant_path in plant_paths:
            _result_dir, meta = _load_latest_metadata(plant_path)
            if not meta:
                continue
            if "pixel_size" in meta:
                try:
                    pixel_sizes.append(float(meta["pixel_size"]))
                except (TypeError, ValueError):
                    pass
            src = meta.get("calibration_source")
            if src:
                sources.append(str(src))
        unique_sizes = sorted({round(float(v), 8) for v in pixel_sizes})
        display_size = unique_sizes[0] if len(unique_sizes) == 1 else (
            None if not unique_sizes else unique_sizes[0]
        )
        if registry_entry and display_size is None:
            try:
                display_size = float(registry_entry.get("pixel_size"))
            except (TypeError, ValueError):
                display_size = None
        if len(unique_sizes) > 1:
            source = SOURCE_MISMATCH
        elif sources:
            # Prefer a non-default label if mixed; otherwise the common one
            non_default = [s for s in sources if s != SOURCE_DEFAULT]
            source = non_default[0] if non_default else sources[0]
            if len(set(sources)) > 1 and SOURCE_DEFAULT in sources:
                source = SOURCE_MISMATCH if len(unique_sizes) > 1 else source
        elif registry_entry:
            source = str(registry_entry.get("source") or SOURCE_REGISTRY)
        else:
            source = ""
        status = _status_from_sources(sources, pixel_sizes, registry_entry)
        rows.append({
            "image_path": image_path,
            "rpi": rpi,
            "cam": cam,
            "plant_paths": list(plant_paths),
            "plant_count": len(plant_paths),
            "pixel_size": display_size,
            "source": source,
            "status": status,
            "registry_entry": dict(registry_entry) if registry_entry else None,
            "pixel_sizes": pixel_sizes,
        })
    return rows
