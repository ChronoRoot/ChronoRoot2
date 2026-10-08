"""Canonical metadata keys shared by ChronoRootApp and screening.

Qt widget objectNames stay in the GUI. JSON on disk uses these names.
Load helpers promote old aliases; writers persist canonical keys only.
"""

import json
import os

SCHEMA_VERSION = 1
SOURCE_SINGLE_PLANT = "single_plant"
SOURCE_SCREENING = "screening"
SCREENING_JOB_KIND = "screening_job"
PENDING_ANALYSIS_NAME = ".pending_analysis.json"
PLANT_METADATA_FILENAME = "metadata.json"

# old persisted key -> canonical name (applied only when the canonical key is missing)
LOAD_ALIASES = {
    "plantIdentifier": "Experiment",
    "identifierField": "Experiment",
    "projectField": "MainFolder",
    "projectField_2": "MainFolder",
    "videoField": "Images",
    "video_directory": "Images",
    "rpiField": "rpi",
    "cameraField": "cam",
    "plantField": "plant",
    "saveImagesButton": "saveImages",
    "videoHasQRbutton": "videoHasQR",
    "has_qr": "videoHasQR",
    "processingLimitField": "processingLimit",
    "processingLimitField_3": "processingLimit",
    "captureIntervalField": "timeStep",
    "captureIntervalField_3": "timeStep",
    "time_delta": "timeStep",
    "plateConditionName": "PlateCondition",
    "extraField": "ExtraVariable",
    "emergenceDistanceField": "emergenceDistance",
    "bounding box": "bounding_box",
    "known_distance": "knownDistance",
    "pixel_distance": "pixelDistance",
    "reportGenotypeAxisLabelField": "genotypeAxisLabel",
    "reportPlateConditionAxisLabelField": "plateConditionAxisLabel",
    "reportExtraVariableAxisLabelField": "extraVariableLabel",
}

# GUI objectName -> canonical key written to project_config.json
WIDGET_TO_CANONICAL = {
    "plantIdentifier": "Experiment",
    "videoField": "Images",
    "projectField": "MainFolder",
    "rpiField": "rpi",
    "cameraField": "cam",
    "plantField": "plant",
    "processingLimitField": "processingLimit",
    "captureIntervalField": "timeStep",
    "plateConditionName": "PlateCondition",
    "extraField": "ExtraVariable",
    "saveImagesButton": "saveImages",
    "videoHasQRbutton": "videoHasQR",
    "emergenceDistanceField": "emergenceDistance",
    "analysisEmergenceDistanceField": "emergenceDistance",
    "reportProcessingLimitField": "processingLimit",
    "reportCaptureIntervalField": "timeStep",
    "reportProjectField": "MainFolder",
    "reportGenotypeAxisLabelField": "genotypeAxisLabel",
    "reportPlateConditionAxisLabelField": "plateConditionAxisLabel",
    "reportExtraVariableAxisLabelField": "extraVariableLabel",
}

ROI_KEYS = ("bounding_box", "bounding box", "seed")

SINGLE_PLANT_SNAPSHOT_KEYS = (
    "processingLimit",
    "saveImages",
    "videoHasQR",
    "knownDistance",
    "pixelDistance",
)


def dump_json(path, data):
    """Write JSON with indent=4 so files are human-readable."""
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(data, handle, indent=4)


def load_json(path):
    with open(path, "r") as handle:
        return json.load(handle)


def apply_load_aliases(data):
    """Return a copy with old keys copied onto canonical names when missing."""
    if not isinstance(data, dict):
        return {}
    out = dict(data)
    for old, new in LOAD_ALIASES.items():
        if old not in out:
            continue
        current = out.get(new)
        if new not in out or current in (None, ""):
            out[new] = out[old]
    return out


def canonicalize_persisted(data):
    """Promote aliases, then drop old names, ROI leftovers, and Limit."""
    out = apply_load_aliases(data)
    for old, new in LOAD_ALIASES.items():
        if old != new:
            out.pop(old, None)
    for key in ROI_KEYS:
        out.pop(key, None)
    out.pop("Limit", None)
    out.pop("folders", None)
    out.pop("SegPath", None)
    return out


def canonical_get(data, key, default=None):
    if not data:
        return default
    if key in data and data[key] is not None:
        return data[key]
    for old, new in LOAD_ALIASES.items():
        if new == key and old in data and data[old] is not None:
            return data[old]
    return default


def widget_value(data, object_name, default=None):
    """Look up a GUI field: canonical key, then objectName, then load aliases."""
    canonical = WIDGET_TO_CANONICAL.get(object_name, object_name)
    value = canonical_get(data, canonical, None)
    if value is not None:
        return value
    if object_name in (data or {}):
        return data[object_name]
    return default


def strip_roi_keys(data):
    out = dict(data or {})
    for key in ROI_KEYS:
        out.pop(key, None)
    return out


def get_bounding_box(data):
    return canonical_get(data, "bounding_box")


def get_seed(data):
    return canonical_get(data, "seed")


def has_roi(data):
    bbox = get_bounding_box(data)
    seed = get_seed(data)
    return bbox is not None and seed is not None


def video_has_qr(data, default=True):
    value = canonical_get(data, "videoHasQR", default)
    if value is None:
        return default
    return bool(value)


def video_image_dir(data):
    """Directory containing source PNGs."""
    return canonical_get(data, "ImagePath") or canonical_get(data, "Images") or ""


def result_paths(result_dir):
    images = os.path.join(result_dir, "Images")
    return {
        "result": result_dir,
        "graphs": os.path.join(result_dir, "Graphs"),
        "images": images,
        "rsml": os.path.join(result_dir, "RSML"),
        "seg": os.path.join(images, "Seg.tif"),
        "seg_multi": os.path.join(images, "SegMulti.tif"),
    }


def is_results_dir(path):
    return os.path.basename(os.path.abspath(path or "")).startswith("Results_")


def is_plant_metadata_path(config_path):
    if not config_path:
        return False
    abspath = os.path.abspath(config_path)
    if os.path.basename(abspath) != PLANT_METADATA_FILENAME:
        return False
    return is_results_dir(os.path.dirname(abspath))


def is_pending_analysis_path(config_path):
    return os.path.basename(os.path.abspath(config_path or "")) == PENDING_ANALYSIS_NAME


def pending_analysis_path(project_dir):
    return os.path.join(project_dir, PENDING_ANALYSIS_NAME)


def project_root_from_result(result_dir):
    """Walk up from a Results_* folder to the parent of Analysis/. None if layout mismatches."""
    current = os.path.abspath(result_dir)
    while True:
        parent, name = os.path.split(current)
        if name == "Analysis":
            return parent
        if not parent or parent == current:
            return None
        current = parent


def slot_identity_from_result(result_dir):
    """Experiment / rpi / cam / plant from Analysis/{exp}/{rpi}/cam_*/plant_*/Results_*."""
    from .fileUtilities import convertFromPathSafe

    path = os.path.abspath(result_dir)
    if not is_results_dir(path):
        return None
    plant_dir = os.path.dirname(path)
    cam_dir = os.path.dirname(plant_dir)
    rpi_dir = os.path.dirname(cam_dir)
    exp_dir = os.path.dirname(rpi_dir)
    analysis_dir = os.path.dirname(exp_dir)
    if os.path.basename(analysis_dir) != "Analysis":
        return None
    plant_name = os.path.basename(plant_dir)
    cam_name = os.path.basename(cam_dir)
    plant = plant_name[len("plant_"):] if plant_name.startswith("plant_") else plant_name
    cam = cam_name[len("cam_"):] if cam_name.startswith("cam_") else cam_name
    return {
        "Experiment": convertFromPathSafe(os.path.basename(exp_dir)),
        "rpi": os.path.basename(rpi_dir),
        "cam": cam,
        "plant": plant,
    }


def hydrate_run_config(meta, config_path):
    """Prepare a pipeline conf dict. Plant metadata never supplies MainFolder/folders."""
    conf = apply_load_aliases(meta)
    if not is_plant_metadata_path(config_path):
        return conf

    result_dir = os.path.dirname(os.path.abspath(config_path))
    root = project_root_from_result(result_dir)
    if root:
        conf["MainFolder"] = root
    identity = slot_identity_from_result(result_dir)
    if identity:
        conf.update(identity)
    conf.pop("folders", None)
    if not conf.get("Images"):
        conf["Images"] = conf.get("ImagePath") or ""
    return conf


def load_run_config(config_path):
    """Load a pipeline config and hydrate plant files from disk."""
    return hydrate_run_config(load_json(config_path), config_path)


def cleanup_pending_config(config_path):
    if is_pending_analysis_path(config_path):
        try:
            os.remove(config_path)
        except OSError:
            pass


def _omit_empty(value):
    return value in (None, "")


def build_plant_record(conf, source=SOURCE_SINGLE_PLANT):
    """Slim Results_*/metadata.json payload. Does not copy GUI or report flags."""
    conf = apply_load_aliases(conf)
    images = conf.get("Images") or ""
    image_path = conf.get("ImagePath")
    record = {
        "schema_version": SCHEMA_VERSION,
        "source": source,
        "Experiment": "" if conf.get("Experiment") is None else str(conf.get("Experiment")),
        "PlateCondition": conf.get("PlateCondition", "") or "",
        "ExtraVariable": conf.get("ExtraVariable", "") or "",
        "rpi": "" if conf.get("rpi") is None else str(conf.get("rpi")),
        "cam": "" if conf.get("cam") is None else str(conf.get("cam")),
        "plant": "" if conf.get("plant") is None else str(conf.get("plant")),
        "Images": images,
        "timeStep": conf.get("timeStep"),
    }
    bbox = get_bounding_box(conf)
    seed = get_seed(conf)
    if bbox is not None:
        record["bounding_box"] = bbox
    if seed is not None:
        record["seed"] = seed
    if image_path and image_path != images:
        record["ImagePath"] = image_path
    if conf.get("pixel_size") is not None:
        record["pixel_size"] = conf["pixel_size"]
    if source == SOURCE_SINGLE_PLANT:
        for key in SINGLE_PLANT_SNAPSHOT_KEYS:
            if key in conf:
                record[key] = conf[key]
    if source == SOURCE_SCREENING:
        video = conf.get("Video")
        if not _omit_empty(video):
            record["Video"] = str(video)
        plant_id = conf.get("Plant_id")
        if not _omit_empty(plant_id):
            record["Plant_id"] = str(plant_id)
    return {key: value for key, value in record.items() if value is not None}
