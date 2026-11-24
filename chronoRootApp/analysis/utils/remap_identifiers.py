"""
Remap plant identifiers (Experiment folders) under a project's Analysis/ tree.

Public API
----------
- collect_experiment_counts(analysis_root)
- apply_experiment_remap(analysis_root, mapping)   <-- main procedure
- RemapError
"""

import os
import shutil

from .fileUtilities import convertFromPathSafe, convertToPathSafe, list_result_dirs
from .metadata_schema import dump_json, load_json


class RemapError(Exception):
    """Raised when an identifier remap cannot be applied safely."""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def collect_experiment_counts(analysis_root):
    """Return {readable_identifier: number_of_plant_slots} under Analysis/."""
    counts = {}
    if not analysis_root or not os.path.isdir(analysis_root):
        return counts

    for folder_name in os.listdir(analysis_root):
        exp_dir = os.path.join(analysis_root, folder_name)
        if not os.path.isdir(exp_dir) or folder_name.startswith("__remap_tmp_"):
            continue
        readable = convertFromPathSafe(folder_name)
        counts[readable] = counts.get(readable, 0) + len(list_plant_slots(exp_dir))
    return counts


def apply_experiment_remap(analysis_root, mapping):
    """
    Main procedure: rename / merge Experiment folders and update metadata.

    mapping: {old_readable_name: new_readable_name} for changed identifiers only.

    Why two phases?
      A simple rename fails for swaps (A→B and B→A) because the destination
      already exists. We first move every source into a unique temp folder,
      then move/merge each temp into its final name.
    """
    if not analysis_root or not os.path.isdir(analysis_root):
        raise RemapError("Analysis folder does not exist.")

    # --- Step 1: clean mapping (strip whitespace, drop unchanged rows) ---
    cleaned = {}
    for old, new in (mapping or {}).items():
        old_s = str(old)
        new_s = str(new).strip()
        if new_s and new_s != old_s:
            cleaned[old_s] = new_s
    if not cleaned:
        return

    # Readable name -> current folder name on disk
    folders = experiment_folders(analysis_root)
    for old in cleaned:
        if old not in folders:
            raise RemapError(f"Identifier '{old}' was not found under Analysis.")

    # --- Step 2: refuse merges that would overwrite the same plant slot ---
    check_merge_conflicts(analysis_root, folders, cleaned)

    # Remember which old name each temp came from (for metadata path rewrites)
    temps = []  # list of (temp_path, old_readable, new_readable)

    try:
        # --- Step 3 (Phase A): park each source folder under a unique temp ---
        for i, (old, new) in enumerate(cleaned.items()):
            src = os.path.join(analysis_root, folders[old])
            temp_path = unique_temp_path(analysis_root, i)
            shutil.move(src, temp_path)
            temps.append((temp_path, old, new))

        # --- Step 4 (Phase B): place each temp at its final destination ---
        for temp_path, old, new in temps:
            dest_path = os.path.join(analysis_root, convertToPathSafe(new))
            if os.path.exists(dest_path):
                # Merge: move rpi/cam/plant trees into the existing folder
                merge_plant_trees(temp_path, dest_path)
            else:
                shutil.move(temp_path, dest_path)

            # --- Step 5: patch identifier fields + Analysis paths in metadata ---
            update_metadata_identifiers(dest_path, new_name=new, old_name=old)

    except Exception as exc:
        leftover = [os.path.basename(p) for p, _, _ in temps if os.path.isdir(p)]
        hint = ""
        if leftover:
            hint = (
                " Temporary folders may remain under Analysis: "
                + ", ".join(leftover)
                + "."
            )
        if isinstance(exc, RemapError):
            raise RemapError(str(exc) + hint) from exc
        raise RemapError(f"{exc}{hint}") from exc


# ---------------------------------------------------------------------------
# Helpers used by the main procedure
# ---------------------------------------------------------------------------

def experiment_folders(analysis_root):
    """Map readable identifier -> folder name under Analysis/."""
    result = {}
    for name in os.listdir(analysis_root):
        path = os.path.join(analysis_root, name)
        if os.path.isdir(path) and not name.startswith("__remap_tmp_"):
            result[convertFromPathSafe(name)] = name
    return result


def list_plant_slots(exp_dir):
    """Return absolute paths of plant slots: Analysis/<exp>/<rpi>/<cam>/<plant>."""
    slots = []
    if not os.path.isdir(exp_dir):
        return slots
    for rpi in sorted(os.listdir(exp_dir)):
        rpi_path = os.path.join(exp_dir, rpi)
        if not os.path.isdir(rpi_path):
            continue
        for cam in sorted(os.listdir(rpi_path)):
            cam_path = os.path.join(rpi_path, cam)
            if not os.path.isdir(cam_path):
                continue
            for plant in sorted(os.listdir(cam_path)):
                plant_path = os.path.join(cam_path, plant)
                if os.path.isdir(plant_path):
                    slots.append(plant_path)
    return slots


def unique_temp_path(analysis_root, index):
    """Pick an unused __remap_tmp_* path under Analysis/."""
    n = index
    while True:
        path = os.path.join(analysis_root, f"__remap_tmp_{n}")
        if not os.path.exists(path):
            return path
        n += 1000


def check_merge_conflicts(analysis_root, folders, mapping):
    """
    After remap, several old identifiers may land in the same folder.
    Fail early if two of them contain the same rpi/cam/plant path.
    """
    # Group sources by their final path-safe folder name
    by_dest = {}
    for readable, folder_name in folders.items():
        new_name = mapping.get(readable, readable)
        safe = convertToPathSafe(new_name)
        by_dest.setdefault(safe, []).append((readable, new_name, folder_name))

    for sources in by_dest.values():
        if len(sources) < 2:
            continue
        claimed = {}
        dest_label = sources[0][1]
        for readable, new_name, folder_name in sources:
            dest_label = new_name
            exp_dir = os.path.join(analysis_root, folder_name)
            for plant_path in list_plant_slots(exp_dir):
                rel = os.path.relpath(plant_path, exp_dir)
                if rel in claimed:
                    raise RemapError(
                        f"Cannot merge into '{dest_label}': plant slot "
                        f"'{rel}' exists in both '{claimed[rel]}' and '{readable}'."
                    )
                claimed[rel] = readable


def merge_plant_trees(src_exp, dest_exp):
    """Move every plant slot from src_exp into dest_exp. Never overwrite."""
    os.makedirs(dest_exp, exist_ok=True)
    for plant_path in list_plant_slots(src_exp):
        rel = os.path.relpath(plant_path, src_exp)
        dest_plant = os.path.join(dest_exp, rel)
        if os.path.exists(dest_plant):
            raise RemapError(f"Plant slot already exists at destination: {rel}")
        os.makedirs(os.path.dirname(dest_plant), exist_ok=True)
        shutil.move(plant_path, dest_plant)
    remove_empty_dirs(src_exp)


def remove_empty_dirs(path):
    """Delete empty directories bottom-up, including path itself if empty."""
    if not os.path.isdir(path):
        return
    for root, _dirs, _files in os.walk(path, topdown=False):
        try:
            if not os.listdir(root):
                os.rmdir(root)
        except OSError:
            pass


def update_metadata_identifiers(exp_dir, new_name, old_name=None):
    """
    Update identifier-related fields in every Results_*/metadata.json
    under this Experiment folder.
    """
    old_safe = convertToPathSafe(old_name) if old_name is not None else None
    new_safe = convertToPathSafe(new_name)

    for plant_path in list_plant_slots(exp_dir):
        for result_dir in list_result_dirs(plant_path):
            meta_path = os.path.join(result_dir, "metadata.json")
            if not os.path.isfile(meta_path):
                continue
            try:
                meta = load_json(meta_path)
            except (OSError, ValueError):
                continue

            patch_metadata_dict(meta, new_name, old_name, old_safe, new_safe)
            dump_json(meta_path, meta)


def patch_metadata_dict(meta, new_name, old_name, old_safe, new_safe):
    """Apply identifier and path updates to one metadata dict (in place)."""
    # Core biological label used by reports / postprocess
    meta["Experiment"] = new_name

    # UI / pipeline aliases that mirror Experiment when present
    for key in ("plantIdentifier", "fileKey", "identifierField"):
        if key in meta:
            meta[key] = new_name

    # sequenceLabel is typically "<Experiment>_<Images>_<plant>"
    if old_name is not None and "sequenceLabel" in meta:
        label = meta["sequenceLabel"]
        if isinstance(label, str) and label.startswith(old_name + "_"):
            meta["sequenceLabel"] = new_name + label[len(old_name):]

    # Rewrite Analysis/<old_safe>/ path segments so absolute paths stay valid
    if old_safe is not None and old_safe != new_safe:
        old_seg = os.sep + "Analysis" + os.sep + old_safe + os.sep
        new_seg = os.sep + "Analysis" + os.sep + new_safe + os.sep
        rewrite_analysis_paths(meta, old_seg, new_seg)


def rewrite_analysis_paths(obj, old_seg, new_seg):
    """Recursively replace Analysis/<old>/ with Analysis/<new>/ in string values."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            if isinstance(value, str):
                if old_seg in value:
                    obj[key] = value.replace(old_seg, new_seg)
            else:
                rewrite_analysis_paths(value, old_seg, new_seg)
    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            if isinstance(value, str):
                if old_seg in value:
                    obj[i] = value.replace(old_seg, new_seg)
            else:
                rewrite_analysis_paths(value, old_seg, new_seg)
