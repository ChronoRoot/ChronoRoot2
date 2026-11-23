""" 
ChronoRoot: High-throughput phenotyping by deep learning reveals novel temporal parameters of plant root system architecture
Copyright (C) 2020 Nicolás Gaggion

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

from analysis.dataWork import dataWork
from analysis.qr import qr_detect, get_pixel_size, load_path
from analysis.report import plot_individual_plant
from analysis.lateral_angles import getAngles
from analysis.utils.fileUtilities import (
    convertFromPathSafe,
    get_latest_result_dir,
    list_video_pngs,
    processing_limit_frame_count,
)
from analysis.utils.metadata_schema import (
    apply_load_aliases,
    dump_json,
    load_json,
    video_has_qr,
    video_image_dir,
)
from analysis.utils.calibration_registry import (
    DEFAULT_PIXEL_SIZE,
    SOURCE_DEFAULT,
    SOURCE_EXISTING,
    SOURCE_MANUAL_PROJECT,
    SOURCE_MISMATCH,
    SOURCE_QR,
    SOURCE_REGISTRY,
    get_calibration,
    normalize_image_path,
    plate_key_from_meta,
    upsert_calibration,
)
import json
import os
import pathlib
import pandas as pd
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

MAX_PLANT_WORKERS = 4


def _file_stamp(path):
    try:
        st = os.stat(path)
        return {"mtime_ns": int(st.st_mtime_ns), "size": int(st.st_size)}
    except OSError:
        return None


def postprocess_fingerprint(raw_csv_path, pixel_size, calibration_source, conf):
    """Inputs that must match for a plant to be considered unchanged."""
    return {
        "raw": _file_stamp(raw_csv_path),
        "pixel_size": round(float(pixel_size), 8),
        "calibration_source": str(calibration_source or ""),
        "processingLimit": conf.get("processingLimit"),
        "timeStep": conf.get("timeStep"),
        "emergenceDistance": conf.get("emergenceDistance"),
    }


def _load_latest_metadata(plant_path):
    """Return (result_dir, metadata dict) or (None, None)."""
    result_dir = get_latest_result_dir(plant_path)
    if result_dir is None:
        return None, None
    meta_path = os.path.join(result_dir, "metadata.json")
    if not os.path.isfile(meta_path):
        return result_dir, None
    try:
        with open(meta_path, "r") as f:
            return result_dir, apply_load_aliases(json.load(f))
    except (OSError, json.JSONDecodeError):
        return result_dir, None


def resolve_plate_pixel_size(conf, plants):
    """
    Resolve pixel_size once for a shared video/plate group.

    Returns (pixel_size, source, detail).
    """
    sample_meta = None
    sizes = []
    for plant in plants:
        _result_dir, meta = _load_latest_metadata(plant)
        if meta is None:
            continue
        if sample_meta is None:
            sample_meta = meta
        if "pixel_size" in meta:
            try:
                sizes.append(float(meta["pixel_size"]))
            except (TypeError, ValueError):
                pass

    project_dir = (conf.get("MainFolder") or "").strip()
    image_path = normalize_image_path(video_image_dir(sample_meta)) if sample_meta else ""
    registry_entry = get_calibration(project_dir, image_path) if project_dir and image_path else None

    unique = {round(v, 8) for v in sizes}
    if len(unique) == 1:
        return sizes[0], SOURCE_EXISTING, "from plant metadata"
    if len(unique) > 1:
        detail = f"plants disagree on pixel_size: {sorted(unique)}"
        if registry_entry and "pixel_size" in registry_entry:
            return (
                float(registry_entry["pixel_size"]),
                SOURCE_MISMATCH,
                detail + "; using registry",
            )
        return sizes[0], SOURCE_MISMATCH, detail + "; using first plant value"

    if registry_entry and "pixel_size" in registry_entry:
        return float(registry_entry["pixel_size"]), SOURCE_REGISTRY, "from calibrations.json"

    if sample_meta is None:
        return DEFAULT_PIXEL_SIZE, SOURCE_DEFAULT, "no plant metadata"

    if not video_has_qr(conf):
        try:
            pixel_size = float(conf["knownDistance"]) / float(conf["pixelDistance"])
            return pixel_size, SOURCE_MANUAL_PROJECT, "project known/pixel fields"
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            return DEFAULT_PIXEL_SIZE, SOURCE_DEFAULT, "invalid project manual scale"

    if not image_path:
        return DEFAULT_PIXEL_SIZE, SOURCE_DEFAULT, "no ImagePath for QR scan"

    # Same discovery as preview: local PNGs, else segmentation_metadata input_path.
    images, _resolved = list_video_pngs({"Images": image_path})
    for image in images[:20]:
        qr = qr_detect(image)
        if qr is not None:
            pixel_size = 10 / get_pixel_size(qr[0])
            if project_dir:
                upsert_calibration(
                    project_dir, image_path, pixel_size=pixel_size, source=SOURCE_QR,
                )
            return pixel_size, SOURCE_QR, "QR detected"

    return (
        DEFAULT_PIXEL_SIZE,
        SOURCE_DEFAULT,
        f"QR not detected in first {min(20, len(images))} frames of {image_path}",
    )


def process_one_plant(plant_path, pixel_size, conf, plot_label, calibration_source=""):
    """
    Post-process a single plant slot. Top-level for ProcessPoolExecutor.

    Returns (plant_path, ok, error_message, skipped).
    """
    try:
        target_res = get_latest_result_dir(plant_path)
        if target_res is None:
            return plant_path, False, "no Results_* folder", False

        pfile = os.path.join(target_res, "Results_raw.csv")
        if not os.path.isfile(pfile):
            return plant_path, False, "missing Results_raw.csv", False

        processed_csv = os.path.join(target_res, "PostProcess_Hour.csv")
        meta_file = os.path.join(target_res, "metadata.json")
        plant_metadata = apply_load_aliases(load_json(meta_file))
        fingerprint = postprocess_fingerprint(
            pfile, pixel_size, calibration_source, conf,
        )
        if (
            os.path.isfile(processed_csv)
            and plant_metadata.get("postprocess_fingerprint") == fingerprint
        ):
            return plant_path, True, "skipped unchanged", True

        plant_metadata["pixel_size"] = pixel_size
        if calibration_source:
            plant_metadata["calibration_source"] = calibration_source
        dump_json(meta_file, plant_metadata)

        n_limit = processing_limit_frame_count(conf)
        dataWork(conf, pfile, target_res, N_exp=n_limit)

        if os.path.exists(processed_csv):
            data = pd.read_csv(processed_csv)
            plot_individual_plant(target_res, data, plot_label, conf)

        getAngles(conf, target_res)

        plant_metadata = apply_load_aliases(load_json(meta_file))
        plant_metadata["pixel_size"] = pixel_size
        if calibration_source:
            plant_metadata["calibration_source"] = calibration_source
        plant_metadata["postprocess_fingerprint"] = fingerprint
        dump_json(meta_file, plant_metadata)
        return plant_path, True, "", False
    except Exception as exc:
        return plant_path, False, str(exc), False


def collect_plate_groups(analysis_root):
    """
    Walk Analysis/*/*/*/* and group plant slots by shared video/plate key.

    Returns dict: plate_key -> list of (plant_path, plot_label, experiment_name)
    """
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
        pfile = os.path.join(result_dir, "Results_raw.csv")
        if not os.path.isfile(pfile):
            print(f"  Skipping {plant_path}: missing Results_raw.csv")
            continue

        rel = os.path.relpath(plant_path, analysis_root)
        parts = rel.split(os.sep)
        if len(parts) < 4:
            continue
        exp_folder, rpi, cam, plant = parts[0], parts[1], parts[2], parts[3]
        plot_label = f"{exp_folder}_{rpi}_{cam}_{plant}"
        experiment_name = convertFromPathSafe(exp_folder)

        key = plate_key_from_meta(meta, plant_path)
        groups[key].append((plant_path, plot_label, experiment_name))

    return groups


def _format_plate_log(plate_key):
    """Short label: robot/video from ImagePath, or cam folder basename."""
    if len(plate_key) >= 3:
        image_path = plate_key[0].rstrip(os.sep)
        video = os.path.basename(image_path) or image_path
        robot = os.path.basename(os.path.dirname(image_path))
        if robot:
            return f"{robot}/{video}"
        return video
    return os.path.basename(str(plate_key[0]).rstrip(os.sep)) or str(plate_key[0])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ChronoRoot Post-processing")
    parser.add_argument("--config", type=str, help="Path to the configuration file")

    conf = apply_load_aliases(load_json(parser.parse_args().config))
    analysis = os.path.join(conf["MainFolder"], "Analysis")

    experiment_dirs = load_path(analysis, "*")

    print("Post processing started.")

    # --- Cleanup Phase ---
    for exp_dir in experiment_dirs:
        rpis = load_path(exp_dir, "*")
        for rpi in rpis:
            cams = load_path(rpi, "*")
            for cam in cams:
                plants = load_path(cam, "*")
                for plant in plants:
                    results = load_path(plant, "*")
                    if len(results) == 0:
                        os.rmdir(plant)

    # Pixel size is resolved once per plate, including a single QR scan when
    # the plate does not already have one. Plants from every plate then share
    # one pool, so a short plate does not leave workers idle.
    plate_groups = collect_plate_groups(analysis)
    jobs = []
    plate_total = {}
    scale_issues = []
    for plate_key in sorted(plate_groups.keys(), key=lambda k: str(k)):
        jobs_meta = plate_groups[plate_key]
        if not jobs_meta:
            continue
        plant_paths = [plant_path for plant_path, _label, _exp in jobs_meta]
        pixel_size, source, detail = resolve_plate_pixel_size(conf, plant_paths)
        plate_label = _format_plate_log(plate_key)
        plate_total[plate_key] = (plate_label, len(jobs_meta))
        print(f"  Scale {plate_label}: pixel_size={pixel_size:.6g} source={source} ({detail})")
        if source in (SOURCE_DEFAULT, SOURCE_MISMATCH):
            scale_issues.append((plate_label, source, detail, pixel_size))
        for plant_path, plot_label, _exp in jobs_meta:
            jobs.append((plant_path, pixel_size, conf, plot_label, plate_key, source))

    if jobs:
        workers = min(MAX_PLANT_WORKERS, len(jobs))
        done_for = {key: 0 for key in plate_total}
        ok_for = {key: 0 for key in plate_total}
        skipped_for = {key: 0 for key in plate_total}
        skipped_total = 0
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    process_one_plant, plant_path, pixel_size, conf, plot_label, source,
                ): plate_key
                for plant_path, pixel_size, conf, plot_label, plate_key, source in jobs
            }
            for fut in as_completed(futures):
                plate_key = futures[fut]
                plant_path, ok, err, skipped = fut.result()
                done_for[plate_key] += 1
                if ok:
                    ok_for[plate_key] += 1
                    if skipped:
                        skipped_for[plate_key] += 1
                        skipped_total += 1
                else:
                    print(f"  Error {os.path.basename(plant_path)}: {err}")
                plate_label, total = plate_total[plate_key]
                if done_for[plate_key] == total:
                    skipped_n = skipped_for[plate_key]
                    if skipped_n:
                        print(
                            f"  {plate_label}: {ok_for[plate_key]}/{total} done "
                            f"({skipped_n} unchanged skipped)"
                        )
                    else:
                        print(f"  {plate_label}: {ok_for[plate_key]}/{total} done")
        if skipped_total:
            print(f"Skipped {skipped_total} unchanged plant(s).")

    if scale_issues:
        print("Plate scale warnings:")
        for plate_label, source, detail, pixel_size in scale_issues:
            print(f"  [{source}] {plate_label}: {detail} (using pixel_size={pixel_size:.6g})")
        print(
            "Next: Review plate calibrations → Test QR or Set scale on affected "
            "plates → Process all plants again → Generate report "
            "(then Analyze more plants as needed)."
        )

    print("Post processing finished.")
