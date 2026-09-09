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
from analysis.utils.fileUtilities import convertFromPathSafe, get_latest_result_dir
import json
import os
import pathlib
import pandas as pd
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

MAX_PLANT_WORKERS = 4


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
            return result_dir, json.load(f)
    except (OSError, json.JSONDecodeError):
        return result_dir, None


def plate_key_from_meta(meta, plant_path):
    """
    Identity of the physical plate/video, independent of genotype folder.

    Same ImagePath + rpi + cam share calibration even across Experiments.
    """
    if meta and meta.get("ImagePath"):
        image_path = os.path.abspath(os.path.expanduser(str(meta["ImagePath"])))
        rpi = str(meta.get("rpi", ""))
        cam = str(meta.get("cam", ""))
        if not cam:
            # Fallback: cam_* folder name under plant slot
            cam = os.path.basename(os.path.dirname(plant_path))
        if not rpi:
            rpi = os.path.basename(os.path.dirname(os.path.dirname(plant_path)))
        return (image_path, rpi, cam)
    # No ImagePath: isolate by camera folder on disk
    cam_dir = os.path.dirname(plant_path)
    return (os.path.abspath(cam_dir),)


def resolve_plate_pixel_size(conf, plants):
    """
    Resolve pixel_size once for a shared video/plate group.

    Prefer an existing pixel_size from any plant in the group; otherwise
    manual calibration or QR from the first plant with usable metadata.
    """
    sample_meta = None
    for plant in plants:
        _result_dir, meta = _load_latest_metadata(plant)
        if meta is None:
            continue
        if sample_meta is None:
            sample_meta = meta
        if "pixel_size" in meta:
            return float(meta["pixel_size"])

    if sample_meta is None:
        return 0.04

    if not conf.get("videoHasQRbutton", True):
        return float(conf["knownDistance"]) / float(conf["pixelDistance"])

    image_path = sample_meta.get("ImagePath")
    if not image_path:
        return 0.04

    images = load_path(image_path, "*.png")
    pixel_size = 0.04
    for image in images[:20]:
        qr = qr_detect(image)
        if qr is not None:
            pixel_size = 10 / get_pixel_size(qr[0])
            break
    return pixel_size


def process_one_plant(plant_path, pixel_size, conf, plot_label):
    """
    Post-process a single plant slot. Top-level for ProcessPoolExecutor.

    Returns (plant_path, ok, error_message).
    """
    try:
        target_res = get_latest_result_dir(plant_path)
        if target_res is None:
            return plant_path, False, "no Results_* folder"

        pfile = os.path.join(target_res, "Results_raw.csv")
        if not os.path.isfile(pfile):
            return plant_path, False, "missing Results_raw.csv"

        meta_file = os.path.join(target_res, "metadata.json")
        with open(meta_file, "r") as f:
            plant_metadata = json.load(f)
        plant_metadata["pixel_size"] = pixel_size
        with open(meta_file, "w") as f:
            json.dump(plant_metadata, f)

        n_limit = conf["Limit"] if conf.get("Limit", 0) != 0 else None
        dataWork(conf, pfile, target_res, N_exp=n_limit)

        processed_csv = os.path.join(target_res, "PostProcess_Hour.csv")
        if os.path.exists(processed_csv):
            data = pd.read_csv(processed_csv)
            plot_individual_plant(target_res, data, plot_label, conf)

        getAngles(conf, target_res)
        return plant_path, True, ""
    except Exception as exc:
        return plant_path, False, str(exc)


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
        experiment_name = ""
        if meta:
            experiment_name = convertFromPathSafe(
                str(meta.get("Experiment", exp_folder))
            )
        else:
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

    conf = json.load(open(parser.parse_args().config, "r"))
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

    # --- Processing: group by shared video/plate, then parallel plants ---
    plate_groups = collect_plate_groups(analysis)

    for plate_key in sorted(plate_groups.keys(), key=lambda k: str(k)):
        jobs_meta = plate_groups[plate_key]
        plant_paths = [p for p, _label, _exp in jobs_meta]
        pixel_size = resolve_plate_pixel_size(conf, plant_paths)
        plate_label = _format_plate_log(plate_key)
        print(f"{plate_label} ({len(jobs_meta)} plants)")

        jobs = [
            (plant_path, pixel_size, conf, plot_label)
            for plant_path, plot_label, _exp in jobs_meta
        ]
        if not jobs:
            continue

        workers = min(MAX_PLANT_WORKERS, len(jobs))
        n_ok = 0
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(process_one_plant, *job) for job in jobs]
            for fut in as_completed(futures):
                plant_path, ok, err = fut.result()
                if ok:
                    n_ok += 1
                else:
                    print(f"  Error {os.path.basename(plant_path)}: {err}")
        print(f"  {n_ok}/{len(jobs)} done")

    print("Post processing finished.")
