"""
ChronoRoot: High-throughput phenotyping by deep learning reveals novel temporal parameters of plant root system architecture
Copyright (C) 2020 Nicolás Gaggion

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

from analysis.plantAnalysis import plantAnalysis
import argparse
from analysis.utils.fileUtilities import natural_key
from analysis.utils.metadata_schema import (
    hydrate_run_config,
    is_plant_metadata_path,
    load_json,
    video_has_qr,
)
import os
import traceback
from multiprocessing import Pool

_SKIP_DIRS = {"Graphs", "RSML", "Images"}


def plant_metadata_paths(path):
    """Latest Results_*/metadata.json under a project, Analysis, or experiment directory.

    A project directory is the parent of Analysis/. An experiment directory is
    one genotype folder inside Analysis/. A single metadata.json file is
    accepted as well. Older Results_* folders for the same plant are skipped.
    """
    path = os.path.abspath(path)
    if os.path.isfile(path):
        if not is_plant_metadata_path(path):
            raise SystemExit(f"Not a plant Results metadata file: {path}")
        return [path]
    if not os.path.isdir(path):
        raise SystemExit(f"Path not found: {path}")

    analysis = os.path.join(path, "Analysis")
    root = analysis if os.path.isdir(analysis) else path

    found = []
    for dirpath, dirnames, _filenames in os.walk(root):
        if os.path.basename(dirpath).startswith("Results_"):
            meta = os.path.join(dirpath, "metadata.json")
            if os.path.isfile(meta) and is_plant_metadata_path(meta):
                found.append(meta)
            dirnames.clear()
            continue
        dirnames[:] = [name for name in dirnames if name not in _SKIP_DIRS]

    by_slot = {}
    for meta in found:
        slot = os.path.dirname(os.path.dirname(meta))
        by_slot.setdefault(slot, []).append(meta)
    chosen = []
    for group in by_slot.values():
        group.sort(key=natural_key)
        chosen.append(group[-1])
    chosen.sort(key=natural_key)
    return chosen


def prepare_rerun(metadata_path, processing_limit=None):
    """Same run setup as 1_analysis, using the saved ROI, seed, and limits."""
    conf = hydrate_run_config(load_json(metadata_path), metadata_path)
    experiment = conf.get("Experiment") or conf.get("identifier")
    conf["Experiment"] = experiment
    conf["fileKey"] = experiment
    conf["sequenceLabel"] = (
        str(experiment) + "_" + str(conf.get("Images", "")) + "_" + str(conf.get("plant", ""))
    )
    conf["Plant"] = conf.get("Plant") or "Arabidopsis thaliana"
    if processing_limit is not None:
        conf["processingLimit"] = processing_limit
    if not video_has_qr(conf):
        known = conf.get("knownDistance")
        pixels = conf.get("pixelDistance")
        if known not in (None, "") and pixels not in (None, "", 0, "0"):
            conf["pixel_size"] = float(known) / float(pixels)
    return conf


def analyze_experiment(metadata_path, processing_limit=None):
    try:
        conf = prepare_rerun(metadata_path, processing_limit)
        print(
            f"Rerunning {conf.get('Experiment')} rpi {conf.get('rpi')} "
            f"cam {conf.get('cam')} plant {conf.get('plant')} "
            f"(processing limit {conf.get('processingLimit')} × 24 hours)",
            flush=True,
        )
        plantAnalysis(conf, True)
    except Exception:
        print(f"Failed {metadata_path}", flush=True)
        traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Rerun plant analysis for every saved plant under a project or experiment. "
            "Each finished plant replaces its previous Results_* folder in that same project."
        )
    )
    parser.add_argument(
        "path",
        help=(
            "Project directory (parent of Analysis/), an Analysis/ directory, "
            "one experiment directory, or a single Results_*/metadata.json"
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="How many plants to analyze at once (default: 4)",
    )
    parser.add_argument(
        "--processing-limit",
        type=float,
        default=None,
        help=(
            "Days of images to reprocess for every plant, same unit as the GUI "
            "processing limit. 0 keeps each full sequence. "
            "Default: the limit saved in each plant's metadata."
        ),
    )

    args = parser.parse_args()
    if args.workers < 1:
        raise SystemExit("--workers must be at least 1")
    processing_limit = None
    if args.processing_limit is not None:
        if args.processing_limit < 0:
            raise SystemExit("--processing-limit cannot be negative")
        processing_limit = int(round(args.processing_limit))

    experiments = plant_metadata_paths(args.path)
    if not experiments:
        raise SystemExit(f"No Results_*/metadata.json files under {args.path}")

    print(f"Found {len(experiments)} plants under {os.path.abspath(args.path)}", flush=True)
    if processing_limit is None:
        print("Processing limit: each plant's saved value", flush=True)
    elif processing_limit == 0:
        print("Processing limit: full sequence for every plant", flush=True)
    else:
        print(f"Processing limit: {processing_limit} × 24 hours for every plant", flush=True)
    jobs = [(metadata_path, processing_limit) for metadata_path in experiments]
    if args.workers == 1:
        for job in jobs:
            analyze_experiment(*job)
    else:
        with Pool(args.workers) as pool:
            pool.starmap(analyze_experiment, jobs)
