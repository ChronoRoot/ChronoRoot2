"""ROI and seed selection orchestration for single-plant analysis."""

import os
import sys
import pathlib
from collections import defaultdict

from PyQt5.QtWidgets import QApplication, QDialog

from analysis.utils.fileUtilities import (
    plant_slot_path,
    get_latest_result_dir,
    load_result_metadata,
    convertFromPathSafe,
)
from analysis.utils.metadata_schema import get_bounding_box, video_image_dir


def _ensure_qapplication():
    app = QApplication.instance()
    if app is not None:
        return app
    try:
        import PyQt5
        plugin_path = os.path.join(os.path.dirname(PyQt5.__file__), 'Qt', 'plugins', 'platforms')
        if os.path.isdir(plugin_path):
            os.environ.setdefault('QT_QPA_PLATFORM_PLUGIN_PATH', plugin_path)
        return QApplication(sys.argv)
    except Exception:
        return None


def _run_roi_seed_dialogs(conf, images, seg_files, own_previous_roi=None):
    import plant_viewer

    previous_rois, _warnings_text = collect_previous_rois_and_warnings(conf)

    experiment = conf.get('Experiment', '')
    plant_num = conf.get('plant', '?')
    plant_label = f"plant_{plant_num} - {experiment}"

    roi_dialog = plant_viewer.PlantROISelectorWindow(
        images,
        seg_files,
        previous_rois,
        time_delta=conf.get('timeStep', 15),
        plant_label=plant_label,
        own_previous_roi=own_previous_roi,
    )
    if roi_dialog.exec_() != QDialog.Accepted:
        return None, None

    roi = roi_dialog.get_roi()
    if roi is None:
        return None, None

    x1, y1, x2, y2 = roi
    bbox = [y1, y2, x1, x2]

    seed_dialog = plant_viewer.SeedSelectorWindow(images, seg_files, bbox, conf)
    if seed_dialog.exec_() != QDialog.Accepted:
        return None, None

    seed = seed_dialog.get_seed()
    if seed is None:
        return None, None

    return bbox, seed


def _format_warnings(analysis_root, cam_plants, video_slots):
    """Build a single detailed warning message, or empty string if none."""
    sections = []

    for cam_path, plants in sorted(cam_plants.items()):
        videos = {video for _, video in plants}
        if len(videos) <= 1:
            continue
        rel_cam = os.path.relpath(cam_path, analysis_root)
        experiment = convertFromPathSafe(rel_cam.split(os.sep)[0])
        rpi_cam = '/'.join(rel_cam.split(os.sep)[1:])
        lines = [f"  {experiment} / {rpi_cam}"]
        for video in sorted(videos):
            lines.append(f"    {video}")
        sections.append('\n'.join(lines))

    if sections:
        sections.insert(0, 'Same camera, different video sources:')

    shared_video_sections = []
    for video_abs, slots in sorted(video_slots.items()):
        hardware_keys = {(rpi, cam) for _, rpi, cam, _ in slots}
        if len(hardware_keys) <= 1:
            continue
        lines = [f"  {video_abs}"]
        for rpi, cam in sorted(hardware_keys):
            lines.append(f"    {rpi} / {cam}")
        shared_video_sections.append('\n'.join(lines))

    if shared_video_sections:
        if sections:
            sections.append('')
        sections.append('Same video used on different hardware:')
        sections.extend(shared_video_sections)

    if not sections:
        return ''
    return 'Video path inconsistencies detected:\n\n' + '\n\n'.join(sections)


def _pending_slot_parts(conf):
    """Return (slot, video, cam_path, experiment, rpi, cam, plant_folder) or None."""
    try:
        current_slot = os.path.abspath(plant_slot_path(conf))
        video_src = video_image_dir(conf) or conf.get('Images')
        if not video_src:
            return None
        current_video = os.path.abspath(video_src)
        analysis_root = os.path.join(conf['MainFolder'], 'Analysis')
        rel = os.path.relpath(current_slot, analysis_root)
        parts = rel.split(os.sep)
        if len(parts) < 4:
            return None
        experiment_folder, rpi, cam, plant_folder = parts[0], parts[1], parts[2], parts[3]
        cam_path = os.path.join(analysis_root, experiment_folder, rpi, cam)
        experiment = convertFromPathSafe(experiment_folder)
        return current_slot, current_video, cam_path, experiment, rpi, cam, plant_folder
    except (KeyError, TypeError, ValueError, OSError):
        return None


def _inject_pending_plant(cam_plants, video_slots, conf):
    """Replace this slot's saved video with the plant about to be launched."""
    pending = _pending_slot_parts(conf)
    if pending is None:
        return
    _slot, current_video, cam_path, experiment, rpi, cam, plant_folder = pending

    if cam_path in cam_plants:
        cam_plants[cam_path] = [
            (plant, video) for plant, video in cam_plants[cam_path]
            if plant != plant_folder
        ]
    for video_abs in list(video_slots.keys()):
        video_slots[video_abs] = [
            entry for entry in video_slots[video_abs]
            if not (
                entry[0] == experiment
                and entry[1] == rpi
                and entry[2] == cam
                and entry[3] == plant_folder
            )
        ]
        if not video_slots[video_abs]:
            del video_slots[video_abs]

    cam_plants[cam_path].append((plant_folder, current_video))
    video_slots[current_video].append((experiment, rpi, cam, plant_folder))


def collect_previous_rois_and_warnings(conf, include_pending=False):
    """Scan Analysis tree for prior ROIs on the same video and path warnings."""
    analysis_root = os.path.join(conf['MainFolder'], 'Analysis')
    current_slot = os.path.abspath(plant_slot_path(conf))
    current_video = os.path.abspath(video_image_dir(conf) or conf['Images'])

    previous_rois = []
    cam_plants = defaultdict(list)
    video_slots = defaultdict(list)

    if os.path.isdir(analysis_root):
        for plant_slot in pathlib.Path(analysis_root).glob('*/*/*/*'):
            if not plant_slot.is_dir():
                continue

            slot_str = str(plant_slot)
            rel = os.path.relpath(slot_str, analysis_root)
            parts = rel.split(os.sep)
            if len(parts) < 4:
                continue

            experiment_folder, rpi, cam, plant_folder = parts[0], parts[1], parts[2], parts[3]
            cam_path = os.path.join(analysis_root, experiment_folder, rpi, cam)
            experiment = convertFromPathSafe(experiment_folder)
            result_dir = get_latest_result_dir(slot_str)
            meta = load_result_metadata(result_dir) if result_dir else {}

            video = video_image_dir(meta)
            if video:
                video_abs = os.path.abspath(video)
                cam_plants[cam_path].append((plant_folder, video_abs))
                video_slots[video_abs].append((experiment, rpi, cam, plant_folder))

            if os.path.abspath(slot_str) == current_slot:
                continue
            if not result_dir or get_bounding_box(meta) is None:
                continue

            meta_video = video_image_dir(meta)
            if not meta_video or os.path.abspath(meta_video) != current_video:
                continue

            y1, y2, x1, x2 = get_bounding_box(meta)
            plant_num = meta.get('plant', plant_folder.replace('plant_', ''))
            label = f"{experiment}\nplant_{plant_num}"
            previous_rois.append((label, x1, y1, x2, y2))

    if include_pending:
        _inject_pending_plant(cam_plants, video_slots, conf)

    warnings_text = _format_warnings(analysis_root, cam_plants, video_slots)
    return previous_rois, warnings_text


def video_path_warnings(conf):
    """Conflict text including the plant about to be launched, or empty."""
    _rois, warnings_text = collect_previous_rois_and_warnings(conf, include_pending=True)
    return warnings_text


def select_roi_and_seed(conf, images, seg_files, own_previous_roi=None):
    """
    Interactive ROI and seed selection. Requires an existing QApplication (run.py).
    Returns (bbox, seed) or (None, None) on cancel.
    """
    if QApplication.instance() is None:
        return try_select_roi_and_seed(conf, images, seg_files, own_previous_roi=own_previous_roi)
    return _run_roi_seed_dialogs(conf, images, seg_files, own_previous_roi=own_previous_roi)


def try_select_roi_and_seed(conf, images, seg_files, own_previous_roi=None):
    """
    Try interactive ROI/seed selection; creates QApplication if needed.
    Returns (bbox, seed) or (None, None) on failure or cancel.
    """
    try:
        if _ensure_qapplication() is None:
            return None, None
        return _run_roi_seed_dialogs(conf, images, seg_files, own_previous_roi=own_previous_roi)
    except Exception:
        return None, None
