"""Re-run screening tracking into a new project, keeping ROIs and calibration.

Video and segmentation directories stay the same. Only the mask subfolder
(Ensemble or Fold_0) can change. Launch from this directory:

  python reprocess_clone.py /path/to/source_project /path/to/new_project
  python reprocess_clone.py SRC DST --seg Fold_0 --workers 4
"""

import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import chrono_root_backend  # noqa: F401
from analysis.utils.metadata_schema import (
    apply_load_aliases,
    canonicalize_persisted,
    dump_json,
    load_json,
    video_image_dir,
)

SEG_CHOICES = ('Ensemble', 'Fold_0')
APP_DIR = os.path.dirname(os.path.abspath(__file__))


def _load_json(path):
    return load_json(path)


def _analysis_config(analysis_dir, dest_project, seg_subdir):
    """Build a process_config for dest from an existing analysis folder."""
    cfg_path = os.path.join(analysis_dir, 'process_config.json')
    rois_path = os.path.join(analysis_dir, 'group_rois.json')
    info_path = os.path.join(analysis_dir, 'group_info.json')
    meta_path = os.path.join(analysis_dir, 'metadata.json')

    if os.path.isfile(cfg_path):
        config = canonicalize_persisted(_load_json(cfg_path))
    else:
        if not os.path.isfile(meta_path):
            return None, 'missing process_config.json and metadata.json'
        meta = apply_load_aliases(_load_json(meta_path))
        info = apply_load_aliases(_load_json(info_path)) if os.path.isfile(info_path) else {}
        config = {
            'video_dir': meta.get('video_dir') or video_image_dir(meta),
            'segmentation_dir': meta.get('segmentation_directory') or meta.get('segmentation_dir'),
            'analysis_id': meta.get('analysis_id') or os.path.basename(analysis_dir),
            'group_names': info.get('group_names') or meta.get('group_names') or [],
            'seed_counts': info.get('seed_counts') or [0] * len(info.get('group_names') or []),
            'timeStep': info.get('timeStep', 15),
            'videoHasQR': True,
            'show_tracking': False,
            'PlateCondition': info.get('PlateCondition', '') or '',
            'ExtraVariable': info.get('ExtraVariable', '') or '',
            'rpi': info.get('rpi') or meta.get('rpi') or '',
            'cam': info.get('cam') or meta.get('cam') or '',
        }

    if os.path.isfile(rois_path):
        config['group_rois'] = _load_json(rois_path)
    if not config.get('group_rois'):
        return None, 'no group_rois (needed to skip interactive ROI selection)'

    if os.path.isfile(info_path):
        info = _load_json(info_path)
        if info.get('pixel_size') not in (None, ''):
            config['pixel_size'] = info['pixel_size']

    config['project_dir'] = dest_project
    config['analysis_id'] = config.get('analysis_id') or os.path.basename(analysis_dir)
    config['segmentation_subdir'] = seg_subdir
    return config, None


def clone_and_reprocess(source_project, dest_project, segmentation_subdir='Ensemble', workers=4):
    source_project = os.path.abspath(source_project)
    dest_project = os.path.abspath(dest_project)
    if segmentation_subdir not in SEG_CHOICES:
        raise ValueError(f'segmentation_subdir must be one of {SEG_CHOICES}')
    src_root = os.path.join(source_project, 'analysis')
    if not os.path.isdir(src_root):
        raise ValueError(f'No analysis folder in source project: {src_root}')

    os.makedirs(dest_project, exist_ok=True)
    jobs = []
    for name in sorted(os.listdir(src_root)):
        analysis_dir = os.path.join(src_root, name)
        if not os.path.isdir(analysis_dir):
            continue
        config, error = _analysis_config(analysis_dir, dest_project, segmentation_subdir)
        if error:
            print(f'Skip {name}: {error}')
            continue
        out_dir = os.path.join(dest_project, 'analysis', config['analysis_id'])
        os.makedirs(out_dir, exist_ok=True)
        config_path = os.path.join(out_dir, 'process_config.json')
        dump_json(config_path, config)
        jobs.append((config['analysis_id'], config_path))

    if not jobs:
        raise ValueError('No analyses with saved group ROIs found to clone')

    print(f'Cloning {len(jobs)} analyses to {dest_project}')
    print(f'Segmentation subdir: {segmentation_subdir}')
    workers = max(1, int(workers))
    failed = []
    with ThreadPoolExecutor(max_workers=min(workers, len(jobs))) as pool:
        futures = {
            pool.submit(
                subprocess.run,
                [sys.executable, 'process_video.py', '--config', config_path],
                cwd=APP_DIR,
            ): analysis_id
            for analysis_id, config_path in jobs
        }
        for future in as_completed(futures):
            analysis_id = futures[future]
            result = future.result()
            if result.returncode == 0:
                print(f'Done {analysis_id}')
            else:
                failed.append(analysis_id)
                print(f'Failed {analysis_id} (exit {result.returncode})')

    if failed:
        raise SystemExit(f'Finished with failures: {", ".join(failed)}')
    print('Clone reprocess finished.')


def main():
    parser = argparse.ArgumentParser(
        description='Clone a screening project and re-run tracking in parallel.',
    )
    parser.add_argument('source_project', help='Existing project directory')
    parser.add_argument('dest_project', help='New project directory for the clone')
    parser.add_argument(
        '--seg', choices=SEG_CHOICES, default='Ensemble',
        help='Mask subfolder under segmentation_dir (default: Ensemble)',
    )
    parser.add_argument('--workers', type=int, default=4, help='Parallel video jobs')
    args = parser.parse_args()
    clone_and_reprocess(args.source_project, args.dest_project, args.seg, args.workers)


if __name__ == '__main__':
    main()
