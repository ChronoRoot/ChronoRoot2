"""Turn screening tracking tables into chronoRootApp plant files via dataWork."""

import os
import json
import re
from collections import defaultdict, Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from robot_ids import cam_folder_name, resolve_rpi_cam
from chrono_root_backend import (
    convertToPathSafe,
    dataWork,
    data_file,
    ensure_factor_columns,
    normalize_factor_value,
    plot_individual_plant,
    apply_time_windows,
)

MAX_PLANT_WORKERS = 4

RSA_RAW_COLUMNS = [
    'FileName',
    'Frame',
    'MainRootLength',
    'LateralRootsLength',
    'NumberOfLateralRoots',
    'TotalLength',
    'HypocotylLength',
    'Area',
    'DenseRootArea',
]

NEAR_ZERO_MM = 0.1
SPIKE_SIGMA = 10


def plant_result_dir(project_dir, experiment, rpi, cam, plant_index, video=None):
    safe_exp = convertToPathSafe(str(experiment))
    robot = video if video not in (None, '') else rpi
    robot_name = convertToPathSafe(str(robot) if robot not in (None, '') else 'unspecified')
    cam_name = convertToPathSafe(cam_folder_name(cam))
    return os.path.join(
        project_dir,
        'Analysis',
        safe_exp,
        robot_name,
        cam_name,
        f'plant_{plant_index}',
        'Results_0',
    )


def _filename_for_datawork(row):
    name = str(row['FileName'])
    nums = re.findall(r'\d+', name)
    if len(nums) >= 5:
        return name
    stamp = datetime(2000, 1, 1) + timedelta(hours=float(row.get('ElapsedHours', 0) or 0))
    return stamp.strftime('%Y-%m-%d_%H-%M-00.png')


def _write_results_raw(plant_rows, result_dir):
    missing = [c for c in RSA_RAW_COLUMNS if c not in plant_rows.columns]
    if missing:
        raise ValueError(f'Results_raw missing columns: {missing}')
    raw = plant_rows[RSA_RAW_COLUMNS].copy()
    raw['FileName'] = plant_rows.apply(_filename_for_datawork, axis=1)
    raw_path = os.path.join(result_dir, 'Results_raw.csv')
    raw.to_csv(raw_path, index=False)
    return raw_path


def _write_metadata(plant_rows, result_dir, pixel_size, time_step):
    first = plant_rows.iloc[0]
    metadata = {
        'Experiment': str(first['Experiment']),
        'PlateCondition': normalize_factor_value(first.get('PlateCondition', '')),
        'ExtraVariable': normalize_factor_value(first.get('ExtraVariable', '')),
        'Plant_id': str(first['Plant_id']),
        'pixel_size': float(pixel_size),
        'timeStep': int(time_step) if float(time_step) == int(float(time_step)) else float(time_step),
        'Video': str(first.get('Video', '')),
        'rpi': str(first.get('rpi', '')),
        'cam': str(first.get('cam', '')),
    }
    path = os.path.join(result_dir, 'metadata.json')
    with open(path, 'w') as handle:
        json.dump(metadata, handle, indent=4)
    return metadata


def _attach_area_mm(hourly, pixel_size):
    scale = float(pixel_size) ** 2
    if 'Area' in hourly.columns:
        hourly['Area (mm2)'] = hourly['Area'] * scale
        hourly = hourly.drop(columns=['Area'])
    if 'DenseRootArea' in hourly.columns:
        hourly['DenseRootArea (mm2)'] = hourly['DenseRootArea'] * scale
        hourly = hourly.drop(columns=['DenseRootArea'])
    return hourly


def _validate_plant_series(plant_rows, max_len):
    """QC from the old plant_analysis growth filter. Returns (ok, reason)."""
    n_frames = len(plant_rows)
    if n_frames < 0.95 * max_len:
        return False, 'incomplete_series'

    pixel_size = float(plant_rows['pixel_size'].iloc[0]) if 'pixel_size' in plant_rows.columns else 1.0
    lengths = plant_rows['MainRootLength'].to_numpy(dtype=float)
    if lengths.size == 0 or np.nanmax(lengths) * pixel_size < NEAR_ZERO_MM:
        return False, 'near_zero_length'

    clamped = lengths.copy()
    for i in range(1, len(clamped)):
        if clamped[i] < clamped[i - 1]:
            clamped[i] = clamped[i - 1]
    if len(clamped) > 1:
        speed = np.diff(clamped)
        threshold = np.mean(speed) + SPIKE_SIGMA * np.std(speed)
        if np.any(speed > threshold):
            return False, 'growth_spike'
    return True, None


def _filter_valid_series(merged_seeds):
    """Drop incomplete / empty / spiked tracks per video. Survivors are unchanged."""
    if 'Video' not in merged_seeds.columns:
        work = merged_seeds.copy()
        work['Video'] = 'default'
    else:
        work = merged_seeds

    dropped = []
    keep_keys = set()
    for video, video_rows in work.groupby('Video', sort=True):
        lengths = video_rows.groupby('Plant_id').size()
        max_len = int(lengths.max()) if not lengths.empty else 0
        for plant_id, plant_rows in video_rows.groupby('Plant_id', sort=True):
            plant_rows = plant_rows.sort_values('Frame')
            ok, reason = _validate_plant_series(plant_rows, max_len)
            if ok:
                keep_keys.add((str(video), str(plant_id)))
            else:
                dropped.append({
                    'Video': video,
                    'Plant_id': plant_id,
                    'Experiment': plant_rows['Experiment'].iloc[0] if 'Experiment' in plant_rows.columns else '',
                    'n_frames': len(plant_rows),
                    'max_len': max_len,
                    'reason': reason,
                })

    if dropped:
        reason_counts = Counter(d['reason'] for d in dropped)
        parts = ', '.join(f'{r}={c}' for r, c in sorted(reason_counts.items()))
        print(f'QC dropped {len(dropped)} plants', flush=True)
        print(f'  reason breakdown: {parts}', flush=True)

    if not keep_keys:
        return work.iloc[0:0], dropped

    key = work['Video'].astype(str) + '\t' + work['Plant_id'].astype(str)
    keep = {f'{video}\t{plant_id}' for video, plant_id in keep_keys}
    return work.loc[key.isin(keep)].copy(), dropped


def process_one_screening_plant(job):
    import os, sys
    # Re-setup sys.path for subprocess
    _app_dir = os.path.dirname(os.path.abspath(__file__))
    _root_dir = os.path.abspath(os.path.join(_app_dir, '..'))
    _chron_dir = os.path.join(_root_dir, 'chronoRootApp')
    for _p in (_app_dir, _chron_dir):
        if _p not in sys.path:
            sys.path.insert(0, _p)
    import numpy as np
    import pandas as pd
    from analysis.dataWork import dataWork
    from analysis.report import plot_individual_plant
    from analysis.utils.fileUtilities import convertToPathSafe

    result_dir = job['result_dir']
    raw_path = job['raw_path']
    metadata = job['metadata']
    pixel_size = job['pixel_size']
    conf = job['conf']
    plant_id = job['plant_id']
    rpi = job['rpi']
    cam = job['cam']
    experiment = job['experiment']

    try:
        dataWork(conf, raw_path, result_dir, N_exp=None)
    except Exception as exc:
        return plant_id, None, str(exc)

    hourly_path = os.path.join(result_dir, 'PostProcess_Hour.csv')
    if not os.path.isfile(hourly_path):
        return plant_id, None, 'missing PostProcess_Hour.csv'

    hourly = pd.read_csv(hourly_path)
    # attach area in mm^2
    scale = float(pixel_size) ** 2
    if 'Area' in hourly.columns:
        hourly['Area (mm2)'] = hourly['Area'] * scale
        hourly = hourly.drop(columns=['Area'])
    if 'DenseRootArea' in hourly.columns:
        hourly['DenseRootArea (mm2)'] = hourly['DenseRootArea'] * scale
        hourly = hourly.drop(columns=['DenseRootArea'])

    hourly['Experiment'] = metadata['Experiment']
    hourly['Plant_id'] = metadata['Plant_id']
    hourly['PlateCondition'] = metadata['PlateCondition']
    hourly['ExtraVariable'] = metadata['ExtraVariable']
    hourly['Video'] = metadata['Video']
    hourly['rpi'] = rpi
    hourly['cam'] = cam
    hourly.to_csv(hourly_path, index=False)

    plot_name = f"{convertToPathSafe(experiment)}_{metadata['Plant_id']}.png"
    try:
        plot_individual_plant(result_dir, hourly, plot_name, conf)
    except Exception as exc:
        pass  # plot failure is non-fatal

    return plant_id, hourly, None


def postprocess_tracking(merged_seeds, conf):
    """Write Analysis/ plant slots, run dataWork, return Temporal_Data dataframe."""
    project_dir = conf['MainFolder']
    time_step = conf['timeStep']
    plant_frames = []

    if merged_seeds.empty:
        raise ValueError('No tracking rows to postprocess')

    valid_seeds, dropped = _filter_valid_series(merged_seeds)
    dropped_path = data_file(conf, 'dropped_plants.csv')
    if dropped:
        pd.DataFrame(dropped).to_csv(dropped_path, index=False)
        print(f'Wrote {dropped_path} ({len(dropped)} dropped plants)')
    elif os.path.isfile(dropped_path):
        os.remove(dropped_path)

    if valid_seeds.empty:
        raise ValueError('Postprocess produced no hourly tables: all plants failed series QC')

    # Count plants before and after QC per combination
    def _count_plants(df, key_cols):
        if df.empty:
            return {}
        for col in key_cols:
            if col not in df.columns:
                df = df.copy()
                df[col] = 'Unspecified'
        return df.groupby(key_cols)['Plant_id'].nunique().to_dict()

    combo_cols = ['Experiment', 'PlateCondition', 'ExtraVariable']
    before_counts = _count_plants(merged_seeds, combo_cols)
    after_counts = _count_plants(valid_seeds, combo_cols)

    # Per-reason drops per combo
    drops_by_combo = defaultdict(Counter)
    for d in dropped:
        plant_rows_ms = merged_seeds[merged_seeds['Plant_id'] == d['Plant_id']]
        key = (
            str(d.get('Experiment', 'Unspecified')),
            str(plant_rows_ms['PlateCondition'].iloc[0]) if not plant_rows_ms.empty else 'Unspecified',
            str(plant_rows_ms['ExtraVariable'].iloc[0]) if not plant_rows_ms.empty else 'Unspecified',
        )
        drops_by_combo[key][d['reason']] += 1

    all_reasons = ['incomplete_series', 'near_zero_length', 'growth_spike']
    rows = []
    all_keys = set(before_counts.keys()) | set(after_counts.keys())
    for key in sorted(all_keys):
        exp, plate, extra = key
        n_before = before_counts.get(key, 0)
        n_kept = after_counts.get(key, 0)
        n_dropped = n_before - n_kept
        row = {'Experiment': exp, 'PlateCondition': plate, 'ExtraVariable': extra,
               'n_before': n_before, 'n_dropped': n_dropped, 'n_kept': n_kept}
        for r in all_reasons:
            row[r] = drops_by_combo.get(key, Counter()).get(r, 0)
        rows.append(row)
        print(f'{exp} | {plate} | {extra}: kept {n_kept} (dropped {n_dropped})', flush=True)

    if rows:
        counts_path = data_file(conf, 'qc_plant_counts.csv')
        pd.DataFrame(rows).to_csv(counts_path, index=False)
        print(f'Wrote {counts_path}', flush=True)
    print()

    # Build result dirs and jobs
    grouped = valid_seeds.groupby(['Video', 'Plant_id'], sort=True)
    plant_index_by_key = {}
    next_index_by_exp_video = {}
    jobs = []

    for (video, plant_id), plant_rows in grouped:
        plant_rows = plant_rows.sort_values('Frame').reset_index(drop=True)
        experiment = str(plant_rows['Experiment'].iloc[0])
        key = (experiment, video)
        if (video, plant_id) not in plant_index_by_key:
            next_index_by_exp_video[key] = next_index_by_exp_video.get(key, 0) + 1
            plant_index_by_key[(video, plant_id)] = next_index_by_exp_video[key]
        plant_index = plant_index_by_key[(video, plant_id)]

        pixel_size = plant_rows['pixel_size'].iloc[0] if 'pixel_size' in plant_rows.columns else 1.0
        rpi, cam = resolve_rpi_cam(
            rpi=plant_rows['rpi'].iloc[0] if 'rpi' in plant_rows.columns else '',
            cam=plant_rows['cam'].iloc[0] if 'cam' in plant_rows.columns else '',
            analysis_id=video,
        )
        plant_rows = plant_rows.copy()
        plant_rows['rpi'] = rpi
        plant_rows['cam'] = cam
        result_dir = plant_result_dir(
            project_dir, experiment, rpi, cam, plant_index, video=video,
        )
        os.makedirs(result_dir, exist_ok=True)

        raw_path = _write_results_raw(plant_rows, result_dir)
        metadata = _write_metadata(plant_rows, result_dir, pixel_size, time_step)

        jobs.append({
            'result_dir': result_dir,
            'raw_path': raw_path,
            'metadata': metadata,
            'pixel_size': pixel_size,
            'conf': conf,
            'experiment': experiment,
            'plant_id': plant_id,
            'rpi': rpi,
            'cam': cam,
        })

    total = len(jobs)
    done = 0
    last_decile = 0
    print('Postprocessing started.')
    with ProcessPoolExecutor(max_workers=min(MAX_PLANT_WORKERS, max(1, len(jobs)))) as executor:
        future_to_pid = {executor.submit(process_one_screening_plant, job): job['plant_id'] for job in jobs}
        for future in as_completed(future_to_pid):
            done += 1
            plant_id_res, hourly, err = future.result()
            if err:
                print(f'Postprocess skipped {plant_id_res}: {err}')
            else:
                plant_frames.append(hourly)
            decile = int(10 * done / total) if total else 10
            if done == total or decile > last_decile:
                pct = 100 if done == total else decile * 10
                print(f'Postprocess {done}/{total} ({pct}%)', flush=True)
                last_decile = decile

    if not plant_frames:
        raise ValueError('Postprocess produced no hourly tables')

    all_data = pd.concat(plant_frames, ignore_index=True)
    all_data = apply_time_windows(all_data, conf)
    all_data = ensure_factor_columns(all_data)
    temporal_path = data_file(conf, 'Temporal_Data.csv')
    all_data.to_csv(temporal_path, index=False)
    print(f'Wrote {temporal_path}')
    return all_data
