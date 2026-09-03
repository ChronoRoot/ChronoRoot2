"""Turn screening tracking tables into chronoRootApp plant files via dataWork."""

import os
import json
import re
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from chrono_root_backend import (
    convertToPathSafe,
    dataWork,
    data_file,
    ensure_factor_columns,
    normalize_factor_value,
    plot_individual_plant,
)

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


def plant_result_dir(project_dir, experiment, analysis_id, plant_index):
    safe_exp = convertToPathSafe(str(experiment))
    safe_video = convertToPathSafe(str(analysis_id))
    return os.path.join(
        project_dir,
        'Analysis',
        safe_exp,
        safe_video,
        'cam_0',
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
    if n_frames < max_len:
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
                print(f'QC dropped {plant_id} ({video}): {reason} ({len(plant_rows)}/{max_len} frames)')

    if not keep_keys:
        return work.iloc[0:0], dropped

    key = work['Video'].astype(str) + '\t' + work['Plant_id'].astype(str)
    keep = {f'{video}\t{plant_id}' for video, plant_id in keep_keys}
    return work.loc[key.isin(keep)].copy(), dropped


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

    grouped = valid_seeds.groupby(['Video', 'Plant_id'], sort=True)
    plant_index_by_key = {}
    next_index_by_exp_video = {}

    for (video, plant_id), plant_rows in grouped:
        plant_rows = plant_rows.sort_values('Frame').reset_index(drop=True)
        experiment = str(plant_rows['Experiment'].iloc[0])
        key = (experiment, video)
        if (video, plant_id) not in plant_index_by_key:
            next_index_by_exp_video[key] = next_index_by_exp_video.get(key, 0) + 1
            plant_index_by_key[(video, plant_id)] = next_index_by_exp_video[key]
        plant_index = plant_index_by_key[(video, plant_id)]

        pixel_size = plant_rows['pixel_size'].iloc[0] if 'pixel_size' in plant_rows.columns else 1.0
        result_dir = plant_result_dir(project_dir, experiment, video, plant_index)
        os.makedirs(result_dir, exist_ok=True)

        raw_path = _write_results_raw(plant_rows, result_dir)
        metadata = _write_metadata(plant_rows, result_dir, pixel_size, time_step)

        try:
            dataWork(conf, raw_path, result_dir, N_exp=None)
        except Exception as exc:
            print(f'Postprocess skipped {plant_id}: {exc}')
            continue

        hourly_path = os.path.join(result_dir, 'PostProcess_Hour.csv')
        if not os.path.isfile(hourly_path):
            print(f'Postprocess skipped {plant_id}: missing PostProcess_Hour.csv')
            continue

        hourly = pd.read_csv(hourly_path)
        hourly = _attach_area_mm(hourly, pixel_size)
        hourly['Experiment'] = metadata['Experiment']
        hourly['Plant_id'] = metadata['Plant_id']
        hourly['PlateCondition'] = metadata['PlateCondition']
        hourly['ExtraVariable'] = metadata['ExtraVariable']
        hourly['Video'] = metadata['Video']
        hourly.to_csv(hourly_path, index=False)

        plot_name = f"{convertToPathSafe(experiment)}_{metadata['Plant_id']}.png"
        try:
            plot_individual_plant(result_dir, hourly, plot_name, conf)
        except Exception as exc:
            print(f'Individual plot skipped for {plant_id}: {exc}')

        plant_frames.append(hourly)

    if not plant_frames:
        raise ValueError('Postprocess produced no hourly tables')

    all_data = pd.concat(plant_frames, ignore_index=True)
    all_data = ensure_factor_columns(all_data)
    temporal_path = data_file(conf, 'Temporal_Data.csv')
    all_data.to_csv(temporal_path, index=False)
    print(f'Wrote {temporal_path}')
    return all_data
