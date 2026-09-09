"""Detect clock-overlap groups, align elapsed time, and crop report windows."""

import json
import os
import re
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd

SYNTHETIC_YEAR = 2000
DEFAULT_CLOCK_TICKS = ['00:00']
ID_COLUMNS = (
    'Experiment', 'Plant_id', 'PlateCondition', 'ExtraVariable',
    'Video', 'rpi', 'cam',
)
_FILENAME_DIGITS = re.compile(r'\d+')


def closest_timed_path(paths, target):
    """Return the path whose filename timestamp is closest to target."""
    target = parse_datetime(target)
    if target is None:
        return None
    best = None
    best_delta = None
    for path in paths or []:
        ts = datetime_from_filename(path)
        if ts is None:
            continue
        delta = abs((ts - target).total_seconds())
        if best is None or delta < best_delta:
            best = path
            best_delta = delta
    return best


def datetime_from_filename(name):
    """Parse YYYY-MM-DD_HH-MM style stamps from an image or csv filename."""
    nums = _FILENAME_DIGITS.findall(os.path.basename(str(name)))
    if len(nums) < 5:
        return None
    try:
        return pd.Timestamp(
            int(nums[0]), int(nums[1]), int(nums[2]), int(nums[3]), int(nums[4]),
        )
    except (ValueError, TypeError):
        return None


def parse_datetime(value):
    if value is None or value == '':
        return None
    if isinstance(value, pd.Timestamp):
        ts = value
    elif isinstance(value, datetime):
        ts = pd.Timestamp(value)
    else:
        ts = pd.to_datetime(value, errors='coerce')
    if ts is None or pd.isna(ts):
        return None
    if getattr(ts, 'tzinfo', None) is not None:
        ts = ts.tz_localize(None)
    return pd.Timestamp(ts).floor('min')


def format_datetime(value):
    ts = parse_datetime(value)
    if ts is None:
        return ''
    return ts.strftime('%Y-%m-%dT%H:%M:%S')


def is_synthetic_date(value):
    ts = parse_datetime(value)
    return ts is not None and int(ts.year) <= SYNTHETIC_YEAR


def report_folder_name(conf):
    name = str((conf or {}).get('reportFolderName') or 'Report').strip() or 'Report'
    name = os.path.basename(name.replace('\\', '/'))
    if name in ('.', '..', ''):
        name = 'Report'
    return name


def figure_clock_ticks_enabled(conf):
    """Clock-time markers: saved flag, else on for real-time sync and off for anchor."""
    conf = conf or {}
    if conf.get('showFigureClockTicks') is not None:
        return bool(conf.get('showFigureClockTicks'))
    return (conf.get('timeSyncMode') or 'clock') != 'anchor'


def parse_clock_ticks(conf):
    if not figure_clock_ticks_enabled(conf):
        return []
    raw = (conf or {}).get('figureClockTicks')
    if raw in (None, ''):
        raw = DEFAULT_CLOCK_TICKS
    if isinstance(raw, str):
        parts = [p.strip() for p in raw.replace(';', ',').split(',') if p.strip()]
    else:
        parts = [str(p).strip() for p in raw if str(p).strip()]
    ticks = []
    for part in parts:
        try:
            hour_s, minute_s = part.split(':', 1)
            hour = int(hour_s)
            minute = int(minute_s)
        except (ValueError, TypeError):
            continue
        if 0 <= hour <= 23 and 0 <= minute <= 59:
            ticks.append((hour, minute, f'{hour:02d}:{minute:02d}'))
    return ticks


def parse_hour_list(raw):
    if raw in (None, ''):
        return []
    if isinstance(raw, (list, tuple)):
        parts = raw
    else:
        parts = str(raw).replace(';', ',').split(',')
    hours = []
    for part in parts:
        text = str(part).strip()
        if not text:
            continue
        try:
            hours.append(int(round(float(text))))
        except (TypeError, ValueError):
            continue
    return hours


def snapshot_hours(conf, kind='convex'):
    """Elapsed snapshot hours for convex-hull or lateral-angle figures."""
    if conf is None:
        return []
    kind = 'angles' if kind == 'angles' else 'convex'
    specific = 'snapshotHoursAngles' if kind == 'angles' else 'snapshotHoursConvex'
    hours = parse_hour_list(conf.get(specific))
    if hours:
        return hours
    hours = parse_hour_list(conf.get('snapshotHours'))
    if hours:
        return hours
    legacy = 'daysAngles' if kind == 'angles' else 'daysConvexHull'
    return parse_hour_list(conf.get(legacy))


def elapsed_hour_windows(data, dt, hour_col='ElapsedTime (h)'):
    """Yield (start, end) integer hour bins covering the elapsed range."""
    if data is None or data.empty or hour_col not in data.columns:
        return
    try:
        dt = max(int(dt), 1)
    except (TypeError, ValueError):
        dt = 1
    series = pd.to_numeric(data[hour_col], errors='coerce').dropna()
    if series.empty:
        return
    min_h = int(np.floor(series.min()))
    max_h = int(np.floor(series.max()))
    start = min_h
    while start <= max_h:
        end = min(start + dt, max_h + 1)
        yield start, end
        start = end


def _ensure_dates(series):
    return pd.to_datetime(series, errors='coerce')


def elapsed_hours_from_t0(dates, t0):
    """Integer hour bins: floor((Date - t0) / 1 hour).

    pandas/NumPy .round uses banker's rounding (half to even). Offsets of
    n + 0.5 hours then collapse onto even elapsed hours and odd hours become
    empty after reindex. Floor keeps consecutive hourly stamps 1 h apart.
    """
    series = pd.to_datetime(pd.Series(dates), errors='coerce')
    t0_ts = parse_datetime(t0)
    if t0_ts is None:
        return pd.Series(np.nan, index=series.index, dtype='float64')
    delta = series - t0_ts
    hours = delta.dt.total_seconds() / 3600.0
    elapsed = np.floor(hours.to_numpy(dtype=float))
    result = pd.Series(elapsed, index=series.index, dtype='float64')
    return result.where(series.notna(), np.nan)


def _plant_key_columns(data):
    cols = [c for c in ('Experiment', 'Video', 'Plant_id') if c in data.columns]
    if 'Plant_id' not in cols:
        raise ValueError('Elapsed-time alignment requires a Plant_id column')
    return cols


def plant_date_spans(data):
    """Return a DataFrame of per-plant Date min/max (and optional Video)."""
    if data is None or data.empty or 'Date' not in data.columns:
        return pd.DataFrame(columns=['Experiment', 'Plant_id', 'Video', 'date_min', 'date_max'])
    work = data.copy()
    work['Date'] = _ensure_dates(work['Date'])
    work = work.dropna(subset=['Date'])
    if work.empty:
        return pd.DataFrame(columns=['Experiment', 'Plant_id', 'Video', 'date_min', 'date_max'])
    if 'Plant_id' not in work.columns:
        work['Plant_id'] = 'plant'
    if 'Experiment' not in work.columns:
        work['Experiment'] = 'unspecified'
    group_cols = _plant_key_columns(work)
    rows = []
    for key, part in work.groupby(group_cols, sort=False):
        if not isinstance(key, tuple):
            key = (key,)
        record = dict(zip(group_cols, key))
        record['date_min'] = part['Date'].min()
        record['date_max'] = part['Date'].max()
        if 'Video' in part.columns:
            record['Video'] = part['Video'].iloc[0]
        else:
            record['Video'] = ''
        record['n_plants'] = 1
        rows.append(record)
    return pd.DataFrame(rows)


def _union_find_clusters(spans):
    n = len(spans)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(n):
        start_i, end_i = spans[i]
        for j in range(i + 1, n):
            start_j, end_j = spans[j]
            if start_i <= end_j and start_j <= end_i:
                union(i, j)

    clusters = defaultdict(list)
    for i in range(n):
        clusters[find(i)].append(i)
    return list(clusters.values())


def _group_dict(group_id, members, mode='clock'):
    date_min = min(m['date_min'] for m in members)
    date_max = max(m['date_max'] for m in members)
    start = date_min
    t0 = start
    n_plants = 0
    for member in members:
        try:
            n_plants += int(member.get('n_plants') or 1)
        except (TypeError, ValueError):
            n_plants += 1
    return {
        'id': int(group_id),
        'n_plants': max(n_plants, len(members)),
        'spanStart': format_datetime(date_min),
        'spanEnd': format_datetime(date_max),
        'start': format_datetime(start),
        't0': format_datetime(t0),
        'synthetic': bool(members and is_synthetic_date(members[0]['date_min'])),
    }


def _spans_frame(data):
    if data is None or (hasattr(data, 'empty') and data.empty):
        return pd.DataFrame()
    if 'date_min' in data.columns and 'date_max' in data.columns:
        work = data.copy()
        work['date_min'] = work['date_min'].map(parse_datetime)
        work['date_max'] = work['date_max'].map(parse_datetime)
        work = work.dropna(subset=['date_min', 'date_max'])
        if 'n_plants' not in work.columns:
            work['n_plants'] = 1
        if 'Video' not in work.columns:
            work['Video'] = ''
        if 'Experiment' not in work.columns:
            work['Experiment'] = 'unspecified'
        if 'Plant_id' not in work.columns:
            work['Plant_id'] = work.get('Video', pd.Series(['plant'] * len(work)))
        return work
    return plant_date_spans(data)


def detect_time_groups(data, mode='clock'):
    """Cluster plants whose Date ranges overlap (synthetic dates by Video)."""
    spans = _spans_frame(data)
    if spans.empty:
        return []

    real_rows = []
    synthetic_buckets = defaultdict(list)
    records = spans.to_dict('records')
    for rec in records:
        if is_synthetic_date(rec['date_min']) and is_synthetic_date(rec['date_max']):
            bucket = rec.get('Video') or rec.get('Experiment') or 'synthetic'
            synthetic_buckets[str(bucket)].append(rec)
        else:
            real_rows.append(rec)

    groups = []
    group_id = 1
    if real_rows:
        intervals = [(r['date_min'], r['date_max']) for r in real_rows]
        for cluster in _union_find_clusters(intervals):
            members = [real_rows[i] for i in cluster]
            groups.append(_group_dict(group_id, members, mode=mode))
            group_id += 1
    for _bucket, members in sorted(synthetic_buckets.items(), key=lambda kv: kv[0]):
        groups.append(_group_dict(group_id, members, mode='clock'))
        group_id += 1

    groups.sort(key=lambda g: g['spanStart'] or '')
    for i, group in enumerate(groups, start=1):
        group['id'] = i
    return groups


def _intervals_overlap(a_start, a_end, b_start, b_end):
    if None in (a_start, a_end, b_start, b_end):
        return False
    return a_start <= b_end and b_start <= a_end


def match_time_group(groups, date_min, date_max):
    """Pick the saved group whose stored span overlaps this plant the most."""
    if not groups:
        return None
    date_min = parse_datetime(date_min)
    date_max = parse_datetime(date_max)
    best = None
    best_overlap = pd.Timedelta(0)
    for group in groups:
        span_start = parse_datetime(group.get('spanStart'))
        span_end = parse_datetime(group.get('spanEnd'))
        if not _intervals_overlap(date_min, date_max, span_start, span_end):
            continue
        overlap_start = max(date_min, span_start)
        overlap_end = min(date_max, span_end)
        overlap = overlap_end - overlap_start
        if best is None or overlap > best_overlap:
            best = group
            best_overlap = overlap
    return best


def default_duration_hours(groups, data=None):
    lengths = []
    for group in groups or []:
        start = parse_datetime(group.get('spanStart'))
        end = parse_datetime(group.get('spanEnd'))
        if start is not None and end is not None:
            lengths.append((end - start).total_seconds() / 3600.0)
    if lengths:
        return int(max(1, int(np.ceil(min(lengths)))))
    if data is not None and 'ElapsedTime (h)' in data.columns:
        series = pd.to_numeric(data['ElapsedTime (h)'], errors='coerce').dropna()
        if not series.empty:
            return int(max(1, round(series.max() - series.min() + 1)))
    return 40


def shared_clock_origin(groups):
    """Earliest group start; used as t0 when synchronizing by real time."""
    starts = []
    for group in groups or []:
        ts = parse_datetime(group.get('start')) or parse_datetime(group.get('spanStart'))
        if ts is not None:
            starts.append(ts)
    if not starts:
        return None
    return min(starts)


def resolved_time_groups(conf, data=None):
    mode = (conf or {}).get('timeSyncMode') or 'clock'
    saved = list((conf or {}).get('timeGroups') or [])
    if saved:
        return saved, mode
    detected = detect_time_groups(data, mode=mode) if data is not None else []
    return detected, mode


def _window_for_group(group, duration_hours, t0_override=None):
    start = parse_datetime(group.get('start')) or parse_datetime(group.get('spanStart'))
    if t0_override is not None:
        t0 = parse_datetime(t0_override)
    else:
        t0 = parse_datetime(group.get('t0')) or start
    if start is None:
        return None
    if duration_hours is None:
        span_end = parse_datetime(group.get('spanEnd')) or start
        end = span_end
    else:
        end = start + pd.Timedelta(hours=float(duration_hours))
    if t0 is None:
        t0 = start
    return start, t0, end


def apply_time_windows(data, conf):
    """Recompute ElapsedTime from each group's t0, crop, and pad a shared grid."""
    if data is None or data.empty:
        return data
    if 'Date' not in data.columns:
        print('Analysis period skipped: no Date column in hourly data', flush=True)
        return data

    work = data.copy()
    work['Date'] = _ensure_dates(work['Date'])
    if 'Plant_id' not in work.columns:
        work['Plant_id'] = np.arange(len(work)).astype(str)
    if 'Video' in work.columns:
        work['Video'] = work['Video'].fillna('').astype(str)
    if 'Experiment' in work.columns:
        work['Experiment'] = work['Experiment'].fillna('').astype(str)

    groups = list((conf or {}).get('timeGroups') or [])
    if not groups:
        print(
            'Analysis period skipped: no saved time groups '
            '(set analysis period first)',
            flush=True,
        )
        return work

    duration = (conf or {}).get('timeDurationHours')
    if duration in (None, ''):
        duration_hours = default_duration_hours(groups, work)
    else:
        duration_hours = float(duration)

    mode = (conf or {}).get('timeSyncMode') or 'clock'
    clock_t0 = shared_clock_origin(groups) if mode != 'anchor' else None

    key_cols = _plant_key_columns(work)
    aligned = []
    elapsed_bounds = []
    n_empty = 0
    n_unmatched = 0
    for key, part in work.groupby(key_cols, sort=False):
        part = part.sort_values('Date')
        dates = part['Date'].dropna()
        if dates.empty:
            print(f'Skipping plant {key}: no Date values', flush=True)
            continue
        group = match_time_group(groups, dates.min(), dates.max())
        if group is None:
            print(
                f'Skipping plant {key}: no overlapping analysis-period group',
                flush=True,
            )
            n_unmatched += 1
            continue
        window = _window_for_group(group, duration_hours, t0_override=clock_t0)
        if window is None:
            print(f'Skipping plant {key}: analysis-period group has no start', flush=True)
            n_unmatched += 1
            continue
        start, t0, end = window
        cropped = part[(part['Date'] >= start) & (part['Date'] <= end)].copy()
        if cropped.empty:
            print(
                f'Skipping plant {key}: no samples in analysis window '
                f'{format_datetime(start)}–{format_datetime(end)}',
                flush=True,
            )
            n_empty += 1
            continue
        elapsed = elapsed_hours_from_t0(cropped['Date'], t0)
        cropped = cropped.loc[elapsed.notna()].copy()
        if cropped.empty:
            print(f'Skipping plant {key}: elapsed hours could not be computed', flush=True)
            n_empty += 1
            continue
        cropped['ElapsedTime (h)'] = elapsed.loc[cropped.index].astype(int)
        cropped['_t0'] = t0
        bound_hours = elapsed_hours_from_t0(pd.Series([start]), t0)
        if bound_hours.isna().any():
            continue
        e0 = int(bound_hours.iloc[0])
        real_max = int(cropped['ElapsedTime (h)'].max())
        elapsed_bounds.append((e0, real_max))
        aligned.append(cropped)

    if n_empty or n_unmatched:
        print(
            f'Analysis period dropped {n_empty} plants with empty windows '
            f'and {n_unmatched} unmatched plants',
            flush=True,
        )

    if not aligned:
        print('Analysis period: no plants overlapped the configured window', flush=True)
        return work.iloc[0:0].copy()

    grid_start = min(b[0] for b in elapsed_bounds)
    grid_end = max(b[1] for b in elapsed_bounds)
    if grid_end < grid_start:
        grid_end = grid_start
    target_hours = list(range(int(grid_start), int(grid_end) + 1))

    padded = []
    for frame in aligned:
        t0 = frame['_t0'].iloc[0]
        identity = {}
        for col in ID_COLUMNS:
            if col in frame.columns and len(frame[col].dropna()):
                identity[col] = frame[col].dropna().iloc[0]
        indexed = frame.drop(columns=['_t0']).drop_duplicates('ElapsedTime (h)', keep='last')
        indexed = indexed.set_index('ElapsedTime (h)')
        reindexed = indexed.reindex(target_hours)
        for col, value in identity.items():
            reindexed[col] = reindexed[col].fillna(value)
        reindexed = reindexed.reset_index()
        if 'index' in reindexed.columns and 'ElapsedTime (h)' not in reindexed.columns:
            reindexed = reindexed.rename(columns={'index': 'ElapsedTime (h)'})
        kept_dates = reindexed['Date'].copy() if 'Date' in reindexed.columns else None
        reindexed['Date'] = pd.to_datetime(t0) + pd.to_timedelta(
            pd.to_numeric(reindexed['ElapsedTime (h)'], errors='coerce'), unit='h'
        )
        if kept_dates is not None:
            real = kept_dates.notna()
            reindexed.loc[real, 'Date'] = pd.to_datetime(kept_dates.loc[real])
        padded.append(reindexed)

    result = pd.concat(padded, ignore_index=True)
    if 'NewDay' in result.columns:
        dates = pd.to_datetime(result['Date'])
        result['NewDay'] = (dates.dt.hour == 0) & (dates.dt.minute == 0)
    return result


INITIAL_STAGE_COLUMNS = (
    'MainRootLength (mm)',
    'LateralRootsLength (mm)',
    'TotalLength (mm)',
    'HypocotylLength (mm)',
    'NumberOfLateralRoots',
    'Area (mm2)',
    'DenseRootArea (mm2)',
    'Area',
    'DenseRootArea',
)


def subtract_initial_stage(data, conf):
    """Subtract each plant's first in-window non-NaN from length/area/LR series."""
    if data is None or data.empty:
        return data
    if not (conf or {}).get('measureRelativeToInitial'):
        return data
    work = data.copy()
    cols = [c for c in INITIAL_STAGE_COLUMNS if c in work.columns]
    if not cols:
        return work
    for col in cols:
        work[col] = pd.to_numeric(work[col], errors='coerce')
    key_cols = _plant_key_columns(work)
    for _key, part in work.groupby(key_cols, sort=False):
        idx = part.index
        for col in cols:
            series = work.loc[idx, col]
            first = series.first_valid_index()
            if first is None:
                continue
            baseline = series.loc[first]
            if pd.isna(baseline):
                continue
            work.loc[idx, col] = series - baseline
    return work


def hourly_covers_analysis_period(hourly_df, conf):
    """True if each group's [start, start+duration] lies inside Hour Date spans."""
    if hourly_df is None or hourly_df.empty or 'Date' not in hourly_df.columns:
        return False
    groups = list((conf or {}).get('timeGroups') or [])
    if not groups:
        return False
    duration = (conf or {}).get('timeDurationHours')
    if duration in (None, ''):
        duration_hours = default_duration_hours(groups, hourly_df)
    else:
        duration_hours = float(duration)

    mode = (conf or {}).get('timeSyncMode') or 'clock'
    clock_t0 = shared_clock_origin(groups) if mode != 'anchor' else None

    work = hourly_df.copy()
    work['Date'] = _ensure_dates(work['Date'])
    if 'Plant_id' not in work.columns:
        return False
    key_cols = _plant_key_columns(work)
    group_bounds = {}
    for _key, part in work.groupby(key_cols, sort=False):
        dates = part['Date'].dropna()
        if dates.empty:
            continue
        group = match_time_group(groups, dates.min(), dates.max())
        if group is None:
            continue
        gid = group.get('id')
        dmin, dmax = dates.min(), dates.max()
        if gid not in group_bounds:
            group_bounds[gid] = [dmin, dmax]
        else:
            group_bounds[gid][0] = min(group_bounds[gid][0], dmin)
            group_bounds[gid][1] = max(group_bounds[gid][1], dmax)

    if not group_bounds:
        return False
    for i, group in enumerate(groups):
        window = _window_for_group(group, duration_hours, t0_override=clock_t0)
        if window is None:
            return False
        start, _t0, end = window
        gid = group.get('id')
        if gid is None:
            gid = i + 1
        bounds = group_bounds.get(gid)
        if bounds is None:
            bounds = group_bounds.get(i + 1)
        if bounds is None:
            return False
        if bounds[0] > start or bounds[1] < end:
            return False
    return True


def elapsed_clock_ticks(data, conf=None, clock_times=None):
    """Elapsed hours whose Date matches configured clock times of day."""
    if data is None or data.empty:
        return []
    if 'ElapsedTime (h)' not in data.columns:
        return []
    ticks = parse_clock_ticks(conf) if clock_times is None else clock_times
    if not ticks:
        return []
    if 'Date' not in data.columns:
        return []
    dates = _ensure_dates(data['Date'])
    hours = pd.to_numeric(data['ElapsedTime (h)'], errors='coerce')
    elapsed_int = pd.Series(np.floor(hours.to_numpy(dtype=float)), index=hours.index)
    elapsed_int = elapsed_int.where(hours.notna())
    found = []
    seen = set()
    for clock_h, clock_m, _label in ticks:
        mask = (dates.dt.hour == clock_h) & (dates.dt.minute == clock_m)
        if not mask.any() and clock_m == 0:
            mask = dates.dt.hour == clock_h
        values = elapsed_int[mask].dropna().astype(int)
        for value in sorted(values.unique()):
            value = int(value)
            if value in seen:
                continue
            at_h = dates[elapsed_int == value].dropna()
            if at_h.empty:
                continue
            clocks = {
                (int(ts.hour), int(ts.minute)) for ts in at_h
            }
            if len(clocks) != 1:
                continue
            seen.add(value)
            found.append(value)
    return found


def clock_tick_label(data, elapsed_hour):
    if data is None or data.empty or 'Date' not in data.columns:
        return '00:00'
    hours = pd.to_numeric(data['ElapsedTime (h)'], errors='coerce')
    match = data.loc[hours.round(0) == int(elapsed_hour)]
    if match.empty:
        return '00:00'
    ts = parse_datetime(match['Date'].iloc[0])
    if ts is None:
        return '00:00'
    return ts.strftime('%H:%M')


def draw_clock_ticks(ax, data, conf, *, twin_axis=True):
    """Draw axvline markers (and optional top labels) at real-clock ticks."""
    ticks = elapsed_clock_ticks(data, conf)
    if not ticks:
        return ticks
    for hour in ticks:
        ax.axvline(hour, color='0.55', linestyle='--', linewidth=0.9, zorder=0)
    if twin_axis:
        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xticks(ticks)
        ax2.set_xticklabels(
            [clock_tick_label(data, hour) for hour in ticks],
            rotation=45, ha='left', fontsize=9,
        )
        ax2.tick_params(axis='x', which='major', length=6, width=1, color='black')
    return ticks


def draw_clock_ticks_on_axes(axes, data, conf):
    for ax in axes:
        draw_clock_ticks(ax, data, conf, twin_axis=True)


def hourly_files_exist(main_folder):
    """True if any PostProcess_Hour.csv exists under Analysis/ (does not read files)."""
    analysis = os.path.join(main_folder, 'Analysis') if main_folder else ''
    if not analysis or not os.path.isdir(analysis):
        return False
    for _dirpath, _dirnames, filenames in os.walk(analysis):
        if 'PostProcess_Hour.csv' in filenames:
            return True
    return False


def collect_hourly_frames(analysis_folder):
    """Load PostProcess_Hour.csv frames (Date fallback from Results_raw names)."""
    frames = []
    if not analysis_folder or not os.path.isdir(analysis_folder):
        return frames
    for dirpath, _dirnames, filenames in os.walk(analysis_folder):
        if 'PostProcess_Hour.csv' not in filenames:
            continue
        path = os.path.join(dirpath, 'PostProcess_Hour.csv')
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if 'Date' not in df.columns or df['Date'].isna().all():
            raw_path = os.path.join(dirpath, 'Results_raw.csv')
            dates = []
            if os.path.isfile(raw_path):
                try:
                    raw = pd.read_csv(raw_path, usecols=lambda c: c in ('FileName',))
                    dates = [datetime_from_filename(n) for n in raw['FileName']]
                except Exception:
                    dates = []
            if dates and any(d is not None for d in dates):
                df = df.copy()
                if len(dates) >= len(df):
                    df['Date'] = dates[:len(df)]
                else:
                    continue
            else:
                continue
        df = df.copy()
        df['Date'] = _ensure_dates(df['Date'])
        if 'Plant_id' not in df.columns:
            plant_folder = os.path.basename(os.path.dirname(dirpath))
            df['Plant_id'] = plant_folder
        frames.append(df)
    return frames


def collect_hourly_data(main_folder):
    analysis = os.path.join(main_folder, 'Analysis') if main_folder else ''
    frames = collect_hourly_frames(analysis)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_hourly_tables(conf):
    """Load PostProcess_Hour.csv tables from the project's Analysis tree."""
    return collect_hourly_data((conf or {}).get('MainFolder') or '')


def _span_record(plant_id, date_min, date_max, experiment='', video='', n_plants=1):
    date_min = parse_datetime(date_min)
    date_max = parse_datetime(date_max)
    if date_min is None or date_max is None:
        return None
    if date_max < date_min:
        date_min, date_max = date_max, date_min
    return {
        'Experiment': experiment or 'unspecified',
        'Plant_id': str(plant_id),
        'Video': video or '',
        'date_min': date_min,
        'date_max': date_max,
        'n_plants': int(n_plants or 1),
    }


def _datetimes_from_names(names):
    stamps = [datetime_from_filename(n) for n in names or [] if n]
    stamps = [s for s in stamps if s is not None]
    if not stamps:
        return None, None
    return min(stamps), max(stamps)


def _n_plants_from_screening(analysis_dir, meta):
    seeds_path = os.path.join(analysis_dir, 'seeds.tsv')
    if os.path.isfile(seeds_path):
        try:
            seeds = pd.read_csv(seeds_path, sep='\t', usecols=lambda c: c in ('Plant_id', 'FileName'))
            if 'Plant_id' in seeds.columns and seeds['Plant_id'].nunique():
                return int(seeds['Plant_id'].nunique())
        except Exception:
            pass
    group_path = os.path.join(analysis_dir, 'group_info.json')
    info = meta
    if os.path.isfile(group_path):
        try:
            with open(group_path, 'r') as handle:
                info = json.load(handle)
        except Exception:
            pass
    counts = info.get('seed_counts') if isinstance(info, dict) else None
    if counts:
        try:
            return int(sum(int(c or 0) for c in counts))
        except (TypeError, ValueError):
            pass
    return 1


def _collect_screening_spans(main_folder):
    root = os.path.join(main_folder, 'analysis') if main_folder else ''
    if not os.path.isdir(root):
        return pd.DataFrame()
    rows = []
    for analysis_id in sorted(os.listdir(root)):
        analysis_dir = os.path.join(root, analysis_id)
        if not os.path.isdir(analysis_dir):
            continue
        meta = {}
        meta_path = os.path.join(analysis_dir, 'metadata.json')
        if os.path.isfile(meta_path):
            try:
                with open(meta_path, 'r') as handle:
                    meta = json.load(handle) or {}
            except Exception:
                meta = {}
        date_min = parse_datetime(meta.get('first_datetime'))
        date_max = parse_datetime(meta.get('last_datetime'))
        if date_min is None or date_max is None:
            date_min, date_max = _datetimes_from_names(
                [meta.get('first_image'), meta.get('last_image')]
            )
        if date_min is None or date_max is None:
            seeds_path = os.path.join(analysis_dir, 'seeds.tsv')
            if os.path.isfile(seeds_path):
                try:
                    seeds = pd.read_csv(seeds_path, sep='\t', usecols=lambda c: c in ('FileName',))
                    date_min, date_max = _datetimes_from_names(seeds['FileName'].tolist())
                except Exception:
                    date_min, date_max = None, None
        record = _span_record(
            analysis_id, date_min, date_max,
            experiment=str(meta.get('analysis_id') or analysis_id),
            video=analysis_id,
            n_plants=_n_plants_from_screening(analysis_dir, meta),
        )
        if record is not None:
            rows.append(record)
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def _collect_results_raw_spans(main_folder):
    analysis = os.path.join(main_folder, 'Analysis') if main_folder else ''
    if not os.path.isdir(analysis):
        return pd.DataFrame()
    rows = []
    for dirpath, _dirnames, filenames in os.walk(analysis):
        if 'Results_raw.csv' not in filenames:
            continue
        raw_path = os.path.join(dirpath, 'Results_raw.csv')
        try:
            raw = pd.read_csv(raw_path, usecols=lambda c: c in ('FileName', 'Plant_id'))
        except Exception:
            continue
        if 'FileName' not in raw.columns:
            continue
        date_min, date_max = _datetimes_from_names(raw['FileName'].tolist())
        plant_id = ''
        if 'Plant_id' in raw.columns and raw['Plant_id'].notna().any():
            plant_id = str(raw['Plant_id'].dropna().iloc[0])
        if not plant_id:
            plant_id = os.path.basename(os.path.dirname(dirpath))
        experiment = os.path.basename(analysis)
        meta_path = os.path.join(dirpath, 'metadata.json')
        video = ''
        if os.path.isfile(meta_path):
            try:
                with open(meta_path, 'r') as handle:
                    meta = json.load(handle) or {}
                experiment = str(meta.get('Experiment') or experiment)
                video = str(meta.get('Video') or '')
                plant_id = str(meta.get('Plant_id') or plant_id)
            except Exception:
                pass
        record = _span_record(plant_id, date_min, date_max, experiment=experiment, video=video)
        if record is not None:
            rows.append(record)
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def collect_acquisition_spans(main_folder):
    """Clock spans from tracking metadata / image names, without Hour CSV."""
    screening = _collect_screening_spans(main_folder)
    if not screening.empty:
        return screening
    raw = _collect_results_raw_spans(main_folder)
    if not raw.empty:
        return raw
    hourly = collect_hourly_data(main_folder)
    if hourly.empty:
        return pd.DataFrame()
    return plant_date_spans(hourly)


PERIOD_GATE_MESSAGE = (
    "Set an analysis period for the current videos/plants before postprocessing. "
    "The period is missing or was saved for a different set of analyses."
)


def time_period_sources(main_folder):
    """Fingerprint of the acquisition set used to set the analysis period."""
    spans = collect_acquisition_spans(main_folder)
    if spans is None or spans.empty:
        return []
    analysis_root = os.path.join(main_folder, 'analysis') if main_folder else ''
    if os.path.isdir(analysis_root) and 'Video' in spans.columns:
        videos = sorted({
            str(v).strip() for v in spans['Video'].dropna() if str(v).strip()
        })
        if videos:
            return videos
    keys = []
    for _, row in spans.iterrows():
        keys.append('\t'.join([
            str(row.get('Experiment') or '').strip(),
            str(row.get('Video') or '').strip(),
            str(row.get('Plant_id') or '').strip(),
        ]))
    return sorted(keys)


def analysis_period_is_current(conf, main_folder):
    groups = list((conf or {}).get('timeGroups') or [])
    saved = [str(s) for s in list((conf or {}).get('timePeriodSources') or [])]
    current = [str(s) for s in time_period_sources(main_folder)]
    if not groups or not saved or not current:
        return False
    return saved == current


def groups_to_config(groups):
    payload = []
    for group in groups or []:
        payload.append({
            'id': int(group.get('id') or 0),
            'spanStart': group.get('spanStart') or '',
            'spanEnd': group.get('spanEnd') or '',
            'start': group.get('start') or group.get('spanStart') or '',
            't0': group.get('t0') or group.get('start') or '',
            'n_plants': int(group.get('n_plants') or 0),
        })
    return payload
