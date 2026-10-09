""" 
ChronoRoot: High-throughput phenotyping by deep learning reveals novel temporal parameters of plant root system architecture
Copyright (C) 2020 Nicolás Gaggion

Modified version with improved time handling for irregular sampling
"""

import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import pandas as pd
import os
from scipy import signal
import json
import warnings

from .time_windows import datetime_from_filename, elapsed_hours_from_t0, period_frame_indices
from .utils.fileUtilities import expected_hourly_rows


def _finite_time_spans(dates, finite, max_gap):
    """Index ranges [start, end) of samples that can be filtered together.

    A missing picture is not a sample and not a neighbor. A non-finite value
    or a time gap wider than max_gap starts a new range, so a median filter
    or a difference never reads across the hole.
    """
    dates = pd.to_datetime(pd.Series(dates)).reset_index(drop=True)
    finite = np.asarray(finite, dtype=bool)
    spans = []
    start = None
    for i, ok in enumerate(finite):
        if not ok:
            if start is not None:
                spans.append((start, i))
                start = None
            continue
        if start is None:
            start = i
            continue
        if dates.iloc[i] - dates.iloc[i - 1] >= max_gap:
            spans.append((start, i))
            start = i
    if start is not None:
        spans.append((start, len(finite)))
    return spans


def _median_filter_spans(values, spans, requested):
    """Median-filter each finite run. Short runs keep their own values."""
    out = np.array(values, dtype=float, copy=True)
    for start, end in spans:
        n = end - start
        if n < 3:
            continue
        limit = n if n % 2 else n - 1
        kernel = min(int(requested), limit)
        if kernel % 2 == 0:
            kernel -= 1
        if kernel >= 3:
            out[start:end] = signal.medfilt(out[start:end], kernel)
    return out


def _gradient_spans(values, hours, spans):
    """mm per hour inside each finite run, using the real hour coordinates.

    The missing hour stays NaN. The slope is not taken across that hole, and
    it is not taken as if neighboring photos were one index step apart.
    """
    out = np.full(len(values), np.nan, dtype=float)
    hours = np.asarray(hours, dtype=float)
    values = np.asarray(values, dtype=float)
    for start, end in spans:
        y = values[start:end]
        x = hours[start:end]
        if len(y) < 2 or not np.all(np.isfinite(x)) or np.any(np.diff(x) <= 0):
            continue
        edge = 2 if len(y) >= 3 else 1
        out[start:end] = np.gradient(y, x, edge_order=edge)
    return out


def dataWork(conf, pfile, folder, N_exp = None, debug=False, time_tolerance=0.5):
    """
    Process root measurement data.
    COMPATIBILITY: Works with both Pandas < 2.0 (using 'H'/'T') and Pandas 2.2+ (using 'h'/'min').
    """
    
    # --- CROSS-VERSION COMPATIBILITY CHECK ---
    # We define the frequency aliases dynamically to avoid errors on mixed versions.
    try:
        # Try to parse the modern lowercase alias (Pandas 2.2+)
        pd.tseries.frequencies.to_offset('h')
        FREQ_HOUR = 'h'
        FREQ_MIN = 'min'
    except (ValueError, TypeError):
        # Fallback for ancient Pandas versions that strict-require uppercase
        FREQ_HOUR = 'H'
        FREQ_MIN = 'T'
    # -----------------------------------------

    data = pd.read_csv(pfile)

    # Check for required columns
    required_cols = ['FileName', 'MainRootLength', 'LateralRootsLength', 'NumberOfLateralRoots']
    missing_cols = [col for col in required_cols if col not in data.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # If there is no HypocotylLength column, add it with zeros
    if 'HypocotylLength' not in data.columns:
        data['HypocotylLength'] = 0.0
    
    # Keep photographed frames only. A filename we cannot date is not a
    # timepoint, and a gap between photos is not filled with a copy of the
    # previous measurement.
    stamps = []
    keep_idx = []
    for i, name in enumerate(data['FileName'].tolist()):
        ts = datetime_from_filename(name)
        if ts is None:
            warnings.warn(f"Cannot parse date from filename: {name}")
            continue
        stamps.append(ts)
        keep_idx.append(i)
    if not keep_idx:
        raise ValueError(f"No timestamps found in {pfile}")
    data = data.iloc[keep_idx].copy()
    data['Date'] = pd.to_datetime(stamps)
    data = data.sort_values(by=['Date']).reset_index(drop=True)

    try:
        timeStep = float(conf['timeStep'])
    except (TypeError, ValueError):
        timeStep = 15.0
    if timeStep <= 0:
        timeStep = 15.0
    timeStep_td = pd.Timedelta(minutes=timeStep)
    # The old filler treated a jump of two capture intervals as a missing
    # frame. The same threshold splits the series for filtering.
    frame_gap = timeStep_td * 2

    if N_exp is not None:
        try:
            n_exp = int(N_exp)
        except (TypeError, ValueError):
            n_exp = None
        if n_exp is not None and n_exp > 0 and len(data) > n_exp:
            data = data.iloc[:n_exp].copy()

    # The analysis period is the timelapse that gets measured. Frames before
    # the window start and after its end are left out, the same way a
    # processing limit drops frames past its last day.
    kept = period_frame_indices(data['FileName'].tolist(), conf)
    if not kept:
        raise ValueError("No frames fall inside the analysis period")
    if len(kept) != len(data):
        data = data.iloc[kept].reset_index(drop=True)

    # Reads the pixel size
    path = os.path.abspath(os.path.join(folder, 'metadata.json'))
    try:
        with open(path) as f:
            metadata = json.load(f)
        pixel_size = metadata['pixel_size']
    except (FileNotFoundError, KeyError):
        print(f"Warning: Could not read pixel_size from {path}, defaulting to 1.0")
        pixel_size = 1.0
    
    # A NaN here is a bad measurement on a real frame, not a missing hour.
    # It stays NaN. Missing hours are introduced later by the hourly grid.
    # Copy: recent pandas hands back a read-only array, and the 6-hour
    # cleanup and the shrink correction both write into these values.
    mainRoot = np.array(pd.to_numeric(data['MainRootLength'], errors='coerce'), dtype=float, copy=True)
    lateralRoots = np.array(pd.to_numeric(data['LateralRootsLength'], errors='coerce'), dtype=float, copy=True)
    numlateralRoots = np.array(pd.to_numeric(data['NumberOfLateralRoots'], errors='coerce'), dtype=float, copy=True)
    hypocotylLength = np.array(pd.to_numeric(data['HypocotylLength'], errors='coerce'), dtype=float, copy=True)

    # Early zeros are only trusted when a real frame about 6 h earlier is
    # also zero. A missing frame must not stand in for that earlier photo,
    # and it must not be written as zero.
    dates = data['Date']
    half_step = timeStep_td / 2
    six_hours = pd.Timedelta(hours=6)
    for t in range(len(data)):
        delta = (dates - (dates.iloc[t] - six_hours)).abs()
        j = int(np.argmin(delta.to_numpy()))
        if delta.iloc[j] > half_step or j >= t:
            continue
        if (
            np.isfinite(numlateralRoots[j]) and np.isfinite(numlateralRoots[t])
            and numlateralRoots[j] == 0 and numlateralRoots[t] == 0
        ):
            finite = np.isfinite(lateralRoots[:t])
            lateralRoots[:t] = np.where(finite, 0.0, lateralRoots[:t])
            finite = np.isfinite(numlateralRoots[:t])
            numlateralRoots[:t] = np.where(finite, 0.0, numlateralRoots[:t])
        if (
            np.isfinite(mainRoot[j]) and np.isfinite(mainRoot[t])
            and mainRoot[j] == 0 and mainRoot[t] == 0
        ):
            finite = np.isfinite(mainRoot[:t])
            mainRoot[:t] = np.where(finite, 0.0, mainRoot[:t])
        if (
            np.isfinite(hypocotylLength[j]) and np.isfinite(hypocotylLength[t])
            and hypocotylLength[j] == 0 and hypocotylLength[t] == 0
        ):
            finite = np.isfinite(hypocotylLength[:t])
            hypocotylLength[:t] = np.where(finite, 0.0, hypocotylLength[:t])

    def _frame_spans(series):
        return _finite_time_spans(dates, np.isfinite(series), frame_gap)

    mainRoot = _median_filter_spans(mainRoot, _frame_spans(mainRoot), 9)
    lateralRoots = _median_filter_spans(lateralRoots, _frame_spans(lateralRoots), 9)
    numlateralRoots = _median_filter_spans(numlateralRoots, _frame_spans(numlateralRoots), 9)
    hypocotylLength = _median_filter_spans(hypocotylLength, _frame_spans(hypocotylLength), 9)

    # Lengths are not allowed to shrink inside a photographed run. The value
    # is not carried across a gap. Lateral and hypocotyl lengths only hold
    # when the previous sample was already above zero, as before.
    for series, hold_only_if_positive in (
        (mainRoot, False),
        (numlateralRoots, True),
        (lateralRoots, True),
        (hypocotylLength, True),
    ):
        for start, end in _frame_spans(series):
            for i in range(start + 1, end):
                previous = series[i - 1]
                if series[i] < previous and (not hold_only_if_positive or previous > 0):
                    series[i] = previous

    # Multiply by pixel size
    mainRoot_mm = mainRoot.copy() * pixel_size
    lateralRoots_mm = lateralRoots.copy() * pixel_size
    hypocotyl_mm = hypocotylLength.copy() * pixel_size
    
    data['MainRootLength (mm)'] = mainRoot_mm
    data['LateralRootsLength (mm)'] = lateralRoots_mm
    data['NumberOfLateralRoots'] = numlateralRoots
    data['TotalLength (mm)'] = mainRoot_mm + lateralRoots_mm
    data['HypocotylLength (mm)'] = hypocotyl_mm

    data['FileName'].to_csv(
        os.path.abspath(os.path.join(folder, 'FilesAfterPostprocessing.csv')),
        index=False,
    )

    # Pixel columns and the filename are not hourly means. Drop whichever
    # of them this file actually has.
    data = data.drop(columns=[
        col for col in (
            'FileName', 'Frame', 'MainRootLength', 'LateralRootsLength',
            'TotalLength', 'HypocotylLength',
        )
        if col in data.columns
    ])
    
    # Create elapsed time column, in hours
    data['ElapsedTime (h)'] = ((data['Date'] - data['Date'][0]).dt.total_seconds() / 3600).round(2)
    
    # Create NewDay column
    data['NewDay'] = (data['Date'].dt.hour == 0) & (data['Date'].dt.minute == 0)

    data.to_csv(os.path.abspath(os.path.join(folder, 'PostProcess_Original.csv')), index=False)

    # Downsample to hourly data
    data = data.set_index('Date')
    data.index.name = 'Date'
    
    # --- USE DYNAMIC ALIAS HERE ---
    reference_timestamp = data.index[0].floor(FREQ_HOUR)
    
    # --- USE DYNAMIC ALIAS HERE (e.g., '60min' or '60T') ---
    # Hours with no photo are NaN. Do not repeat the last measurement to
    # fill out the processing limit.
    hour_data = data.resample(f'60{FREQ_MIN}', origin=reference_timestamp).mean()

    if N_exp is not None:
        expected_hour_count = expected_hourly_rows(N_exp, timeStep)
        if expected_hour_count is not None and len(hour_data) > expected_hour_count:
            hour_data = hour_data.iloc[:expected_hour_count]
    
    data = hour_data.reset_index()
    
    if 'Date' not in data.columns and 'index' in data.columns:
        data = data.rename(columns={'index': 'Date'})
    
    data['NewDay'] = (data['Date'].dt.hour == 0) & (data['Date'].dt.minute == 0)
    data['ElapsedTime (h)'] = elapsed_hours_from_t0(data['Date'], data['Date'].iloc[0])
    data['NumberOfLateralRoots'] = pd.to_numeric(
        data['NumberOfLateralRoots'], errors='coerce',
    ).round(0)

    hour_dates = data['Date']
    hour_gap = pd.Timedelta(hours=1, minutes=30)
    elapsed_hours = pd.to_numeric(data['ElapsedTime (h)'], errors='coerce').to_numpy(dtype=float)

    def _hour_spans(series):
        return _finite_time_spans(hour_dates, np.isfinite(series), hour_gap)

    main_mm = pd.to_numeric(data['MainRootLength (mm)'], errors='coerce').to_numpy(dtype=float)
    lateral_mm = pd.to_numeric(data['LateralRootsLength (mm)'], errors='coerce').to_numpy(dtype=float)
    total_mm = pd.to_numeric(data['TotalLength (mm)'], errors='coerce').to_numpy(dtype=float)
    hypocotyl_mm = pd.to_numeric(data['HypocotylLength (mm)'], errors='coerce').to_numpy(dtype=float)
    n_lateral = pd.to_numeric(data['NumberOfLateralRoots'], errors='coerce').to_numpy(dtype=float)

    mainRootGrad = _gradient_spans(main_mm, elapsed_hours, _hour_spans(main_mm))
    lateralRootsGrad = _gradient_spans(lateral_mm, elapsed_hours, _hour_spans(lateral_mm))
    totalRootsGrad = _gradient_spans(total_mm, elapsed_hours, _hour_spans(total_mm))
    hypocotylGrad = _gradient_spans(hypocotyl_mm, elapsed_hours, _hour_spans(hypocotyl_mm))

    # A missing length stays a missing ratio. A real zero length keeps the
    # old fallback (100% main, density 0) because that photo was measured.
    mainOverTotal = np.full(len(data), np.nan, dtype=float)
    both = np.isfinite(total_mm) & np.isfinite(main_mm)
    positive = both & (total_mm > 0)
    mainOverTotal[positive] = main_mm[positive] / total_mm[positive] * 100.0
    mainOverTotal[both & ~positive] = 100.0

    lateralDensity = np.full(len(data), np.nan, dtype=float)
    both = np.isfinite(lateral_mm) & np.isfinite(main_mm)
    positive = both & (main_mm > 0)
    lateralDensity[positive] = lateral_mm[positive] / main_mm[positive]
    lateralDensity[both & ~positive] = 0.0

    discreteLateralDensity = np.full(len(data), np.nan, dtype=float)
    both = np.isfinite(n_lateral) & np.isfinite(main_mm)
    positive = both & (main_mm > 0)
    discreteLateralDensity[positive] = 10.0 * n_lateral[positive] / main_mm[positive]
    discreteLateralDensity[both & ~positive] = 0.0

    mainOverTotal = _median_filter_spans(mainOverTotal, _hour_spans(mainOverTotal), 5)
    lateralDensity = _median_filter_spans(lateralDensity, _hour_spans(lateralDensity), 5)
    discreteLateralDensity = _median_filter_spans(
        discreteLateralDensity, _hour_spans(discreteLateralDensity), 5,
    )

    # Add calculated columns
    data['MainRootLengthGrad (mm/h)'] = mainRootGrad
    data['LateralRootsLengthGrad (mm/h)'] = lateralRootsGrad
    data['TotalLengthGrad (mm/h)'] = totalRootsGrad
    data['HypocotylLengthGrad (mm/h)'] = hypocotylGrad
    data['MainOverTotal (%)'] = mainOverTotal
    data['LateralDensity (mm/mm)'] = lateralDensity
    data['DiscreteLateralDensity (LR/cm)'] = discreteLateralDensity
    
    data.to_csv(os.path.abspath(os.path.join(folder, 'PostProcess_Hour.csv')), index=False)
    
    return data