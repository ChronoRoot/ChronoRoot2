""" 
ChronoRoot: High-throughput phenotyping by deep learning reveals novel temporal parameters of plant root system architecture
Copyright (C) 2020 Nicolás Gaggion

Modified version with improved time handling for irregular sampling
"""

import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import re
import pandas as pd
import os
from scipy import signal
import json
import warnings

from .time_windows import elapsed_hours_from_t0
from .utils.fileUtilities import expected_hourly_rows

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
    shape = data.shape
    N = shape[0]
    
    # Check for required columns
    required_cols = ['FileName', 'MainRootLength', 'LateralRootsLength', 'NumberOfLateralRoots']
    missing_cols = [col for col in required_cols if col not in data.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # If there is no HypocotylLength column, add it with zeros
    if 'HypocotylLength' not in data.columns:
        data['HypocotylLength'] = 0.0
    
    # Read info from the filename
    dates = []
    for i in range(0,N):
        name = data['FileName'][i]
        nums = re.findall(r'\d+', name)
        if len(nums) < 5:
            warnings.warn(f"Cannot parse date from filename: {name}")
            continue
        date = nums[0] + '-' + nums[1] + '-' + nums[2] + '-' + nums[3] + ':' + nums[4] 
        dates.append(date)

    data.insert(data.shape[1], 'Date', dates)
    data['Date'] = pd.to_datetime(data['Date'], format='%Y-%m-%d-%H:%M')
    
    # Sort by date first
    data = data.sort_values(by=['Date'])
    data = data.reset_index(drop=True)
    
    # Smart time handling with tolerance
    timeStep = conf['timeStep']
    timeStep_td = pd.Timedelta(minutes=timeStep)
        
    # Find and fill only REAL gaps (larger than timeStep * 2)
    filled_data = []
    filled_data.append(data.iloc[0].to_dict())  # Add first row
    
    for i in range(1, len(data)):
        current_row = data.iloc[i]
        prev_row = filled_data[-1]
        
        time_gap = current_row['Date'] - prev_row['Date']
        expected_gaps = int(time_gap / timeStep_td)
        
        # Only fill if there's a gap larger than 2x timeStep
        if expected_gaps >= 2:
            # Fill missing timepoints
            for j in range(1, expected_gaps):
                interpolated_date = prev_row['Date'] + j * timeStep_td
                interpolated_row = prev_row.copy()
                interpolated_row['Date'] = interpolated_date
                interpolated_row['FileName'] = f"INTERPOLATED_{j}"  # Mark as interpolated
                filled_data.append(interpolated_row)
        
        filled_data.append(current_row.to_dict())
    
    # Convert back to DataFrame
    data = pd.DataFrame(filled_data)
        
    # Sort again after any adjustments
    data = data.sort_values(by=['Date'])
    data = data.reset_index(drop=True)
    N = len(data)
    
    # Handle duplicate timestamps that might arise from snapping
    for i in range(1, N):
        if data['Date'][i] <= data['Date'][i-1]:
            data.loc[i, 'Date'] = data['Date'][i-1] + timeStep_td
    
    # Trims or expands the data to the desired number of measurements
    if N_exp is not None:
        if N > N_exp:
            data = data.iloc[0:N_exp]
        else:
            # add rows with last data
            for i in range(N, N_exp):
                last_row = data.iloc[-1].to_dict()
                last_row['Date'] = last_row['Date'] + timeStep_td
                data = pd.concat([data, pd.DataFrame([last_row])], ignore_index=True)
    
    # Reads the pixel size
    path = os.path.abspath(os.path.join(folder, 'metadata.json'))
    try:
        with open(path) as f:
            metadata = json.load(f)
        pixel_size = metadata['pixel_size']
    except (FileNotFoundError, KeyError):
        print(f"Warning: Could not read pixel_size from {path}, defaulting to 1.0")
        pixel_size = 1.0
    
    # Beginning of the data processing
    mainRoot = data['MainRootLength'].to_numpy().astype('float')
    lateralRoots = data['LateralRootsLength'].to_numpy().astype('float')
    numlateralRoots = data['NumberOfLateralRoots'].to_numpy().astype('int')
    hypocotylLength = data['HypocotylLength'].to_numpy().astype('float') 

    # Check for NaN or invalid values
    if np.any(np.isnan(mainRoot)):
        warnings.warn(f"NaN values found in MainRootLength at indices: {np.where(np.isnan(mainRoot))[0]}")
    if np.any(np.isnan(lateralRoots)):
        warnings.warn(f"NaN values found in LateralRootsLength at indices: {np.where(np.isnan(lateralRoots))[0]}")
    if np.any(np.isnan(numlateralRoots)):
        warnings.warn(f"NaN values found in NumberOfLateralRoots at indices: {np.where(np.isnan(numlateralRoots))[0]}")
    if np.any(np.isnan(hypocotylLength)):
        warnings.warn(f"NaN values found in HypocotylLength at indices: {np.where(np.isnan(hypocotylLength))[0]}")
        
    # Replace NaN with 0 for processing
    mainRoot = np.nan_to_num(mainRoot, nan=0.0)
    lateralRoots = np.nan_to_num(lateralRoots, nan=0.0)
    numlateralRoots = np.nan_to_num(numlateralRoots, nan=0.0)
    hypocotylLength = np.nan_to_num(hypocotylLength, nan=0.0)

    # Remove spurious lateral roots at the beginning
    space = int((6 * 60) / timeStep)  # 6 hours worth of timeSteps

    for t in range(space, len(numlateralRoots)):
        if t-space >= 0 and t < len(numlateralRoots):
            if numlateralRoots[t-space] == 0 and numlateralRoots[t] == 0:
                lateralRoots[:t] = 0
                numlateralRoots[:t] = 0
            if mainRoot[t-space] == 0 and mainRoot[t] == 0:
                mainRoot[:t] = 0
            if hypocotylLength[t-space] == 0 and hypocotylLength[t] == 0:
                hypocotylLength[:t] = 0
    
    # Smooth
    mainRoot = signal.medfilt(mainRoot, 9) 
    lateralRoots = signal.medfilt(lateralRoots, 9) 
    numlateralRoots = signal.medfilt(numlateralRoots, 9)
    hypocotylLength = signal.medfilt(hypocotylLength, 9)

    # Check that the values never decrease
    for i in range(1, len(mainRoot)):
        dif = mainRoot[i] < mainRoot[i-1]
        if dif:
            mainRoot[i] = mainRoot[i-1]
        
        dif = numlateralRoots[i] < numlateralRoots[i-1]
        if dif and numlateralRoots[i-1] > 0:
            numlateralRoots[i] = numlateralRoots[i-1]

        dif = lateralRoots[i] < lateralRoots[i-1]
        if dif and lateralRoots[i-1] > 0:
            lateralRoots[i] = lateralRoots[i-1]
        
        dif = hypocotylLength[i] < hypocotylLength[i-1]
        if dif and hypocotylLength[i-1] > 0:
            hypocotylLength[i] = hypocotylLength[i-1]

    # Multiply by pixel size
    mainRoot_mm = mainRoot.copy() * pixel_size
    lateralRoots_mm = lateralRoots.copy() * pixel_size
    hypocotyl_mm = hypocotylLength.copy() * pixel_size
    
    data['MainRootLength (mm)'] = mainRoot_mm
    data['LateralRootsLength (mm)'] = lateralRoots_mm
    data['NumberOfLateralRoots'] = numlateralRoots
    data['TotalLength (mm)'] = mainRoot_mm + lateralRoots_mm
    data['HypocotylLength (mm)'] = hypocotyl_mm

    # Save file names (excluding interpolated ones if desired)
    original_files = data[~data['FileName'].str.contains('INTERPOLATED', na=False)]['FileName']
    original_files.to_csv(os.path.abspath(os.path.join(folder, 'FilesAfterPostprocessing.csv')), index=False)

    # Remove original columns
    try:
        data = data.drop(columns=['FileName', 'Frame', 'MainRootLength', 'LateralRootsLength', 'TotalLength', 'HypocotylLength'])
    except:
        try:
            data = data.drop(columns=['FileName', 'MainRootLength', 'LateralRootsLength', 'TotalLength', 'HypocotylLength'])
        except KeyError:
             pass
    
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
    hour_data = data.resample(f'60{FREQ_MIN}', origin=reference_timestamp).mean()
    
    # Handle N_exp for hourly data
    if N_exp is not None:
        expected_hour_count = expected_hourly_rows(N_exp, timeStep)
        if expected_hour_count is None:
            expected_hour_count = len(hour_data)
        
        if len(hour_data) < expected_hour_count:
            missing_hours = expected_hour_count - len(hour_data)
            last_hour = hour_data.index[-1]
            
            # --- USE DYNAMIC ALIAS HERE ---
            new_hours = pd.date_range(start=last_hour + pd.Timedelta(hours=1), 
                                     periods=missing_hours, freq=f'60{FREQ_MIN}')
            new_data = pd.DataFrame(index=new_hours)
            for column in hour_data.columns:
                new_data[column] = hour_data[column].iloc[-1]
            hour_data = pd.concat([hour_data, new_data])
        elif len(hour_data) > expected_hour_count:
            hour_data = hour_data.iloc[:expected_hour_count]
    
    data = hour_data.reset_index()
    
    if 'Date' not in data.columns and 'index' in data.columns:
        data = data.rename(columns={'index': 'Date'})
    
    data['NewDay'] = (data['Date'].dt.hour == 0) & (data['Date'].dt.minute == 0)
    data['ElapsedTime (h)'] = elapsed_hours_from_t0(data['Date'], data['Date'].iloc[0])
    data['NumberOfLateralRoots'] = data['NumberOfLateralRoots'].round(0)

    # Calculate gradients
    mainRootGrad = np.gradient(data['MainRootLength (mm)'].to_numpy(), edge_order=2)
    lateralRootsGrad = np.gradient(data['LateralRootsLength (mm)'].to_numpy(), edge_order=2)
    totalRootsGrad = np.gradient(data['TotalLength (mm)'].to_numpy(), edge_order=2)
    hypocotylGrad = np.gradient(data['HypocotylLength (mm)'].to_numpy(), edge_order=2) 
    
    # Calculate ratios
    total_length = data['TotalLength (mm)'].to_numpy()
    main_length = data['MainRootLength (mm)'].to_numpy()
    
    with np.errstate(divide='ignore', invalid='ignore'):
        mainOverTotal = np.where(total_length > 0, 
                                 main_length / total_length * 100, 
                                 100.0)
    mainOverTotal = signal.medfilt(mainOverTotal, 5)

    with np.errstate(divide='ignore', invalid='ignore'):
        lateralDensity = np.where(main_length > 0,
                                  data['LateralRootsLength (mm)'].to_numpy() / main_length,
                                  0.0)
    lateralDensity = signal.medfilt(lateralDensity, 5)

    with np.errstate(divide='ignore', invalid='ignore'):
        discreteLateralDensity = np.where(main_length > 0,
                                          10 * data['NumberOfLateralRoots'].to_numpy() / main_length,
                                          0.0)
    discreteLateralDensity = signal.medfilt(discreteLateralDensity, 5)

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