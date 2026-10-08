"""
Lateral Root Angle Analysis Module

This module provides functions for analyzing lateral root angles from RSML 
(Root System Markup Language) files. It includes functionality for:
- Parsing RSML files and extracting root geometry
- Calculating tip angles and emergence angles
- Tracking lateral roots across time series
- Statistical analysis and visualization of angle data

Dependencies:
    - xml.etree.ElementTree: XML parsing for RSML files
    - matplotlib/seaborn: Visualization
    - numpy/pandas: Data manipulation
    - scipy.stats: Statistical tests
    - cv2: Image processing
"""

import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
import os
import csv
import pandas as pd
from .utils import report_utils as utils
import scipy.stats as stats
from .utils.fileUtilities import (
    convertFromPathSafe,
    convertToPathSafe,
    get_latest_result_dir,
    load_result_metadata,
    attach_plant_metadata_columns,
    build_plant_id,
)
from .utils.metadata_schema import get_bounding_box, video_image_dir
from .time_windows import (
    collect_hourly_data,
    default_duration_hours,
    elapsed_hours_from_t0,
    match_time_group,
    parse_datetime,
    resolved_time_groups,
    snapshot_hours as conf_snapshot_hours,
)
from .plot_time_axis import draw_snapshot_day_axis
from .utils.report_paths import (
    MODULE_ANGLES,
    angle_overlays_dir,
    data_file,
    plot_file,
    table_file,
    metric_dir,
)
from .stats_utils import (
    perform_interval_pairwise_stats,
    ensure_factor_columns,
    comparison_modes_for_run,
    _mode_spec,
    _describe_averaging,
    run_pairwise_comparisons,
    _resolve_stats_path,
    _stratify_label,
)
from .report_plots import emit_interval_comparison_plots
from .utils.report_style import genotype_palette_for_data, get_genotype_axis_label
import cv2
import logging

# Suppress matplotlib category warnings and configure backend for non-interactive use
logging.getLogger('matplotlib.category').setLevel(logging.ERROR)
plt.switch_backend('agg')
pd.options.mode.chained_assignment = None


def traverse_root(root_element, is_lateral_root=False):
    """
    Extract coordinate points from a root element in the RSML structure.
    
    Traverses the XML element representing a root and extracts all x,y coordinates
    of the polyline defining the root shape. Points are ordered from base to tip
    (ascending y-coordinate).
    
    Args:
        root_element: XML element containing the root geometry data
        is_lateral_root: If True, access the first child element first 
                        (lateral roots have an extra nesting level)
    
    Returns:
        list: List of [x, y] coordinate pairs ordered from base to tip
    """
    if is_lateral_root:
        root_element = root_element[0]

    # Extract all coordinate points from the polyline
    points = [[int(point.attrib['x']), int(point.attrib['y'])] 
              for point in root_element[0]]
  
    start_point = points[0]
    end_point = points[-1]

    # Ensure points are ordered from base (lower y) to tip (higher y)
    # If start has higher y than end, reverse the list
    if start_point[1] > end_point[1]:
        return points[::-1]
    else:
        return points


def tipAngle(points):
    """
    Calculate the tip angle of a root from base to tip.
    
    The tip angle is measured as the angle between the vertical axis and
    the line connecting the first point (base) to the last point (tip).
    An angle of 0 degrees means the root grows straight down (vertical).
    
    Args:
        points: List of [x, y] coordinate pairs defining the root polyline
    
    Returns:
        float: Tip angle in degrees (0 = vertical, 90 = horizontal)
    """
    start = points[0]
    end = points[-1]
    
    # Calculate hypotenuse (Euclidean distance between start and end)
    hypotenuse = np.sqrt((start[0] - end[0])**2 + (start[1] - end[1])**2)
    
    # Calculate vertical distance (adjacent side for angle calculation)
    vertical_distance = end[1] - start[1]

    # Calculate angle using arccos (angle from vertical)
    angle_rad = np.arccos(vertical_distance / hypotenuse)
    angle_deg = angle_rad * 180 / np.pi

    return angle_deg


def emergenceAngle(points, distance_cm=1, pixel_size=0.04):
    """
    Calculate the emergence angle of a root over a specified initial distance.
    
    The emergence angle captures the initial growth direction of the root,
    measured from the base to a point at a specified physical distance.
    This is more representative of the root's initial trajectory than the tip angle.
    
    Args:
        points: List of [x, y] coordinate pairs defining the root polyline
        distance_cm: Physical distance in cm to measure the angle over
        pixel_size: Size of each pixel in cm (for converting pixels to physical units)
    
    Returns:
        float: Emergence angle in degrees (0 = vertical, 90 = horizontal)
    """
    # Convert physical distance to pixel distance
    distance_pixels = int(distance_cm / pixel_size)
    num_points = len(points)
    
    start = points[0]
    # Use the point at the specified distance, or the last point if root is shorter
    end = points[min(distance_pixels, num_points) - 1]

    # Calculate hypotenuse (Euclidean distance)
    hypotenuse = np.sqrt((start[0] - end[0])**2 + (start[1] - end[1])**2)
    
    # Calculate vertical distance
    vertical_distance = end[1] - start[1]

    # Calculate angle using arccos
    angle_rad = np.arccos(vertical_distance / hypotenuse)
    angle_deg = angle_rad * 180 / np.pi

    return angle_deg


def plotAngles(ax, points, angle, tip=True, distance_cm=1, pixel_size=0.04):
    """
    Plot angle visualization lines on a matplotlib axes.
    
    Draws the reference lines used for angle measurement: the line between
    start and end points, and a vertical reference line. Also annotates
    the calculated angle value.
    
    Args:
        ax: Matplotlib axes object to draw on
        points: List of [x, y] coordinate pairs defining the root
        angle: Pre-calculated angle value to display
        tip: If True, measure to the tip; if False, measure to emergence distance
        distance_cm: Physical distance for emergence angle measurement
        pixel_size: Size of each pixel in cm
    
    Returns:
        None (modifies ax in place)
    """
    if tip:
        start = points[0]
        end = points[-1]
    else:
        distance_pixels = int(distance_cm / pixel_size)
        num_points = len(points)
        start = points[0]
        end = points[min(distance_pixels, num_points) - 1]

    # Draw line from start to end point
    xs = start[0], end[0]
    ys = start[1], end[1]
    ax.plot(xs, ys, linewidth=0.5)

    # Draw vertical reference line
    ax.plot([start[0], start[0]], ys, linewidth=0.5)

    # Annotate with angle value
    ax.text(x=start[0], y=start[1], s=str(round(angle, 2)), fontsize='xx-small')

    return


def find_nearest(point, point_list):
    """
    Find the nearest point in a list to a given reference point.
    
    Uses Euclidean distance to find the closest match. Useful for
    tracking roots across frames by matching starting points.
    
    Args:
        point: Reference point as numpy array [x, y]
        point_list: Array of points to search, shape (N, 2)
    
    Returns:
        tuple: (index of nearest point, distance to nearest point)
    """
    distances = np.linalg.norm(point - point_list, axis=1)
    nearest_index = np.argmin(distances)

    return nearest_index, distances[nearest_index]


def matching(new_starts, all_new_points, old_starts, all_old_points, old_names, num_roots=0):
    """
    Match lateral roots between consecutive frames for tracking.
    
    Implements a simple tracking algorithm that matches roots based on the
    proximity of their starting points. Roots within 20 pixels of a previous
    root's start point are considered the same root. New roots are assigned
    new names, and disappeared roots are preserved in the list.
    
    Args:
        new_starts: List of starting points for roots in current frame
        all_new_points: List of full polyline points for each root in current frame
        old_starts: List of starting points from previous frame
        all_old_points: List of full polyline points from previous frame
        old_names: List of root names from previous frame
        num_roots: Counter for total number of unique roots seen
    
    Returns:
        tuple: (matched_starts, matched_points, matched_names, updated_num_roots)
    """
    matched_names = []
    matched_starts = []
    matched_points = []
    seen_indices = []

    # If no previous data, assign new names to all roots
    if old_starts == [] or old_names == []:
        for j in range(len(new_starts)):
            matched_names.append('LR%s' % num_roots)
            matched_starts.append(new_starts[j])
            matched_points.append(all_new_points[j])
            num_roots += 1
    else:
        old_points_array = np.array(old_starts)
        new_points_array = np.array(new_starts)

        # Match each new root to the nearest old root
        for j in range(new_points_array.shape[0]):
            nearest_idx, distance = find_nearest(new_points_array[j, :], old_points_array)

            # Distance threshold of 20 pixels for matching
            if distance < 20:
                # Match found - use existing name
                matched_names.append(old_names[nearest_idx])
                matched_starts.append(new_starts[j])
                matched_points.append(all_new_points[j])
                seen_indices.append(nearest_idx)
            else:
                # No match - assign new name
                matched_names.append('LR%s' % num_roots)
                matched_starts.append(new_starts[j])
                matched_points.append(all_new_points[j])
                num_roots += 1
        
        # Preserve old roots that weren't matched (may have temporarily disappeared)
        num_old = old_points_array.shape[0]
        seen_indices.sort()

        for j in range(num_old):
            if j not in seen_indices:
                matched_names.append(old_names[j])
                matched_starts.append(old_starts[j])
                matched_points.append(all_old_points[j])

        # Sort all results by name for consistent ordering
        sort_indices = np.argsort(np.array(matched_names))
        matched_names = np.array(matched_names, dtype=object)[sort_indices].tolist()
        matched_starts = np.array(matched_starts, dtype=object)[sort_indices].tolist()
        matched_points = np.array(matched_points, dtype=object)[sort_indices].tolist()

    return matched_starts, matched_points, matched_names, num_roots


def lenRoot(points, pixel_size=0.04):
    """
    Calculate the total length of a root polyline in physical units.
    
    Sums the Euclidean distances between consecutive points and converts
    from pixels to centimeters using the pixel size.
    
    Args:
        points: List of [x, y] coordinate pairs defining the root
        pixel_size: Size of each pixel in cm
    
    Returns:
        float: Total root length in cm
    """
    # Convert to numpy array for vectorized operations
    points_array = np.array(points)
    
    # Calculate differences between consecutive points
    diffs = np.diff(points_array, axis=0)
    
    # Calculate segment lengths and sum
    segment_lengths = np.linalg.norm(diffs, axis=1)
    total_length = np.sum(segment_lengths) * pixel_size
    
    return total_length


def getAngles(conf, path):
    """
    Extract and save angle measurements for all lateral roots in a time series.
    
    Processes all RSML files in the specified path, tracks lateral roots across
    frames, and calculates tip and emergence angles. Results are saved to a CSV file.
    
    Args:
        conf: Configuration dictionary containing 'emergenceDistance' parameter
        path: Path to the analysis results folder containing RSML subfolder
    
    Returns:
        None (writes results to LateralRootsData.csv)
    """
    images = pd.read_csv(os.path.join(path, "FilesAfterPostprocessing.csv"))
    images.dropna(inplace=True)
    images = images['FileName'].tolist()

    rsml_dir = os.path.join(path, "RSML")
    rsml_files = os.listdir(rsml_dir)
    
    # Use basename to match files, avoiding issues with special characters in 'path'
    paths = []
    for image in images:
        rsml_name = os.path.basename(image).replace('.png', '.rsml')
        if rsml_name in rsml_files:
            paths.append(os.path.join(rsml_dir, rsml_name))

    lateral_roots = []
    lateral_root_starts = []
    lateral_root_names = []
    num_roots = 0
    
    filepath = os.path.join(path, 'LateralRootsData.csv')

    metadata = load_result_metadata(path)
    pixel_size = metadata['pixel_size']
    
    with open(filepath, 'w+') as f:
        writer = csv.writer(f)
        writer.writerow(["Img", "Number of lateral roots", "Mean tip angle", 
                        "Mean emergence angle", "First LR tip", "First LR emergence"])

        for step in paths:
            tree = ET.parse(step).getroot()

            plant = tree[1][0][0]
            
            # Check if lateral roots exist (more than just primary root)
            if len(plant) > 1:
                lateral_root_elements = plant[1:]
                
                points_list = []
                start_points = []

                # First: perform tracking of lateral roots
                for lr_element in lateral_root_elements:
                    points = traverse_root(lr_element, is_lateral_root=True)
                    points_list.append(points)
                    start_points.append(points[0])

                lateral_root_starts, lateral_roots, lateral_root_names, num_roots = matching(
                    start_points, points_list, 
                    lateral_root_starts, lateral_roots, lateral_root_names, 
                    num_roots
                )
                
                # Second: estimate angles for all tracked roots
                tip_angles = []
                emergence_angles = []

                for root in lateral_roots:
                    tip_angle = tipAngle(root)
                    emergence_angle = emergenceAngle(
                        root, 
                        float(conf['emergenceDistance']), 
                        pixel_size
                    )
                    tip_angles.append(tip_angle)
                    emergence_angles.append(emergence_angle)

            # Robustly extract the image name for the CSV column
            imgname = os.path.basename(step).replace('.rsml', '.png')
            
            if num_roots == 0:
                measures = [imgname, 0, 0, 0, 0, 0]
            else:
                measures = [imgname, num_roots, round(np.mean(tip_angles), 3), 
                            round(np.mean(emergence_angles), 3), 
                            round(tip_angles[0], 3), round(emergence_angles[0], 3)]
            writer.writerow(measures)
    
    return


def dataWork(df, t0, start=None, end=None):
    """Process angle rows onto elapsed hours relative to t0."""
    datetime_strings = df['Img'].str.extract(r'(\d{4})-(\d{2})-(\d{2})_(\d{2})-(\d{2})')
    datetime_strings.columns = ['Year', 'Month', 'Day', 'Hour', 'Minute']
    datetime_strings = datetime_strings.astype(int)

    df = df.drop(['Img'], axis=1)
    df['Date'] = pd.to_datetime(datetime_strings[['Year', 'Month', 'Day', 'Hour', 'Minute']])
    t0 = parse_datetime(t0)
    start = parse_datetime(start)
    end = parse_datetime(end)
    if t0 is None:
        t0 = df['Date'].min()
    if start is not None:
        df = df[df['Date'] >= start]
    if end is not None:
        df = df[df['Date'] <= end]
    if df.empty:
        return pd.DataFrame()

    elapsed = elapsed_hours_from_t0(df['Date'], t0)
    df = df.loc[elapsed.notna()].copy()
    if df.empty:
        return pd.DataFrame()
    df['ElapsedTime (h)'] = elapsed.loc[df.index].astype(int)
    df['Day'] = df['ElapsedTime (h)']
    df['Hour'] = df['Date'].dt.hour
    df = df.drop(['Date'], axis=1)
    df = df.groupby(['ElapsedTime (h)', 'Day', 'Hour']).mean(numeric_only=True).reset_index()
    return df


def avoidIncreasingValues(data, metric, tolerance=0.3):
    """
    Apply smoothing to prevent unrealistic jumps in time series data.
    
    First applies a median filter to remove spikes, then enforces that values
    cannot increase by more than the specified tolerance (30% by default).
    This corrects for tracking errors that might cause angle values to jump.
    
    Args:
        data: DataFrame containing the time series
        metric: Column name to process
        tolerance: Maximum allowed relative increase between consecutive values
    
    Returns:
        DataFrame: Data with smoothed metric column
    """
    # Apply median filter to remove spikes
    data[metric] = data[metric].rolling(window=5, min_periods=1).median()
    
    # Enforce maximum increase constraint (skip initial transient period)
    series = data[metric]
    for j in range(12, len(series)):
        if series.iloc[j] < series.iloc[j-1] * (1 + tolerance):
            continue
        else:
            series.iloc[j] = series.iloc[j-1]
    data[metric] = series
    
    return data


def getFirstLateralRoots(conf, df):
    """
    Extract and synchronize first lateral root data across all plants.
    
    For each plant, finds when the first lateral root appeared and tracks
    its tip angle over a 72-hour window. Data is synchronized so that
    time=0 corresponds to first LR emergence for each plant.
    """
    columns = ['Time', 'First LR tip', 'Experiment', 'Plant_id', 'Real']
    empty = pd.DataFrame(columns=columns)
    output_path = data_file(conf, 'Synchronized_FirstLR.csv')
    if df is None or df.empty or 'First LR tip' not in df.columns or 'Plant_id' not in df.columns:
        empty.to_csv(output_path, index=False)
        return empty

    df = df.copy()
    df['Real'] = 1
    plant_key = df['Plant_id'].astype(str)

    max_hours = 72
    synchronized_data = empty.copy()

    for plant in df['Plant_id'].unique():
        plant_data = df.loc[plant_key == str(plant)].copy()
        plant_data = plant_data.loc[plant_data['First LR tip'] > 0]
        num_rows = plant_data.shape[0]

        if num_rows > 0:
            plant_data = avoidIncreasingValues(plant_data, 'First LR tip')
            plant_data = plant_data.iloc[:(max_hours + 1), :]

            while num_rows <= max_hours:
                last_row = plant_data.iloc[-1:].copy()
                last_row['Real'] = np.nan
                plant_data = pd.concat([plant_data, last_row], ignore_index=True)
                num_rows += 1

            plant_data = plant_data.reset_index(drop=True)
            plant_data['Time'] = plant_data.index
            if 'Experiment' not in plant_data.columns:
                plant_data['Experiment'] = ''
            plant_data = plant_data[columns]
            synchronized_data = pd.concat([synchronized_data, plant_data], ignore_index=True)

    synchronized_data.to_csv(output_path, index=False)
    return synchronized_data


def makeLateralAnglesPlots(conf):
    """
    Generate all lateral root angle analysis plots and statistics.
    
    Main entry point for angle analysis. Loads data from all experiments,
    generates violin plots for emergence angles, tracks first lateral roots,
    and performs statistical comparisons between experiments.
    
    Args:
        conf: Configuration dictionary containing:
              - 'MainFolder': Root path for analysis
              - 'daysAngles': Comma-separated days to analyze
              - 'everyXhourFieldAngles': Interval for statistical analysis
              - 'averagePerPlantStats': Whether to average per plant before stats
    
    Returns:
        None (generates plots and CSV files in Report folder)
    """
    parent_folder = conf['MainFolder']
    analysis = os.path.join(conf['MainFolder'], "Analysis")
    experiments = utils.load_paths(analysis, '*')

    emergence_slug = 'mean_emergence_angle'
    first_lr_slug = 'first_lr_tip_angle'

    all_data = pd.DataFrame()
    groups, _mode = resolved_time_groups(conf, None)
    if not groups:
        groups, _mode = resolved_time_groups(conf, collect_hourly_data(parent_folder))
    hours = conf_snapshot_hours(conf, kind='angles')
    duration = conf.get('timeDurationHours')
    if duration in (None, '') and groups:
        duration = default_duration_hours(groups)
        
    for exp in experiments:
        plants = utils.load_paths(exp, '*/*/*')
        # Use basename to get the literal folder name for logic
        raw_exp_folder = os.path.basename(exp)
        
        # Determine the name for the 'Experiment' column (the display name)
        # We check the first plant's metadata for the preferred name
        display_name = convertFromPathSafe(raw_exp_folder)

        for plant in plants:
            results = get_latest_result_dir(plant)
            if results is None:
                continue

            plant_name = plant.replace(exp, '').replace('/', '_')
            file = os.path.join(results, 'LateralRootsData.csv')
            file2 = os.path.join(results, 'PostProcess_Original.csv')

            if not os.path.exists(file) or not os.path.exists(file2):
                continue

            meta = load_result_metadata(results)
            data2 = pd.read_csv(file2)
            data2.dropna(inplace=True)
            
            date1 = pd.to_datetime(data2.loc[0, "Date"], dayfirst=False)
            date2 = pd.to_datetime(data2.iloc[-1]["Date"], dayfirst=False)
            group = match_time_group(groups, date1, date2) if groups else None
            t0 = parse_datetime(group.get('t0')) if group else date1
            start = parse_datetime(group.get('start')) if group else date1
            end = None
            if start is not None and duration not in (None, ''):
                end = start + pd.Timedelta(hours=float(duration))
            elif group:
                end = parse_datetime(group.get('spanEnd')) or date2
            else:
                end = date2
                        
            data = pd.read_csv(file)
            data = dataWork(data, t0, start, end)

            if data.empty:
                continue

            hardware_id = build_plant_id(
                str(meta.get('rpi', '')),
                f"cam_{meta.get('cam', '')}",
                f"plant_{meta.get('plant', '')}",
            )
            data['Plant_id'] = display_name + plant_name if not hardware_id.strip('_') else hardware_id
            data = attach_plant_metadata_columns(
                data, meta, experiment_fallback=display_name,
            )

            all_data = pd.concat([all_data, data], ignore_index=True)

    all_data.to_csv(data_file(conf, 'LateralRoots_Data.csv'), index=False)
    # Filter data for specified analysis days
    frame = []
    if not all_data.empty:
        if 'ElapsedTime (h)' not in all_data.columns and 'Day' in all_data.columns:
            all_data['ElapsedTime (h)'] = all_data['Day']
        snapshot = hours
        elapsed = pd.to_numeric(all_data['ElapsedTime (h)'], errors='coerce')
        for hour in snapshot:
            # Only plants imaged at this hour. A missing hour is not filled
            # by the nearest other measurement.
            aux = all_data.loc[elapsed == int(hour)].copy()
            if 'Mean emergence angle' not in aux.columns or aux.empty:
                continue
            aux = aux[pd.to_numeric(aux['Mean emergence angle'], errors='coerce').notna()]
            aux = aux[aux['Mean emergence angle'] > 0]
            if not aux.empty:
                frame.append(aux)
    
    # Generate emergence angle plots
    if len(frame) > 0:
        frame = pd.concat(frame)
        if not frame.empty:
            n_exp = len(frame['Experiment'].unique())
            geno_palette = genotype_palette_for_data(frame)
            geno_label = get_genotype_axis_label(conf)

            fig, ax = plt.subplots(figsize=(8, 9), dpi=200, constrained_layout=True)
            sns.violinplot(
                x='ElapsedTime (h)', y='Mean emergence angle', data=frame,
                hue='Experiment', inner=None, zorder=1, legend=False,
                palette=geno_palette, ax=ax,
            )
            sns.swarmplot(
                x='ElapsedTime (h)', y='Mean emergence angle', data=frame,
                hue='Experiment', dodge=True, size=4, palette=geno_palette,
                edgecolor='black', linewidth=0.5, zorder=3, ax=ax,
            )
            ax.set_ylim(-20, 120)
            
            # Fix legend to only show experiments once
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles[0:n_exp], labels[0:n_exp], loc=4, title=geno_label)

            ax.set_title('Mean emergence angle — by genotype')
            draw_snapshot_day_axis(ax)

            plt.savefig(plot_file(conf, MODULE_ANGLES, emergence_slug, f'{emergence_slug}_violin.png'),
                       dpi=300, bbox_inches='tight')
            plt.savefig(plot_file(conf, MODULE_ANGLES, emergence_slug, f'{emergence_slug}_violin.svg'),
                       dpi=300, bbox_inches='tight')
            
            # Perform statistical analysis
            performStatisticalAnalysisAngles(conf, frame, 'Mean emergence angle')

            # Generate summary table
            summary_data = frame.groupby(['ElapsedTime (h)', 'Experiment']).agg(
                {'Mean emergence angle': ['count', 'mean', 'std']}
            )
            summary_data.columns = [' '.join(col).strip() for col in summary_data.columns.values]
            summary_data = summary_data.reset_index()
            summary_data.columns = ['ElapsedTime (h)', 'Experiment', 'N Samples', 
                                   'Mean Emergence Angle (Mean)', 'Mean Emergence Angle (std)']
            summary_data = summary_data.round(3)
            summary_data = summary_data.sort_values(by='ElapsedTime (h)', ascending=True)
            summary_data.to_csv(table_file(conf, MODULE_ANGLES, emergence_slug, 'summary_table.csv'),
                               index=False)
    
    # Generate first lateral root analysis
    synchronized_data = getFirstLateralRoots(conf, all_data)
    
    if synchronized_data.empty:
        print("Warning: No data available for First Lateral Root plots. Skipping.")
    else:
        # Plot number of plants with first LR over time
        plt.figure(figsize=(6, 4), dpi=200)
        geno_palette = genotype_palette_for_data(synchronized_data)
        geno_label = get_genotype_axis_label(conf)
        sns.lineplot(y="Real", x='Time', hue="Experiment", data=synchronized_data,
                    errorbar=None, estimator=np.count_nonzero, palette=geno_palette)
        plt.title('First lateral root — number of plants per experiment')
        plt.ylabel('Number of first LR')
        plt.xlabel('Time elapsed since emergence (h)')
        plt.legend(title=geno_label)
        
        plt.savefig(plot_file(conf, MODULE_ANGLES, first_lr_slug, f'{first_lr_slug}_count.png'),
                   dpi=300, bbox_inches='tight')
        plt.savefig(plot_file(conf, MODULE_ANGLES, first_lr_slug, f'{first_lr_slug}_count.svg'),
                   dpi=300, bbox_inches='tight')
        
        # Remove extended (non-real) data points for angle analysis
        synchronized_data.dropna(inplace=True)

        if not synchronized_data.empty:
            # Plot tip angle decay over time
            plt.figure(figsize=(6, 4), dpi=200)
            sns.lineplot(y="First LR tip", x='Time', hue="Experiment",
                        data=synchronized_data, errorbar='se', palette=geno_palette)
            plt.title('First LR tip angle — decay over time since emergence')
            plt.xlabel('Time (h)')
            plt.ylabel('Angle')
            plt.legend(title=geno_label)

            plt.savefig(plot_file(conf, MODULE_ANGLES, first_lr_slug, f'{first_lr_slug}_decay.png'),
                       dpi=300, bbox_inches='tight')
            plt.savefig(plot_file(conf, MODULE_ANGLES, first_lr_slug, f'{first_lr_slug}_decay.svg'),
                       dpi=300, bbox_inches='tight')

            performStatisticalAnalysisFirstLR(conf, synchronized_data.copy(), 'First LR tip')

            # Generate summary table for first LR tip angles
            summary_df_list = []
            
            data = synchronized_data.copy()
            data['Experiment'] = data['Experiment'].astype(str)
            dt = int(conf['everyXhourFieldAngles'])

            for hour in range(0, 73 - dt, dt):
                end = min(hour + dt, 72)
                hours = np.arange(hour, end)
                subdata = data[data['Time'].isin(hours)]

                if not subdata.empty:
                    subdata["First LR tip"] = subdata["First LR tip"].astype(float)
                    subdata = subdata.groupby(['Experiment', 'Plant_id'])["First LR tip"].mean(
                        numeric_only=True
                    ).reset_index()
                    subdata = subdata.groupby(['Experiment']).agg(
                        {'First LR tip': ['count', 'mean', 'std']}
                    )

                    subdata.columns = [' '.join(col).strip() for col in subdata.columns.values]
                    subdata = subdata.reset_index()
                    subdata['Interval'] = str(hour) + '-' + str(end)
                    
                    subdata.columns = ['Experiment', 'N Samples', 'First LR tip (Mean)', 
                                      'First LR tip (std)', 'Hours interval']
                    subdata = subdata.round(3)
                    summary_df_list.append(subdata)
            
            if len(summary_df_list) > 0:
                summary_df = pd.concat(summary_df_list)
                col = summary_df.pop("Hours interval")
                summary_df.insert(0, col.name, col)
                summary_df.to_csv(table_file(conf, MODULE_ANGLES, first_lr_slug, 'summary_table.csv'),
                                 index=False)
    return


def performStatisticalAnalysisAngles(conf, data, metric):
    data = ensure_factor_columns(data)
    if 'ElapsedTime (h)' not in data.columns and 'Day' in data.columns:
        data['ElapsedTime (h)'] = data['Day']
    data['ElapsedTime (h)'] = data['ElapsedTime (h)'].astype(int).astype(str)
    hours = [str(h) for h in conf_snapshot_hours(conf, kind='angles')]
    slug = 'mean_emergence_angle'
    table_path = table_file(conf, MODULE_ANGLES, slug, 'summary_table.csv')
    perform_interval_pairwise_stats(
        conf, data, metric, output_dir=None,
        interval_col='ElapsedTime (h)',
        intervals=hours,
        plant_id_col='Plant_id',
        interval_label='ElapsedTime (h)',
        module=MODULE_ANGLES,
        metric_slug_name=slug,
        table_file_path=table_path,
    )
    emit_interval_comparison_plots(
        conf, data, metric, metric_dir(conf, MODULE_ANGLES, slug),
        module=MODULE_ANGLES, metric_slug_name=slug,
        metric_label='Mean emergence angle',
        x_col='ElapsedTime (h)',
    )
    return


def performStatisticalAnalysisFirstLR(conf, data, metric):
    data = ensure_factor_columns(data)
    data['Experiment'] = data['Experiment'].astype(str)
    slug = 'first_lr_tip_angle'
    dt = int(conf['everyXhourFieldAngles'])

    for mode in comparison_modes_for_run(conf, data):
        spec = _mode_spec(mode, conf=conf)
        output_path = _resolve_stats_path(
            conf, MODULE_ANGLES, slug, mode,
        )
        with open(output_path, 'w') as f:
            f.write('Using Mann Whitney U test to compare groups\n')
            f.write(f"{spec['header']}\n")
            f.write(f'{_describe_averaging(conf)}\n\n')
            for hour in range(0, 73 - dt, dt):
                end = min(hour + dt, 72)
                f.write(f'Hour: {hour}-{end}\n')
                hours = np.arange(hour, end)
                subdata = data[data['Time'].isin(hours)].copy()
                if conf.get('averagePerPlantStats', False):
                    subdata[metric] = subdata[metric].astype(float)
                    cols = ['Experiment', 'Plant_id']
                    if spec['stratify_col'] and spec['stratify_col'] in subdata.columns:
                        cols = [spec['stratify_col']] + cols
                    subdata = subdata.groupby(cols).mean(numeric_only=True).reset_index()
                run_pairwise_comparisons(
                    f, subdata, metric,
                    group_col=spec['group_col'],
                    stratify_col=spec['stratify_col'],
                    stratify_label=_stratify_label(spec, conf),
                    conf=conf,
                    plant_id_col='Plant_id',
                )
                f.write('\n')

    emit_interval_comparison_plots(
        conf, data, metric, metric_dir(conf, MODULE_ANGLES, slug),
        module=MODULE_ANGLES, metric_slug_name=slug,
        x_col='Time', metric_label='First LR tip angle',
    )
    return


def estimateAngles(path, ax, img, i=-1, tip=False):
    """
    Estimate and visualize lateral root angles on an image.
    
    Parses the RSML file, calculates angles for all lateral roots,
    and plots the angle measurement lines on the provided axes.
    
    Args:
        path: Path to the analysis results folder
        ax: Matplotlib axes to draw on
        img: Image array to display as background
        i: Index of the RSML file to use (-1 = last file)
        tip: If True, show tip angles; if False, show emergence angles
    
    Returns:
        ax: Modified matplotlib axes with angle visualizations
    """
    plt.ioff()
    paths = utils.load_paths(os.path.join(path, 'RSML'))

    # Initialize tracking variables
    lateral_roots = []
    lateral_root_starts = []
    lateral_root_names = []
    num_roots = 0

    step = paths[i]
    tree = ET.parse(step).getroot()

    with open(os.path.join(path, 'metadata.json')) as f:
        metadata = json.load(f)

    y1, y2, x1, x2 = get_bounding_box(metadata)
    h = y2 - y1
    w = x2 - x1

    plant = tree[1][0][0]
    
    # Check if lateral roots exist
    if len(plant) > 1:
        lateral_root_elements = plant[1:]
        
        points_list = []
        start_points = []

        # First: perform tracking
        for lr_element in lateral_root_elements:
            points = traverse_root(lr_element, is_lateral_root=True)
            points_list.append(points)
            start_points.append(points[0])

        lateral_root_starts, lateral_roots, lateral_root_names, num_roots = matching(
            start_points, points_list, 
            lateral_root_starts, lateral_roots, lateral_root_names, 
            num_roots
        )
        
        # Second: estimate angles
        tip_angles = []
        emergence_angles = []

        for root in lateral_roots:
            tip_angle = tipAngle(root)
            emergence_angle = emergenceAngle(root, 2, metadata['pixel_size'])
            tip_angles.append(tip_angle)
            emergence_angles.append(emergence_angle)

        # Display image and overlay angle measurements
        ax.imshow(img)
        
        for i in range(len(points_list)):
            if not tip:
                plotAngles(ax, points_list[i], emergence_angles[i], tip=tip)
            else:
                plotAngles(ax, points_list[i], tip_angles[i], tip=tip)
        ax.axis('off')
    
    return ax


def plotLateralAnglesOnTop(conf):
    """
    Generate angle visualization overlays for all plants.
    
    Creates images showing the emergence and tip angle measurements
    overlaid on the original root images for visual verification.
    
    Args:
        conf: Configuration dictionary with 'MainFolder' path
    
    Returns:
        None (saves PNG and SVG images to Report folder)
    """
    exp_path = os.path.join(conf['MainFolder'], "Analysis")

    for exp in os.listdir(exp_path):
        safe_exp = convertToPathSafe(exp)
        save_path = angle_overlays_dir(conf, safe_exp)
        robot_path = os.path.join(exp_path, exp)

        for robot in os.listdir(robot_path):
            safe_robot = convertToPathSafe(robot)
            cam_path = os.path.join(robot_path, robot)

            for cam in os.listdir(cam_path):
                safe_cam = convertToPathSafe(cam)
                plant_path = os.path.join(cam_path, cam)

                for plant in os.listdir(plant_path):
                    safe_plant = convertToPathSafe(plant)
                    results_path = get_latest_result_dir(os.path.join(plant_path, plant))
                    if results_path is None:
                        continue
                    
                    if os.path.exists(results_path):
                        metadata = load_result_metadata(results_path)
                        
                        i = -1

                        # Load list of postprocessed images
                        images = pd.read_csv(os.path.join(results_path, "FilesAfterPostprocessing.csv"))
                        images.dropna(inplace=True)
                        images = images["FileName"].tolist()

                        rsml_files = os.listdir(os.path.join(results_path, "RSML"))

                        # Filter to images with corresponding RSML
                        images = [image for image in images 
                                 if image.split('/')[-1].replace('.png', '.rsml') in rsml_files]
                        
                        video_dir = video_image_dir(metadata) or metadata.get("folder")
                        images = [os.path.join(video_dir, image) for image in images]

                        # Load and crop the image
                        bbox = get_bounding_box(metadata)
                        crop = cv2.imread(images[i])[bbox[0]:bbox[1], bbox[2]:bbox[3]]
                        
                        # Save cropped image
                        save = exp + "_" + robot + "_" + cam + "_" + plant + "_crop.png"
                        cv2.imwrite(os.path.join(save_path, save), crop)

                        # Generate emergence angles visualization
                        fig, ax = plt.subplots(figsize=(16, 8), dpi=200)
                        estimateAngles(results_path, ax, crop.copy(), i)
                        plt.title("Emergence Angles")

                        save = f"{safe_exp}_{safe_robot}_{safe_cam}_{safe_plant}_emergence_angles.svg"
                        plt.savefig(os.path.join(save_path, save), bbox_inches="tight", dpi=200)

                        plt.close()
                        plt.clf()
                        plt.cla()

                        # Generate tip angles visualization
                        fig, ax = plt.subplots(figsize=(16, 8), dpi=200)
                        estimateAngles(results_path, ax, crop.copy(), i, True)
                        plt.title("Tip Angles")

                        filename = f"{safe_exp}_{safe_robot}_{safe_cam}_{safe_plant}_tip_angles.svg"
                        plt.savefig(os.path.join(save_path, filename), bbox_inches="tight", dpi=200)
                        
                        plt.close()
                        plt.clf()
                        plt.cla()