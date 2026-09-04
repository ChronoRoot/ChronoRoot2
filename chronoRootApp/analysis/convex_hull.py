import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import numpy as np
import cv2
import networkx as nx
import json
import scipy.stats as stats
import logging
from typing import Tuple, List

# Import shared utilities
from analysis.utils import report_utils as utils
from analysis.utils.fileUtilities import convertFromPathSafe, normalize_factor_value
from analysis.utils.report_style import genotype_palette_for_data, get_genotype_axis_label
from analysis.time_windows import (
    closest_timed_path,
    match_time_group,
    parse_datetime,
    resolved_time_groups,
    snapshot_hours as conf_snapshot_hours,
)
from analysis.utils.report_paths import (
    MODULE_CONVEX,
    metric_slug,
    overview_dir,
    plot_file,
)

# Suppress minor logging and warnings for cleaner output
logging.getLogger('matplotlib.category').setLevel(logging.ERROR)
plt.switch_backend('agg')

# Constants
PADDING_X = 200 # Pixel padding used for alignment
PIXEL_SIZE_MM = 0.04 # Conversion factor

def calculate_atlas_geometry(experiment_paths: List[str]) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """
    Scans all experiments to determine the maximum biological bounding box relative to the seed.
    Returns:
        canvas_shape: (height, width)
        seed_center: (y, x) coordinates for the seed on the new canvas
    """
    max_left = 0
    max_right = 0
    max_depth = 0
    BUFFER = 50 

    for exp_path in experiment_paths:
        # Load all results folders
        result_paths = utils.load_paths(exp_path, '*/*/*/Results*')
        
        for r_path in result_paths:            
            # 1. Load Last Segmentation Image
            seg_folder = os.path.join(r_path, 'Images/Seg/')
            seg_files = utils.load_paths(seg_folder, "*.png")
            if not seg_files: continue
            
            # Read image
            img = cv2.imread(seg_files[-1], 0)
            
            # Apply standard padding for alignment
            img = np.pad(img, ((0,0), (PADDING_X, 0)))
            
            # 2. Load Graph to find tips for rotation
            graph_folder = os.path.join(r_path, 'Graphs/')
            # Try to find corresponding graph file
            graph_filename = os.path.basename(seg_files[-1]).replace('png', 'xml.gz')
            graph_path = os.path.join(graph_folder, graph_filename)
            
            try: 
                g = nx.read_graphml(graph_path)
            except: 
                continue

            # Find root endpoints (Start and Tip)
            end1 = end2 = None
            for n in g.nodes():
                if g.nodes[n]['type'] == 'FTip': 
                    end1 = np.array([g.nodes[n]["pos_x"], g.nodes[n]["pos_y"]])
                if g.nodes[n]['type'] == 'Ini': 
                    end2 = np.array([g.nodes[n]["pos_x"], g.nodes[n]["pos_y"]])
            
            if end1 is None or end2 is None: continue
            
            seed = end1 if end1[1] < end2[1] else end2
            seed[0] += PADDING_X 
            
            # Determine which is the bottom tip (highest Y value)
            root_tip = end1 if end1[1] > end2[1] else end2
            root_tip[0] += PADDING_X

            # Calculate Angle to vertical
            v_vertical = np.array([0, 1])
            v_root = utils.unit_vector(root_tip - seed)   
            angle_rad = utils.angle_between(v_vertical, v_root)
            
            # Determine direction of rotation
            angle_deg = -np.rad2deg(angle_rad) if v_root[0] > 0 else np.rad2deg(angle_rad)
            
            # Rotate Image to straighten root
            rot_center = (float(seed[0]), float(seed[1]))
            rot_mat = cv2.getRotationMatrix2D(rot_center, angle_deg, 1.0)
            
            result = cv2.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv2.INTER_LINEAR)
            _, result = cv2.threshold(result, 10, 255, cv2.THRESH_BINARY)
            
            # 3. Measure Size Relative to Seed
            points = cv2.findNonZero(result)
            if points is None: continue
            
            x, y, w, h = cv2.boundingRect(points)
            
            # Calculate extent relative to seed
            dist_left = seed[0] - x
            dist_right = (x + w) - seed[0]
            dist_down = (y + h) - seed[1]
            
            max_left = max(max_left, dist_left)
            max_right = max(max_right, dist_right)
            max_depth = max(max_depth, dist_down)

    # Calculate final geometry
    center_x = int(max_left + BUFFER)
    center_y = 100 # Fixed top margin
    
    canvas_width = int(center_x + max_right + BUFFER)
    canvas_height = int(center_y + max_depth + BUFFER)
    
    return (canvas_height, canvas_width), (center_y, center_x)


def generate_root_atlases(save_path, snapshot_hours=None, timestep=15, canvas_shape=None, center_coords=None, rotate_root=True, conf=None, days=None):
    """
    Generates the accumulated heatmaps (atlases) and calculates convex hull metrics.
    snapshot_hours are elapsed hours relative to each plant group's t0.
    """
    dest_seed_y, dest_seed_x = center_coords 
    exp_dir_name = os.path.basename(save_path)
    
    real_exp_name = convertFromPathSafe(exp_dir_name)
    sample_meta = utils.load_paths(save_path, '*/*/*/metadata.json')
    if sample_meta:
        try:
            with open(sample_meta[0], 'r') as f:
                real_exp_name = json.load(f).get('Experiment', real_exp_name)
        except: pass
    
    # Initialize Accumulators
    atlas_hull_mask = np.zeros(canvas_shape, dtype='uint8')
    temp_mask = np.zeros(canvas_shape, dtype='uint8')
    atlas_contours_rgb = np.zeros([canvas_shape[0], canvas_shape[1], 3], dtype='uint8')
    atlas_root_density = np.zeros(canvas_shape, dtype='float64')

    result_paths = utils.load_paths(save_path, '*/*/*/Results*')
    hours = list(snapshot_hours if snapshot_hours is not None else (days or []))
    groups, _mode = resolved_time_groups(conf, None) if conf else ([], 'clock')
    if not groups:
        hourly_frames = []
        for r_path in result_paths:
            csv_files = utils.load_paths(r_path, 'PostProcess_Hour.csv')
            if csv_files:
                try:
                    hourly_frames.append(pd.read_csv(csv_files[0]))
                except Exception:
                    continue
        if hourly_frames:
            groups, _mode = resolved_time_groups(conf or {}, pd.concat(hourly_frames, ignore_index=True))

    all_frames_list = []
    set_of_atlases = []

    for hour in hours:
        hour = int(hour)
        
        # Reset Atlases for the new day
        atlas_contours_rgb.fill(255) # White background
        atlas_hull_mask.fill(0)
        temp_mask.fill(0)
        atlas_root_density.fill(0)
        
        # Metrics storage for this day
        metrics = {
            'area_bbox': [], 'area_chull': [], 'aspect_ratio': [],
            'density_lat': [], 'density_tot': [],
            'density_lat_bbox': [], 'density_tot_bbox': [],
            'width': [], 'height': [],
            'plate_condition': [], 'extra_variable': [], 'plant_id': [],
        }
        
        for r_path in result_paths:
            # --- Load Data ---
            meta_path = os.path.join(r_path, 'metadata.json')
            if not os.path.exists(meta_path): continue
            with open(meta_path) as f: metadata = json.load(f)

            csv_files = utils.load_paths(r_path, 'PostProcess_Hour.csv')
            if not csv_files: continue
            df_temporal = pd.read_csv(csv_files[0])

            seg_path = os.path.join(r_path, 'Images/Seg/')
            seg_files = utils.load_paths(seg_path, "*.png")
            if not seg_files:
                continue

            t0 = None
            if 'Date' in df_temporal.columns:
                dates = pd.to_datetime(df_temporal['Date'], errors='coerce').dropna()
                if not dates.empty:
                    group = match_time_group(groups, dates.min(), dates.max()) if groups else None
                    t0 = parse_datetime(group.get('t0')) if group else parse_datetime(dates.min())
            target = (t0 + pd.Timedelta(hours=hour)) if t0 is not None else None
            current_seg_file = closest_timed_path(seg_files, target) if target is not None else None
            if current_seg_file is None:
                imgs_per_day = int(24 * (60 / max(int(timestep or 15), 1)))
                img_idx = int(hour * imgs_per_day / 24) if hour >= 0 else 0
                if img_idx >= len(seg_files):
                    current_seg_file = seg_files[-1]
                else:
                    current_seg_file = seg_files[img_idx]
            
            img = cv2.imread(current_seg_file, 0)
            
            # Load Graph
            graph_folder = os.path.join(r_path, 'Graphs/')
            graph_file = current_seg_file.replace(seg_path, graph_folder).replace('png', 'xml.gz')
            
            try: 
                g = nx.read_graphml(graph_file)
            except: 
                # Fallback to last available graph
                fallback_graphs = utils.load_paths(graph_folder, "*.xml.gz")
                if not fallback_graphs: continue
                g = nx.read_graphml(fallback_graphs[-1])

            # --- Pre-processing (Padding) ---
            img = np.pad(img, ((0,0), (PADDING_X, 0)))
            
            # Find endpoints
            end1 = end2 = None
            for n in g.nodes():
                if g.nodes[n]['type'] == 'FTip': end1 = np.array([g.nodes[n]["pos_x"], g.nodes[n]["pos_y"]])
                if g.nodes[n]['type'] == 'Ini': end2 = np.array([g.nodes[n]["pos_x"], g.nodes[n]["pos_y"]])
            
            if end1 is None or end2 is None: continue
            
            # Determine seed
            seed = end1 if end1[1] < end2[1] else end2
            seed[0] += PADDING_X
            
            # Determine tip
            root_tip = end1 if end1[1] > end2[1] else end2
            root_tip[0] += PADDING_X

            # --- Rotation Logic ---
            angle_deg = 0.0
            if rotate_root:
                v_vertical = np.array([0, 1])
                v_root = utils.unit_vector(root_tip - seed)
                angle_rad = utils.angle_between(v_vertical, v_root)
                angle_deg = -np.rad2deg(angle_rad) if v_root[0] > 0 else np.rad2deg(angle_rad)

            # --- Spatial Transformation ---
            # 1. Rotation Matrix around original seed
            rot_center = (float(seed[0]), float(seed[1]))
            rot_mat = cv2.getRotationMatrix2D(rot_center, angle_deg, 1.0)

            # 2. Translate to align seed with destination center
            rot_mat[0, 2] += (dest_seed_x - seed[0])
            rot_mat[1, 2] += (dest_seed_y - seed[1])

            # 3. Warp to final canvas
            warped_img = cv2.warpAffine(img, rot_mat, (canvas_shape[1], canvas_shape[0]), flags=cv2.INTER_LINEAR)
            _, warped_img = cv2.threshold(warped_img, 10, 255, cv2.THRESH_BINARY)
            
            # --- Hull & Atlas Calculation ---
            contours, _ = cv2.findContours(warped_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours: continue
            
            # Calculate Convex Hulls for all fragments (usually just one main root)
            hulls = [cv2.convexHull(c, False) for c in contours]
            
            temp_mask.fill(0)
            for j in range(len(contours)):
                cv2.drawContours(temp_mask, hulls, j, 1, -1)
                # Draw outline on the RGB map
                cv2.drawContours(atlas_contours_rgb, hulls, j, (255, 0, 0), 3)
            
            atlas_hull_mask += temp_mask
            atlas_root_density += warped_img / 255.0
            
            # --- Metrics Calculation ---
            # Find largest contour for analysis
            contour_sizes = [(cv2.contourArea(c), c) for c in contours]
            biggest_contour = max(contour_sizes, key=lambda x: x[0])[1]
            
            # Check if root is actually connected to the seed area
            is_inside = cv2.pointPolygonTest(biggest_contour, (dest_seed_x, dest_seed_y), False) > 0
            dist_to_seed = cv2.pointPolygonTest(biggest_contour, (dest_seed_x, dest_seed_y), True)
            
            if is_inside or dist_to_seed < 30:
                hull_big = cv2.convexHull(biggest_contour, False)
                _,_,w,h = cv2.boundingRect(hull_big)
                
                area_bbox = w * h * (PIXEL_SIZE_MM**2)
                area_chull = cv2.contourArea(hull_big) * (PIXEL_SIZE_MM**2)
                
                # Match temporal data to the selected frame clock time
                csv_idx = len(df_temporal) - 1
                if 'Date' in df_temporal.columns and target is not None:
                    dates = pd.to_datetime(df_temporal['Date'], errors='coerce')
                    valid = dates.dropna()
                    if not valid.empty:
                        csv_idx = int((valid - target).abs().idxmin())
                elif 'ElapsedTime (h)' in df_temporal.columns:
                    elapsed = pd.to_numeric(df_temporal['ElapsedTime (h)'], errors='coerce')
                    valid = elapsed.dropna()
                    if not valid.empty:
                        csv_idx = int((valid - hour).abs().idxmin())
                if csv_idx >= len(df_temporal):
                    csv_idx = len(df_temporal) - 1
                
                try:
                    total_len = df_temporal['TotalLength (mm)'][csv_idx]
                    lateral_len = df_temporal['LateralRootsLength (mm)'][csv_idx]
                    
                    # Store Metrics
                    metrics['area_bbox'].append(area_bbox)
                    metrics['area_chull'].append(area_chull)
                    metrics['aspect_ratio'].append(h/w if w > 0 else 0)
                    metrics['density_lat'].append(lateral_len/area_chull if area_chull > 0 else 0)
                    metrics['density_tot'].append(total_len/area_chull if area_chull > 0 else 0)
                    metrics['density_lat_bbox'].append(lateral_len/area_bbox if area_bbox > 0 else 0)
                    metrics['density_tot_bbox'].append(total_len/area_bbox if area_bbox > 0 else 0)
                    metrics['width'].append(w * PIXEL_SIZE_MM)
                    metrics['height'].append(h * PIXEL_SIZE_MM)
                    metrics['plate_condition'].append(
                        normalize_factor_value(metadata.get('PlateCondition', ''))
                    )
                    metrics['extra_variable'].append(
                        normalize_factor_value(metadata.get('ExtraVariable', ''))
                    )
                    metrics['plant_id'].append(
                        f"{metadata.get('rpi', '')}_cam_{metadata.get('cam', '')}_plant_{metadata.get('plant', '')}"
                    )
                except Exception as e:
                    pass

        # Construct DataFrame for the day
        day_df = pd.DataFrame({
            'Convex Hull Aspect Ratio': metrics['aspect_ratio'],
            'Lateral Root Area Density': metrics['density_lat'],
            'Lateral Root Area Density BBOX': metrics['density_lat_bbox'],
            'Total Root Area Density': metrics['density_tot'],
            'Total Root Area Density BBOX': metrics['density_tot_bbox'],
            'Convex Hull Area': metrics['area_chull'],
            'Bounding Box Area': metrics['area_bbox'],
            'Convex Hull Width': metrics['width'],
            'Convex Hull Height': metrics['height']
        })
        day_df['ElapsedTime (h)'] = hour
        day_df['Day'] = hour
        day_df['Experiment'] = real_exp_name
        day_df['PlateCondition'] = metrics['plate_condition']
        day_df['ExtraVariable'] = metrics['extra_variable']
        day_df['Plant_id'] = metrics['plant_id']
        
        all_frames_list.append(day_df)
        set_of_atlases.append([atlas_hull_mask.copy(), atlas_contours_rgb.copy(), atlas_root_density.copy()])
    
    final_df = pd.concat(all_frames_list, ignore_index=True) if all_frames_list else pd.DataFrame()
    return set_of_atlases, final_df


def visualize_single_atlas(atlas_hull, atlas_contours, atlas_roots, save_path, exp_dir_name, day=None):
    """Plots the three-panel atlas visualization."""
    readable_name = convertFromPathSafe(exp_dir_name)
    plt.ioff()
    plt.figure(figsize=(9, 4))

    plt.subplot(1,3,1)
    plt.imshow(atlas_roots, cmap='jet', vmin=0, vmax=25)
    plt.title("Accumulated Roots")
    plt.axis('off')

    plt.subplot(1,3,2)
    plt.imshow(atlas_contours, cmap='jet', vmin=0, vmax=25)
    plt.title("Convex Hull Contours")
    plt.axis('off')

    plt.subplot(1,3,3)
    plt.imshow(atlas_hull, alpha=0.6)
    plt.title("Accumulated Convex Hulls")
    plt.axis('off')

    title_suffix = f" - t={day} h" if day is not None else " - Last snapshot"
    full_title = f"{readable_name}{title_suffix}"
    plt.suptitle(full_title)
    
    save_filename = f"{exp_dir_name}_t_{day}.png" if day is not None else f"{exp_dir_name}_Last_Day.png"
    plt.savefig(os.path.join(save_path, save_filename), dpi=300, bbox_inches='tight')
    plt.close('all')


def visualize_combined_atlases(conf):
    """Stacks atlas images vertically for qualitative comparison across days."""
    folder = overview_dir(conf, MODULE_CONVEX)
    images_per_day = {}

    if not os.path.exists(folder):
        return

    for filename in os.listdir(folder):
        if not filename.endswith(('png', 'jpg', 'jpeg')):
            continue
        if "_t_" in filename:
            try:
                day = filename.rsplit('_t_', 1)[1].split('.png')[0]
            except Exception:
                continue
        elif "_Day_" in filename:
            try:
                day = filename.split('_Day_')[1].split('.png')[0]
            except Exception:
                continue
        elif "Last_Day" in filename:
            day = "Last"
        else:
            continue

        images_per_day.setdefault(day, []).append(filename)

    sorted_days = sorted([d for d in images_per_day.keys() if d != "Last"], key=lambda x: int(x))
    if "Last" in images_per_day:
        sorted_days.append("Last")

    for day in sorted_days:
        images = images_per_day[day]
        images.sort(key=utils.natural_key)

        fig, axs = plt.subplots(len(images), 1, figsize=(9, 4 * len(images)))
        if len(images) == 1:
            axs = [axs]

        for i, image_name in enumerate(images):
            image_path = os.path.join(folder, image_name)
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            axs[i].imshow(img)
            axs[i].axis('off')

        plt.subplots_adjust(wspace=0, hspace=0)
        plt.savefig(os.path.join(overview_dir(conf, MODULE_CONVEX), f'combined_atlas_day_{day}.png'), dpi=300, bbox_inches='tight')
        plt.close('all')


def plot_hull_metrics_summary(conf, frame):
    """Generates per-metric violin plots and overview summary CSV."""
    plt.ioff()

    metrics_config = [
        ('Convex Hull Area', 'Convex Hull Area', 'Area (mm²)', False),
        ('Lateral Root Area Density', 'Lateral Roots Area Density', 'LR / area (mm/mm²)', True),
        ('Convex Hull Aspect Ratio', 'Aspect Ratio', 'H/W Ratio', False),
        ('Total Root Area Density', 'Total Root Area Density', 'TR / area (mm/mm²)', True),
        ('Convex Hull Width', 'Width', 'Width (mm)', False),
        ('Convex Hull Height', 'Height', 'Height (mm)', False),
    ]

    geno_palette = genotype_palette_for_data(frame)
    geno_label = get_genotype_axis_label(conf)

    for y_col, title, y_label, filter_zeros in metrics_config:
        slug = metric_slug(y_col)
        plt.figure()
        plot_data = frame.copy()
        if filter_zeros:
            plot_data = plot_data[plot_data[y_col] > 0].reset_index(drop=True)

        sns.violinplot(x='ElapsedTime (h)', y=y_col, data=plot_data, hue='Experiment', inner='quartile',
                       palette=geno_palette)
        plt.title(f'{y_col} — {title}')
        plt.ylabel(y_label)
        leg = plt.gca().get_legend()
        if leg is not None:
            leg.set_title(geno_label)
        plt.savefig(plot_file(conf, MODULE_CONVEX, slug, f'{slug}_violin.png'), dpi=300, bbox_inches='tight')
        plt.close('all')

    numeric_cols = [c for c, *_ in metrics_config]
    summary = frame.groupby(['ElapsedTime (h)', 'Experiment'])[numeric_cols].agg(['mean', 'std']).round(3)
    summary.to_csv(os.path.join(overview_dir(conf, MODULE_CONVEX), 'Summary_table.csv'))


def analyze_hull_statistics(conf, data, metric):
    from .stats_utils import perform_interval_pairwise_stats
    from .report_plots import emit_interval_comparison_plots
    from .utils.report_paths import metric_dir, table_file

    data = data.copy()
    if 'ElapsedTime (h)' not in data.columns and 'Day' in data.columns:
        data['ElapsedTime (h)'] = data['Day']
    data['ElapsedTime (h)'] = data['ElapsedTime (h)'].astype(str)
    hours = [str(h) for h in conf_snapshot_hours(conf)]
    slug = metric_slug(metric)
    table_path = table_file(conf, MODULE_CONVEX, slug, 'summary_table.csv')
    perform_interval_pairwise_stats(
        conf, data, metric, output_dir=None,
        interval_col='ElapsedTime (h)',
        intervals=hours,
        plant_id_col='Plant_id',
        interval_label='ElapsedTime (h)',
        module=MODULE_CONVEX,
        metric_slug_name=slug,
        table_file_path=table_path,
    )
    emit_interval_comparison_plots(
        conf, data, metric, metric_dir(conf, MODULE_CONVEX, slug),
        module=MODULE_CONVEX, metric_slug_name=slug,
        metric_label=metric, x_col='ElapsedTime (h)',
    )