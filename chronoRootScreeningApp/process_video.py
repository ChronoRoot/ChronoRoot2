import os
import re
import json
import numpy as np
import cv2
from sort import Sort
import pandas as pd
import argparse
from datetime import datetime
from typing import Dict, Any, Tuple, List
import matplotlib
import sys
from qr import qr_detect, get_pixel_size
from collections import defaultdict
matplotlib.use('Agg')

from skimage.morphology import skeletonize
from robot_ids import resolve_rpi_cam

# ignore future warnings from pandas
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

class GroupROISelector:
    def __init__(self, image_path: str, group_names: List[str]):
        # Read and normalize image
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")
        self.original_image = img
        self.display_image = (cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)).astype(np.uint8)
        
        # Scale image to a reasonable size
        height, width = self.display_image.shape
        scale = min(1.0, 1200 / height)
        self.scale = scale
        self.display_image = cv2.resize(self.display_image, None, fx=scale, fy=scale)
        
        self.groups = {}
        self.group_names = group_names
        
    def select_roi_for_group(self, group_name: str) -> bool:
        instructions = (
            f"Select ROI for {group_name}\n"
            "1. Click and drag to select region\n"
            "2. Press ENTER TWICE to confirm selection\n"
            "3. Press 'r' to redo selection\n"
            "4. Press 'q' to quit analysis"
        )
        window_name = f"Select ROI for {group_name}"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        while True:
            # Create fresh copy of image for display
            img_copy = self.display_image.copy()
            
            # Add instructions to the image
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            thickness = 2
            color = (255, 255, 255)
            
            y0 = 30
            for i, line in enumerate(instructions.split('\n')):
                y = y0 + i * 25
                cv2.putText(img_copy, line, (10, y), font, font_scale, color, thickness)
            
            # Draw ROI
            roi = cv2.selectROI(window_name, img_copy, fromCenter=False, showCrosshair=True)
            
            if roi[2] == 0 or roi[3] == 0:  # If ROI has no area
                key = cv2.waitKey(1)
                if key == ord('q'):  # Check if user pressed 'q' to quit
                    cv2.destroyWindow(window_name)
                    return False
                print("Invalid selection, please try again or press 'q' to quit")
                continue
                
            # Scale ROI back to original image coordinates
            orig_roi = tuple(int(x / self.scale) for x in roi)
            
            # Draw the selection for confirmation
            cv2.rectangle(img_copy, (roi[0], roi[1]), 
                         (roi[0] + roi[2], roi[1] + roi[3]), (255, 255, 255), 2)
            cv2.imshow(window_name, img_copy)
            
            print(f"\nSelection made for {group_name}.")
            print("Press 'r' to redo selection or 'q' to quit analysis")
            
            key = cv2.waitKey(0)
            if key == ord('q'):
                cv2.destroyWindow(window_name)
                return False
            elif key == ord('r'):
                continue
            else:
                self.groups[group_name] = (
                    orig_roi[0], 
                    orig_roi[1],
                    orig_roi[0] + orig_roi[2],
                    orig_roi[1] + orig_roi[3]
                )
                cv2.destroyWindow(window_name)
                return True
            
        return False

    def select_groups(self) -> Dict[str, Tuple[int, int, int, int]]:
        for group_name in self.group_names:
            print(f"\nSelecting region for {group_name}")
            if not self.select_roi_for_group(group_name):
                print("\nROI selection cancelled. Stopping analysis.")
                return None  # Return None to indicate cancellation
        return self.groups

def natural_sort_key(s):
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', s)]


def _datetime_from_image_name(name):
    nums = re.findall(r'\d+', os.path.basename(str(name)))
    if len(nums) < 5:
        return None
    try:
        return datetime(int(nums[0]), int(nums[1]), int(nums[2]), int(nums[3]), int(nums[4]))
    except (ValueError, TypeError):
        return None


def record_acquisition_span(analysis_dir, image_files):
    """Store first/last picture names and parsed clock times on tracking metadata."""
    names = [f for f in (image_files or []) if f]
    if not names:
        return
    first_image = os.path.basename(str(names[0]))
    last_image = os.path.basename(str(names[-1]))
    first_dt = _datetime_from_image_name(first_image)
    last_dt = _datetime_from_image_name(last_image)
    metadata_path = os.path.join(analysis_dir, 'metadata.json')
    metadata = {}
    if os.path.isfile(metadata_path):
        try:
            with open(metadata_path, 'r') as handle:
                metadata = json.load(handle) or {}
        except Exception:
            metadata = {}
    metadata['first_image'] = first_image
    metadata['last_image'] = last_image
    if first_dt is not None:
        metadata['first_datetime'] = first_dt.strftime('%Y-%m-%dT%H:%M:%S')
    if last_dt is not None:
        metadata['last_datetime'] = last_dt.strftime('%Y-%m-%dT%H:%M:%S')
    with open(metadata_path, 'w') as handle:
        json.dump(metadata, handle, indent=4)


def get_group_for_position(x: int, y: int, groups: Dict[str, Tuple[int, int, int, int]]) -> str:
    for group_name, (x1, y1, x2, y2) in groups.items():
        if x1 <= x <= x2 and y1 <= y <= y2:
            return group_name
    return "Unknown"

def save_metadata(analysis_dir: str, params: Dict[str, Any], start_time: str = None,
                  completion_time: str = None, status: str = None, error: str = None) -> None:
    """
    Save or update metadata about the analysis.
    
    Args:
        analysis_dir: Path to the analysis directory
        params: Analysis parameters
        start_time: Optional start time, if provided creates new metadata
        completion_time: Optional completion time, if provided updates existing metadata
    """
    metadata_path = os.path.join(analysis_dir, 'metadata.json')
    
    if start_time:
        # Create new metadata file
        metadata = {
            'analysis_id': params['analysis_id'],
            'group_names': params['group_names'],
            'num_groups': len(params['group_names']),
            'PlateCondition': params.get('PlateCondition', '') or '',
            'ExtraVariable': params.get('ExtraVariable', '') or '',
            'video_directory': params['video_dir'],
            'segmentation_directory': params['segmentation_dir'],
            'rpi': params.get('rpi', ''),
            'cam': params.get('cam', ''),
            'start_time': start_time,
            'completion_time': None,
            'status': 'In Progress'
        }
    else:
        # Update existing metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        if completion_time:
            metadata['completion_time'] = completion_time
            metadata['status'] = status or 'Complete'
        if error is not None:
            metadata['error'] = error
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)


def mark_analysis_failed(analysis_dir: str, params: Dict[str, Any], error: str) -> None:
    metadata_path = os.path.join(analysis_dir, 'metadata.json')
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    else:
        metadata = {
            'analysis_id': params.get('analysis_id'),
            'group_names': params.get('group_names', []),
            'num_groups': len(params.get('group_names', [])),
            'video_directory': params.get('video_dir'),
            'segmentation_directory': params.get('segmentation_dir'),
            'start_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
    metadata['completion_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    metadata['status'] = 'Failed'
    metadata['error'] = error
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)


def load_group_rois(group_rois: Any, group_names: List[str]) -> Dict[str, Tuple[int, int, int, int]]:
    if isinstance(group_rois, str):
        with open(group_rois, 'r') as f:
            group_rois = json.load(f)

    if not isinstance(group_rois, dict):
        raise ValueError("group_rois must be a JSON object mapping group names to rectangles")

    groups = {}
    for name in group_names:
        if name not in group_rois:
            raise ValueError(f"Missing ROI for group: {name}")
        coords = group_rois[name]
        if not isinstance(coords, (list, tuple)) or len(coords) != 4:
            raise ValueError(f"Invalid ROI coordinates for group {name}")
        groups[name] = tuple(int(v) for v in coords)
    return groups


def draw_tracking(img, bbox, group, seedpos, crop_seg_mask, original_coords):
    """
    Draw tracking visualization with bounding box, ID and segmentation overlay.
    
    Args:
        img: Original image to draw on
        bbox: Bounding box coordinates [x, y, x_w, y_h]
        group: Group name string
        seedpos: Seed position ID
        crop_seg_mask: Cropped segmentation mask
        original_coords: Tuple of (x, y) coordinates for the crop origin
    """
    x, y, x_w, y_h = map(int, bbox)
    
    # Draw bounding box
    cv2.rectangle(img, (x, y), (x_w, y_h), (0, 255, 0), 2)
    
    # Draw ID text
    text = f"{group}_{seedpos}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    
    # Background rectangle for text
    cv2.rectangle(img,
                 (x, y - text_size[1] - 5),
                 (x + text_size[0], y),
                 (0, 255, 0), -1)
    
    # Draw text
    cv2.putText(img, text, (x, y - 5),
                font, font_scale, (0, 0, 0), thickness)
    
    # Draw segmentation overlay if available
    if crop_seg_mask is not None and np.any(crop_seg_mask):
        ox, oy = original_coords
        mask_height, mask_width = crop_seg_mask.shape
        
        # Get the image region we'll modify
        region = img[oy:oy+mask_height, ox:ox+mask_width]
        
        # Create color overlays directly on the region
        region[(crop_seg_mask == 4)] = [0, 0, 255]  # Hypocotyl in red
        region[(crop_seg_mask > 0) & (crop_seg_mask != 4)] = [255, 165, 0]  # Rest in orange

    return img

def get_contour_centroid(contour):
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        return (cx, cy)
    return None


def process_video(params: Dict[str, Any]):
    """Process video with enhanced results management and merge detection."""

    analysis_dir = os.path.join(params['project_dir'], 'analysis', params['analysis_id'])
    vis_dir = os.path.join(analysis_dir, 'visualizations')

    os.makedirs(analysis_dir, exist_ok=True)
    if params['show_tracking']:
        os.makedirs(vis_dir, exist_ok=True)

    start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    save_metadata(analysis_dir, params, start_time=start_time)

    group_info = {
        'group_names': params['group_names'],
        'seed_counts': params['seed_counts'],
        'PlateCondition': params.get('PlateCondition', '') or '',
        'ExtraVariable': params.get('ExtraVariable', '') or '',
        'time_delta': params.get('time_delta'),
        'rpi': params.get('rpi', ''),
        'cam': params.get('cam', ''),
    }

    with open(os.path.join(analysis_dir, 'group_info.json'), 'w') as f:
        json.dump(group_info, f, indent=4)

    image_files = [f for f in os.listdir(params['video_dir'])
                  if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp'))]
    image_files.sort(key=natural_sort_key)

    if not image_files:
        raise ValueError("No image files found in the video directory")

    record_acquisition_span(analysis_dir, image_files)

    if params.get('group_rois'):
        groups = load_group_rois(params['group_rois'], params['group_names'])
        with open(os.path.join(analysis_dir, 'group_rois.json'), 'w') as f:
            json.dump({k: list(v) for k, v in groups.items()}, f, indent=4)
    else:
        last_image_path = os.path.join(params['video_dir'], image_files[-1])
        roi_selector = GroupROISelector(last_image_path, params['group_names'])
        groups = roi_selector.select_groups()
        if groups is None:
            import shutil
            shutil.rmtree(analysis_dir)
            return None
        with open(os.path.join(analysis_dir, 'group_rois.json'), 'w') as f:
            json.dump({k: list(v) for k, v in groups.items()}, f, indent=4)

    mot_tracker = Sort(max_age=8, min_hits=2, iou_threshold=0.5)
    known_track_ids = set()
    max_init_frame = 8
    current_frame = 0
    prev_binary = None
    t = 0
    excluded_tracks = set()  # Keep track of merged objects
    
    # Use list instead of DataFrame for accumulation 
    all_data_rows = []
    column_names = [
        "Plant_id", "Experiment", "ElapsedHours", "Area",
        "Perim.", "Frame", "SeedPos", "FileName",
        "HypocotylLength", "MainRootLength", "TotalLength",
        "LateralRootsLength", "NumberOfLateralRoots",
        "DenseRootArea",
        "PlateCondition", "ExtraVariable", "pixel_size",
        "BoundingBox_X", "BoundingBox_Y", "BoundingBox_Width", "BoundingBox_Height",
    ]


    # Calculate pixel size from calibration method
    pixel_size = 0.004  # Default value in case something fails
    try:
        if params.get('pixel_size') not in (None, ''):
            pixel_size = float(params['pixel_size'])
            print(f"Using stored pixel size: {pixel_size:.6f} mm/pixel")
        elif params['has_qr']:
            # QR code calibration
            j = 0
            for img_file in image_files:
                image_file = os.path.join(params['video_dir'], img_file)
                qr = qr_detect(image_file)
                if qr is not None:
                    pixel_size = 10 / get_pixel_size(qr[0])
                    print(f"Pixel size calculated from QR code: {pixel_size:.6f} mm/pixel")
                    break
                j += 1
                if j > 10:
                    print("No QR code found in the first 10 images, using default value")
                    break

        else:
            # Manual calibration
            known_distance = params['known_distance']  # in mm
            pixel_distance = params['pixel_distance']  # in pixels
            pixel_size = known_distance / pixel_distance
            print(f"Pixel size from manual calibration: {pixel_size:.6f} mm/pixel")
            
    except Exception as e:
        print(f"Error in pixel size calculation: {str(e)}, using default value {pixel_size:.6f} mm/pixel")

    group_info['pixel_size'] = pixel_size
    with open(os.path.join(analysis_dir, 'group_info.json'), 'w') as f:
        json.dump(group_info, f, indent=4)

    seg_subdir = params.get('segmentation_subdir') or 'Ensemble'

    # Process frames
    for img_file in image_files:
        seg_file = os.path.join(params['segmentation_dir'], seg_subdir, img_file)
        original_seg = cv2.imread(seg_file, cv2.IMREAD_GRAYSCALE)
        if original_seg is None:
            print(f'Warning: Could not read segmentation file: {seg_file}')
            continue

        # Load visualization image if needed
        vis_image = None
        if params['show_tracking']:
            vis_image = cv2.imread(os.path.join(params['video_dir'], img_file))
            if vis_image is None:
                print(f"Warning: Could not load image for visualization: {img_file}")
                continue

        # Process binary segmentation
        binary_seg = original_seg > 0
        if prev_binary is None:
            prev_binary = binary_seg.copy()
        else:
            binary_seg = np.bitwise_or(binary_seg, prev_binary)
            prev_binary = binary_seg.copy()

        # Find contours and create detections
        contours, _ = cv2.findContours(binary_seg.astype(np.uint8), cv2.RETR_EXTERNAL, 
                                     cv2.CHAIN_APPROX_SIMPLE)
        dets = []
        for contour in contours:
            if cv2.contourArea(contour) < 10:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            dets.append([x, y, x + w, y + h, 1])
        
        if not dets:
            print('No seeds found in', img_file)
            continue
        
        # Update tracking
        trackers = mot_tracker.update(np.array(dets))
        current_frame += 1

        if current_frame <= max_init_frame:
            known_track_ids.update(trackers[:, 4])
        else:
            # Filter out previously excluded tracks first
            trackers = trackers[np.isin(trackers[:, 4], list(known_track_ids - excluded_tracks))]

        # Create a dict to track which detections are assigned to each contour
        contour_assignments = defaultdict(list)

        # Assign trackers to contours
        for i, contour in enumerate(contours):
            cont_x, cont_y, cont_w, cont_h = cv2.boundingRect(contour)
            
            # Check which trackers' centers fall within this contour's bbox
            for tracker in trackers:
                x, y, x_w, y_h, ID = tracker
                x_center = int((x + x_w) // 2)
                y_center = int((y + y_h) // 2)
                
                if (cont_x <= x_center <= cont_x + cont_w and 
                    cont_y <= y_center <= cont_y + cont_h):
                    contour_assignments[i].append(ID)

        # Detect merging and remove merged trackers
        newly_merged = set()
        for assigned_ids in contour_assignments.values():
            if len(assigned_ids) > 1:
                newly_merged.update(assigned_ids)

        # Add to excluded tracks for future frames
        excluded_tracks.update(newly_merged)

        # Remove merged trackers from current frame
        if newly_merged:
            valid_tracks = ~np.isin(trackers[:, 4], list(newly_merged))
            trackers = trackers[valid_tracks]

        # Process only non-merged tracks
        rows = []
        for tracker in trackers:
            x, y, x_w, y_h, ID = tracker
            x = int(x)
            y = int(y)
            w = int(x_w - x)
            h = int(y_h - y)
            x_center = int((x + x_w) // 2)
            y_center = int((y + y_h) // 2)

            # Find the corresponding contour
            best_contour = None
            for i, contour in enumerate(contours):
                if ID in contour_assignments[i]:
                    best_contour = contour
                    break
                    
            if best_contour is None:
                continue
            
            # Get ROI bounds and create mask
            x_min = max(0, x)
            y_min = max(0, y)
            x_max = min(original_seg.shape[1], x + w)
            y_max = min(original_seg.shape[0], y + h)
            
            roi_contour = best_contour - np.array([[x_min, y_min]])
            roi_mask = np.zeros((y_max - y_min, x_max - x_min), dtype=np.uint8)
            cv2.drawContours(roi_mask, [roi_contour], 0, 1, -1)
            
            # Apply mask to ROI
            crop = original_seg[y_min:y_max, x_min:x_max] * roi_mask

            # Calculate measurements in pixels (dataWork converts lengths to mm)
            dense_area_covered = float(np.sum(crop > 0))
            perimeter = cv2.arcLength(best_contour, True) * pixel_size

            hypocotyl_length = 0
            hypocotyl = crop == 4
            if np.sum(hypocotyl) > 10:
                kernel = np.ones((3, 3), np.uint8)
                hypocotyl = cv2.morphologyEx(hypocotyl.astype(np.uint8), cv2.MORPH_OPEN, kernel)
                hypocotyl = cv2.morphologyEx(hypocotyl, cv2.MORPH_CLOSE, kernel)
                hypocotyl_skeleton = skeletonize(hypocotyl)
                hypocotyl_length = float(np.sum(hypocotyl_skeleton))
            
            mainroot_length = 0
            mainroot = crop == 1
            if np.sum(mainroot) > 10:
                kernel = np.ones((3, 3), np.uint8)
                mainroot = cv2.morphologyEx(mainroot.astype(np.uint8), cv2.MORPH_OPEN, kernel)
                mainroot = cv2.morphologyEx(mainroot, cv2.MORPH_CLOSE, kernel)
                mainroot_skeleton = skeletonize(mainroot)
                mainroot_length = float(np.sum(mainroot_skeleton))
            
            totalroot_length = 0
            totalroot = np.bitwise_or(crop == 1, crop == 2)
            if np.sum(totalroot) > 10:
                kernel = np.ones((3, 3), np.uint8)
                totalroot = cv2.morphologyEx(totalroot.astype(np.uint8), cv2.MORPH_OPEN, kernel)
                totalroot = cv2.morphologyEx(totalroot, cv2.MORPH_CLOSE, kernel)
                totalroot_skeleton = skeletonize(totalroot)
                totalroot_length = float(np.sum(totalroot_skeleton))

            dense_root_area = float(np.sum(totalroot))
            lateral_length = max(0.0, totalroot_length - mainroot_length)

            experiment = get_group_for_position(x_center, y_center, groups)
            plate_condition = params.get('PlateCondition', '') or ''
            extra_variable = params.get('ExtraVariable', '') or ''
            if params['show_tracking']:
                vis_image = draw_tracking(
                    vis_image,
                    [x, y, x_w, y_h],
                    experiment,
                    int(ID),
                    crop,
                    (x_min, y_min)
                )
            
            plant_id = f"{params['analysis_id']}_{experiment}_{int(ID)}"
            rows.append([
                plant_id, experiment, t * params['time_delta'] / 60,
                dense_area_covered,
                perimeter, t + 1, ID, img_file,
                hypocotyl_length, mainroot_length, totalroot_length,
                lateral_length, 0,
                dense_root_area,
                plate_condition, extra_variable, pixel_size,
                x, y, w, h,
            ])   
        
        # Save visualization
        if params['show_tracking']:
            cv2.imwrite(os.path.join(vis_dir, f"tracking_{img_file}"), vis_image)

        # --- OPTIMIZATION: Extend the list instead of DataFrame concat ---
        if rows:
            all_data_rows.extend(rows)
        t += 1

    # --- Create DataFrame ONCE at the end ---
    dataframe = pd.DataFrame(all_data_rows, columns=column_names)
    dataframe['rpi'] = params.get('rpi', '')
    dataframe['cam'] = params.get('cam', '')

    # Post-processing
    if not dataframe.empty:
        dataframe = dataframe.sort_values(by=['SeedPos', 'Frame']).reset_index(drop=True)
        # Ensure SeedPos can be converted to int safely
        try:
             dataframe['SeedPos'] = dataframe['SeedPos'].astype(int) - np.min(dataframe['SeedPos'].astype(int))
        except:
             pass # Handle cases where SeedPos might be empty

    # Remove unknown group seeds after all processing
    dataframe = dataframe[dataframe['Experiment'] != 'Unknown']

    # Save results
    results_path = os.path.join(analysis_dir, 'seeds.tsv')
    dataframe.to_csv(results_path, sep='\t', index=False)

    # Update metadata
    completion_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    save_metadata(analysis_dir, params, completion_time=completion_time)
    record_acquisition_span(analysis_dir, image_files)

    print(f"Results saved in: {results_path}")

    return dataframe


def validate_directories(video_dir: str, project_dir: str, segmentation_dir: str = None,
                         segmentation_subdir: str = 'Ensemble'):
    """Validate directory structure and files."""
    if not os.path.exists(video_dir):
        raise ValueError(f"Video directory does not exist: {video_dir}")

    if segmentation_dir is None:
        segmentation_dir = os.path.join(video_dir, "Segmentation")

    subdir = segmentation_subdir or 'Ensemble'
    seg_dir = os.path.join(segmentation_dir, subdir)
    if not os.path.exists(seg_dir) and subdir == 'Ensemble':
        seg_dir = os.path.join(segmentation_dir, "Seg")
    if not os.path.exists(seg_dir):
        raise ValueError(f"Segmentation directory not found: {seg_dir}")

    if not os.path.exists(project_dir):
        raise ValueError(f"Project directory does not exist: {project_dir}")

    image_files = [f for f in os.listdir(video_dir)
                  if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp'))]

    seg_files = [f for f in os.listdir(seg_dir)
                 if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp'))]

    if not image_files:
        raise ValueError(f"No image files found in video directory: {video_dir}")

    if not seg_files:
        raise ValueError(f"No segmentation files found in: {seg_dir}")

    return True


def attach_rpi_cam(params: Dict[str, Any]) -> Dict[str, Any]:
    rpi, cam = resolve_rpi_cam(
        rpi=params.get('rpi'),
        cam=params.get('cam'),
        video_dir=params.get('video_dir'),
        analysis_id=params.get('analysis_id'),
    )
    params['rpi'] = rpi
    params['cam'] = cam
    return params


def build_params_from_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, 'r') as f:
        config = json.load(f)

    required = [
        'video_dir', 'segmentation_dir', 'project_dir', 'analysis_id',
        'group_names', 'seed_counts', 'time_delta', 'has_qr', 'show_tracking'
    ]
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"Missing required config fields: {', '.join(missing)}")

    params = {
        'video_dir': config['video_dir'],
        'segmentation_dir': config['segmentation_dir'],
        'project_dir': config['project_dir'],
        'analysis_id': config['analysis_id'],
        'time_delta': config['time_delta'],
        'has_qr': bool(config['has_qr']),
        'show_tracking': bool(config['show_tracking']),
        'group_names': config['group_names'],
        'seed_counts': config['seed_counts'],
        'PlateCondition': config.get('PlateCondition', '') or '',
        'ExtraVariable': config.get('ExtraVariable', '') or '',
        'rpi': config.get('rpi', ''),
        'cam': config.get('cam', ''),
        'segmentation_subdir': config.get('segmentation_subdir') or 'Ensemble',
    }
    if config.get('pixel_size') not in (None, ''):
        params['pixel_size'] = float(config['pixel_size'])

    if params.get('pixel_size') not in (None, ''):
        pass
    elif params['has_qr']:
        pass
    else:
        if config.get('known_distance') is None or config.get('pixel_distance') is None:
            raise ValueError("Manual calibration requires known_distance and pixel_distance in config")
        params['known_distance'] = float(config['known_distance'])
        params['pixel_distance'] = int(config['pixel_distance'])

    if 'group_rois' in config and config['group_rois']:
        params['group_rois'] = config['group_rois']

    return attach_rpi_cam(params)


def build_params_from_args(args) -> Dict[str, Any]:
    if len(args.group_info) % 2 != 0:
        raise ValueError("--group-info must have pairs of names and counts")

    group_names = args.group_info[::2]
    seed_counts = [int(count) for count in args.group_info[1::2]]

    params = {
        'video_dir': args.video_dir,
        'segmentation_dir': args.segmentation_dir,
        'project_dir': args.project_dir,
        'analysis_id': args.analysis_id,
        'time_delta': args.time_delta,
        'has_qr': args.has_qr,
        'show_tracking': args.show_tracking,
        'group_names': group_names,
        'seed_counts': seed_counts,
        'PlateCondition': getattr(args, 'plate_condition', '') or '',
        'ExtraVariable': getattr(args, 'extra_variable', '') or '',
    }

    if not args.has_qr:
        params['known_distance'] = args.known_distance
        params['pixel_distance'] = args.pixel_distance

    if args.group_rois:
        params['group_rois'] = args.group_rois

    return attach_rpi_cam(params)


def main():
    parser = argparse.ArgumentParser(description='Process seed tracking video')
    parser.add_argument('--config', help='JSON config file with all processing parameters')

    # Legacy CLI arguments (used when --config is not provided)
    parser.add_argument('--video-dir', help='Directory containing the video frames')
    parser.add_argument('--segmentation-dir', help='Directory containing segmentation masks')
    parser.add_argument('--project-dir', help='Project directory for output')
    parser.add_argument('--analysis-id', help='Unique identifier for this analysis')
    parser.add_argument('--has-qr', action='store_true', help='Use QR code for calibration')
    parser.add_argument('--known-distance', type=float, help='Known physical distance in mm')
    parser.add_argument('--pixel-distance', type=int,
                        help='Pixel distance corresponding to known physical distance')
    parser.add_argument('--time-delta', type=float, default=15,
                        help='Time between slices in minutes')
    parser.add_argument('--show-tracking', action='store_true',
                        help='Flag to show tracking visualization')
    parser.add_argument('--group-info', nargs='+',
                        help='Alternating group names and seed counts (e.g., "GroupA" "10" "GroupB" "15")')
    parser.add_argument('--group-rois', help='JSON file with precomputed group ROIs')
    parser.add_argument('--plate-condition', default='', help='Plate growth condition for this video')
    parser.add_argument('--extra-variable', default='', help='Extra variable for this video')

    args = parser.parse_args()

    try:
        if args.config:
            params = build_params_from_config(args.config)
        else:
            required = ['video_dir', 'segmentation_dir', 'project_dir', 'analysis_id', 'group_info']
            missing = [name for name in required if getattr(args, name.replace('-', '_'), None) in (None, [])]
            if missing:
                parser.error(f"Missing required arguments: {', '.join(missing)} (or provide --config)")

            if not args.has_qr:
                if args.known_distance is None or args.pixel_distance is None:
                    parser.error("--known-distance and --pixel-distance are required for manual calibration")
            elif args.known_distance is not None:
                parser.error("--known-distance cannot be used with --has-qr")

            params = build_params_from_args(args)

        validate_directories(
            params['video_dir'], params['project_dir'], params['segmentation_dir'],
            params.get('segmentation_subdir') or 'Ensemble',
        )

        print(f"Starting analysis: {params['analysis_id']}")
        print("Groups to analyze:")
        for name, count in zip(params['group_names'], params['seed_counts']):
            print(f"  - {name}: {count} seeds")

        if params.get('pixel_size') not in (None, ''):
            pass
        elif params['has_qr']:
            print("Using QR code calibration")
        else:
            print(f"Using manual calibration: {params['known_distance']}mm = {params['pixel_distance']}px")

        df = process_video(params)
        if df is None:
            print("Processing cancelled during ROI selection.")
            sys.exit(1)

        print("\nProcessing completed successfully")
        print(f"Results saved in: {os.path.join(params['project_dir'], 'analysis', params['analysis_id'])}")
    except Exception as e:
        print(f"Error during processing: {e}", file=sys.stderr)
        if args.config:
            try:
                with open(args.config, 'r') as f:
                    config = json.load(f)
                analysis_dir = os.path.join(
                    config['project_dir'], 'analysis', config['analysis_id']
                )
                if os.path.exists(analysis_dir):
                    mark_analysis_failed(analysis_dir, config, str(e))
            except Exception:
                pass
        elif args.project_dir and args.analysis_id:
            analysis_dir = os.path.join(args.project_dir, 'analysis', args.analysis_id)
            if os.path.exists(analysis_dir):
                try:
                    mark_analysis_failed(
                        analysis_dir,
                        {
                            'analysis_id': args.analysis_id,
                            'group_names': args.group_info[::2] if args.group_info else [],
                            'video_dir': args.video_dir,
                            'segmentation_dir': args.segmentation_dir,
                        },
                        str(e),
                    )
                except Exception:
                    pass
        sys.exit(1)

if __name__ == "__main__":
    main()