""" 
ChronoRoot: High-throughput phenotyping by deep learning reveals novel temporal parameters of plant root system architecture
Copyright (C) 2020 Nicolás Gaggion

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

from .utils.fileUtilities import (
    createSaveFolder,
    getImages,
    saveMetadata,
    cleanup_superseded_results,
    plant_slot_path,
    processing_limit_frame_count,
)
from .utils.metadata_schema import get_bounding_box, get_seed
from .imageUtils.seg import extract_root_segmentation, extract_skeleton
from .imageUtils.plot import saveImages
from .graphUtils.save import saveGraph, saveProps
from .graphUtils.graph import createGraph
from .graphUtils.graphTrim import trimGraph
from .graphUtils.graphTrack import graphInit, matchGraphs
from .rsmlUtils.rsml import createTree, saveRSML

import os 
import csv
import datetime

def getImgName(image, conf):
    return image.replace(conf['ImagePath'],'').replace('/','')

def setupPlantAnalysis(conf, replicate):
    """
    Setup analysis by loading images and determining the region of interest.
    
    Args:
        conf: Configuration dictionary
        replicate: If True, use previously saved ROI and seed from conf.
                   If False, use conf ROI when present, otherwise try interactive selection.
    
    Returns:
        image_paths: List of paths to original images
        segmentation_paths: List of paths to segmentation masks
        roi_bounds: Bounding box coordinates [y_min, y_max, x_min, x_max]
        current_root_base: Current root base position (x, y) - updates during tracking
        fixed_seed_position: Original seed position (x, y) - never changes
    """
    # Load image and segmentation file paths
    image_paths, segmentation_paths = getImages(conf)
    
    limit_frames = processing_limit_frame_count(conf)
    if limit_frames is not None:
        image_paths = image_paths[:limit_frames]
        segmentation_paths = segmentation_paths[:limit_frames]

    roi_bounds = get_bounding_box(conf)
    seed_position = get_seed(conf)
    if replicate:
        if roi_bounds is None or seed_position is None:
            print('Missing saved ROI/seed in config.')
            return None, None, None, None, None
    elif roi_bounds is None or seed_position is None:
        from .utils.roi_selection import try_select_roi_and_seed
        roi_bounds, seed_position = try_select_roi_and_seed(conf, image_paths, segmentation_paths)
        if seed_position is None:
            print(
                'Interactive ROI selection failed or was cancelled. '
                'Use Analyze Plant or Redo Analysis in the GUI, or check your display/Qt setup.'
            )
            return None, None, None, None, None
        conf['bounding_box'] = roi_bounds
        conf['seed'] = seed_position

    fixed_seed_position = seed_position.copy()
    current_root_base = seed_position.copy()
    
    return image_paths, segmentation_paths, roi_bounds, current_root_base, fixed_seed_position
    
def plantAnalysis(conf, replicate=False):
    # Setup: get images, ROI bounds, and seed position
    images, segmentation_paths, roi_bounds, current_root_base, fixed_seed_position = setupPlantAnalysis(conf, replicate)
    
    if images is None:
        print('Analysis cancelled by user')
        return
    
    # Create output folders and save metadata
    output_folders = createSaveFolder(conf)
    conf['folders'] = output_folders
    conf = saveMetadata(roi_bounds, current_root_base, conf)
    
    total_frames = len(images)
    print(f'Number of frames: {total_frames}')
    
    # Setup CSV for measurements
    measurements_file = os.path.join(output_folders['result'], "Results_raw.csv")
    
    # Initialize logging
    analysis_log = []
    frame_errors = []  # 0 = success, 1 = error for each frame
    
    with open(measurements_file, 'w+') as csv_file:
        csv_writer = csv.writer(csv_file)
        header = ['FileName', 'Frame', 'MainRootLength', 'LateralRootsLength', 'NumberOfLateralRoots', 'TotalLength', 'HypocotylLength']
        csv_writer.writerow(header)
        
        # ====================================================================
        # PHASE 1: Find first frame with valid root structure
        # ====================================================================
        print('Searching for initial valid segmentation...')
        
        first_valid_frame = None
        initial_root_mask = None
        initial_skeleton_overlay = None
        initial_graph = None
        initial_rsml = None
        initial_lateral_count = None
        initial_hypocotyl_length = None
        initial_hypocotyl_skeleton = None
        
        for frame_idx in range(total_frames):
            print(f'Checking frame {frame_idx + 1} of {total_frames}', end='\r')
            
            # Try to extract root segmentation
            try:
                # Pass None for previous_bbox during initialization
                root_mask, hypocotyl_skeleton, hypocotyl_length, found_root, mc_filtered_mask, current_bbox = extract_root_segmentation(
                    segmentation_paths[frame_idx], 
                    roi_bounds, 
                    current_root_base,
                    fixed_seed_position,
                    previous_bbox=None
                )
                
                if not found_root:
                    frame_name = getImgName(images[frame_idx], conf)
                    saveProps(frame_name, frame_idx, False, csv_writer, 0, 0)
                    saveImages(conf, images, frame_idx, root_mask, None, None, None)
                    frame_errors.append(0)
                    continue
                
                # Try to extract skeleton structure
                skeleton, branch_points, end_points, is_valid_skeleton = extract_skeleton(root_mask)
                                
                if not is_valid_skeleton:
                    frame_name = getImgName(images[frame_idx], conf)
                    saveProps(frame_name, frame_idx, False, csv_writer, 0, 0)
                    saveImages(conf, images, frame_idx, root_mask, None, None, None)
                    frame_errors.append(0)
                    continue
            except Exception as e:
                frame_name = getImgName(images[frame_idx], conf)
                saveProps(frame_name, frame_idx, False, csv_writer, 0, 0)
                saveImages(conf, images, frame_idx, root_mask, None, None, None)
                frame_errors.append(0)
                continue
            
            skeleton_color = skeleton.copy() * mc_filtered_mask.copy()  # Color skeleton by multiclass mask
            
            # Try to create graph structure
            graph, updated_root_base, skeleton_overlay = createGraph(
                skeleton_color.copy(), 
                current_root_base, 
                end_points, 
                branch_points
            )

            try:
                graph = graphInit(graph)
                rsml_tree, lateral_root_count = createTree(conf, frame_idx, images, graph, skeleton, skeleton_overlay)
            except Exception as e:
                frame_name = getImgName(images[frame_idx], conf)
                saveProps(frame_name, frame_idx, False, csv_writer, 0, 0)
                saveImages(conf, images, frame_idx, root_mask, None, None, None)
                frame_errors.append(0)
                continue
            
            # Success! Store the initial valid frame
            first_valid_frame = frame_idx
            initial_root_mask = root_mask
            initial_skeleton_overlay = skeleton_overlay
            initial_graph = graph
            initial_rsml = rsml_tree
            initial_hypocotyl_skeleton = hypocotyl_skeleton
            initial_hypocotyl_length = hypocotyl_length
            initial_lateral_count = lateral_root_count
            current_root_base = updated_root_base
            
            print(f'\nFound initial valid structure at frame {frame_idx}')
            break
                
            
        # Check if we found any valid frame
        if first_valid_frame is None:
            print('\nERROR: No valid segmentation found in entire sequence')
            log_path = os.path.join(output_folders['result'], "log.txt")
            with open(log_path, 'w+') as log_file:
                log_file.write('No valid segmentation found in entire sequence\n')
                log_file.write(f'Checked all {total_frames} frames\n')
                log_file.write(f'Error Rate: 1.0')
            return
        
        # Save the first valid frame
        growth_start_frame = first_valid_frame
        print(f'Growth begins at frame {growth_start_frame}')
        analysis_log.append(f'Frame {growth_start_frame}: Growth begins\n')
        
        frame_name = getImgName(images[growth_start_frame], conf)
        saveImages(conf, images, growth_start_frame, initial_root_mask, initial_graph, initial_skeleton_overlay, initial_hypocotyl_skeleton)
        saveGraph(initial_graph, conf, frame_name)
        saveRSML(initial_rsml, conf, frame_name)
        saveProps(frame_name, growth_start_frame, initial_graph, csv_writer, initial_lateral_count, initial_hypocotyl_length)
        
        # ====================================================================
        # PHASE 2: Track growth over time
        # ====================================================================
        print('Tracking growth...')
        
        # Current state (will be updated each frame)
        current_root_mask = initial_root_mask
        current_skeleton_overlay = initial_skeleton_overlay
        current_graph = initial_graph
        current_rsml = initial_rsml
        current_lateral_count = initial_lateral_count
        current_hypocotyl_length = initial_hypocotyl_length
        current_hypocotyl_skeleton = initial_hypocotyl_skeleton
        
        error_count = 0
        consecutive_errors = 0
        tracking_lost = False
        
        for frame_idx in range(growth_start_frame + 1, total_frames):
            print(f'Processing frame {frame_idx + 1} of {total_frames}', end='\r')
            
            frame_failed = False
            seg_exception = None
            
            try:
                new_root_mask, new_hypocotyl_skeleton, new_hypocotyl_length, found_root, mc_filtered_mask, new_bbox = extract_root_segmentation(
                    segmentation_paths[frame_idx],
                    roi_bounds,
                    current_root_base,
                    fixed_seed_position,
                    previous_bbox=current_bbox
                )
            except Exception as e:
                found_root = False
                seg_exception = e
            
            if not found_root:
                frame_failed = True
                if seg_exception is not None:
                    analysis_log.append(f'Frame {frame_idx}: Error in segmentation - {str(seg_exception)}\n')
                else:
                    analysis_log.append(f'Frame {frame_idx}: Error in segmentation\n')
                
                consecutive_errors += 1
                if consecutive_errors >= 20:
                    current_bbox = None
                    current_root_base = fixed_seed_position.copy()
                    tracking_lost = True
                    consecutive_errors = 0
                    print(f'\n\nWARNING: 20 consecutive segmentation errors at frame {frame_idx}')
                    print('Dropping tracking memory and reinitializing from seed')
                    analysis_log.append(f'Frame {frame_idx}: 20 consecutive segmentation errors, dropped tracking memory\n')
            else:
                consecutive_errors = 0
            
            if not frame_failed:
                new_skeleton, branch_points, end_points, is_valid_skeleton = extract_skeleton(new_root_mask)
                
                if not is_valid_skeleton:
                    frame_failed = True
                    analysis_log.append(f'Frame {frame_idx}: Error in skeletonization\n')
            
            if not frame_failed:
                new_skeleton_color = new_skeleton.copy() * mc_filtered_mask.copy()  # Color skeleton by multiclass mask
                
                try:
                    new_graph, updated_root_base, new_skeleton_overlay = createGraph(
                        new_skeleton_color.copy(),
                        current_root_base,
                        end_points,
                        branch_points
                    )
                except Exception as e:
                    frame_failed = True
                    analysis_log.append(f'Frame {frame_idx}: Error in graph creation - {str(e)}\n')
            
            if not frame_failed:
                try:
                    new_graph = trimGraph(
                        new_graph
                    )
                except Exception as e:
                    frame_failed = True
                    analysis_log.append(f'Frame {frame_idx}: Error in graph trimming - {str(e)}\n')
            
            if not frame_failed:
                try:
                    if tracking_lost:
                        print(f'\nFrame {frame_idx}: Tracking memory lost, reinitializing graph')
                        new_graph = graphInit(new_graph)
                        analysis_log.append(f'Frame {frame_idx}: Tracking memory lost, reinitialized graph\n')
                    else:
                        new_graph = matchGraphs(current_graph, new_graph)
                except Exception as e:
                    # Matching failed - decide whether to reinitialize or fail
                    frames_since_start = frame_idx - growth_start_frame
                    
                    if tracking_lost or frames_since_start < 300:
                        # Early in tracking, or bbox memory was dropped - try reinitializing
                        try:
                            print(f'\nFrame {frame_idx}: Matching failed, reinitializing graph')
                            new_graph = graphInit(new_graph)
                            analysis_log.append(f'Frame {frame_idx}: Tracking failed, reinitialized graph\n')
                        except Exception as e2:
                            frame_failed = True
                            analysis_log.append(f'Frame {frame_idx}: Error in tracking and reinitialization - {str(e2)}\n')
                    else:
                        # Later in tracking - this is a real error
                        frame_failed = True
                        analysis_log.append(f'Frame {frame_idx}: Error in tracking - {str(e)}\n')

            if not frame_failed:
                try:
                    new_rsml, new_lateral_count = createTree(
                        conf,
                        frame_idx,
                        images,
                        new_graph,
                        new_skeleton.copy(),
                        new_skeleton_overlay.copy()
                    )
                except Exception as e:
                    # RSML creation failed - graph is still good, just can't export
                    # Don't fail the frame, just log it
                    analysis_log.append(f'Frame {frame_idx}: Warning in RSML creation - {str(e)}\n')
                    new_rsml = current_rsml  # Use previous RSML
                    new_lateral_count = current_lateral_count
            
            if frame_failed:
                # Frame failed - keep previous state (mask/graph still last-good)
                if frame_idx - growth_start_frame >= 50:
                    error_count += 1
                
                frame_errors.append(1)
            else:
                # Frame succeeded - update state
                current_root_mask = new_root_mask
                current_skeleton_overlay = new_skeleton_overlay
                current_graph = new_graph
                current_rsml = new_rsml
                current_lateral_count = new_lateral_count
                current_hypocotyl_length = new_hypocotyl_length
                current_hypocotyl_skeleton = new_hypocotyl_skeleton
                current_root_base = updated_root_base
                current_bbox = new_bbox
                tracking_lost = False
                
                frame_errors.append(0)
            
            frame_name = getImgName(images[frame_idx], conf)
            saveImages(conf, images, frame_idx, current_root_mask, current_graph, current_skeleton_overlay, current_hypocotyl_skeleton)
            saveGraph(current_graph, conf, frame_name)
            saveRSML(current_rsml, conf, frame_name)
            saveProps(frame_name, frame_idx, current_graph, csv_writer, current_lateral_count, current_hypocotyl_length)
        
        print('\n\nGrowth tracking complete')
        print('Saving outputs...')
        log_path = os.path.join(output_folders['result'], "log.txt")
        with open(log_path, 'w+') as log_file:
            log_file.write("Analysis completed: " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n")
            log_file.write(f"Growth start frame: {growth_start_frame}\n")
            log_file.write(f"Total frames analyzed: {total_frames}\n")
            log_file.write(f"Growth frames tracked: {total_frames - growth_start_frame}\n")
            
            # Calculate error statistics (only for frames after initialization)
            growth_frames = total_frames - growth_start_frame
            if growth_frames > 0:
                error_rate = round(error_count / growth_frames, 3)
                log_file.write(f"Total errors: {error_count}\n")
                log_file.write(f"Error rate: {error_rate}")
            else:
                log_file.write("Total errors: 0\n")
                log_file.write("Error rate: 1.0")
            log_file.write("\n")
        
        detailed_log_path = os.path.join(output_folders['result'], "detailed_log.txt")
        with open(detailed_log_path, 'w+') as log_file:
            # Write detailed log
            log_file.write("Detailed log:\n")
            for log_entry in analysis_log:
                log_file.write(log_entry)
        
        print(f'Results saved to {output_folders["result"]}')
        if error_count > 0:
            print(f'Warning: {error_count} errors occurred during tracking (see log.txt)')

        cleanup_superseded_results(plant_slot_path(conf), output_folders['result'])
    
    return