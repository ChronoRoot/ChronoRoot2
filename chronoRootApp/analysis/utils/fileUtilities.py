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

import pathlib
import re
import os
import shutil

def natural_key(string_):
    """See http://www.codinghorror.com/blog/archives/001018.html"""
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]

def loadPath(search_path, ext = '*.*'):
    data_root = pathlib.Path(search_path)
    all_files = list(data_root.glob(ext))
    all_files = [str(path) for path in all_files]
    all_files.sort(key = natural_key)
    
    return all_files

def convertToPathSafe(name):
    name = name.replace('.', '_dot_')
    name = name.replace('/', '_slash_')
    name = name.replace('\\', '_backslash_')
    return name
def convertFromPathSafe(name):
    name = name.replace('_dot_', '.')
    name = name.replace('_slash_', '/')
    name = name.replace('_backslash_', '\\')
    return name


def list_result_dirs(plant_path):
    """Return all Results_* directories under a plant slot, naturally sorted."""
    if not plant_path or not os.path.isdir(plant_path):
        return []
    return loadPath(plant_path, 'Results_*')


def get_latest_result_dir(plant_path):
    """Return the latest Results_* folder for a plant slot, or None."""
    results = list_result_dirs(plant_path)
    return results[-1] if results else None


def plant_slot_path(conf):
    """Filesystem path for a plant hardware slot (parent of Results_* folders)."""
    identifier = convertToPathSafe(conf['Experiment'])
    rpi = str(conf['rpi'])
    cam = f"cam_{conf['cam']}"
    plant = f"plant_{conf['plant']}"
    return os.path.join(conf['MainFolder'], 'Analysis', identifier, rpi, cam, plant)


def plant_slot_has_finished_analysis(conf):
    """True if the plant slot has at least one completed Results_* run."""
    slot = plant_slot_path(conf)
    for result_dir in list_result_dirs(slot):
        log_path = os.path.join(result_dir, 'log.txt')
        if os.path.exists(log_path):
            return True
    return False


def processing_limit_frame_count(conf):
    """First-N-days tracking cap as a frame count, using capture interval."""
    limit = (conf or {}).get('processingLimit', None)
    if limit in (None, '', 0, '0'):
        return None
    try:
        days = int(round(float(limit)))
    except (TypeError, ValueError):
        return None
    if days <= 0:
        return None
    try:
        time_step = int((conf or {}).get('timeStep') or 15)
    except (TypeError, ValueError):
        time_step = 15
    if time_step <= 0:
        time_step = 15
    frames_per_day = (24 * 60) // time_step
    return days * frames_per_day


def expected_hourly_rows(n_frames, time_step):
    """How many hourly bins a processing-limit frame cap should produce."""
    try:
        step = float(time_step)
    except (TypeError, ValueError):
        step = 15.0
    if step <= 0:
        step = 15.0
    frames_per_hour = max(1, int(round(60.0 / step)))
    try:
        n_frames = int(n_frames)
    except (TypeError, ValueError):
        return None
    if n_frames <= 0:
        return None
    return (n_frames + frames_per_hour - 1) // frames_per_hour


def cleanup_superseded_results(plant_path, keep_result_path):
    """Delete all Results_* folders except the one to keep."""
    keep_result_path = os.path.abspath(keep_result_path)
    for result_dir in list_result_dirs(plant_path):
        if os.path.abspath(result_dir) != keep_result_path and os.path.isdir(result_dir):
            shutil.rmtree(result_dir, ignore_errors=True)


def createSaveFolder(conf):
    # Create the folder for the general results
    analysis = os.path.join(conf['MainFolder'], 'Analysis')
    os.makedirs(analysis, exist_ok=True)
    
    # Create the folder for the identifier
    identifier = convertToPathSafe(conf['Experiment'])
    
    id_path = os.path.join(analysis, identifier)
    os.makedirs(id_path, exist_ok=True)

    # Create the folder for the rpi
    rpi = str(conf['rpi'])
    rpi_path = os.path.join(id_path, rpi)
    os.makedirs(rpi_path, exist_ok=True)
    
    # Create the folder for the cam
    cam = "cam_" + str(str(conf['cam']))
    cam_path = os.path.join(rpi_path, cam)
    os.makedirs(cam_path, exist_ok=True)
    
    # Create the folder for the plant
    plant = "plant_" + str(str(conf['plant']))
    plant_path = os.path.join(cam_path, plant)
    os.makedirs(plant_path, exist_ok=True)
    
    # Create the folder for the results
    for j in range(0, 50):
        result_path = os.path.join(plant_path, 'Results_%s'%j)
        if not os.path.exists(result_path):
            os.makedirs(result_path)
            break
    
    # create folders for outputs
    graphsPath = os.path.join(result_path, 'Graphs')
    os.makedirs(graphsPath, exist_ok=True)
    
    imagePath = os.path.join(result_path, 'Images')
    os.makedirs(imagePath, exist_ok=True)
    
    outSegPath = os.path.join(imagePath, 'Seg')
    os.makedirs(outSegPath, exist_ok=True)
        
    multiPath = os.path.join(imagePath, 'SegMulti')
    os.makedirs(multiPath, exist_ok=True)
    
    if conf['saveImages']:
        inPath = os.path.join(imagePath, 'Input')
        os.makedirs(inPath, exist_ok=True)

    rsmlPath = os.path.join(result_path, 'RSML')
    os.makedirs(rsmlPath, exist_ok=True)
    
    # creates a dictionary with all the paths
    paths = {'analysis': analysis, 'result': result_path, 'graphs': graphsPath, 'images': imagePath, 'rsml': rsmlPath}

    return paths

def getImages(conf):
    # Get the list of images    
    images = loadPath(conf['Images'], ext = "*.png") 

    conf['ImagePath'] = conf['Images']
                
    # Check if there is no images, then look for a file called "segmentation_metadata.json"
    if len(images) == 0:
        metadata_path = os.path.join(conf['Images'], 'Segmentation', 'segmentation_metadata.json')
        #print("No images found in the specified folder. Looking for segmentation metadata file at: ", metadata_path)
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            images_path = metadata.get('input_path', None)
            if images_path and os.path.exists(images_path):
                images = loadPath(images_path, ext="*.png")
                conf['ImagePath'] = images_path
    
    # Get the list of segmentation images
    SegPath = os.path.join(conf['Images'], 'Segmentation', 'Ensemble')
    if not os.path.exists(SegPath):
        SegPath = os.path.join(conf['Images'], 'Seg')
    
    segFiles = loadPath(SegPath, ext = "*.png") 

    # Save configuration
    
    conf['SegPath'] = SegPath
        
    return images, segFiles

import json

UNSPECIFIED_FACTOR = "unspecified"


def normalize_factor_value(value):
    if value is None:
        return UNSPECIFIED_FACTOR
    text = str(value).strip()
    return text if text else UNSPECIFIED_FACTOR


def load_result_metadata(res_folder):
    meta_path = os.path.join(res_folder, 'metadata.json')
    if not os.path.exists(meta_path):
        return {}
    try:
        with open(meta_path, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def attach_plant_metadata_columns(data, meta, rpi_name=None, cam_name=None, plant_name=None,
                                  experiment_fallback=None):
    """Attach experiment and grouping columns from per-plant metadata."""
    experiment = meta.get('Experiment', experiment_fallback)
    if experiment is not None:
        data['Experiment'] = experiment
    data['PlateCondition'] = normalize_factor_value(meta.get('PlateCondition', ''))
    data['ExtraVariable'] = normalize_factor_value(meta.get('ExtraVariable', ''))
    if rpi_name is not None:
        data['rpi'] = rpi_name
    if cam_name is not None:
        data['cam'] = cam_name
    if plant_name is not None:
        data['plant'] = plant_name
    return data


def build_plant_id(rpi_name, cam_name, plant_name, extra_variable=None):
    """Hardware slot identity only; ExtraVariable is a separate stats column."""
    return f"{rpi_name}_{cam_name}_{plant_name}"


def saveMetadata(bbox, seed, conf):
    metadata = {}
    metadata['bounding box'] = bbox
    metadata['seed'] = seed

    # combine metadata and conf
    metadata.update(conf)

    metapath = os.path.join(metadata['folders']['result'], 'metadata.json')

    with open(metapath, 'w') as fp:
        json.dump(metadata, fp)

    metapath = os.path.join(metadata['MainFolder'], 'lastAnalysis.json')
    with open(metapath, 'w') as fp:
        json.dump(metadata, fp)

    return metadata