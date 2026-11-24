""" 
ChronoRoot: High-throughput phenotyping by deep learning reveals novel temporal parameters of plant root system architecture
Copyright (C) 2020 Nicolás Gaggion

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

    10|This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

from analysis.plantAnalysis import plantAnalysis
from analysis.utils.metadata_schema import (
    cleanup_pending_config,
    has_roi,
    load_run_config,
    video_has_qr,
)
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ChronoRoot: High-throughput phenotyping by deep learning reveals novel temporal parameters of plant root system architecture')
    parser.add_argument('--config', type=str, help='Path to the configuration file (default: config.json)')
    parser.add_argument('--rerun', action='store_true', default=False, help='Reruns the analysis, even if the results already exist')
    parser.add_argument('--restart', action='store_true', default=False, help='Reruns the analysis from scratch, ignoring any previous bounding box or root start point')
    
    args = parser.parse_args()
        
    conf = load_run_config(args.config)
    
    if args.restart:
        conf.pop('bounding_box', None)
        conf.pop('bounding box', None)
        conf.pop('seed', None)
    
    experiment = conf.get("Experiment") or conf.get("identifierField")
    conf['Experiment'] = experiment
    conf['fileKey'] = experiment
    conf['sequenceLabel'] = str(experiment) + "_" + str(conf.get('Images', '')) + "_" + str(conf.get('plant', ''))
    conf['Plant'] = 'Arabidopsis thaliana'
    
    if not video_has_qr(conf):
        pixel_size = float(conf['knownDistance']) / float(conf['pixelDistance'])
        conf['pixel_size'] = pixel_size

    try:
        plantAnalysis(conf, replicate=has_roi(conf))
    finally:
        cleanup_pending_config(args.config)
