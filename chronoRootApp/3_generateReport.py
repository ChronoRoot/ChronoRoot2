""" 
ChronoRoot: High-throughput phenotyping by deep learning reveals novel temporal parameters of plant root system architecture
Copyright (C) 2020 Nicolás Gaggion
"""

import shutil
import pandas as pd
import os
import json
import argparse

from analysis.utils import report_utils as utils
from analysis import convex_hull
from analysis.report import (
    plot_individual_plant,
    plot_info_all,
    performStatisticalAnalysisForMetrics,
    generateTableTemporal,
)
from analysis.stats_utils import (
    get_enabled_comparison_modes,
    log_auto_disabled_modes,
)

from analysis.fourier_analysis import makeFourierPlots
from analysis.lateral_angles import makeLateralAnglesPlots, plotLateralAnglesOnTop
from analysis.fpca_analysis import performFPCA
from analysis.utils.fileUtilities import (
    convertFromPathSafe,
    get_latest_result_dir,
    load_result_metadata,
    attach_plant_metadata_columns,
    build_plant_id,
    normalize_factor_value,
)
from analysis.time_windows import apply_time_windows, snapshot_hours
from analysis.utils.report_paths import (
    data_file,
    individual_plots_dir,
    overview_dir,
    MODULE_CONVEX,
    purge_disabled_comparison_outputs,
    report_root,
)


def _experiment_display_name(exp_dir):
    exp_dir_name = os.path.basename(exp_dir)
    real_exp_name = convertFromPathSafe(exp_dir_name)
    first_meta = utils.load_paths(exp_dir, '*/*/*/metadata.json')
    if first_meta:
        try:
            with open(first_meta[0], 'r') as f:
                meta_data = json.load(f)
                real_exp_name = meta_data.get('Experiment', real_exp_name)
        except Exception:
            pass
    return exp_dir_name, real_exp_name


def _normalize_temporal_dataframe(all_data):
    all_data['Experiment'] = all_data['Experiment'].astype(str)
    if 'PlateCondition' in all_data.columns:
        all_data['PlateCondition'] = all_data['PlateCondition'].apply(normalize_factor_value)
    if 'ExtraVariable' in all_data.columns:
        all_data['ExtraVariable'] = all_data['ExtraVariable'].apply(normalize_factor_value)
    return all_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ChronoRoot: Report Generation')
    parser.add_argument('--config', type=str, help='Path to the configuration file (default: config.json)')

    args = parser.parse_args()
    conf = json.load(open(args.config, 'r'))

    analysis_folder = os.path.join(conf['MainFolder'], 'Analysis')
    experiments = utils.load_paths(analysis_folder, '*')

    utils.ensure_directory(report_root(conf))

    print("Report generation began. This may take a while.")

    temporal_parameters = conf.get('temporalMetrics') or [
        'MainRootLength (mm)', 'LateralRootsLength (mm)', 'TotalLength (mm)',
        'NumberOfLateralRoots', 'DiscreteLateralDensity (LR/cm)', 'MainOverTotal (%)',
        'HypocotylLength (mm)',
    ]

    temporal_data_path = data_file(conf, 'Temporal_Data.csv')
    convex_data_path = data_file(conf, 'Convex_Hull_Data.csv')

    # --- Phase 1: temporal aggregate + individual plots ---
    print('Phase 1/3: aggregating temporal data...')
    plant_frames = []
    n_experiments = len(experiments)

    for exp_index, exp_dir in enumerate(experiments, start=1):
        exp_dir_name, real_exp_name = _experiment_display_name(exp_dir)
        print(f'Phase 1/3: experiment {exp_index}/{n_experiments}: {real_exp_name}')

        iplots_exp_folder = individual_plots_dir(conf, exp_dir_name)

        rpi_paths = utils.load_paths(exp_dir, '*')
        for rpi in rpi_paths:
            rpi_name = os.path.basename(rpi)
            cam_paths = utils.load_paths(rpi, '*')
            for cam in cam_paths:
                cam_name = os.path.basename(cam)
                plant_paths = utils.load_paths(cam, '*')
                for plant in plant_paths:
                    plant_name = os.path.basename(plant)
                    res_folder = get_latest_result_dir(plant)
                    if res_folder is None:
                        continue

                    meta = load_result_metadata(res_folder)
                    plant_id = build_plant_id(rpi_name, cam_name, plant_name)
                    file_csv = os.path.join(res_folder, 'PostProcess_Hour.csv')

                    if not os.path.exists(file_csv):
                        continue

                    data = pd.read_csv(file_csv)
                    data['Plant_id'] = plant_id
                    data = attach_plant_metadata_columns(
                        data, meta,
                        rpi_name=rpi_name,
                        cam_name=cam_name,
                        plant_name=plant_name,
                        experiment_fallback=real_exp_name,
                    )
                    plant_frames.append(data)

                    plot_filename = f"{exp_dir_name}_{plant_id}.png"
                    iplot_cache = os.path.join(res_folder, plot_filename)
                    report_dest = os.path.join(iplots_exp_folder, plot_filename)

                    if not os.path.exists(iplot_cache):
                        plot_individual_plant(iplots_exp_folder, data, plot_filename)
                        if os.path.exists(report_dest):
                            shutil.copy(report_dest, iplot_cache)
                    else:
                        shutil.copy(iplot_cache, report_dest)

    all_data = pd.concat(plant_frames, ignore_index=True) if plant_frames else pd.DataFrame()
    if not all_data.empty:
        all_data = apply_time_windows(all_data, conf)
    all_data.to_csv(temporal_data_path, index=False)
    print(f'Phase 1/3: wrote {temporal_data_path}')

    all_data = _normalize_temporal_dataframe(all_data)

    config_modes = get_enabled_comparison_modes(conf)
    effective_modes = get_enabled_comparison_modes(conf, all_data)
    conf['effectiveComparisonModes'] = effective_modes
    log_auto_disabled_modes(conf, config_modes, effective_modes)

    # --- Phase 2: convex hull ---
    convex_hull_df = pd.DataFrame()
    if conf.get('doConvex'):
        print('Phase 2/3: convex hull analysis...')
        global_shape, global_center = convex_hull.calculate_atlas_geometry(experiments)
        snapshot_h = snapshot_hours(conf, kind='convex')
        timestep = int(conf['timeStep'])
        convex_overview = overview_dir(conf, MODULE_CONVEX)
        convex_frames = []
        n_experiments = len(experiments)

        for exp_index, exp_dir in enumerate(experiments, start=1):
            exp_dir_name, real_exp_name = _experiment_display_name(exp_dir)
            print(f'Phase 2/3: convex hull ({exp_index}/{n_experiments}): {real_exp_name}')

            atlases, current_convex_df = convex_hull.generate_root_atlases(
                exp_dir,
                snapshot_hours=snapshot_h,
                timestep=timestep,
                canvas_shape=global_shape,
                center_coords=global_center,
                rotate_root=True,
                conf=conf,
            )

            if not current_convex_df.empty:
                current_convex_df['Experiment'] = real_exp_name
                convex_frames.append(current_convex_df)

            if conf.get('saveImagesConvex') and atlases:
                for i, hour in enumerate(snapshot_h):
                    at_hull, at_cont, at_root = atlases[i]
                    convex_hull.visualize_single_atlas(
                        at_hull, at_cont, at_root,
                        convex_overview, exp_dir_name, hour,
                    )
            elif atlases:
                at_hull, at_cont, at_root = atlases[-1]
                convex_hull.visualize_single_atlas(
                    at_hull, at_cont, at_root,
                    convex_overview, exp_dir_name,
                    snapshot_h[-1] if snapshot_h else None,
                )

        convex_hull_df = pd.concat(convex_frames, ignore_index=True) if convex_frames else pd.DataFrame()
        if not convex_hull_df.empty:
            convex_hull_df.to_csv(convex_data_path, index=False)
            print(f'Phase 2/3: wrote {convex_data_path}')

    purge_disabled_comparison_outputs(conf, effective_modes)

    # --- Phase 3: figures and statistics ---
    print('Phase 3/3: generating report figures and statistics...')

    performStatisticalAnalysisForMetrics(conf, all_data, temporal_parameters)

    plot_info_all(conf, all_data)
    generateTableTemporal(conf, all_data)

    if conf.get('doFPCA'):
        print('Phase 3/3: FPCA analysis')
        performFPCA(args.config)

    if conf.get('doConvex') and not convex_hull_df.empty:
        print('Phase 3/3: convex hull summary and statistics')
        convex_hull.plot_hull_metrics_summary(conf, convex_hull_df)
        convex_hull.visualize_combined_atlases(conf)

        saved_modes = conf.get('effectiveComparisonModes')
        conf['effectiveComparisonModes'] = get_enabled_comparison_modes(conf, convex_hull_df)

        convex_params = [
            'Convex Hull Area', 'Lateral Root Area Density', 'Total Root Area Density',
            'Convex Hull Aspect Ratio', 'Convex Hull Height', 'Convex Hull Width',
        ]
        for param in convex_params:
            print(f'Phase 3/3: convex hull — {param}')
            convex_hull.analyze_hull_statistics(conf, convex_hull_df, param)

        conf['effectiveComparisonModes'] = saved_modes

    if conf.get('doFourier'):
        print('Phase 3/3: Fourier analysis')
        makeFourierPlots(conf)

    if conf.get('doLateralAngles'):
        print('Phase 3/3: lateral angles analysis')
        makeLateralAnglesPlots(conf)
        plotLateralAnglesOnTop(conf)

    print("Report generation finished.")
