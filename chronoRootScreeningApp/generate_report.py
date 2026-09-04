"""Screening report generation: postprocess via dataWork, then chronoRootApp reports."""

import argparse
import json
import os
import warnings

import pandas as pd

warnings.simplefilter(action='ignore', category=FutureWarning)

import chrono_root_backend  # noqa: F401  # chronoRootApp on sys.path

from chrono_root_backend import (
    MODULE_TEMPORAL,
    data_file,
    emit_scalar_comparison_plots,
    ensure_factor_columns,
    generateTableTemporal,
    get_enabled_comparison_modes,
    log_auto_disabled_modes,
    makeFourierPlots,
    metric_dir,
    performFPCA,
    performStatisticalAnalysisForMetrics,
    perform_scalar_pairwise_stats,
    plot_info_all,
    purge_disabled_comparison_outputs,
    temporal_metric_slug,
)
from analysis.utils.report_paths import analysis_dir
from analysis.utils.report_style import genotype_palette_for_data, get_genotype_axis_label
from skfda import FDataGrid
from skfda.preprocessing.dim_reduction import FPCA
from skfda.representation.basis import MonomialBasis
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns

from postprocess import postprocess_tracking, MAX_PLANT_WORKERS
from data_processing.germination_analysis import GerminationAnalyzer
from robot_ids import resolve_rpi_cam

SCREENING_TEMPORAL_METRICS = [
    'MainRootLength (mm)',
    'HypocotylLength (mm)',
    'TotalLength (mm)',
    'Area (mm2)',
    'DenseRootArea (mm2)',
]
RSA_FPCA_METRICS = {
    'MainRootLength (mm)',
    'LateralRootsLength (mm)',
    'TotalLength (mm)',
    'NumberOfLateralRoots',
    'MainOverTotal (%)',
    'DiscreteLateralDensity (LR/cm)',
    'HypocotylLength (mm)',
}
LATERAL_FPCA_METRICS = {
    'LateralRootsLength (mm)',
    'NumberOfLateralRoots',
    'MainOverTotal (%)',
    'DiscreteLateralDensity (LR/cm)',
}


def apply_screening_plot_defaults(conf, selected=None):
    """Skip SORT-unavailable lateral plots; keep RSA/area measures the user selected."""
    selected = list(selected if selected is not None else (conf.get('selectedMetrics') or SCREENING_TEMPORAL_METRICS))
    temporal_selected = [m for m in selected if m != 'GerminationTime']
    conf['includeLateralRootPlots'] = False
    conf['temporalOverviewMetrics'] = temporal_selected
    conf['fpcaMetrics'] = [
        m for m in temporal_selected
        if m in RSA_FPCA_METRICS and m not in LATERAL_FPCA_METRICS
    ]
    return temporal_selected


def merge_analysis_files(project_dir, name_mapping_file=None):
    analysis_dir_root = os.path.join(project_dir, 'analysis')
    all_data = []
    name_mapping = {}
    if name_mapping_file and os.path.exists(name_mapping_file):
        with open(name_mapping_file, 'r') as handle:
            name_mapping = json.load(handle)

    analyses = [
        name for name in os.listdir(analysis_dir_root)
        if os.path.isdir(os.path.join(analysis_dir_root, name))
    ]
    for analysis_id in analyses:
        analysis_path = os.path.join(analysis_dir_root, analysis_id, 'seeds.tsv')
        metadata_path = os.path.join(analysis_dir_root, analysis_id, 'group_info.json')
        if not os.path.exists(analysis_path):
            continue
        try:
            df = pd.read_csv(analysis_path, sep='\t')
            missing = [c for c in ('Experiment', 'Plant_id') if c not in df.columns]
            if missing:
                print(
                    f'Skipping {analysis_path}: missing {missing}. '
                    'Re-run tracking; old Group/UID seeds.tsv files are not supported.'
                )
                continue
            df['Experiment'] = df['Experiment'].astype(str)
            df['Plant_id'] = df['Plant_id'].astype(str)
            df['OriginalExperiment'] = df['Experiment']
            if name_mapping:
                df['Experiment'] = df['Experiment'].map(lambda x: name_mapping.get(x, x))

            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as handle:
                    group_info = json.load(handle)
                counts = pd.DataFrame({
                    'Experiment': [str(x) for x in group_info.get('group_names', [])],
                    'SeedCount': group_info.get('seed_counts', []),
                })
                if name_mapping and not counts.empty:
                    counts['Experiment'] = counts['Experiment'].map(
                        lambda x: name_mapping.get(x, x)
                    )
                if not counts.empty:
                    df = df.merge(counts, on='Experiment', how='left')
                if 'PlateCondition' not in df.columns:
                    df['PlateCondition'] = group_info.get('PlateCondition', '')
                if 'ExtraVariable' not in df.columns:
                    df['ExtraVariable'] = group_info.get('ExtraVariable', '')
                if 'pixel_size' not in df.columns and 'pixel_size' in group_info:
                    df['pixel_size'] = group_info['pixel_size']
                if 'rpi' not in df.columns:
                    df['rpi'] = group_info.get('rpi', '')
                if 'cam' not in df.columns:
                    df['cam'] = group_info.get('cam', '')

            df['Video'] = analysis_id
            rpi_val = ''
            cam_val = ''
            video_dir = ''
            if not df.empty:
                if 'rpi' in df.columns:
                    rpi_val = df['rpi'].iloc[0]
                if 'cam' in df.columns:
                    cam_val = df['cam'].iloc[0]
            tracking_meta_path = os.path.join(analysis_dir_root, analysis_id, 'metadata.json')
            if os.path.exists(tracking_meta_path):
                try:
                    with open(tracking_meta_path, 'r') as handle:
                        tracking_meta = json.load(handle)
                    video_dir = tracking_meta.get('video_directory') or tracking_meta.get('video_dir') or ''
                    if not rpi_val:
                        rpi_val = tracking_meta.get('rpi', '')
                    if not cam_val:
                        cam_val = tracking_meta.get('cam', '')
                except (OSError, ValueError, TypeError):
                    pass
            rpi, cam = resolve_rpi_cam(
                rpi=rpi_val,
                cam=cam_val,
                video_dir=video_dir,
                analysis_id=analysis_id,
            )
            df['rpi'] = rpi
            df['cam'] = cam
            if 'SeedCount' not in df.columns:
                df['SeedCount'] = 0
            else:
                df['SeedCount'] = df['SeedCount'].fillna(0)
            all_data.append(df)
        except Exception as exc:
            print(f'Error reading {analysis_path}: {exc}')

    if not all_data:
        raise ValueError('No valid seeds.tsv files found in any analysis folder')
    return pd.concat(all_data, ignore_index=True)


def _run_fpca_for_metrics(conf, columns):
    """FPCA + comparison stats for screening-only columns (Area, dense root area)."""
    from analysis.utils.report_paths import plot_file

    temporal_data_df = pd.read_csv(data_file(conf, 'Temporal_Data.csv'))
    temporal_data_df = ensure_factor_columns(temporal_data_df)
    temporal_data_df['Experiment'] = temporal_data_df['Experiment'].astype(str)
    temporal_data_df = temporal_data_df.sort_values(by='Experiment')
    temporal_data_df['Plant_id'] = (
        temporal_data_df['Plant_id'].astype(str) + ' (' + temporal_data_df['Experiment'] + ')'
    )
    available = [col for col in columns if col in temporal_data_df.columns]
    if not available:
        return

    inverse_rank_normalize = conf['normFPCA']
    number_of_components = int(conf['numComponentsFPCAField'])
    genotype_palette = genotype_palette_for_data(temporal_data_df)
    genotype_legend = get_genotype_axis_label(conf)
    get_expid = temporal_data_df.set_index('Plant_id')['Experiment'].to_dict()

    plt.switch_backend('agg')
    plt.ioff()

    for magnitude in available:
        mag_slug = temporal_metric_slug(magnitude)
        pivoted = temporal_data_df.pivot(
            columns='Plant_id', values=magnitude, index='ElapsedTime (h)'
        ).dropna()
        if pivoted.empty or pivoted.shape[1] < 2:
            print(f'FPCA skipped for {magnitude}: not enough complete series')
            continue

        fpca = FPCA(n_components=number_of_components, components_basis=MonomialBasis)
        fpc_values = fpca.fit_transform(FDataGrid(pivoted.transpose()))
        fpc_df = pd.DataFrame(fpc_values).set_index(pivoted.columns)
        fpc_df.columns = [f'PC{i}' for i in range(1, fpca.n_components + 1)]
        fpc_df = fpc_df.reset_index()
        fpc_df['Experiment'] = fpc_df.Plant_id.map(get_expid)
        plant_meta = temporal_data_df.drop_duplicates('Plant_id').set_index('Plant_id')
        for col in ['PlateCondition', 'ExtraVariable']:
            if col in plant_meta.columns:
                fpc_df[col] = fpc_df.Plant_id.map(plant_meta[col].to_dict())
        fpc_df = ensure_factor_columns(fpc_df)
        for j in range(1, fpca.n_components + 1):
            fpc_df[f'PC{j}_IRN'] = norm.ppf(fpc_df[f'PC{j}'].rank() / (len(fpc_df) + 1))

        suffix = '_IRN' if inverse_rank_normalize else ''
        plt.figure(figsize=(8, 4))
        ax = plt.subplot(1, 2, 1)
        sns.lineplot(
            x='ElapsedTime (h)', y=magnitude, hue='Experiment',
            data=temporal_data_df, errorbar='se', palette=genotype_palette, ax=ax,
        )
        ax.set_title(magnitude)
        ax.legend(title=genotype_legend)
        ax = plt.subplot(1, 2, 2)
        sns.scatterplot(
            data=fpc_df, x='PC1' + suffix, y='PC2' + suffix,
            hue='Experiment', palette=genotype_palette, s=100, ax=ax,
        )
        ax.set_title('PC1 vs PC2')
        ax.legend(title=genotype_legend, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        for ext in ('png', 'svg'):
            plt.savefig(
                plot_file(conf, MODULE_TEMPORAL, mag_slug, f'{mag_slug}_overview.{ext}', 'fpca'),
                dpi=300, bbox_inches='tight',
            )
        plt.close('all')

        for fpc1 in range(1, number_of_components + 1):
            pc_col = f'PC{fpc1}{suffix}'
            pc_label = f'{magnitude} — PC{fpc1}'
            pc_dir = analysis_dir(conf, MODULE_TEMPORAL, mag_slug, 'fpca', f'pc{fpc1}')
            perform_scalar_pairwise_stats(
                conf, fpc_df, pc_col, output_dir=None, plant_id_col='Plant_id',
                module=MODULE_TEMPORAL, metric_slug_name=mag_slug,
                subpath=('fpca', f'pc{fpc1}'), analysis_type='fpca',
            )
            emit_scalar_comparison_plots(
                conf, fpc_df, pc_col, pc_dir,
                module=MODULE_TEMPORAL, metric_slug_name=mag_slug,
                analysis_type='fpca', metric_label=pc_label,
            )


def _germination_pairwise_stats(conf, germ_analyzer):
    germ = germ_analyzer.germination_data
    if germ is None or germ.empty:
        return
    scalar = germ.dropna(subset=['GerminationTime']).copy()
    if scalar.empty:
        return
    if 'Experiment' not in scalar.columns:
        print('Germination pairwise stats skipped: no Experiment column')
        return
    scalar = ensure_factor_columns(scalar)
    slug = 'germination_time'
    base_dir = metric_dir(conf, 'germination', slug)
    perform_scalar_pairwise_stats(
        conf, scalar, 'GerminationTime', output_dir=None, plant_id_col='Plant_id',
        module='germination', metric_slug_name=slug, analysis_type='scalar',
    )
    emit_scalar_comparison_plots(
        conf, scalar, 'GerminationTime', base_dir,
        module='germination', metric_slug_name=slug,
        analysis_type='scalar', metric_label='Germination time',
    )


def main():
    parser = argparse.ArgumentParser(description='ChronoRoot Screening: Report Generation')
    parser.add_argument('--config', required=True, help='Path to report_config.json')
    args = parser.parse_args()

    conf = json.load(open(args.config, 'r'))
    project_dir = conf['MainFolder']
    selected = conf.get('selectedMetrics') or SCREENING_TEMPORAL_METRICS
    temporal_selected = apply_screening_plot_defaults(conf, selected)
    do_growth = conf.get('doPlantGrowth', True)
    do_germination = conf.get('doGermination', True)
    do_fpca = conf.get('doFPCA', False)
    do_fourier = conf.get('doFourier', False)

    print('Merging tracking files...')
    combined = merge_analysis_files(project_dir, conf.get('nameMapping'))
    raw_path = data_file(conf, 'Raw_Data.tsv')
    combined.to_csv(raw_path, sep='\t', index=False)
    print(f'Wrote {raw_path}')

    all_data = pd.DataFrame()
    if do_growth:
        print(f'Postprocessing tracking through dataWork (up to {MAX_PLANT_WORKERS} workers)...')
        all_data = postprocess_tracking(combined, conf)
        all_data = ensure_factor_columns(all_data)

        config_modes = get_enabled_comparison_modes(conf)
        effective_modes = get_enabled_comparison_modes(conf, all_data)
        conf['effectiveComparisonModes'] = effective_modes
        log_auto_disabled_modes(conf, config_modes, effective_modes)
        purge_disabled_comparison_outputs(conf, effective_modes)

        temporal_selected = [m for m in temporal_selected if m != 'GerminationTime']
        performStatisticalAnalysisForMetrics(conf, all_data, temporal_selected)

        plot_info_all(conf, all_data)
        generateTableTemporal(conf, all_data)

        if do_fpca:
            print('FPCA analysis')
            if conf.get('fpcaMetrics'):
                performFPCA(conf)
            extra = [m for m in temporal_selected if m not in RSA_FPCA_METRICS]
            if extra:
                _run_fpca_for_metrics(conf, extra)

        if do_fourier:
            print('Fourier analysis')
            try:
                makeFourierPlots(conf)
            except Exception as exc:
                print(f'Fourier analysis skipped: {exc}')

    if do_germination:
        print('Germination analysis...')
        germ_dir = os.path.join(project_dir, 'Report')
        os.makedirs(germ_dir, exist_ok=True)
        germ_analyzer = GerminationAnalyzer(
            data=combined,
            output_dir=germ_dir,
            dt=conf.get('timeStep', 15),
            add_time_before_photo=conf.get('addTimeBeforePhoto', 0),
            store_for_each_video=conf.get('germinationEachVideo', False),
            time_cut=conf.get('germinationTimeCut', 0),
        )
        germ_analyzer.analyze()
        if 'GerminationTime' in selected:
            if not all_data.empty:
                saved = conf.get('effectiveComparisonModes')
                germ_rows = germ_analyzer.germination_data
                if germ_rows is not None and not germ_rows.empty:
                    germ_rows = ensure_factor_columns(germ_rows)
                    conf['effectiveComparisonModes'] = get_enabled_comparison_modes(conf, germ_rows)
                _germination_pairwise_stats(conf, germ_analyzer)
                conf['effectiveComparisonModes'] = saved
            else:
                _germination_pairwise_stats(conf, germ_analyzer)

    print('Report generation finished.')
    print(f'Results saved in: {os.path.join(project_dir, "Report")}')


if __name__ == '__main__':
    main()
