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

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
import scipy.stats as stats
import numpy as np
import logging
logging.getLogger('matplotlib.category').setLevel(logging.ERROR)

# remove FutureWarning
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

plt.switch_backend('agg')

from .utils.fileUtilities import convertFromPathSafe, convertToPathSafe
from .utils.report_paths import (
    MODULE_TEMPORAL,
    REPLICAS_DIR,
    analysis_dir,
    clear_replicas_subdir,
    metric_dir,
    overview_dir,
    table_file,
    temporal_metric_slug,
)
from .stats_utils import (
    perform_temporal_pairwise_stats,
    ensure_factor_columns,
    iter_replica_runs,
)
from .report_plots import emit_temporal_comparison_plots, relative_title, subplot_grid
from .time_windows import draw_clock_ticks, elapsed_hour_windows
from .utils.report_style import genotype_palette_for_data, get_genotype_axis_label


DEFAULT_TEMPORAL_METRICS = [
    'MainRootLength (mm)', 'LateralRootsLength (mm)', 'TotalLength (mm)',
    'NumberOfLateralRoots', 'DiscreteLateralDensity (LR/cm)', 'MainOverTotal (%)',
    'HypocotylLength (mm)',
]

DEFAULT_OVERVIEW_METRICS = [
    ('MainRootLength (mm)', 'Main root length'),
    ('LateralRootsLength (mm)', 'Lateral root length'),
    ('TotalLength (mm)', 'Total root length'),
    ('NumberOfLateralRoots', 'Number of lateral roots'),
    ('DiscreteLateralDensity (LR/cm)', 'Discrete lateral root density'),
    ('MainOverTotal (%)', 'Main root / total length (%)'),
]

LATERAL_SUMMARY_COLUMNS = (
    'LateralRootsLength (mm)',
    'NumberOfLateralRoots',
    'DiscreteLateralDensity (LR/cm)',
    'MainOverTotal (%)',
)

METRIC_TITLES = {
    'MainRootLength (mm)': 'Main root length',
    'LateralRootsLength (mm)': 'Lateral root length',
    'TotalLength (mm)': 'Total root length',
    'NumberOfLateralRoots': 'Number of lateral roots',
    'DiscreteLateralDensity (LR/cm)': 'Discrete lateral root density',
    'MainOverTotal (%)': 'Main root / total length (%)',
    'HypocotylLength (mm)': 'Hypocotyl length',
    'Area (mm2)': 'Plant area',
    'DenseRootArea (mm2)': 'Dense root area',
}

MAX_METRIC_WORKERS = 4


def _include_lateral_root_plots(conf):
    if conf is None:
        return True
    return conf.get('includeLateralRootPlots', True)


def _add_day_axis(ax, dataframe, day_tick_size=12, conf=None):
    return draw_clock_ticks(ax, dataframe, conf, twin_axis=True)


def plot_individual_plant(savepath, dataframe, name, conf=None):
    plt.ioff()
    
    # Define font sizes for consistency across subplots
    LABEL_SIZE = 18
    TICK_SIZE = 16
    TITLE_SIZE = 18
    LEGEND_SIZE = 16
    DAY_TICK_SIZE = 12

    include_laterals = _include_lateral_root_plots(conf)

    if include_laterals:
        fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(9, 10), dpi=150, sharex=True)
    else:
        fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(9, 6), dpi=150)
        ax2 = None

    dataframe.plot(x='ElapsedTime (h)', y='MainRootLength (mm)', ax=ax1, color='g', label='Main Root Length')
    if include_laterals and 'LateralRootsLength (mm)' in dataframe.columns:
        dataframe.plot(x='ElapsedTime (h)', y='LateralRootsLength (mm)', ax=ax1, color='b', label='Lateral Roots Length')
    dataframe.plot(x='ElapsedTime (h)', y='HypocotylLength (mm)', ax=ax1, color='r', label='Hypocotyl Length')
    
    ax1.set_title('%s' % convertFromPathSafe(name), pad=40, fontsize=TITLE_SIZE)
    ax1.set_ylabel('Length (mm)', fontsize=LABEL_SIZE)
    ax1.tick_params(axis='y', which='major', labelsize=TICK_SIZE)
    ax1.legend(fontsize=LEGEND_SIZE, loc='upper left')

    if ax2 is not None:
        ax1.set_xlabel('')
        dataframe.plot(x='ElapsedTime (h)', y='NumberOfLateralRoots', ax=ax2, color='m', legend=False)
        ax2.set_ylabel('Number of Lateral Roots', fontsize=LABEL_SIZE)
        ax2.set_xlabel('Elapsed Time (h)', fontsize=LABEL_SIZE)
        ax2.tick_params(axis='both', which='major', labelsize=TICK_SIZE)
    else:
        ax1.set_xlabel('Elapsed Time (h)', fontsize=LABEL_SIZE)

    _add_day_axis(ax1, dataframe, DAY_TICK_SIZE, conf=conf)

    plt.tight_layout()

    fig.savefig(os.path.join(savepath, name), dpi=150, bbox_inches='tight')
        
    plt.cla()
    plt.clf()
    plt.close('all')

def performStatisticalAnalysis(conf, data, metric):
    data = ensure_factor_columns(data)
    slug = temporal_metric_slug(metric)
    table_path = table_file(conf, MODULE_TEMPORAL, slug, 'summary_table.csv')
    perform_temporal_pairwise_stats(
        conf, data, metric,
        module=MODULE_TEMPORAL, metric_slug_name=slug,
        table_file_path=table_path,
    )
    _write_metric_summary_table(conf, data, metric, slug)
    base_dir = metric_dir(conf, MODULE_TEMPORAL, slug)
    emit_temporal_comparison_plots(
        conf, data, metric, base_dir,
        module=MODULE_TEMPORAL, metric_slug_name=slug,
        metric_label=metric,
    )
    clear_replicas_subdir(base_dir)
    for _extra, extra_slug, subset, replica_conf in iter_replica_runs(conf, data):
        replica_dir = analysis_dir(conf, MODULE_TEMPORAL, slug, REPLICAS_DIR, extra_slug)
        replica_subpath = (REPLICAS_DIR, extra_slug)
        perform_temporal_pairwise_stats(
            replica_conf, subset, metric,
            module=MODULE_TEMPORAL, metric_slug_name=slug,
            subpath=replica_subpath,
        )
        emit_temporal_comparison_plots(
            replica_conf, subset, metric, replica_dir,
            module=MODULE_TEMPORAL, metric_slug_name=slug,
            metric_label=metric,
        )
    return


def _run_one_temporal_metric(job):
    """Spawn-safe worker: one temporal metric's stats and comparison plots."""
    conf = job['conf']
    data = job['data']
    metric = job['metric']
    print(f'Temporal analysis — {metric}', flush=True)
    performStatisticalAnalysis(conf, data, metric)
    return metric


def performStatisticalAnalysisForMetrics(conf, data, metrics, max_workers=MAX_METRIC_WORKERS):
    """Run performStatisticalAnalysis for each metric, in parallel when there are several."""
    selected = []
    for metric in metrics:
        if metric in data.columns:
            selected.append(metric)
        else:
            print(f'Skipping {metric}: column not in Temporal_Data', flush=True)
    metrics = selected
    if not metrics:
        return
    if len(metrics) == 1:
        print(f'Temporal analysis — {metrics[0]}', flush=True)
        performStatisticalAnalysis(conf, data, metrics[0])
        print('Metrics 1/1', flush=True)
        return

    ctx = multiprocessing.get_context('spawn')
    workers = min(max_workers, len(metrics))
    total = len(metrics)
    print(f'Temporal analysis ({total} metrics, up to {workers} workers)...', flush=True)
    jobs = [{'conf': conf, 'data': data, 'metric': metric} for metric in metrics]
    done = 0
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as executor:
        futures = [executor.submit(_run_one_temporal_metric, job) for job in jobs]
        for future in as_completed(futures):
            metric = future.result()
            done += 1
            print(f'Metrics {done}/{total} ({metric} done)', flush=True)


def _write_metric_summary_table(conf, data, metric, slug):
    """Per-metric descriptive summary across intervals and grouping factors."""
    dt = int(conf['everyXhourField'])
    rows = []
    for start, end in elapsed_hour_windows(data, dt):
        hours = np.arange(start, end)
        subdata = data[data['ElapsedTime (h)'].isin(hours)]
        subdata = subdata.groupby(
            ['Experiment', 'PlateCondition', 'ExtraVariable', 'Plant_id']
        ).mean(numeric_only=True).reset_index()
        grouped = subdata.groupby(['Experiment', 'PlateCondition', 'ExtraVariable']).agg(
            n_plants=('Plant_id', 'nunique'),
            mean=(metric, 'mean'),
            sd=(metric, 'std'),
        ).reset_index()
        grouped['hours_interval'] = f'{start}-{end - 1}'
        rows.append(grouped)

    if not rows:
        return

    result = pd.concat(rows, ignore_index=True)
    result = result.round(3)
    result.to_csv(table_file(conf, MODULE_TEMPORAL, slug, 'summary_table.csv'), index=False)


def _build_temporal_summary_table(data, group_cols, dt, include_laterals=True):
    summary_df = []

    agg_cols = {
        'MainRootLength (mm)': ['count', 'mean', 'std'],
        'LateralRootsLength (mm)': ['mean', 'std'],
        'TotalLength (mm)': ['mean', 'std'],
        'NumberOfLateralRoots': ['mean', 'std'],
        'DiscreteLateralDensity (LR/cm)': ['mean', 'std'],
        'MainOverTotal (%)': ['mean', 'std'],
        'HypocotylLength (mm)': ['mean', 'std'],
    }
    if not include_laterals:
        for col in LATERAL_SUMMARY_COLUMNS:
            agg_cols.pop(col, None)
    agg_cols = {col: spec for col, spec in agg_cols.items() if col in data.columns}
    if not agg_cols:
        return pd.DataFrame()

    for start, end in elapsed_hour_windows(data, dt):
        hours = np.arange(start, end)
        subdata = data[data['ElapsedTime (h)'].isin(hours)]
        subdata = subdata.groupby(group_cols + ['Plant_id']).mean(numeric_only=True).reset_index()
        subdata = subdata.groupby(group_cols).agg(agg_cols)
        subdata.columns = [' '.join(col).strip() for col in subdata.columns.values]
        subdata = subdata.reset_index()
        subdata['Hours interval'] = f'{start}-{end - 1}'
        summary_df.append(subdata)

    if not summary_df:
        return pd.DataFrame()

    result = pd.concat(summary_df)
    if 'MainRootLength (mm) count' in result.columns:
        result.rename(columns={'MainRootLength (mm) count': 'N experiment'}, inplace=True)
    col = result.pop('Hours interval')
    result.insert(0, col.name, col)
    return result


def generateTableTemporal(conf, data):
    data = ensure_factor_columns(data)
    dt = int(conf['everyXhourField'])
    include_laterals = _include_lateral_root_plots(conf)

    tables = [
        (_build_temporal_summary_table(
            data, ['Experiment'], dt, include_laterals), 'summary_by_genotype.csv'),
        (_build_temporal_summary_table(
            data, ['PlateCondition', 'Experiment'], dt, include_laterals),
         'summary_by_plate.csv'),
        (_build_temporal_summary_table(
            data, ['ExtraVariable', 'Experiment'], dt, include_laterals),
         'summary_by_extra_variable.csv'),
    ]

    for table, filename in tables:
        if not table.empty:
            table.to_csv(os.path.join(overview_dir(conf, MODULE_TEMPORAL), filename), index=False)
    
def _overview_metric_pairs(conf, dataframe):
    requested = conf.get('temporalOverviewMetrics')
    if requested:
        pairs = []
        for item in requested:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                col, title = item[0], item[1]
            else:
                col, title = item, METRIC_TITLES.get(item, str(item))
            if col in dataframe.columns:
                pairs.append((col, title))
        if pairs:
            return pairs
    return [(col, title) for col, title in DEFAULT_OVERVIEW_METRICS if col in dataframe.columns]


def _overview_ylabel(col):
    if col == 'NumberOfLateralRoots':
        return 'Number of LR'
    if col == 'DiscreteLateralDensity (LR/cm)':
        return 'Discrete LR density (LRs/cm)'
    if col == 'MainOverTotal (%)':
        return 'Percentage (%)'
    if 'mm2' in col.lower():
        return 'Area (mm²)'
    return 'Length (mm)'


def plot_info_all(conf, dataframe):
    plt.ioff()
    dataframe = ensure_factor_columns(dataframe)
    geno_palette = genotype_palette_for_data(dataframe)
    geno_label = get_genotype_axis_label(conf)
    metrics = _overview_metric_pairs(conf, dataframe)
    if not metrics:
        return

    def _plot_metric(ax, y_col, title):
        plot_df = dataframe.dropna(subset=[y_col]) if y_col in dataframe.columns else dataframe
        sns.lineplot(
            x='ElapsedTime (h)', y=y_col, data=plot_df, hue='Experiment',
            errorbar='se', ax=ax, palette=geno_palette,
        )
        draw_clock_ticks(ax, plot_df, conf, twin_axis=False)
        ax.set_title(title, fontsize=16)
        ax.legend(loc='best', title=geno_label)

    n = len(metrics)
    nrows, ncols = subplot_grid(n)
    fig3 = plt.figure(figsize=(4 * ncols, 4 * nrows), constrained_layout=True)
    gs = fig3.add_gridspec(nrows, ncols)
    axes = [fig3.add_subplot(gs[i // ncols, i % ncols]) for i in range(n)]

    for ax, (col, title) in zip(axes, metrics):
        _plot_metric(ax, col, relative_title(title, conf))
        ax.set_xlabel('Elapsed Time (h)', fontsize=12)
        ax.set_ylabel(_overview_ylabel(col), fontsize=12)

    plt.savefig(os.path.join(overview_dir(conf, MODULE_TEMPORAL), 'all_metrics_subplots.png'), dpi=300, bbox_inches='tight')

    plt.cla()
    plt.clf()
    plt.close('all')

    fig_n = plt.figure(figsize=(4 * ncols, 4 * nrows), constrained_layout=True)
    gs_n = fig_n.add_gridspec(nrows, ncols)
    axes_n = [fig_n.add_subplot(gs_n[i // ncols, i % ncols]) for i in range(n)]
    for ax, (col, title) in zip(axes_n, metrics):
        n_df = dataframe.copy()
        n_df['_n'] = n_df[col].notna().astype(int)
        sns.lineplot(
            x='ElapsedTime (h)', y='_n', data=n_df, hue='Experiment',
            errorbar=None, estimator='sum', ax=ax, palette=geno_palette,
        )
        draw_clock_ticks(ax, n_df, conf, twin_axis=False)
        ax.set_title(relative_title(title, conf), fontsize=16)
        ax.set_xlabel('Elapsed Time (h)', fontsize=12)
        ax.set_ylabel('Number of plants', fontsize=12)
        ax.legend(loc='best', title=geno_label)
    plt.savefig(os.path.join(overview_dir(conf, MODULE_TEMPORAL), 'all_metrics_n.png'), dpi=300, bbox_inches='tight')

    plt.cla()
    plt.clf()
    plt.close('all')