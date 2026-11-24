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

import hashlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import shutil
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

from .utils.fileUtilities import (
    UNSPECIFIED_FACTOR,
    convertFromPathSafe,
    convertToPathSafe,
    normalize_factor_value,
)
from .utils.report_paths import (
    FIGURE_FORMATS,
    GERMINATION_TIME_SLUG,
    MODULE_TEMPORAL,
    REPLICAS_DIR,
    analysis_dir,
    clear_replicas_subdir,
    data_file,
    metric_dir,
    overview_dir,
    save_report_figure,
    table_file,
    temporal_metric_slug,
)
from .utils.report_utils import ensure_directory
from .stats_utils import (
    _mode_spec,
    comparison_modes_for_run,
    get_enabled_comparison_modes,
    perform_scalar_pairwise_stats,
    perform_temporal_pairwise_stats,
    ensure_factor_columns,
    iter_replica_runs,
)
from .report_plots import (
    aggregate_count_lines,
    emit_replica_scalar_comparisons,
    emit_scalar_comparison_plots,
    emit_temporal_comparison_plots,
    relative_title,
    subplot_grid,
)
from .plot_time_axis import (
    decorate_elapsed_time_axis,
    decorate_elapsed_time_axes,
    place_elapsed_time_suptitle,
    push_facet_titles_below_day_axes,
    space_elapsed_time_facet_grid,
)
from .time_windows import elapsed_hour_windows
from .utils.report_style import (
    genotype_palette_for_data,
    get_extra_axis_label,
    get_genotype_axis_label,
    get_plate_axis_label,
)


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

GERMINATION_TIME_COLUMN = 'Germination time (h)'
_MAIN_ROOT_LENGTH_COLUMN = 'MainRootLength (mm)'
_ELAPSED_HOURS_COLUMN = 'ElapsedTime (h)'
_GERMINATION_ID_COLUMNS = ['Plant_id', 'Experiment', 'PlateCondition', 'ExtraVariable']


def germination_time_hours(plant_frame):
    """First elapsed hour where main-root length leaves 0.

    Missing hours are ignored. A plant that stays at 0 has no germination time.
    """
    if plant_frame is None or plant_frame.empty:
        return np.nan
    if _MAIN_ROOT_LENGTH_COLUMN not in plant_frame.columns or _ELAPSED_HOURS_COLUMN not in plant_frame.columns:
        return np.nan
    length = pd.to_numeric(plant_frame[_MAIN_ROOT_LENGTH_COLUMN], errors='coerce')
    elapsed = pd.to_numeric(plant_frame[_ELAPSED_HOURS_COLUMN], errors='coerce')
    growing = length.notna() & elapsed.notna() & (length > 0)
    if not growing.any():
        return np.nan
    return float(elapsed[growing].min())


def _identity_piece(value):
    if pd.isna(value):
        return ''
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    return str(value).strip()


def germination_plant_label(row):
    """Same text as the plant dropdown: Genotype — rpi / camera / plant."""
    experiment = _identity_piece(row.get('Experiment', ''))
    rpi = _identity_piece(row.get('rpi', ''))
    cam = _identity_piece(row.get('cam', ''))
    plant = _identity_piece(row.get('plant', ''))
    if rpi and cam and plant:
        return f'{experiment} — {rpi} / {cam} / {plant}'
    return f'{experiment} — {_identity_piece(row.get("Plant_id", ""))}'


def germination_times_table(data):
    """One row per plant that leaves 0, on the report clock."""
    hardware = [col for col in ('rpi', 'cam', 'plant') if data is not None and col in getattr(data, 'columns', [])]
    columns = _GERMINATION_ID_COLUMNS + hardware + [GERMINATION_TIME_COLUMN]
    if data is None or data.empty or 'Plant_id' not in data.columns:
        return pd.DataFrame(columns=columns)
    if _MAIN_ROOT_LENGTH_COLUMN not in data.columns or _ELAPSED_HOURS_COLUMN not in data.columns:
        return pd.DataFrame(columns=columns)

    data = ensure_factor_columns(data)
    rows = []
    for plant_id, group in data.groupby('Plant_id', sort=False):
        hours = germination_time_hours(group)
        if not np.isfinite(hours):
            continue
        first = group.iloc[0]
        row = {
            'Plant_id': plant_id,
            'Experiment': first['Experiment'],
            'PlateCondition': first['PlateCondition'],
            'ExtraVariable': first['ExtraVariable'],
            GERMINATION_TIME_COLUMN: hours,
        }
        for col in hardware:
            row[col] = first[col]
        rows.append(row)
    return pd.DataFrame(rows, columns=columns)


def _iqr_outlier_mask(values):
    """Tukey fences, the same rule as a boxplot's whiskers."""
    numeric = pd.to_numeric(values, errors='coerce')
    q1 = numeric.quantile(0.25)
    q3 = numeric.quantile(0.75)
    if not np.isfinite(q1) or not np.isfinite(q3):
        return pd.Series(False, index=values.index)
    iqr = q3 - q1
    low = q1 - 1.5 * iqr
    high = q3 + 1.5 * iqr
    return (numeric < low) | (numeric > high)


def _germination_conf(conf):
    """Germination time is a clock time, so it stays out of relative-length folders."""
    germ_conf = dict(conf or {})
    germ_conf['measureRelativeToInitial'] = False
    return germ_conf


def _write_germination_plant_list(conf, germ_df):
    """Latest-to-earliest germination times, outliers listed apart.

    Outliers are outside the 1.5 IQR fences of their genotype. The label is
    the plant-list entry, so those plants can be removed from the project.
    """
    work = germ_df.copy()
    work['_outlier'] = False
    for _experiment, index in work.groupby('Experiment').groups.items():
        work.loc[index, '_outlier'] = _iqr_outlier_mask(
            work.loc[index, GERMINATION_TIME_COLUMN],
        ).to_numpy()

    lines = [
        'Germination times, latest to earliest.',
        'Outliers lie outside the 1.5 IQR fences of their genotype (the boxplot fences).',
        'Each label matches the plant list: Genotype — rpi / camera / plant.',
        '',
    ]

    def _section(title, subset):
        lines.append(title)
        if subset.empty:
            lines.append('(none)')
            lines.append('')
            return
        ordered = subset.sort_values(
            [GERMINATION_TIME_COLUMN, 'Experiment', 'Plant_id'],
            ascending=[False, True, True],
        )
        for _idx, row in ordered.iterrows():
            lines.append(f'{float(row[GERMINATION_TIME_COLUMN]):.2f} h\t{germination_plant_label(row)}')
        lines.append('')

    _section('Outliers', work[work['_outlier']])
    _section('Other plants', work[~work['_outlier']])
    path = table_file(conf, MODULE_TEMPORAL, GERMINATION_TIME_SLUG, 'plant_list.txt')
    with open(path, 'w', encoding='utf-8') as handle:
        handle.write('\n'.join(lines).rstrip() + '\n')
    print(f'Germination time: wrote {path}', flush=True)


def _write_germination_summary_table(conf, germ_df):
    grouped = germ_df.groupby(
        ['Experiment', 'PlateCondition', 'ExtraVariable'],
    ).agg(
        n_plants=('Plant_id', 'nunique'),
        mean=(GERMINATION_TIME_COLUMN, 'mean'),
        sd=(GERMINATION_TIME_COLUMN, 'std'),
    ).reset_index()
    grouped = grouped.round(3)
    grouped.to_csv(
        table_file(conf, MODULE_TEMPORAL, GERMINATION_TIME_SLUG, 'summary_table.csv'),
        index=False,
    )


def performGerminationAnalysis(conf, data):
    """Boxplots and Mann-Whitney tests of main-root germination time.

    Germination time is the first windowed hour where main-root length leaves 0.
    Plants that never leave 0 are omitted.
    """
    germ_conf = _germination_conf(conf)
    if (
        data is None
        or getattr(data, 'empty', True)
        or _MAIN_ROOT_LENGTH_COLUMN not in getattr(data, 'columns', [])
        or 'Plant_id' not in data.columns
    ):
        print('Germination time: no hourly main-root lengths, skipping.', flush=True)
        return

    germ_df = germination_times_table(data)
    out_csv = data_file(germ_conf, 'Germination_Time.csv')
    germ_df.to_csv(out_csv, index=False)
    print(f'Germination time: wrote {out_csv}', flush=True)
    if germ_df.empty:
        print('Germination time: no plant left 0, skipping plots.', flush=True)
        return

    _write_germination_summary_table(germ_conf, germ_df)
    _write_germination_plant_list(germ_conf, germ_df)
    base_dir = metric_dir(germ_conf, MODULE_TEMPORAL, GERMINATION_TIME_SLUG)
    saved_modes = germ_conf.get('effectiveComparisonModes')
    germ_conf['effectiveComparisonModes'] = get_enabled_comparison_modes(germ_conf, germ_df)
    try:
        perform_scalar_pairwise_stats(
            germ_conf, germ_df, GERMINATION_TIME_COLUMN, output_dir=None, plant_id_col='Plant_id',
            module=MODULE_TEMPORAL, metric_slug_name=GERMINATION_TIME_SLUG,
            analysis_type='scalar',
        )
        emit_scalar_comparison_plots(
            germ_conf, germ_df, GERMINATION_TIME_COLUMN, base_dir,
            module=MODULE_TEMPORAL, metric_slug_name=GERMINATION_TIME_SLUG,
            analysis_type='scalar', metric_label='Germination time',
            show_mean=True,
        )
        emit_replica_scalar_comparisons(
            germ_conf, germ_df, GERMINATION_TIME_COLUMN,
            module=MODULE_TEMPORAL, metric_slug_name=GERMINATION_TIME_SLUG,
            analysis_type='scalar', metric_label='Germination time',
            show_mean=True,
        )
    finally:
        germ_conf['effectiveComparisonModes'] = saved_modes


def _include_lateral_root_plots(conf):
    if conf is None:
        return True
    return conf.get('includeLateralRootPlots', True)


def _add_day_axis(ax, dataframe, day_tick_size=12, conf=None, *, annotate=True):
    return decorate_elapsed_time_axis(
        ax, dataframe, conf, day_axis=True, annotate=annotate,
        day_tick_size=day_tick_size,
    )


def plot_individual_plant(savepath, dataframe, name, conf=None):
    plt.ioff()
    os.makedirs(savepath, exist_ok=True)
    
    # Define font sizes for consistency across subplots
    LABEL_SIZE = 18
    TICK_SIZE = 16
    TITLE_SIZE = 18
    LEGEND_SIZE = 16
    DAY_TICK_SIZE = 12

    include_laterals = _include_lateral_root_plots(conf)

    if include_laterals:
        fig, (ax1, ax2) = plt.subplots(
            nrows=2, ncols=1, figsize=(9, 10), dpi=150, sharex=True,
            constrained_layout=True,
        )
    else:
        fig, ax1 = plt.subplots(
            nrows=1, ncols=1, figsize=(9, 6), dpi=150, constrained_layout=True,
        )
        ax2 = None

    dataframe.plot(x='ElapsedTime (h)', y='MainRootLength (mm)', ax=ax1, color='g', label='Main Root Length')
    if include_laterals and 'LateralRootsLength (mm)' in dataframe.columns:
        dataframe.plot(x='ElapsedTime (h)', y='LateralRootsLength (mm)', ax=ax1, color='b', label='Lateral Roots Length')
    dataframe.plot(x='ElapsedTime (h)', y='HypocotylLength (mm)', ax=ax1, color='r', label='Hypocotyl Length')
    
    title_name = name
    title_root, title_ext = os.path.splitext(name)
    if title_ext.lower() in FIGURE_FORMATS:
        title_name = title_root
    ax1.set_title('%s' % convertFromPathSafe(title_name), fontsize=TITLE_SIZE)
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

    _add_day_axis(ax1, dataframe, DAY_TICK_SIZE, conf=conf, annotate=(ax2 is None))
    if ax2 is not None:
        decorate_elapsed_time_axis(
            ax2, dataframe, conf, day_axis=False, annotate=True,
        )

    root, ext = os.path.splitext(name)
    png_name = root + '.png' if ext.lower() in FIGURE_FORMATS or ext == '' else name
    save_report_figure(fig, os.path.join(savepath, png_name), conf, dpi=150, bbox_inches='tight')

    plt.cla()
    plt.clf()
    plt.close('all')


def individual_plot_stamp(dataframe, conf):
    """Fingerprint of the series and the options that change this figure."""
    cols = [
        col for col in (
            'ElapsedTime (h)', 'MainRootLength (mm)', 'LateralRootsLength (mm)',
            'HypocotylLength (mm)', 'NumberOfLateralRoots',
        )
        if col in dataframe.columns
    ]
    text = dataframe[cols].to_csv(index=False)
    conf = conf or {}
    text += '|' + '|'.join(str(conf.get(key)) for key in (
        'includeLateralRootPlots', 'figureClockTicks', 'showFigureClockTicks',
        'timeDurationHours', 'exportSvg', 'exportPdf',
    ))
    return hashlib.sha256(text.encode()).hexdigest()


def save_individual_plant_plot(savepath, dataframe, name, conf, cache_path):
    """Draw the plant figure, or copy the stored PNG when the series is unchanged."""
    os.makedirs(savepath, exist_ok=True)
    root, ext = os.path.splitext(name)
    png_name = root + '.png' if ext.lower() in FIGURE_FORMATS or ext == '' else name
    dest = os.path.join(savepath, png_name)
    cache_png = os.path.splitext(cache_path)[0] + '.png'
    stamp = individual_plot_stamp(dataframe, conf)
    stamp_path = cache_png + '.stamp'
    if os.path.isfile(cache_png) and os.path.isfile(stamp_path) and os.path.isfile(dest):
        with open(stamp_path, encoding='utf-8') as handle:
            stored = handle.read()
        if stored == stamp:
            return False
    plot_individual_plant(savepath, dataframe, png_name, conf)
    if os.path.isfile(dest) and os.path.abspath(dest) != os.path.abspath(cache_png):
        shutil.copy(dest, cache_png)
    if os.path.isfile(cache_png):
        with open(stamp_path, 'w', encoding='utf-8') as handle:
            handle.write(stamp)
    return True

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


def _overview_factor_levels(dataframe, column):
    if column not in dataframe.columns:
        return []
    levels = []
    seen = set()
    for value in dataframe[column].dropna().unique():
        normalized = normalize_factor_value(value)
        if normalized == UNSPECIFIED_FACTOR or normalized in seen:
            continue
        seen.add(normalized)
        levels.append(normalized)
    return sorted(levels, key=str)


def _condition_overview_title(factor_label, value):
    factor = str(factor_label or '').strip() or 'Condition'
    return f'{factor}: {value}'


def _write_temporal_overview_pair(conf, dataframe, out_dir, metrics, geno_label,
                                  condition_title=None):
    """Write all_metrics_subplots.png and all_metrics_n.png into out_dir."""
    if dataframe is None or dataframe.empty or not metrics:
        return
    ensure_directory(out_dir)
    geno_palette = genotype_palette_for_data(dataframe)

    n = len(metrics)
    nrows, ncols = subplot_grid(n)
    # Relative suffix belongs on the figure title only, not every metric panel.
    figure_title = relative_title(condition_title or 'Temporal overview', conf)

    def _finish_overview_figure(fig, axes_grid):
        # One shared legend outside (not repeated on every metric panel).
        handles, labels = None, None
        for ax in np.ravel(axes_grid):
            if ax is None:
                continue
            leg = ax.get_legend()
            if leg is not None:
                if handles is None:
                    handles, labels = ax.get_legend_handles_labels()
                leg.remove()
        if handles:
            fig.legend(
                handles, labels, title=geno_label,
                loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=True,
            )
        decorate_elapsed_time_axes(axes_grid, dataframe, conf)
        push_facet_titles_below_day_axes(axes_grid)
        space_elapsed_time_facet_grid(fig, axes_grid)
        place_elapsed_time_suptitle(fig, figure_title)

    def _draw_overview(path, plot_fn, ylabel):
        fig = plt.figure(figsize=(5.5 * ncols, 4.2 * nrows))
        gs = fig.add_gridspec(nrows, ncols, hspace=0.32, wspace=0.28)
        axes_grid = np.empty((nrows, ncols), dtype=object)
        for i, (col, title) in enumerate(metrics):
            r, c = divmod(i, ncols)
            ax = fig.add_subplot(gs[r, c])
            axes_grid[r, c] = ax
            plot_fn(ax, col)
            ax.set_title(title, fontsize=14)
            ax.set_xlabel('Elapsed Time (h)', fontsize=12)
            ax.set_ylabel(ylabel(col), fontsize=12)
        _finish_overview_figure(fig, axes_grid)
        save_report_figure(
            fig, path, conf, dpi=300, bbox_inches='tight', pad_inches=0.15,
        )
        plt.close('all')

    def _metric_lines(ax, y_col):
        plot_df = dataframe.dropna(subset=[y_col]) if y_col in dataframe.columns else dataframe
        sns.lineplot(
            x='ElapsedTime (h)', y=y_col, data=plot_df, hue='Experiment',
            errorbar='se', ax=ax, palette=geno_palette,
        )

    def _count_lines(ax, y_col):
        n_df = aggregate_count_lines(dataframe, 'ElapsedTime (h)', 'Experiment', y_col)
        sns.lineplot(
            x='ElapsedTime (h)', y='_n', data=n_df, hue='Experiment',
            errorbar=None, ax=ax, palette=geno_palette,
        )

    _draw_overview(
        os.path.join(out_dir, 'all_metrics_subplots.png'),
        _metric_lines,
        _overview_ylabel,
    )
    _draw_overview(
        os.path.join(out_dir, 'all_metrics_n.png'),
        _count_lines,
        lambda _col: 'Number of plants',
    )


def plot_info_all(conf, dataframe):
    plt.ioff()
    dataframe = ensure_factor_columns(dataframe)
    geno_label = get_genotype_axis_label(conf)
    metrics = _overview_metric_pairs(conf, dataframe)
    if not metrics:
        return

    base = overview_dir(conf, MODULE_TEMPORAL)
    plates = _overview_factor_levels(dataframe, 'PlateCondition')
    extras = _overview_factor_levels(dataframe, 'ExtraVariable')
    modes = set(comparison_modes_for_run(conf, dataframe))
    # Qualitative genotype-within-condition overviews follow the matching
    # advanced modes (not every plate/extra mode checkbox).
    split_plates = len(plates) >= 2 and 'genotype_by_plate' in modes
    split_extras = len(extras) >= 2 and 'genotype_by_extra' in modes
    write_general = ('by_genotype' in modes) or (not split_plates and not split_extras)

    # Drop previous split trees / flat overviews so remaps do not leave stale pools.
    for split_name in ('by_plate', 'by_extra'):
        split_root = os.path.join(base, split_name)
        if os.path.isdir(split_root):
            shutil.rmtree(split_root, ignore_errors=True)
    for name in ('all_metrics_subplots.png', 'all_metrics_n.png'):
        path = os.path.join(base, name)
        if os.path.isfile(path):
            os.remove(path)

    if write_general:
        general_title = (
            _mode_spec('by_genotype', conf=conf)['header']
            if (split_plates or split_extras)
            else None
        )
        _write_temporal_overview_pair(
            conf, dataframe, base, metrics, geno_label,
            condition_title=general_title,
        )
    if split_plates:
        plate_label = get_plate_axis_label(conf)
        for plate in plates:
            subset = dataframe[dataframe['PlateCondition'].astype(str) == str(plate)]
            out_dir = os.path.join(base, 'by_plate', convertToPathSafe(str(plate)))
            _write_temporal_overview_pair(
                conf, subset, out_dir, metrics, geno_label,
                condition_title=_condition_overview_title(plate_label, plate),
            )
    if split_extras:
        extra_label = get_extra_axis_label(conf)
        for extra in extras:
            subset = dataframe[dataframe['ExtraVariable'].astype(str) == str(extra)]
            out_dir = os.path.join(base, 'by_extra', convertToPathSafe(str(extra)))
            _write_temporal_overview_pair(
                conf, subset, out_dir, metrics, geno_label,
                condition_title=_condition_overview_title(extra_label, extra),
            )