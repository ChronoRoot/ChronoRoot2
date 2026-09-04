"""Shared Report/ folder path helpers — metric-first with paired plot/stats naming."""

import os
import re

from .report_utils import ensure_directory
from ..time_windows import report_folder_name

MODULE_TEMPORAL = 'temporal_parameters'
MODULE_CONVEX = 'convex_hull'
MODULE_ANGLES = 'angles'

OVERVIEW_SLUG = 'overview'
INDIVIDUAL_PLOTS_DIR = 'individual_plots'
RELATIVE_MEASUREMENTS_DIR = 'relative measurements'

TEMPORAL_METRICS = [
    'MainRootLength (mm)',
    'LateralRootsLength (mm)',
    'TotalLength (mm)',
    'NumberOfLateralRoots',
    'DiscreteLateralDensity (LR/cm)',
    'MainOverTotal (%)',
    'HypocotylLength (mm)',
]

CONVEX_METRICS = [
    'Convex Hull Area',
    'Lateral Root Area Density',
    'Total Root Area Density',
    'Convex Hull Aspect Ratio',
    'Convex Hull Height',
    'Convex Hull Width',
]

FOURIER_PARENT_METRICS = {
    'MR': 'main_root_length',
    'TR': 'total_root_length',
    'HY': 'hypocotyl_length',
}

ANGLE_METRICS = {
    'mean_emergence_angle': 'Mean emergence angle',
    'first_lr_tip_angle': 'First LR tip',
}

def metric_slug(display_name: str) -> str:
    """Convert a display metric name to a filesystem-safe folder slug."""
    text = str(display_name).strip().lower()
    text = text.replace('%', ' percent')
    text = re.sub(r'[^\w\s-]', '', text)
    text = re.sub(r'[\s-]+', '_', text).strip('_')
    return text or 'metric'


def report_root(conf) -> str:
    return os.path.join(conf['MainFolder'], report_folder_name(conf))


def data_file(conf, filename: str) -> str:
    path = os.path.join(report_root(conf), 'data')
    ensure_directory(path)
    return os.path.join(path, filename)


def module_dir(conf, module: str) -> str:
    path = os.path.join(report_root(conf), module)
    ensure_directory(path)
    return path


def measure_relative_to_initial(conf) -> bool:
    return bool((conf or {}).get('measureRelativeToInitial'))


def metric_dir(conf, module: str, slug: str) -> str:
    path = os.path.join(module_dir(conf, module), slug)
    if measure_relative_to_initial(conf) and module == MODULE_TEMPORAL:
        path = os.path.join(path, RELATIVE_MEASUREMENTS_DIR)
    ensure_directory(path)
    return path


def temporal_data_file(conf) -> str:
    """Temporal_Data.csv; relative runs write a copy under data/relative measurements/."""
    if measure_relative_to_initial(conf):
        path = os.path.join(report_root(conf), 'data', RELATIVE_MEASUREMENTS_DIR)
        ensure_directory(path)
        return os.path.join(path, 'Temporal_Data.csv')
    return data_file(conf, 'Temporal_Data.csv')


def analysis_dir(conf, module: str, metric_slug_name: str, *subpath: str) -> str:
    """Metric folder or nested sub-analysis path (e.g. fpca/pc1, growth_speed)."""
    path = os.path.join(metric_dir(conf, module, metric_slug_name), *subpath)
    ensure_directory(path)
    return path


def comparison_stats_path(base_dir: str, mode: str, metric_slug: str = '') -> str:
    ensure_directory(base_dir)
    stem = f'{metric_slug}_{mode}' if metric_slug else mode
    return os.path.join(base_dir, f'{stem}_stats.txt')


def comparison_plot_path(base_dir: str, mode: str, metric_slug: str = '') -> str:
    ensure_directory(base_dir)
    stem = f'{metric_slug}_{mode}' if metric_slug else mode
    return os.path.join(base_dir, f'{stem}.png')


def stats_file(conf, module: str, slug: str, comparison_mode: str, *subpath: str) -> str:
    base = analysis_dir(conf, module, slug, *subpath) if subpath else metric_dir(conf, module, slug)
    return comparison_stats_path(base, comparison_mode, metric_slug=slug)


def plot_file(conf, module: str, slug: str, filename: str, *subpath: str) -> str:
    base = analysis_dir(conf, module, slug, *subpath) if subpath else metric_dir(conf, module, slug)
    ensure_directory(base)
    return os.path.join(base, filename)


def table_file(conf, module: str, slug: str, filename: str, *subpath: str) -> str:
    base = analysis_dir(conf, module, slug, *subpath) if subpath else metric_dir(conf, module, slug)
    ensure_directory(base)
    return os.path.join(base, filename)


def overview_dir(conf, module: str) -> str:
    return metric_dir(conf, module, OVERVIEW_SLUG)


def angle_overlays_dir(conf, experiment_slug: str) -> str:
    path = os.path.join(module_dir(conf, MODULE_ANGLES), f'overlays_{experiment_slug}')
    ensure_directory(path)
    return path


def individual_plots_dir(conf, experiment_slug: str) -> str:
    path = os.path.join(report_root(conf), INDIVIDUAL_PLOTS_DIR, experiment_slug)
    ensure_directory(path)
    return path


def temporal_metric_slug(metric_column: str) -> str:
    mapping = {
        'MainRootLength (mm)': 'main_root_length',
        'LateralRootsLength (mm)': 'lateral_root_length',
        'TotalLength (mm)': 'total_root_length',
        'NumberOfLateralRoots': 'number_of_lateral_roots',
        'DiscreteLateralDensity (LR/cm)': 'lateral_root_density',
        'MainOverTotal (%)': 'main_over_total',
        'HypocotylLength (mm)': 'hypocotyl_length',
    }
    return mapping.get(metric_column, metric_slug(metric_column))


def purge_disabled_comparison_outputs(conf, effective_modes):
    """Remove comparison PNG/stats for modes not in the effective list.

    Only touches directories that already exist so unused modules
    (angles, convex hull, unselected temporal slugs) are not created.
    """
    from ..stats_utils import ALL_COMPARISON_MODES

    effective = set(effective_modes or [])
    disabled = set(ALL_COMPARISON_MODES) - effective
    if not disabled:
        return

    root = report_root(conf)
    metric_targets = [
        (MODULE_TEMPORAL, [temporal_metric_slug(m) for m in TEMPORAL_METRICS]),
        (MODULE_CONVEX, [metric_slug(m) for m in CONVEX_METRICS]),
        (MODULE_ANGLES, list(ANGLE_METRICS.keys())),
    ]
    for module, slugs in metric_targets:
        module_path = os.path.join(root, module)
        if not os.path.isdir(module_path):
            continue
        for slug in slugs:
            base = os.path.join(module_path, slug)
            bases = [base]
            if module == MODULE_TEMPORAL:
                rel_base = os.path.join(base, RELATIVE_MEASUREMENTS_DIR)
                if os.path.isdir(rel_base):
                    bases.append(rel_base)
            for out_dir in bases:
                if not os.path.isdir(out_dir):
                    continue
                for mode in disabled:
                    stem = f'{slug}_{mode}'
                    for filename in (f'{stem}.png', f'{stem}_stats.txt', f'{stem}_count.png'):
                        path = os.path.join(out_dir, filename)
                        if os.path.isfile(path):
                            os.remove(path)

    for parent_slug in FOURIER_PARENT_METRICS.values():
        growth_dir = os.path.join(root, MODULE_TEMPORAL, parent_slug, 'growth_speed')
        if not os.path.isdir(growth_dir):
            continue
        for mode in disabled:
            stem = f'{parent_slug}_{mode}'
            for filename in (f'{stem}.png', f'{stem}_stats.txt', f'{stem}_count.png'):
                path = os.path.join(growth_dir, filename)
                if os.path.isfile(path):
                    os.remove(path)
