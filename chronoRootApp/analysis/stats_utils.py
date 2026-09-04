"""Shared statistical comparison utilities for ChronoRoot report modules."""

import os

import numpy as np
import pandas as pd
import scipy.stats as stats

from .utils.fileUtilities import UNSPECIFIED_FACTOR, normalize_factor_value
from .utils.report_paths import stats_file as report_stats_file
from .utils.report_style import get_extra_axis_label, get_genotype_axis_label, get_plate_axis_label

ALL_COMPARISON_MODES = (
    'by_genotype',
    'genotype_by_plate',
    'genotype_by_extra',
    'by_plate_condition',
    'by_extra_variable',
    'plate_within_genotype',
    'extra_within_genotype',
)

_PLATE_MODES = frozenset({
    'genotype_by_plate', 'by_plate_condition', 'plate_within_genotype',
})
_EXTRA_MODES = frozenset({
    'genotype_by_extra', 'by_extra_variable', 'extra_within_genotype',
})


def _modes_from_config(conf):
    modes = []
    if conf.get('statsByGenotype', True):
        modes.append('by_genotype')
    if conf.get('statsGenotypeByPlate', True):
        modes.append('genotype_by_plate')
    if conf.get('statsGenotypeByExtra', True):
        modes.append('genotype_by_extra')
    if conf.get('statsByPlateCondition', True):
        modes.append('by_plate_condition')
    if conf.get('statsByExtraVariable', True):
        modes.append('by_extra_variable')
    if conf.get('statsPlateWithinGenotype', True):
        modes.append('plate_within_genotype')
    if conf.get('statsExtraWithinGenotype', True):
        modes.append('extra_within_genotype')
    return modes or ['by_genotype']


def _mode_applicable_for_data(mode, data, conf):
    data = ensure_factor_columns(data)
    spec = _mode_spec(mode, conf=conf)
    if spec['group_col'] == 'PlateCondition' and len(_usable_groups(data['PlateCondition'].unique())) < 2:
        return False
    if spec['group_col'] == 'ExtraVariable' and len(_usable_groups(data['ExtraVariable'].unique())) < 2:
        return False
    if spec['stratify_col'] == 'PlateCondition' and len(_usable_groups(data['PlateCondition'].unique())) < 1:
        return False
    if spec['stratify_col'] == 'ExtraVariable' and len(_usable_groups(data['ExtraVariable'].unique())) < 1:
        return False
    if spec['stratify_col'] == 'Experiment' and len(_usable_groups(data['Experiment'].unique())) < 1:
        return False
    return True


def get_enabled_comparison_modes(conf, data=None):
    modes = _modes_from_config(conf)
    if data is None:
        return modes
    return [mode for mode in modes if _mode_applicable_for_data(mode, data, conf)]


def comparison_modes_for_run(conf, data=None):
    """Effective comparison modes for the current report run."""
    if conf.get('effectiveComparisonModes') is not None:
        return conf['effectiveComparisonModes']
    if data is not None:
        return get_enabled_comparison_modes(conf, data)
    return get_enabled_comparison_modes(conf)


def log_auto_disabled_modes(conf, config_modes, effective_modes):
    skipped = set(config_modes) - set(effective_modes)
    if not skipped:
        return

    plate_label = get_plate_axis_label(conf)
    extra_label = get_extra_axis_label(conf)
    plate_skipped = skipped & _PLATE_MODES
    extra_skipped = skipped & _EXTRA_MODES
    other_skipped = skipped - plate_skipped - extra_skipped

    if plate_skipped:
        names = ', '.join(sorted(plate_skipped))
        print(f'Report: {plate_label} does not vary across plants — skipping: {names}')
    if extra_skipped:
        names = ', '.join(sorted(extra_skipped))
        print(f'Report: {extra_label} does not vary across plants — skipping: {names}')
    for mode in sorted(other_skipped):
        print(f'Report: comparison mode not applicable to data — skipping: {mode}')


def _mode_spec(mode, conf=None, extra_label=None):
    if conf is not None:
        genotype_label = get_genotype_axis_label(conf)
        plate_label = get_plate_axis_label(conf)
        extra_label = get_extra_axis_label(conf)
    else:
        genotype_label = 'Genotype'
        plate_label = 'Plate condition'
        extra_label = extra_label or 'Run'

    specs = {
        'by_genotype': {
            'group_col': 'Experiment',
            'stratify_col': None,
            'suffix': f'by {genotype_label}',
            'header': f'Comparing {genotype_label} across all data',
        },
        'genotype_by_plate': {
            'group_col': 'Experiment',
            'stratify_col': 'PlateCondition',
            'suffix': f'{genotype_label} by {plate_label}',
            'header': f'Comparing {genotype_label} within each {plate_label}',
        },
        'genotype_by_extra': {
            'group_col': 'Experiment',
            'stratify_col': 'ExtraVariable',
            'suffix': f'{genotype_label} by {extra_label}',
            'header': f'Comparing {genotype_label} within each {extra_label}',
        },
        'by_plate_condition': {
            'group_col': 'PlateCondition',
            'stratify_col': None,
            'suffix': f'by {plate_label}',
            'header': f'Comparing {plate_label} values directly',
        },
        'by_extra_variable': {
            'group_col': 'ExtraVariable',
            'stratify_col': None,
            'suffix': f'by {extra_label}',
            'header': f'Comparing {extra_label} values directly',
        },
        'plate_within_genotype': {
            'group_col': 'PlateCondition',
            'stratify_col': 'Experiment',
            'suffix': f'{plate_label} within {genotype_label}',
            'header': f'Comparing {plate_label} within each {genotype_label}',
        },
        'extra_within_genotype': {
            'group_col': 'ExtraVariable',
            'stratify_col': 'Experiment',
            'suffix': f'{extra_label} within {genotype_label}',
            'header': f'Comparing {extra_label} within each {genotype_label}',
        },
    }
    return specs[mode]


def _stratify_label(spec, conf):
    if spec['stratify_col'] == 'PlateCondition':
        return get_plate_axis_label(conf)
    if spec['stratify_col'] == 'ExtraVariable':
        return get_extra_axis_label(conf)
    if spec['stratify_col'] == 'Experiment':
        return get_genotype_axis_label(conf)
    return spec['stratify_col']


def _describe_averaging(conf):
    if conf.get('averagePerPlantStats', False):
        return 'Test values: mean per plant within each time interval, then compared across plants.'
    return 'Test values: all hourly observations in each interval (no per-plant averaging).'


def _resolve_stats_path(conf, module, metric_slug_name, mode, output_dir=None, metric=None,
                        file_prefix=None, suffix=None, subpath=()):
    if module and metric_slug_name:
        return report_stats_file(conf, module, metric_slug_name, mode, *subpath)
    if file_prefix and suffix:
        return os.path.join(output_dir, f'{file_prefix} Stats - {suffix}.txt')
    if metric and suffix:
        safe_metric = metric.replace('/', ' over ')
        return os.path.join(output_dir, f'{safe_metric} Stats - {suffix}.txt')
    return os.path.join(output_dir, f'{mode}_stats.txt')


def ensure_factor_columns(data):
    data = data.copy()
    if 'PlateCondition' not in data.columns:
        data['PlateCondition'] = UNSPECIFIED_FACTOR
    if 'ExtraVariable' not in data.columns:
        data['ExtraVariable'] = UNSPECIFIED_FACTOR
    data['PlateCondition'] = data['PlateCondition'].apply(normalize_factor_value)
    data['ExtraVariable'] = data['ExtraVariable'].apply(normalize_factor_value)
    data['Experiment'] = data['Experiment'].astype(str)
    return data


def _usable_groups(values, min_groups=2):
    groups = [v for v in pd.Series(values).dropna().unique() if normalize_factor_value(v) != UNSPECIFIED_FACTOR]
    return groups if len(groups) >= min_groups else []


def _average_per_plant(subdata, group_cols, plant_id_col, metric, conf):
    if not conf.get('averagePerPlantStats', False):
        return subdata
    cols = [c for c in group_cols + [plant_id_col] if c in subdata.columns]
    return subdata.groupby(cols).mean(numeric_only=True).reset_index()


def _write_pairwise_block(f, subdata, metric, group_col, group_names, label_prefix=''):
    for i in range(len(group_names) - 1):
        for j in range(i + 1, len(group_names)):
            g1, g2 = group_names[i], group_names[j]
            v1 = subdata[subdata[group_col].astype(str) == str(g1)][metric].dropna()
            v2 = subdata[subdata[group_col].astype(str) == str(g2)][metric].dropna()
            prefix = f'{label_prefix}' if label_prefix else ''
            try:
                if len(v1) == 0 or len(v2) == 0:
                    raise ValueError('empty group')
                _, p = stats.mannwhitneyu(v1, v2)
                p = round(float(p), 6)
                f.write(f'{prefix}Number of samples {g1}: {len(v1)} - ')
                f.write(f'Number of samples {g2}: {len(v2)}\n')
                f.write(f'{prefix}Mean {g1}: {round(v1.mean(), 2)} - ')
                f.write(f'Mean {g2}: {round(v2.mean(), 2)}\n')
                if p < 0.05:
                    f.write(f'{prefix}Groups {g1} and {g2} are significantly different. P-value: {p}\n')
                else:
                    f.write(f'{prefix}Groups {g1} and {g2} are not significantly different. P-value: {p}\n')
            except Exception:
                f.write(f'{prefix}Groups {g1} and {g2} could not be compared\n')


def run_pairwise_comparisons(f, subdata, metric, group_col, stratify_col=None,
                             stratify_label=None, conf=None, plant_id_col='Plant_id'):
    conf = conf or {}
    subdata = ensure_factor_columns(subdata)
    group_cols = [group_col]
    if stratify_col:
        group_cols = [stratify_col, group_col]

    subdata = _average_per_plant(subdata, group_cols, plant_id_col, metric, conf)

    if stratify_col:
        strata = _usable_groups(subdata[stratify_col].unique())
        if not strata:
            f.write(f'Insufficient {stratify_label or stratify_col} levels for stratified comparison.\n')
            return
        for stratum in strata:
            stratum_data = subdata[subdata[stratify_col].astype(str) == str(stratum)]
            groups = _usable_groups(stratum_data[group_col].unique())
            if len(groups) < 2:
                f.write(f'{stratify_label or stratify_col}: {stratum} — insufficient groups for comparison.\n')
                continue
            f.write(f'{stratify_label or stratify_col}: {stratum}\n')
            _write_pairwise_block(f, stratum_data, metric, group_col, groups)
    else:
        groups = _usable_groups(subdata[group_col].unique())
        if len(groups) < 2:
            f.write('Insufficient groups for comparison.\n')
            return
        _write_pairwise_block(f, subdata, metric, group_col, groups)


def perform_temporal_pairwise_stats(conf, data, metric, output_dir=None, plant_id_col='Plant_id',
                                    module=None, metric_slug_name=None, subpath=(),
                                    analysis_type='temporal', table_file_path=None):
    data = ensure_factor_columns(data)
    dt = int(conf['everyXhourField'])
    max_hour = data['ElapsedTime (h)'].max()
    if pd.isna(max_hour):
        return
    n_steps = int(round((max_hour + 1) / dt, 0))

    for mode in comparison_modes_for_run(conf, data):
        spec = _mode_spec(mode, conf=conf)

        output_path = _resolve_stats_path(
            conf, module, metric_slug_name, mode,
            output_dir=output_dir, metric=metric, suffix=spec['suffix'], subpath=subpath,
        )
        with open(output_path, 'w') as f:
            f.write('Using Mann Whitney U test to compare groups\n')
            f.write(f"{spec['header']}\n")
            f.write(f'{_describe_averaging(conf)}\n\n')

            for step in range(n_steps):
                end = int(min(dt * (step + 1), max_hour))
                hours = np.arange(dt * step, end)
                subdata = data[data['ElapsedTime (h)'].isin(hours)]
                f.write(f'Hours from {step * dt} to {end}\n')
                run_pairwise_comparisons(
                    f, subdata, metric,
                    group_col=spec['group_col'],
                    stratify_col=spec['stratify_col'],
                    stratify_label=_stratify_label(spec, conf),
                    conf=conf,
                    plant_id_col=plant_id_col,
                )
                f.write('\n')


def perform_interval_pairwise_stats(conf, data, metric, output_dir, interval_col, intervals,
                                    plant_id_col='Plant_id', interval_label='Interval',
                                    average_group_cols=None, file_prefix=None,
                                    module=None, metric_slug_name=None, subpath=(),
                                    analysis_type='interval', table_file_path=None):
    data = ensure_factor_columns(data)
    average_group_cols = average_group_cols or ['Experiment', plant_id_col]

    for mode in comparison_modes_for_run(conf, data):
        spec = _mode_spec(mode, conf=conf)

        output_path = _resolve_stats_path(
            conf, module, metric_slug_name, mode,
            output_dir=output_dir, metric=metric, file_prefix=file_prefix, suffix=spec['suffix'],
            subpath=subpath,
        )

        with open(output_path, 'w') as f:
            f.write('Using Mann Whitney U test to compare groups\n')
            f.write(f"{spec['header']}\n")
            f.write(f'{_describe_averaging(conf)}\n\n')

            for interval in intervals:
                subdata = data[data[interval_col].astype(str) == str(interval)]
                f.write(f'{interval_label}: {interval}\n')
                run_pairwise_comparisons(
                    f, subdata, metric,
                    group_col=spec['group_col'],
                    stratify_col=spec['stratify_col'],
                    stratify_label=_stratify_label(spec, conf),
                    conf=conf,
                    plant_id_col=plant_id_col,
                )
                f.write('\n')


def perform_scalar_pairwise_stats(conf, data, metric, output_dir, plant_id_col='Plant_id',
                                  file_prefix=None, module=None, metric_slug_name=None,
                                  subpath=(), analysis_type='scalar', table_file_path=None):
    """Pairwise comparisons on one row per plant (e.g. FPCA components)."""
    data = ensure_factor_columns(data)

    for mode in comparison_modes_for_run(conf, data):
        spec = _mode_spec(mode, conf=conf)

        output_path = _resolve_stats_path(
            conf, module, metric_slug_name, mode,
            output_dir=output_dir, metric=metric, file_prefix=file_prefix, suffix=spec['suffix'],
            subpath=subpath,
        )

        with open(output_path, 'w') as f:
            f.write('Using Mann Whitney U test to compare groups\n')
            f.write(f"{spec['header']}\n")
            f.write(f'{_describe_averaging(conf)}\n\n')
            run_pairwise_comparisons(
                f, data, metric,
                group_col=spec['group_col'],
                stratify_col=spec['stratify_col'],
                stratify_label=_stratify_label(spec, conf),
                conf=conf,
                plant_id_col=plant_id_col,
            )


def write_fourier_comparison_stats(f, subdata, group_col, metric, group_names, label_prefix=''):
    for i in range(len(group_names) - 1):
        for j in range(i + 1, len(group_names)):
            g1, g2 = group_names[i], group_names[j]
            v1 = subdata[subdata[group_col].astype(str) == str(g1)][metric]
            v2 = subdata[subdata[group_col].astype(str) == str(g2)][metric]
            v1 = v1.dropna()
            v2 = v2.dropna()
            prefix = label_prefix
            try:
                if len(v1) == 0 or len(v2) == 0:
                    continue
                _, p = stats.mannwhitneyu(v1, v2)
                p_val = round(float(p), 6)
                sig_text = 'SIGNIFICANT' if p < 0.05 else 'NOT SIGNIFICANT'
                stars = '**' if p < 0.001 else ('*' if p < 0.05 else 'ns')
                f.write(f'{prefix}Comparison: {g1} vs {g2}\n')
                f.write(f'{prefix}  - Samples: {len(v1)} vs {len(v2)}\n')
                f.write(f'{prefix}  - Mean: {v1.mean():.4f} vs {v2.mean():.4f}\n')
                f.write(f'{prefix}  - Std Dev: {v1.std():.4f} vs {v2.std():.4f}\n')
                f.write(f'{prefix}  - Result: p={p_val}, {sig_text} ({stars})\n')
            except Exception as exc:
                f.write(f'{prefix}Error comparing {g1} and {g2}: {exc}\n')


def perform_fourier_pairwise_stats(conf, subdata, metric, f, plant_id_col='i', type_col='Type', modes=None):
    subdata = ensure_factor_columns(subdata)
    if type_col == 'Type' and 'Experiment' in subdata.columns and 'Type' not in subdata.columns:
        subdata['Type'] = subdata['Experiment']
    if 'Experiment' not in subdata.columns and type_col in subdata.columns:
        subdata['Experiment'] = subdata[type_col]

    enabled = modes if modes is not None else comparison_modes_for_run(conf, subdata)
    for mode in enabled:
        spec = _mode_spec(mode, conf=conf)
        group_col = spec['group_col']
        if group_col == 'Experiment' and type_col in subdata.columns:
            compare_col = type_col
        else:
            compare_col = group_col

        f.write(f"\n--- {spec['header']} ---\n")
        cols = [compare_col]
        if spec['stratify_col']:
            cols = [spec['stratify_col'], compare_col]
        if conf.get('averagePerPlantStats', False) and plant_id_col in subdata.columns:
            subdata = subdata.groupby(cols + [plant_id_col]).mean(numeric_only=True).reset_index()

        if spec['stratify_col']:
            strata = _usable_groups(subdata[spec['stratify_col']].unique())
            for stratum in strata:
                stratum_data = subdata[subdata[spec['stratify_col']].astype(str) == str(stratum)]
                groups = _usable_groups(stratum_data[compare_col].unique())
                if len(groups) < 2:
                    continue
                f.write(f"{spec['stratify_col']}: {stratum}\n")
                write_fourier_comparison_stats(f, stratum_data, compare_col, metric, groups)
        else:
            groups = _usable_groups(subdata[compare_col].unique())
            if len(groups) >= 2:
                write_fourier_comparison_stats(f, subdata, compare_col, metric, groups)
