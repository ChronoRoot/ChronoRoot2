"""PCA of every report trait at each convex-hull snapshot hour.

Reads an existing report folder. Does not use the project config and does not
import the report pipeline.
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


TEMPORAL_FEATURES = (
    'MainRootLength (mm)',
    'LateralRootsLength (mm)',
    #'TotalLength (mm)',
    'NumberOfLateralRoots',
    'DiscreteLateralDensity (LR/cm)',
    'MainOverTotal (%)',
    #'HypocotylLength (mm)',
    #'LateralDensity (mm/mm)',
    #'Area (mm2)',
    #'DenseRootArea (mm2)',
)

CONVEX_FEATURES = (
    'Convex Hull Area',
    'Lateral Root Area Density',
    'Total Root Area Density',
    'Convex Hull Aspect Ratio',
    'Convex Hull Height',
    'Convex Hull Width',
    #'Bounding Box Area',
    #'Lateral Root Area Density BBOX',
    #'Total Root Area Density BBOX',
)

ANGLE_FEATURES = (
    'Mean tip angle',
    'Mean emergence angle',
    #'First LR tip',
    #'First LR emergence',
)

HOUR_COLUMN = 'ElapsedTime (h)'
JOIN_KEYS = ('Experiment', 'Plant_id')
IDENTITY_COLUMNS = ('Experiment', 'Plant_id', 'PlateCondition', 'ExtraVariable')
OUTPUT_DIRNAME = 'pca_timepoints'
MIN_PLANTS = 3
MIN_FEATURES = 2


def pca_timepoints(report_folder, temporal_only=False):
    """Write trait values and a PC1–PC2 plot for each convex-hull snapshot hour.

    report_folder is the generated report directory (the folder that contains
    data/). Snapshot hours always come from data/Convex_Hull_Data.csv.
    temporal_only keeps the temporal traits and leaves convex-hull and angle
    traits out of the table and the PCA.
    """
    report_folder = os.path.abspath(report_folder)
    if not os.path.isdir(report_folder):
        raise FileNotFoundError(f'Report folder not found: {report_folder}')

    data_dir = os.path.join(report_folder, 'data')
    convex_path = os.path.join(data_dir, 'Convex_Hull_Data.csv')
    temporal_path = os.path.join(data_dir, 'Temporal_Data.csv')
    angles_path = os.path.join(data_dir, 'LateralRoots_Data.csv')

    convex = _read_table(convex_path, 'Convex_Hull_Data.csv is required to read snapshot hours')
    temporal = _read_table(temporal_path, 'Temporal_Data.csv is required')
    hours = _snapshot_hours(convex)
    _require_join_keys(convex, 'Convex_Hull_Data.csv')
    _require_join_keys(temporal, 'Temporal_Data.csv')

    sources = [(temporal, _columns_present(temporal, TEMPORAL_FEATURES))]
    if not temporal_only:
        sources.append((convex, _columns_present(convex, CONVEX_FEATURES)))
        if os.path.isfile(angles_path):
            angles = _read_table(angles_path, 'LateralRoots_Data.csv could not be read')
            _require_join_keys(angles, 'LateralRoots_Data.csv')
            sources.append((angles, _columns_present(angles, ANGLE_FEATURES)))

    feature_columns = []
    for _table, columns in sources:
        for column in columns:
            if column not in feature_columns:
                feature_columns.append(column)
    if not feature_columns:
        raise ValueError('No feature columns found in the report tables')

    frames = []
    variance_rows = []
    score_frames = []
    loading_frames = []
    plots = []
    for hour in hours:
        assembled = _assemble_hour(hour, sources, feature_columns)
        if not assembled.empty:
            frames.append(assembled)
        outcome = _pca_hour(assembled, feature_columns, hour)
        variance_rows.append(outcome['variance'])
        if outcome['scores'] is not None:
            score_frames.append(outcome['scores'])
            loading_frames.append(outcome['loadings'])
            plots.append((hour, outcome['scores'], outcome['variance']))

    features = (
        pd.concat(frames, ignore_index=True)
        if frames else _empty_feature_table(feature_columns)
    )
    scores = (
        pd.concat(score_frames, ignore_index=True)
        if score_frames else _empty_scores()
    )
    loadings = (
        pd.concat(loading_frames, ignore_index=True)
        if loading_frames else pd.DataFrame(columns=[HOUR_COLUMN, 'Feature', 'PC1', 'PC2'])
    )
    variance = pd.DataFrame(variance_rows)

    output_dir = os.path.join(report_folder, OUTPUT_DIRNAME)
    _reset_output_dir(output_dir)
    features_path = os.path.join(output_dir, 'features_by_timepoint.csv')
    features.to_csv(features_path, index=False)
    scores.to_csv(os.path.join(output_dir, 'pca_scores.csv'), index=False)
    loadings.to_csv(os.path.join(output_dir, 'pca_loadings.csv'), index=False)
    variance.to_csv(os.path.join(output_dir, 'variance_explained.csv'), index=False)
    for hour, hour_scores, hour_variance in plots:
        _plot_scores(hour, hour_scores, hour_variance, output_dir)

    print(
        f'Wrote {features_path} ({len(features)} rows, {len(hours)} snapshot hours)',
        flush=True,
    )
    for row in variance_rows:
        hour = row[HOUR_COLUMN]
        if np.isfinite(row['PC1 variance']) and np.isfinite(row['PC2 variance']):
            print(
                f'PCA at {hour} h: {row["Plants used"]} plants, '
                f'{row["Features used"]} features, '
                f'PC1 {100 * row["PC1 variance"]:.1f}%, '
                f'PC2 {100 * row["PC2 variance"]:.1f}%',
                flush=True,
            )
        else:
            print(f'Skipped {hour} h: {row["Note"]}', flush=True)
    return output_dir


def _read_table(path, missing_message):
    if not os.path.isfile(path):
        raise FileNotFoundError(missing_message)
    table = pd.read_csv(path)
    if HOUR_COLUMN not in table.columns:
        raise ValueError(f'{os.path.basename(path)} is missing {HOUR_COLUMN}')
    return table


def _require_join_keys(table, name):
    missing = [column for column in JOIN_KEYS if column not in table.columns]
    if missing:
        raise ValueError(f'{name} is missing {", ".join(missing)}')


def _snapshot_hours(convex):
    values = pd.to_numeric(convex[HOUR_COLUMN], errors='coerce')
    values = values[np.isfinite(values)]
    hours = sorted({_as_hour(value) for value in values})
    if not hours:
        raise ValueError('Convex_Hull_Data.csv has no snapshot hours')
    return hours


def _as_hour(value):
    """Integer hour when the stored value is that hour, otherwise the value itself."""
    value = float(value)
    nearest = round(value)
    if abs(value - nearest) <= 1e-3:
        return int(nearest)
    return value


def _columns_present(table, candidates):
    return [column for column in candidates if column in table.columns]


def _assemble_hour(hour, sources, feature_columns):
    parts = []
    for table, columns in sources:
        if not columns or HOUR_COLUMN not in table.columns:
            continue
        elapsed = _elapsed_hours(table[HOUR_COLUMN])
        part = table.loc[elapsed == hour].copy()
        if part.empty:
            continue
        keep = []
        for column in list(JOIN_KEYS) + list(IDENTITY_COLUMNS) + columns:
            if column in part.columns and column not in keep:
                keep.append(column)
        part = part[keep]
        part = part.dropna(subset=list(JOIN_KEYS))
        if part.empty:
            continue
        for key in JOIN_KEYS:
            part[key] = part[key].astype(str)
        for column in columns:
            part[column] = pd.to_numeric(part[column], errors='coerce')
        part = part.drop_duplicates(list(JOIN_KEYS), keep='last')
        parts.append(part)

    if not parts:
        return _empty_feature_table(feature_columns)

    merged = parts[0]
    for part in parts[1:]:
        merged = merged.merge(part, on=list(JOIN_KEYS), how='outer', suffixes=('', '__src'))
        sourced = [column for column in merged.columns if column.endswith('__src')]
        for column in sourced:
            base = column[:-len('__src')]
            if base in merged.columns:
                merged[base] = merged[base].fillna(merged[column])
            else:
                merged[base] = merged[column]
            merged = merged.drop(columns=[column])

    merged[HOUR_COLUMN] = hour
    for column in feature_columns:
        if column not in merged.columns:
            merged[column] = np.nan
    identity = [column for column in IDENTITY_COLUMNS if column in merged.columns]
    ordered = identity + [HOUR_COLUMN] + list(feature_columns)
    merged = merged[ordered]
    return merged.sort_values(identity + [HOUR_COLUMN], kind='mergesort').reset_index(drop=True)


def _elapsed_hours(series):
    values = pd.to_numeric(series, errors='coerce').to_numpy(dtype=float)
    finite = np.isfinite(values)
    out = np.full(values.shape, np.nan)
    out[finite] = [_as_hour(value) for value in values[finite]]
    return pd.Series(out, index=series.index)


def _empty_feature_table(feature_columns):
    return pd.DataFrame(columns=list(IDENTITY_COLUMNS[:2]) + [HOUR_COLUMN] + list(feature_columns))


def _empty_scores():
    columns = list(IDENTITY_COLUMNS[:2]) + [HOUR_COLUMN, 'PC1', 'PC2']
    return pd.DataFrame(columns=columns)


def _pca_hour(assembled, feature_columns, hour):
    n_rows = len(assembled)
    present = [column for column in feature_columns if column in assembled.columns]
    # A column missing for every plant is not a measurement at this hour.
    # Leaving it in the complete-case check would drop every plant.
    if n_rows and present:
        finite_counts = assembled[present].apply(pd.to_numeric, errors='coerce').notna().sum()
        no_finite = [column for column in present if finite_counts.get(column, 0) == 0]
    else:
        no_finite = []
    usable = [column for column in present if column not in no_finite]

    if n_rows and usable:
        numeric = assembled[usable].apply(pd.to_numeric, errors='coerce')
        complete = numeric.notna().all(axis=1).to_numpy()
        plants = assembled.loc[complete].copy()
        values = numeric.loc[complete].to_numpy(dtype=float)
    else:
        plants = assembled.iloc[0:0].copy()
        values = np.empty((0, 0), dtype=float)

    n_plants = len(plants)
    constant = []
    varying_index = []
    if n_plants and values.size:
        span = values.max(axis=0) - values.min(axis=0)
        for index, column in enumerate(usable):
            if not np.isfinite(span[index]) or span[index] <= 0:
                constant.append(column)
            else:
                varying_index.append(index)
    varying = [usable[index] for index in varying_index]
    n_features = len(varying)

    notes = []
    ran = n_plants >= MIN_PLANTS and n_features >= MIN_FEATURES
    if n_rows == 0:
        notes.append('no plants at this hour')
    elif n_plants < MIN_PLANTS:
        notes.append(f'fewer than {MIN_PLANTS} complete plants')
    elif n_features < MIN_FEATURES:
        notes.append(f'fewer than {MIN_FEATURES} varying features')
    if no_finite:
        notes.append('no finite values: ' + ', '.join(no_finite))
    if constant:
        notes.append('zero variance: ' + ', '.join(constant))

    variance = {
        HOUR_COLUMN: hour,
        'Plants used': n_plants if n_rows else 0,
        'Plants dropped': n_rows - n_plants if n_rows else 0,
        'Features used': n_features,
        'Features dropped': len(no_finite) + len(constant),
        'PC1 variance': np.nan,
        'PC2 variance': np.nan,
        'Note': '; '.join(notes),
    }
    if not ran:
        return {'variance': variance, 'scores': None, 'loadings': None}

    matrix = values[:, varying_index]
    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0, ddof=0)
    standardized = (matrix - mean) / std
    _u, singular, vt = np.linalg.svd(standardized, full_matrices=False)
    total = float(np.sum(singular ** 2))
    ratios = (singular ** 2) / total if total > 0 else np.zeros_like(singular)
    loadings = vt[:2].T.copy()
    scores = standardized @ loadings
    loadings, scores = _positive_max_loading(loadings, scores)

    variance['PC1 variance'] = float(ratios[0])
    variance['PC2 variance'] = float(ratios[1]) if len(ratios) > 1 else np.nan
    if len(ratios) < 2 or not np.isfinite(variance['PC2 variance']):
        variance['Note'] = '; '.join(notes + ['fewer than 2 components'])
        variance['PC1 variance'] = np.nan
        variance['PC2 variance'] = np.nan
        return {'variance': variance, 'scores': None, 'loadings': None}

    identity = [column for column in IDENTITY_COLUMNS if column in plants.columns]
    score_frame = plants[identity].copy()
    score_frame[HOUR_COLUMN] = hour
    score_frame['PC1'] = scores[:, 0]
    score_frame['PC2'] = scores[:, 1]
    loading_frame = pd.DataFrame({
        HOUR_COLUMN: hour,
        'Feature': varying,
        'PC1': loadings[:, 0],
        'PC2': loadings[:, 1],
    })
    return {'variance': variance, 'scores': score_frame, 'loadings': loading_frame}


def _positive_max_loading(loadings, scores):
    for component in range(loadings.shape[1]):
        pivot = int(np.argmax(np.abs(loadings[:, component])))
        if loadings[pivot, component] < 0:
            loadings[:, component] *= -1
            scores[:, component] *= -1
    return loadings, scores


def _plot_scores(hour, scores, variance, output_dir):
    frame = scores.copy()
    frame['Experiment'] = frame['Experiment'].astype(str)
    genotypes = sorted(frame['Experiment'].unique())
    palette = dict(zip(
        genotypes,
        sns.color_palette('tab10', n_colors=max(len(genotypes), 1)),
    ))
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=frame, x='PC1', y='PC2', hue='Experiment',
        palette=palette, s=100,
    )
    plt.title(f'{hour} h', fontsize=14)
    plt.xlabel(f'PC1 ({100 * variance["PC1 variance"]:.1f}%)', fontsize=12)
    plt.ylabel(f'PC2 ({100 * variance["PC2 variance"]:.1f}%)', fontsize=12)
    plt.legend(title='Genotype', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    filename = f't{_hour_label(hour)}_pc1_pc2.png'
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()


def _hour_label(hour):
    if isinstance(hour, (int, np.integer)) or float(hour) == int(hour):
        return str(int(hour))
    return f'{float(hour):g}'


def _reset_output_dir(path):
    os.makedirs(path, exist_ok=True)
    for name in os.listdir(path):
        file_path = os.path.join(path, name)
        if os.path.isfile(file_path):
            os.remove(file_path)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='PCA of report traits at each convex-hull snapshot hour.',
    )
    parser.add_argument(
        'report_folder',
        help='Path to an existing report folder (the folder that contains data/)',
    )
    parser.add_argument(
        '--temporal-only',
        action='store_true',
        help='Use temporal traits only',
    )
    args = parser.parse_args(argv)
    try:
        pca_timepoints(args.report_folder, temporal_only=args.temporal_only)
    except (FileNotFoundError, ValueError) as exc:
        print(exc, file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
