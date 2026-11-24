"""Remix ChronoRoot report tables into invented plate / run / genotype labels."""

from __future__ import annotations

import os
from typing import Dict, Iterable, Mapping, Optional

import numpy as np
import pandas as pd

from ..stats_utils import ensure_factor_columns
from .scenarios import Scenario

TEMPORAL_NAME = 'Temporal_Data.csv'
SCALAR_NAME = 'Germination_Time.csv'
INTERVAL_NAME = 'Convex_Hull_Data.csv'

# Required CSVs inside the --source directory (a ChronoRoot Report .../data folder).
REQUIRED_TABLES = (TEMPORAL_NAME, SCALAR_NAME, INTERVAL_NAME)


def load_source_tables(source_dir: str) -> Dict[str, pd.DataFrame]:
    """Load report CSVs used as trajectory donors from an explicit data directory."""
    if not source_dir:
        raise ValueError('source_dir is required (path to a Report data/ folder)')
    if not os.path.isdir(source_dir):
        raise FileNotFoundError(f'Source data directory not found: {source_dir}')
    tables = {}
    for name in REQUIRED_TABLES:
        path = os.path.join(source_dir, name)
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f'Missing source table: {path}\n'
                f'Pass --source pointing at a ChronoRoot Report data/ folder '
                f'containing {", ".join(REQUIRED_TABLES)}.'
            )
        tables[name] = pd.read_csv(path)
    return tables


def _plant_list(temporal: pd.DataFrame) -> list:
    return sorted(temporal['Plant_id'].astype(str).unique())


def build_plant_factor_map(
    temporal: pd.DataFrame,
    scenario: Scenario,
    *,
    seed: int = 0,
) -> pd.DataFrame:
    """Return one row per Plant_id with remixed Experiment / Plate / Extra.

    Donor trajectories are reused, but Experiment labels are always the invented
    scenario genotype names — never the original genotype strings.
    """
    rng = np.random.default_rng(seed)
    plants = _plant_list(temporal)
    if not plants:
        raise ValueError('No plants in temporal table')

    original = (
        temporal[['Plant_id', 'Experiment']]
        .drop_duplicates('Plant_id')
        .assign(Plant_id=lambda d: d['Plant_id'].astype(str))
        .set_index('Plant_id')['Experiment']
        .astype(str)
        .to_dict()
    )

    # Group by donor Experiment only to keep coherent trajectory pools.
    by_orig: Dict[str, list] = {}
    for pid in plants:
        by_orig.setdefault(original.get(pid, 'unknown'), []).append(pid)
    pools = [sorted(pool) for pool in sorted(by_orig.values(), key=len, reverse=True)]

    n_need = scenario.n_genotypes
    labels = list(scenario.genotypes)
    geno_of: Dict[str, str] = {}

    if n_need <= len(pools):
        # Largest donor pools → invented labels Genotype A, B, …
        for label, pool in zip(labels, pools[:n_need]):
            for pid in pool:
                geno_of[pid] = label
    else:
        # Assign invented labels to every donor pool, then split largest pools
        # for the remaining invented genotypes (clone if a pool is too small).
        for label, pool in zip(labels, pools):
            for pid in pool:
                geno_of[pid] = label
        split_sources = [list(pool) for pool in pools]
        for label in labels[len(pools):]:
            split_sources.sort(key=len, reverse=True)
            src = split_sources[0]
            if len(src) < 2:
                donor = src[0] if src else plants[0]
                new_id = f'{donor}__{label.replace(" ", "_")}'
                geno_of[new_id] = label
                continue
            mid = max(1, len(src) // 2)
            moved, keep = src[:mid], src[mid:]
            split_sources[0] = keep
            for pid in moved:
                geno_of[pid] = label

    flat = [pid for pid in geno_of]
    if not flat:
        raise ValueError('Genotype assignment produced no plants')

    plates = list(scenario.plates)
    extras = list(scenario.extras)
    rows = []
    by_geno: Dict[str, list] = {}
    for pid in flat:
        by_geno.setdefault(geno_of[pid], []).append(pid)
    for lab in by_geno:
        ids = by_geno[lab]
        rng.shuffle(ids)
        by_geno[lab] = ids

    i = 0
    for lab in labels:
        for pid in by_geno.get(lab, []):
            rows.append({
                'Plant_id': pid,
                'Experiment': lab,
                'PlateCondition': plates[i % len(plates)],
                'ExtraVariable': extras[i % len(extras)],
            })
            i += 1

    return pd.DataFrame(rows)


def _clone_plant_rows(frame: pd.DataFrame, src_id: str, new_id: str) -> pd.DataFrame:
    chunk = frame[frame['Plant_id'].astype(str) == str(src_id)].copy()
    if chunk.empty:
        return chunk
    chunk['Plant_id'] = new_id
    return chunk


def apply_factor_map(
    frame: pd.DataFrame,
    factor_map: pd.DataFrame,
    *,
    temporal_donor: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Replace Experiment/Plate/Extra on frame; clone synthetic plant ids if needed."""
    if frame is None or frame.empty:
        return frame
    fmap = factor_map.copy()
    fmap['Plant_id'] = fmap['Plant_id'].astype(str)

    real_ids = set(frame['Plant_id'].astype(str).unique())
    pieces = [frame.copy()]
    donor = temporal_donor if temporal_donor is not None else frame
    for pid in fmap['Plant_id']:
        if pid in real_ids:
            continue
        if '__' not in pid:
            continue
        src = pid.split('__', 1)[0]
        cloned = _clone_plant_rows(donor if 'Plant_id' in donor.columns else frame, src, pid)
        if not cloned.empty:
            # Keep only columns present on frame.
            cloned = cloned[[c for c in cloned.columns if c in frame.columns or c == 'Plant_id']]
            pieces.append(cloned)
            real_ids.add(pid)

    out = pd.concat(pieces, ignore_index=True)
    out['Plant_id'] = out['Plant_id'].astype(str)
    keep = set(fmap['Plant_id'])
    out = out[out['Plant_id'].isin(keep)].copy()
    meta = fmap.set_index('Plant_id')
    out['Experiment'] = out['Plant_id'].map(meta['Experiment'])
    out['PlateCondition'] = out['Plant_id'].map(meta['PlateCondition'])
    out['ExtraVariable'] = out['Plant_id'].map(meta['ExtraVariable'])
    return ensure_factor_columns(out)


def remix_tables(
    tables: Mapping[str, pd.DataFrame],
    scenario: Scenario,
    *,
    seed: int = 0,
) -> Dict[str, pd.DataFrame]:
    """Build remixed temporal / scalar / interval tables for one scenario."""
    temporal = tables[TEMPORAL_NAME]
    factor_map = build_plant_factor_map(temporal, scenario, seed=seed)
    remixed = {
        'factor_map': factor_map,
        'temporal': apply_factor_map(temporal, factor_map, temporal_donor=temporal),
        'scalar': apply_factor_map(tables[SCALAR_NAME], factor_map, temporal_donor=tables[SCALAR_NAME]),
        'interval': apply_factor_map(tables[INTERVAL_NAME], factor_map, temporal_donor=tables[INTERVAL_NAME]),
    }
    # Interval plots expect a 'Day' column in hours-as-day style already present.
    return remixed


def summarize_factors(frame: pd.DataFrame) -> dict:
    frame = ensure_factor_columns(frame)
    return {
        'n_plants': int(frame['Plant_id'].nunique()),
        'genotypes': sorted(frame['Experiment'].astype(str).unique()),
        'plates': sorted(frame['PlateCondition'].astype(str).unique()),
        'extras': sorted(frame['ExtraVariable'].astype(str).unique()),
        'cross_counts': (
            frame.drop_duplicates('Plant_id')
            .groupby(['Experiment', 'PlateCondition', 'ExtraVariable'])
            .size()
            .reset_index(name='n')
            .to_dict(orient='records')
        ),
    }
