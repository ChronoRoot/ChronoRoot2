"""Generate a commentable plot gallery (one figure kind at a time, not a full report).

Usage (from chronoRootApp/):

    python -m analysis.plot_gallery.generate
    python -m analysis.plot_gallery.generate --scenario g4_p2_e2 --kinds temporal
    python -m analysis.plot_gallery.generate --list
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import traceback
from typing import List, Optional, Sequence

import pandas as pd

from ..report import _overview_metric_pairs, _write_temporal_overview_pair
from ..report_plots import (
    plot_comparison_mode,
    plot_interval_comparison_mode,
    plot_scalar_comparison_mode,
)
from ..stats_utils import (
    ALL_COMPARISON_MODES,
    ensure_factor_columns,
    get_enabled_comparison_modes,
)
from ..utils.report_style import get_genotype_axis_label
from .remix import load_source_tables, remix_tables, summarize_factors
from .scenarios import SCENARIOS, Scenario

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
DEFAULT_OUT = os.path.join(REPO_ROOT, 'plot_gallery')

TEMPORAL_METRIC = 'MainRootLength (mm)'
SCALAR_METRIC = 'Germination time (h)'
INTERVAL_METRIC = 'Convex Hull Area'

KIND_TEMPORAL = 'temporal'
KIND_SCALAR = 'scalar'
KIND_INTERVAL = 'interval'
KIND_OVERVIEW = 'overview'
ALL_KINDS = (KIND_TEMPORAL, KIND_SCALAR, KIND_INTERVAL, KIND_OVERVIEW)


def _gallery_conf(scenario_dir: str, modes: Sequence[str], *, relative: bool = False) -> dict:
    return {
        'MainFolder': scenario_dir,
        'reportFolderName': '_report_scratch',
        'exportSvg': True,
        'exportPdf': False,
        'advancedComparisonModes': True,
        'statsByGenotype': True,
        'statsGenotypeByPlate': True,
        'statsGenotypeByExtra': True,
        'statsByPlateCondition': True,
        'statsByExtraVariable': True,
        'statsPlateWithinGenotype': True,
        'statsExtraWithinGenotype': True,
        'genotypeAxisLabel': 'Genotype',
        'plateConditionAxisLabel': 'Plate condition',
        'extraVariableLabel': 'Run',
        'effectiveComparisonModes': list(modes),
        'measureRelativeToInitial': bool(relative),
    }


def _modes_for(conf: dict, data: pd.DataFrame) -> List[str]:
    # Honor forced list, then filter by data applicability.
    conf = dict(conf)
    forced = conf.pop('effectiveComparisonModes', None)
    if forced is None:
        return get_enabled_comparison_modes(conf, data)
    conf_no_force = dict(conf)
    conf_no_force['advancedComparisonModes'] = True
    applicable = set(get_enabled_comparison_modes(conf_no_force, data))
    return [m for m in forced if m in applicable]


def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _write_json(path: str, payload: dict) -> None:
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(payload, fh, indent=2, default=str)
        fh.write('\n')


def _emit_temporal(conf, data, out_dir: str, modes: Sequence[str]) -> List[dict]:
    results = []
    metric = TEMPORAL_METRIC
    for mode in modes:
        path = os.path.join(out_dir, f'{mode}.png')
        ok = False
        err = None
        try:
            ok = bool(
                plot_comparison_mode(
                    conf, data, metric, mode, path,
                    metric_label=metric,
                )
            )
        except Exception as exc:
            err = f'{exc}\n{traceback.format_exc()}'
            ok = False
        results.append({'kind': KIND_TEMPORAL, 'mode': mode, 'path': path, 'ok': ok, 'error': err})
    return results


def _emit_scalar(conf, data, out_dir: str, modes: Sequence[str]) -> List[dict]:
    results = []
    metric = SCALAR_METRIC
    if metric not in data.columns:
        return [{'kind': KIND_SCALAR, 'mode': None, 'path': None, 'ok': False, 'error': f'missing {metric}'}]
    for mode in modes:
        try:
            ok = bool(
                plot_scalar_comparison_mode(
                    conf, data, metric, mode, out_dir,
                    metric_slug_name='germination_time',
                    metric_label=metric,
                    show_mean=True,
                )
            )
            path = os.path.join(out_dir, f'germination_time_{mode}.png')
            # comparison_plot_path naming — locate whatever was written
            if not os.path.isfile(path):
                candidates = [
                    os.path.join(out_dir, name)
                    for name in os.listdir(out_dir)
                    if name.endswith(f'_{mode}.png') or name == f'{mode}.png'
                ]
                path = candidates[0] if candidates else path
            err = None
        except Exception as exc:
            ok = False
            path = os.path.join(out_dir, f'{mode}.png')
            err = f'{exc}\n{traceback.format_exc()}'
        results.append({'kind': KIND_SCALAR, 'mode': mode, 'path': path, 'ok': ok, 'error': err})
    return results


def _emit_interval(conf, data, out_dir: str, modes: Sequence[str]) -> List[dict]:
    results = []
    metric = INTERVAL_METRIC
    if metric not in data.columns:
        return [{'kind': KIND_INTERVAL, 'mode': None, 'path': None, 'ok': False, 'error': f'missing {metric}'}]
    # Prefer a few day snapshots so boxes stay readable.
    frame = data.copy()
    if 'Day' in frame.columns:
        days = sorted(pd.to_numeric(frame['Day'], errors='coerce').dropna().unique())
        if len(days) > 4:
            pick = [days[i] for i in (0, len(days) // 3, 2 * len(days) // 3, -1)]
            frame = frame[frame['Day'].isin(pick)]
    for mode in modes:
        try:
            ok = bool(
                plot_interval_comparison_mode(
                    conf, frame, metric, mode, out_dir,
                    metric_slug_name='convex_hull_area',
                    metric_label=metric,
                    x_col='Day',
                )
            )
            path = os.path.join(out_dir, f'convex_hull_area_{mode}.png')
            if not os.path.isfile(path):
                candidates = [
                    os.path.join(out_dir, name)
                    for name in os.listdir(out_dir)
                    if mode in name and name.endswith('.png')
                ]
                path = candidates[0] if candidates else path
            err = None
        except Exception as exc:
            ok = False
            path = os.path.join(out_dir, f'{mode}.png')
            err = f'{exc}\n{traceback.format_exc()}'
        results.append({'kind': KIND_INTERVAL, 'mode': mode, 'path': path, 'ok': ok, 'error': err})
    return results


def _emit_overview(conf, data, out_dir: str) -> List[dict]:
    results = []
    data = ensure_factor_columns(data)
    metrics = _overview_metric_pairs(conf, data)
    if not metrics:
        return [{'kind': KIND_OVERVIEW, 'mode': 'overview', 'path': None, 'ok': False, 'error': 'no metrics'}]
    geno_label = get_genotype_axis_label(conf)
    _ensure_dir(out_dir)
    try:
        _write_temporal_overview_pair(
            conf, data, out_dir, metrics, geno_label,
            condition_title='Gallery overview',
        )
        path = os.path.join(out_dir, 'all_metrics_subplots.png')
        ok = os.path.isfile(path)
        err = None if ok else 'overview png missing'
    except Exception as exc:
        path = os.path.join(out_dir, 'all_metrics_subplots.png')
        ok = False
        err = f'{exc}\n{traceback.format_exc()}'
    results.append({'kind': KIND_OVERVIEW, 'mode': 'overview', 'path': path, 'ok': ok, 'error': err})
    return results


def _scenario_index_md(scenario: Scenario, summary: dict, results: List[dict]) -> str:
    lines = [
        f'# {scenario.id}',
        '',
        scenario.blurb,
        '',
        f'- genotypes ({scenario.n_genotypes}): {", ".join(summary["genotypes"])}',
        f'- plates ({scenario.n_plates}): {", ".join(summary["plates"])}',
        f'- extras ({scenario.n_extras}): {", ".join(summary["extras"])}',
        f'- plants: {summary["n_plants"]}',
        '',
        '## Figures',
        '',
    ]
    by_kind = {}
    for row in results:
        by_kind.setdefault(row['kind'], []).append(row)
    for kind in ALL_KINDS:
        rows = by_kind.get(kind, [])
        if not rows:
            continue
        lines.append(f'### {kind}')
        lines.append('')
        for row in rows:
            status = 'ok' if row.get('ok') else 'FAIL'
            rel = os.path.basename(row['path']) if row.get('path') else '—'
            mode = row.get('mode') or ''
            lines.append(f'- `{mode}` → `{rel}` [{status}]')
        lines.append('')
    return '\n'.join(lines)


def generate_scenario(
    scenario: Scenario,
    tables: dict,
    out_root: str,
    *,
    kinds: Sequence[str] = ALL_KINDS,
    seed: int = 0,
    relative: bool = False,
) -> dict:
    scenario_dir = _ensure_dir(os.path.join(out_root, scenario.id))
    # Relative review lives under relative/ so absolute gallery stays intact.
    figures_root = (
        _ensure_dir(os.path.join(scenario_dir, 'relative'))
        if relative else scenario_dir
    )
    remixed = remix_tables(tables, scenario, seed=seed)
    summary = summarize_factors(remixed['temporal'])

    data_dir = _ensure_dir(os.path.join(scenario_dir, 'data'))
    remixed['factor_map'].to_csv(os.path.join(data_dir, 'plant_factors.csv'), index=False)
    remixed['temporal'].to_csv(os.path.join(data_dir, 'Temporal_Data.csv'), index=False)
    remixed['scalar'].to_csv(os.path.join(data_dir, 'Germination_Time.csv'), index=False)
    remixed['interval'].to_csv(os.path.join(data_dir, 'Convex_Hull_Data.csv'), index=False)
    _write_json(os.path.join(scenario_dir, 'summary.json'), summary)

    conf = _gallery_conf(figures_root, ALL_COMPARISON_MODES, relative=relative)
    modes = _modes_for(conf, remixed['temporal'])
    conf['effectiveComparisonModes'] = modes

    results: List[dict] = []
    if KIND_TEMPORAL in kinds:
        results.extend(
            _emit_temporal(
                conf, remixed['temporal'],
                _ensure_dir(os.path.join(figures_root, KIND_TEMPORAL)),
                modes,
            )
        )
    if KIND_SCALAR in kinds:
        results.extend(
            _emit_scalar(
                conf, remixed['scalar'],
                _ensure_dir(os.path.join(figures_root, KIND_SCALAR)),
                modes,
            )
        )
    if KIND_INTERVAL in kinds:
        # Interval modes use Day axis; still filter with temporal-applicable modes
        # so plate/extra presence matches.
        results.extend(
            _emit_interval(
                conf, remixed['interval'],
                _ensure_dir(os.path.join(figures_root, KIND_INTERVAL)),
                modes,
            )
        )
    if KIND_OVERVIEW in kinds:
        results.extend(
            _emit_overview(
                conf, remixed['temporal'],
                _ensure_dir(os.path.join(figures_root, KIND_OVERVIEW)),
            )
        )

    index_path = os.path.join(figures_root, 'INDEX.md')
    with open(index_path, 'w', encoding='utf-8') as fh:
        fh.write(_scenario_index_md(scenario, summary, results))
    _write_json(
        os.path.join(figures_root, 'results.json'),
        {'results': results, 'modes': modes, 'relative': relative},
    )
    return {
        'scenario': scenario.id,
        'modes': modes,
        'summary': summary,
        'results': results,
        'relative': relative,
        'figures_root': figures_root,
    }


def _root_index_md(out_root: str, reports: List[dict]) -> str:
    lines = [
        '# Plot gallery',
        '',
        'Remixed from local Report `data/` trajectories with invented plate conditions',
        'and run labels. Each scenario folder holds **one metric per figure kind** so layout',
        'review stays fast (not a full report rebuild).',
        '',
        '## Scenarios',
        '',
        '| id | genotypes | plates | extras | plants | blurb |',
        '|----|-----------|--------|--------|--------|-------|',
    ]
    for rep in reports:
        s = rep['summary']
        sc = next(x for x in SCENARIOS if x.id == rep['scenario'])
        lines.append(
            f"| [`{sc.id}`]({sc.id}/) | {sc.n_genotypes} | {sc.n_plates} | {sc.n_extras} "
            f"| {s['n_plants']} | {sc.blurb} |"
        )
    lines.extend([
        '',
        '## Figure kinds (per scenario)',
        '',
        '- `temporal/` — line plots for every applicable comparison mode (+ `_count` companions)',
        '- `scalar/` — germination-time box plots per mode',
        '- `interval/` — convex-hull-area box+swarm at a few Day snapshots',
        '- `overview/` — multi-metric genotype overview',
        '',
        'Open the `.svg` next to each `.png` to judge Day/title spacing precisely.',
        '',
    ])
    return '\n'.join(lines)


def generate_gallery(
    *,
    source_dir: str,
    out_dir: str = DEFAULT_OUT,
    scenario_ids: Optional[Sequence[str]] = None,
    kinds: Sequence[str] = ALL_KINDS,
    seed: int = 0,
    clean: bool = True,
    relative: bool = False,
) -> List[dict]:
    scenarios = list(SCENARIOS)
    if scenario_ids:
        want = set(scenario_ids)
        scenarios = [s for s in scenarios if s.id in want]
        missing = want - {s.id for s in scenarios}
        if missing:
            raise ValueError(f'Unknown scenarios: {sorted(missing)}')

    if clean and os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    _ensure_dir(out_dir)

    tables = load_source_tables(source_dir)

    reports = []
    for scenario in scenarios:
        reports.append(
            generate_scenario(
                scenario, tables, out_dir, kinds=kinds, seed=seed, relative=relative,
            )
        )

    with open(os.path.join(out_dir, 'INDEX.md'), 'w', encoding='utf-8') as fh:
        fh.write(_root_index_md(out_dir, reports))
    _write_json(os.path.join(out_dir, 'gallery_results.json'), {'reports': reports})
    return reports


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description='Build ChronoRoot plot-layout gallery')
    parser.add_argument(
        '--source', required=False, default=None,
        help='Path to a ChronoRoot Report data/ folder (Temporal_Data.csv, …)',
    )
    parser.add_argument('--out', default=DEFAULT_OUT, help='Gallery output folder')
    parser.add_argument('--scenario', action='append', dest='scenarios',
                        help='Scenario id (repeatable). Default: all.')
    parser.add_argument('--kinds', default=','.join(ALL_KINDS),
                        help=f'Comma list from {",".join(ALL_KINDS)}')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--keep', action='store_true', help='Do not wipe out dir first')
    parser.add_argument(
        '--relative', action='store_true',
        help='measureRelativeToInitial titles; write under <scenario>/relative/',
    )
    parser.add_argument('--list', action='store_true', help='List scenarios and exit')
    args = parser.parse_args(argv)

    if args.list:
        for sc in SCENARIOS:
            print(f'{sc.id:12} g={sc.n_genotypes} p={sc.n_plates} e={sc.n_extras}  {sc.blurb}')
        return 0

    if not args.source:
        raise SystemExit(
            'Missing --source. Pass a local ChronoRoot Report data/ directory, e.g.\n'
            '  python -m analysis.plot_gallery.generate --source /path/to/Project/Report/data'
        )

    kinds = tuple(k.strip() for k in args.kinds.split(',') if k.strip())
    bad = set(kinds) - set(ALL_KINDS)
    if bad:
        raise SystemExit(f'Unknown kinds: {sorted(bad)}')

    try:
        reports = generate_gallery(
            source_dir=args.source,
            out_dir=args.out,
            scenario_ids=args.scenarios,
            kinds=kinds,
            seed=args.seed,
            clean=not args.keep,
            relative=args.relative,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from None
    n_ok = sum(1 for r in reports for row in r['results'] if row.get('ok'))
    n_fail = sum(1 for r in reports for row in r['results'] if not row.get('ok'))
    print(f'Gallery → {args.out}')
    print(f'Scenarios: {len(reports)}  figures ok={n_ok} fail={n_fail}')
    for r in reports:
        fails = [row for row in r['results'] if not row.get('ok')]
        print(f'  {r["scenario"]}: modes={r["modes"]}' + (f' FAILS={len(fails)}' if fails else ''))
        for row in fails[:5]:
            print(f'    - {row["kind"]}/{row.get("mode")}: {row.get("error")}')
    return 0 if n_fail == 0 else 1


if __name__ == '__main__':
    raise SystemExit(main())
