"""Comparison-mode plots paired with statistical report outputs."""

import os

import matplotlib.pyplot as plt
import seaborn as sns

from .stats_utils import (
    _mode_spec,
    comparison_modes_for_run,
    ensure_factor_columns,
)
from .utils.fileUtilities import normalize_factor_value, UNSPECIFIED_FACTOR
from .utils.report_paths import comparison_plot_path
from .time_windows import draw_clock_ticks
from .utils.report_style import (
    apply_factor_axis_labels,
    axis_label_for_column,
    legend_title_for_hue,
    palette_for_hue,
)

plt.switch_backend('agg')


def _plot_title(metric_label, spec):
    return f'{metric_label} — {spec["header"]}'


def _palette_kwargs(data, hue_col, conf):
    palette = palette_for_hue(data, hue_col)
    kwargs = {}
    if palette:
        kwargs['palette'] = palette
    return kwargs


def _set_hue_legend(ax, hue_col, conf):
    leg = ax.get_legend()
    if leg is not None:
        leg.set_title(legend_title_for_hue(hue_col, conf))


_FACTOR_COLUMNS = frozenset({'Experiment', 'PlateCondition', 'ExtraVariable', 'Type'})


def finalize_comparison_axes(plot_obj, conf, *, x_col=None, hue_col=None, facet_col=None):
    """Apply configured factor labels to a matplotlib Axes or seaborn FacetGrid."""
    if hasattr(plot_obj, 'axes'):
        g = plot_obj
        if facet_col in _FACTOR_COLUMNS:
            g.set_titles(col_template='{col_name}')
        x_label = axis_label_for_column(x_col, conf) if x_col in _FACTOR_COLUMNS else None
        for ax in g.axes.flat:
            if x_label:
                ax.set_xlabel(x_label)
            if hue_col:
                _set_hue_legend(ax, hue_col, conf)
    else:
        apply_factor_axis_labels(plot_obj, conf, x_col=x_col, hue_col=hue_col)


def _mode_axes(mode, x_col='ElapsedTime (h)'):
    """Return (x_col, hue_col, facet_col) for a comparison mode."""
    specs = {
        'by_genotype': (x_col, 'Experiment', None),
        'genotype_by_plate': (x_col, 'Experiment', 'PlateCondition'),
        'genotype_by_extra': (x_col, 'Experiment', 'ExtraVariable'),
        'by_plate_condition': (x_col, 'PlateCondition', None),
        'by_extra_variable': (x_col, 'ExtraVariable', None),
        'plate_within_genotype': (x_col, 'PlateCondition', 'Experiment'),
        'extra_within_genotype': (x_col, 'ExtraVariable', 'Experiment'),
    }
    return specs.get(mode, (x_col, None, None))


def _lineplot_kwargs(x_col, y_col, hue_col, facet_col=None, data=None, conf=None):
    kwargs = {
        'x': x_col,
        'y': y_col,
        'hue': hue_col,
        'errorbar': 'se',
        'height': 4,
        'aspect': 1.2,
        'kind': 'line',
    }
    if facet_col:
        kwargs['col'] = facet_col
    if data is not None and conf is not None:
        kwargs.update(_palette_kwargs(data, hue_col, conf))
    return kwargs


def _catplot_kwargs(data, x, y, hue, col, conf, kind='box'):
    kwargs = {
        'data': data, 'x': x, 'y': y, 'hue': hue, 'col': col,
        'kind': kind, 'height': 4, 'aspect': 1.2,
    }
    kwargs.update(_palette_kwargs(data, hue, conf))
    return kwargs


def _count_lineplot_kwargs(x_col, hue_col, facet_col=None, data=None, conf=None):
    kwargs = _lineplot_kwargs(x_col, '_n', hue_col, facet_col=facet_col, data=data, conf=conf)
    kwargs['errorbar'] = None
    kwargs['estimator'] = 'sum'
    return kwargs


def _count_plot_path(output_path):
    root, ext = os.path.splitext(output_path)
    return f'{root}_count{ext or ".png"}'


def _save_temporal_count_plot(conf, data, metric, mode, output_path, x_col, metric_label):
    """Companion N-over-time figure in the same folder as the mean plot."""
    if metric not in data.columns:
        return
    count_data = data.copy()
    count_data['_n'] = count_data[metric].notna().astype(int)
    title = f'{metric_label} — number of plants'
    count_path = _count_plot_path(output_path)
    try:
        if mode == 'by_genotype':
            g = sns.relplot(
                data=count_data,
                **_count_lineplot_kwargs(x_col, 'Experiment', data=count_data, conf=conf),
            )
        elif mode == 'genotype_by_plate':
            g = sns.relplot(
                data=count_data,
                **_count_lineplot_kwargs(x_col, 'Experiment', 'PlateCondition', data=count_data, conf=conf),
                col_order=sorted(
                    [v for v in count_data['PlateCondition'].unique()
                     if normalize_factor_value(v) != UNSPECIFIED_FACTOR],
                    key=str,
                ),
            )
        elif mode == 'genotype_by_extra':
            g = sns.relplot(
                data=count_data,
                **_count_lineplot_kwargs(x_col, 'Experiment', 'ExtraVariable', data=count_data, conf=conf),
                col_order=sorted(
                    [v for v in count_data['ExtraVariable'].unique()
                     if normalize_factor_value(v) != UNSPECIFIED_FACTOR],
                    key=str,
                ),
            )
        elif mode == 'by_plate_condition':
            g = sns.relplot(
                data=count_data,
                **_count_lineplot_kwargs(x_col, 'PlateCondition', data=count_data, conf=conf),
            )
        elif mode == 'by_extra_variable':
            g = sns.relplot(
                data=count_data,
                **_count_lineplot_kwargs(x_col, 'ExtraVariable', data=count_data, conf=conf),
            )
        elif mode == 'plate_within_genotype':
            g = sns.relplot(
                data=count_data,
                **_count_lineplot_kwargs(x_col, 'PlateCondition', 'Experiment', data=count_data, conf=conf),
                col_order=sorted(count_data['Experiment'].unique(), key=str),
            )
        elif mode == 'extra_within_genotype':
            g = sns.relplot(
                data=count_data,
                **_count_lineplot_kwargs(x_col, 'ExtraVariable', 'Experiment', data=count_data, conf=conf),
                col_order=sorted(count_data['Experiment'].unique(), key=str),
            )
        else:
            return
        axes_x, hue_col, facet_col = _mode_axes(mode, x_col)
        finalize_comparison_axes(g, conf, x_col=axes_x, hue_col=hue_col, facet_col=facet_col)
        if x_col in ('ElapsedTime (h)', 'Time'):
            for ax in g.axes.flat:
                draw_clock_ticks(ax, count_data, conf, twin_axis=False)
        g.set_ylabels('Number of plants')
        g.fig.suptitle(title, y=1.02)
        g.savefig(count_path, dpi=300, bbox_inches='tight')
    except Exception:
        pass
    finally:
        plt.close('all')


def plot_comparison_mode(conf, data, metric, mode, output_path, *,
                         x_col='ElapsedTime (h)', metric_label=None, title=None,
                         module=None, metric_slug_name=None, analysis_type='temporal'):
    """Save a line plot matching the grouping used for a comparison mode."""
    data = ensure_factor_columns(data)
    plot_data = data.dropna(subset=[metric]) if metric in data.columns else data

    spec = _mode_spec(mode, conf=conf)
    metric_label = metric_label or metric
    title = title or _plot_title(metric_label, spec)

    plt.ioff()
    try:
        if mode == 'by_genotype':
            g = sns.relplot(data=plot_data, **_lineplot_kwargs(x_col, metric, 'Experiment', data=plot_data, conf=conf))
        elif mode == 'genotype_by_plate':
            g = sns.relplot(
                data=plot_data, **_lineplot_kwargs(x_col, metric, 'Experiment', 'PlateCondition', data=plot_data, conf=conf),
                col_order=sorted(
                    [v for v in plot_data['PlateCondition'].unique() if normalize_factor_value(v) != UNSPECIFIED_FACTOR],
                    key=str,
                ),
            )
        elif mode == 'genotype_by_extra':
            g = sns.relplot(
                data=plot_data, **_lineplot_kwargs(x_col, metric, 'Experiment', 'ExtraVariable', data=plot_data, conf=conf),
                col_order=sorted(
                    [v for v in plot_data['ExtraVariable'].unique() if normalize_factor_value(v) != UNSPECIFIED_FACTOR],
                    key=str,
                ),
            )
        elif mode == 'by_plate_condition':
            g = sns.relplot(data=plot_data, **_lineplot_kwargs(x_col, metric, 'PlateCondition', data=plot_data, conf=conf))
        elif mode == 'by_extra_variable':
            g = sns.relplot(data=plot_data, **_lineplot_kwargs(x_col, metric, 'ExtraVariable', data=plot_data, conf=conf))
        elif mode == 'plate_within_genotype':
            g = sns.relplot(
                data=plot_data, **_lineplot_kwargs(x_col, metric, 'PlateCondition', 'Experiment', data=plot_data, conf=conf),
                col_order=sorted(plot_data['Experiment'].unique(), key=str),
            )
        elif mode == 'extra_within_genotype':
            g = sns.relplot(
                data=plot_data, **_lineplot_kwargs(x_col, metric, 'ExtraVariable', 'Experiment', data=plot_data, conf=conf),
                col_order=sorted(plot_data['Experiment'].unique(), key=str),
            )
        else:
            return False

        axes_x, hue_col, facet_col = _mode_axes(mode, x_col)
        finalize_comparison_axes(g, conf, x_col=axes_x, hue_col=hue_col, facet_col=facet_col)
        if x_col in ('ElapsedTime (h)', 'Time'):
            for ax in g.axes.flat:
                draw_clock_ticks(ax, plot_data, conf, twin_axis=False)
        g.fig.suptitle(title, y=1.02)
        g.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close('all')
        _save_temporal_count_plot(conf, data, metric, mode, output_path, x_col, metric_label)
        return True
    except Exception:
        plt.close('all')
        return False


def plot_scalar_comparison_mode(conf, data, metric, mode, base_dir, *,
                                module=None, metric_slug_name=None, analysis_type='scalar',
                                metric_label=None, x_col='Experiment', plot_kind='box'):
    """Box/swarm plot for scalar metrics (FPCA PCs, etc.)."""
    data = ensure_factor_columns(data)

    spec = _mode_spec(mode, conf=conf)
    metric_label = metric_label or metric
    title = _plot_title(metric_label, spec)
    output_path = comparison_plot_path(base_dir, mode, metric_slug=metric_slug_name or '')
    plt.ioff()

    try:
        if mode == 'by_genotype':
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.boxplot(data=data, x=x_col, y=metric, hue='Experiment', ax=ax,
                        **_palette_kwargs(data, 'Experiment', conf))
            ax.set_title(title)
            finalize_comparison_axes(ax, conf, x_col=x_col, hue_col='Experiment')
        elif mode == 'genotype_by_plate':
            g = sns.catplot(**_catplot_kwargs(data, 'Experiment', metric, 'Experiment', 'PlateCondition', conf, plot_kind))
            finalize_comparison_axes(g, conf, x_col='Experiment', hue_col='Experiment', facet_col='PlateCondition')
            g.fig.suptitle(title, y=1.02)
            g.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close('all')
            return True
        elif mode == 'genotype_by_extra':
            g = sns.catplot(**_catplot_kwargs(data, 'Experiment', metric, 'Experiment', 'ExtraVariable', conf, plot_kind))
            finalize_comparison_axes(g, conf, x_col='Experiment', hue_col='Experiment', facet_col='ExtraVariable')
            g.fig.suptitle(title, y=1.02)
            g.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close('all')
            return True
        elif mode == 'by_plate_condition':
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.boxplot(data=data, x='PlateCondition', y=metric, hue='PlateCondition', ax=ax,
                        **_palette_kwargs(data, 'PlateCondition', conf))
            ax.set_title(title)
            finalize_comparison_axes(ax, conf, x_col='PlateCondition', hue_col='PlateCondition')
        elif mode == 'by_extra_variable':
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.boxplot(data=data, x='ExtraVariable', y=metric, hue='ExtraVariable', ax=ax,
                        **_palette_kwargs(data, 'ExtraVariable', conf))
            ax.set_title(title)
            finalize_comparison_axes(ax, conf, x_col='ExtraVariable', hue_col='ExtraVariable')
        elif mode == 'plate_within_genotype':
            g = sns.catplot(**_catplot_kwargs(data, 'PlateCondition', metric, 'PlateCondition', 'Experiment', conf, plot_kind))
            finalize_comparison_axes(g, conf, x_col='PlateCondition', hue_col='PlateCondition', facet_col='Experiment')
            g.fig.suptitle(title, y=1.02)
            g.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close('all')
            return True
        elif mode == 'extra_within_genotype':
            g = sns.catplot(**_catplot_kwargs(data, 'ExtraVariable', metric, 'ExtraVariable', 'Experiment', conf, plot_kind))
            finalize_comparison_axes(g, conf, x_col='ExtraVariable', hue_col='ExtraVariable', facet_col='Experiment')
            g.fig.suptitle(title, y=1.02)
            g.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close('all')
            return True
        else:
            return False

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close('all')
        return True
    except Exception:
        plt.close('all')
        return False


def plot_interval_comparison_mode(conf, data, metric, mode, base_dir, *,
                                   x_col='Day', module=None, metric_slug_name=None,
                                   analysis_type='interval', metric_label=None):
    """Violin/swarm plot for interval-based metrics (convex hull, angles)."""
    data = ensure_factor_columns(data)

    spec = _mode_spec(mode, conf=conf)
    metric_label = metric_label or metric
    title = _plot_title(metric_label, spec)
    output_path = comparison_plot_path(base_dir, mode, metric_slug=metric_slug_name or '')
    plt.ioff()

    try:
        if mode == 'by_genotype':
            fig, ax = plt.subplots(figsize=(8, 6))
            exp_palette = _palette_kwargs(data, 'Experiment', conf)
            sns.violinplot(data=data, x=x_col, y=metric, hue='Experiment', ax=ax, inner=None, **exp_palette)
            sns.swarmplot(data=data, x=x_col, y=metric, hue='Experiment', ax=ax, dodge=True, size=3, **exp_palette)
            ax.set_title(title)
            finalize_comparison_axes(ax, conf, x_col=x_col, hue_col='Experiment')
        elif mode == 'genotype_by_plate':
            g = sns.catplot(**_catplot_kwargs(data, x_col, metric, 'Experiment', 'PlateCondition', conf, 'violin'))
            finalize_comparison_axes(g, conf, x_col=x_col, hue_col='Experiment', facet_col='PlateCondition')
            g.fig.suptitle(title, y=1.02)
            g.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close('all')
            return True
        elif mode == 'genotype_by_extra':
            g = sns.catplot(**_catplot_kwargs(data, x_col, metric, 'Experiment', 'ExtraVariable', conf, 'violin'))
            finalize_comparison_axes(g, conf, x_col=x_col, hue_col='Experiment', facet_col='ExtraVariable')
            g.fig.suptitle(title, y=1.02)
            g.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close('all')
            return True
        elif mode == 'by_plate_condition':
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.violinplot(data=data, x=x_col, y=metric, hue='PlateCondition', ax=ax,
                           **_palette_kwargs(data, 'PlateCondition', conf))
            ax.set_title(title)
            finalize_comparison_axes(ax, conf, x_col=x_col, hue_col='PlateCondition')
        elif mode == 'by_extra_variable':
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.violinplot(data=data, x=x_col, y=metric, hue='ExtraVariable', ax=ax,
                           **_palette_kwargs(data, 'ExtraVariable', conf))
            ax.set_title(title)
            finalize_comparison_axes(ax, conf, x_col=x_col, hue_col='ExtraVariable')
        elif mode == 'plate_within_genotype':
            g = sns.catplot(**_catplot_kwargs(data, x_col, metric, 'PlateCondition', 'Experiment', conf, 'violin'))
            finalize_comparison_axes(g, conf, x_col=x_col, hue_col='PlateCondition', facet_col='Experiment')
            g.fig.suptitle(title, y=1.02)
            g.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close('all')
            return True
        elif mode == 'extra_within_genotype':
            g = sns.catplot(**_catplot_kwargs(data, x_col, metric, 'ExtraVariable', 'Experiment', conf, 'violin'))
            finalize_comparison_axes(g, conf, x_col=x_col, hue_col='ExtraVariable', facet_col='Experiment')
            g.fig.suptitle(title, y=1.02)
            g.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close('all')
            return True
        else:
            return False

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close('all')
        return True
    except Exception:
        plt.close('all')
        return False


def emit_temporal_comparison_plots(conf, data, metric, base_dir, *, module, metric_slug_name,
                                   analysis_type='temporal', metric_label=None):
    """Generate paired {metric}_{mode}.png for every enabled comparison mode."""
    for mode in comparison_modes_for_run(conf, data):
        output_path = comparison_plot_path(base_dir, mode, metric_slug=metric_slug_name)
        plot_comparison_mode(
            conf, data, metric, mode, output_path,
            metric_label=metric_label or metric,
            module=module, metric_slug_name=metric_slug_name,
            analysis_type=analysis_type,
        )


def emit_interval_comparison_plots(conf, data, metric, base_dir, *, module, metric_slug_name,
                                   analysis_type='interval', x_col='Day', metric_label=None):
    for mode in comparison_modes_for_run(conf, data):
        plot_interval_comparison_mode(
            conf, data, metric, mode, base_dir,
            x_col=x_col, module=module, metric_slug_name=metric_slug_name,
            analysis_type=analysis_type, metric_label=metric_label or metric,
        )


def emit_scalar_comparison_plots(conf, data, metric, base_dir, *, module, metric_slug_name,
                                 analysis_type='scalar', metric_label=None):
    for mode in comparison_modes_for_run(conf, data):
        plot_scalar_comparison_mode(
            conf, data, metric, mode, base_dir,
            module=module, metric_slug_name=metric_slug_name,
            analysis_type=analysis_type, metric_label=metric_label or metric,
        )
