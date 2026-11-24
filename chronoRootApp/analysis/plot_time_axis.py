"""Matplotlib helpers for elapsed-time Day labels, midnight dashes, and snapshot axes."""

import math

import numpy as np
import pandas as pd

from .time_windows import (
    day_ticks_for_limits,
    elapsed_clock_ticks,
    format_days_elapsed,
    parse_clock_ticks,
)

_CLOCK_CAPTION_FLAG = '_chronoroot_clock_caption'


def _as_elapsed_hour_frame(data, hour_col='ElapsedTime (h)'):
    if data is None:
        return data
    if hour_col in data.columns and hour_col != 'ElapsedTime (h)':
        return data.rename(columns={hour_col: 'ElapsedTime (h)'})
    if 'ElapsedTime (h)' not in data.columns and 'Time' in data.columns:
        return data.rename(columns={'Time': 'ElapsedTime (h)'})
    return data


def _clock_caption_text(conf):
    labels = [label for _h, _m, label in parse_clock_ticks(conf)]
    if not labels:
        return ''
    if len(labels) == 1:
        return f'Dashed line indicates {labels[0]} hs'
    return 'Dashed lines indicate ' + ', '.join(labels) + ' hs'


def _annotate_clock_caption(ax, conf):
    if ax is None:
        return
    fig = ax.figure
    if getattr(fig, _CLOCK_CAPTION_FLAG, False):
        return
    text = _clock_caption_text(conf)
    if not text:
        return
    fig.text(
        0.99, 0.01, text,
        ha='right', va='bottom', fontsize=8, color='0.4', style='italic',
    )
    setattr(fig, _CLOCK_CAPTION_FLAG, True)


def _visible_plot_axes(axes):
    visible = []
    for ax in np.ravel(list(axes) if not hasattr(axes, 'ndim') else axes):
        if ax is None:
            continue
        if hasattr(ax, 'get_visible') and not ax.get_visible():
            continue
        if hasattr(ax, 'has_data') and not ax.has_data():
            continue
        visible.append(ax)
    return visible


def _top_row_axes(axes):
    """Axes that sit on the first row of a facet / subplot grid."""
    try:
        arr = np.array(axes, dtype=object)
    except Exception:
        arr = np.array(list(np.ravel(axes)), dtype=object)
    if arr.ndim == 2:
        return [
            ax for ax in arr[0, :]
            if ax is not None and (not hasattr(ax, 'get_visible') or ax.get_visible())
        ]
    flat = [ax for ax in np.ravel(arr) if ax is not None]
    visible = _visible_plot_axes(flat)
    n = len(visible)
    if n <= 1:
        return visible
    # Seaborn col_wrap flattens to 1-D in row-major order.
    ncols = int(math.ceil(n / max(1, int(math.floor(math.sqrt(n))))))
    return visible[:ncols]


def draw_day_axis(ax, data, hour_col='ElapsedTime (h)', tick_size=9):
    """Top twin axis with Day N labels at 24 h steps from t0."""
    x_min, x_max = ax.get_xlim()
    ticks, labels = day_ticks_for_limits(x_min, x_max)
    if not ticks and data is not None and not getattr(data, 'empty', True):
        frame = _as_elapsed_hour_frame(data, hour_col)
        if frame is not None and 'ElapsedTime (h)' in frame.columns:
            hours = pd.to_numeric(frame['ElapsedTime (h)'], errors='coerce')
            if hours.notna().any():
                ticks, labels = day_ticks_for_limits(hours.min(), hours.max())
    ax_days = ax.twiny()
    ax_days.set_xlim(ax.get_xlim())
    if ticks:
        ax_days.set_xticks(ticks)
        ax_days.set_xticklabels(labels, rotation=45, ha='left', fontsize=tick_size)
    else:
        ax_days.set_xticks([])
    ax_days.tick_params(axis='x', which='major', length=8, width=2, color='black')
    ax_days.tick_params(axis='x', which='minor', length=4, width=1, color='black')
    ax._chronoroot_has_day_axis = True
    ax._chronoroot_day_axis = ax_days
    return ax_days


def draw_snapshot_day_axis(ax, tick_size=9):
    """Top axis of days elapsed after t0, aligned to snapshot-hour ticks."""
    if ax is None:
        return None
    try:
        ax.figure.canvas.draw()
    except Exception:
        pass
    xticks = np.asarray(ax.get_xticks(), dtype=float)
    label_texts = [lab.get_text().strip() for lab in ax.get_xticklabels()]
    positions = []
    hours = []
    if label_texts:
        for pos, text in zip(xticks, label_texts):
            if not np.isfinite(pos):
                continue
            value = None
            if text:
                try:
                    value = float(text)
                except (TypeError, ValueError):
                    value = None
            if value is None:
                value = float(pos)
            positions.append(pos)
            hours.append(value)
    else:
        for pos in xticks:
            if np.isfinite(pos):
                positions.append(float(pos))
                hours.append(float(pos))
    if not positions:
        return None
    ax_days = ax.twiny()
    ax_days.set_xlim(ax.get_xlim())
    ax_days.set_xticks(positions)
    ax_days.set_xticklabels(
        [format_days_elapsed(hour) for hour in hours],
        fontsize=tick_size,
    )
    ax_days.set_xlabel('Days elapsed after t0')
    ax_days.tick_params(axis='x', which='major', length=6, width=1, color='black')
    return ax_days


def draw_snapshot_day_axis_on(plot_obj):
    """Apply draw_snapshot_day_axis to an Axes or a FacetGrid."""
    if plot_obj is None:
        return
    if hasattr(plot_obj, 'axes'):
        for ax in _visible_plot_axes(np.ravel(plot_obj.axes)):
            draw_snapshot_day_axis(ax)
    else:
        draw_snapshot_day_axis(plot_obj)


def draw_clock_ticks(ax, data, conf, *, twin_axis=False, annotate=True):
    """Draw dashed axvline markers at real-clock ticks. No HH:MM axis labels."""
    ticks = elapsed_clock_ticks(data, conf)
    if not ticks:
        return ticks
    for hour in ticks:
        ax.axvline(hour, color='0.55', linestyle='--', linewidth=0.9, zorder=0)
    if annotate:
        _annotate_clock_caption(ax, conf)
    return ticks


def decorate_elapsed_time_axis(
    ax, data, conf, *, day_axis=True, annotate=True,
    hour_col='ElapsedTime (h)', day_tick_size=9,
):
    """Day-X top axis, midnight dashes, and a once-per-figure caption."""
    tick_df = _as_elapsed_hour_frame(data, hour_col)
    ticks = draw_clock_ticks(ax, tick_df, conf, twin_axis=False, annotate=annotate)
    if day_axis:
        draw_day_axis(ax, tick_df, hour_col='ElapsedTime (h)', tick_size=day_tick_size)
    return ticks


def decorate_elapsed_time_axes(
    axes, data, conf, *, day_axis=True,
    hour_col='ElapsedTime (h)', day_tick_size=9,
):
    """Decorate a grid of elapsed-time axes; caption once at figure bottom-right.

    Day twin-axes are only drawn on the top row. Putting them on every facet
    piles Day labels into the row above on multi-row FacetGrids.
    """
    visible = _visible_plot_axes(axes)
    top_row = set(_top_row_axes(axes)) if day_axis else set()
    drawn = False
    for ax in visible:
        use_day = day_axis and (ax in top_row)
        ticks = decorate_elapsed_time_axis(
            ax, data, conf, day_axis=use_day, annotate=False,
            hour_col=hour_col, day_tick_size=day_tick_size,
        )
        if ticks:
            drawn = True
    if drawn and visible:
        _annotate_clock_caption(visible[-1], conf)
    return drawn


def draw_clock_ticks_on_axes(axes, data, conf):
    decorate_elapsed_time_axes(axes, data, conf)


# Small gap (points) between the top of Day labels and the facet title baseline.
_FACET_TITLE_GAP_PT = 4
# Small gap between the top of facet titles and the figure title.
_SUPTITLE_GAP_PT = 6


def _renderer_for(fig):
    """Force a draw so tick/title window extents are valid."""
    try:
        fig.canvas.draw()
        return fig.canvas.get_renderer()
    except Exception:
        return None


def _day_label_top_display_y(ax, renderer):
    """Highest display-Y of Day twin tick labels on this axes (or None)."""
    day_ax = getattr(ax, '_chronoroot_day_axis', None)
    if day_ax is None or renderer is None:
        return None
    tops = []
    for lab in day_ax.get_xticklabels():
        if not lab.get_visible() or not (lab.get_text() or '').strip():
            continue
        try:
            tops.append(lab.get_window_extent(renderer=renderer).y1)
        except Exception:
            continue
    return max(tops) if tops else None


def push_facet_titles_below_day_axes(plot_obj):
    """Place facet titles just above measured Day twin labels.

    Applies to all facet modes (genotype_by_plate, plate_within_genotype, …)
    and overview grids. Day tick style is unchanged. Panels without a Day twin
    keep seaborn titles.
    """
    if plot_obj is None:
        return
    if isinstance(plot_obj, np.ndarray):
        axes = list(np.ravel(plot_obj))
    else:
        axes = []
        grid = getattr(plot_obj, 'axes', None)
        if grid is not None and grid is not plot_obj:
            axes = list(np.ravel(grid))
        else:
            axes = [plot_obj]
    fig = None
    for ax in axes:
        if ax is not None and hasattr(ax, 'figure'):
            fig = ax.figure
            break
    renderer = _renderer_for(fig) if fig is not None else None
    for ax in axes:
        if ax is None or not hasattr(ax, 'get_title'):
            continue
        title = ax.get_title()
        if not title:
            continue
        if not getattr(ax, '_chronoroot_has_day_axis', False):
            continue
        ax.set_title('')
        day_top = _day_label_top_display_y(ax, renderer)
        if day_top is None or renderer is None:
            # Fallback: modest pad above the spine (points).
            ax.annotate(
                title,
                xy=(0.5, 1.0),
                xycoords='axes fraction',
                xytext=(0, 36),
                textcoords='offset points',
                ha='center',
                va='bottom',
                fontsize=11,
                annotation_clip=False,
            )
            continue
        # Convert Day-label top + gap into axes-fraction y for this panel.
        gap_px = _FACET_TITLE_GAP_PT * (renderer.points_to_pixels(1.0) if hasattr(renderer, 'points_to_pixels') else renderer.dpi / 72.0)
        _, y_axes = ax.transAxes.inverted().transform((0.0, day_top + gap_px))
        ax.text(
            0.5, y_axes, title,
            transform=ax.transAxes,
            ha='center',
            va='bottom',
            fontsize=11,
            clip_on=False,
        )


def space_elapsed_time_facet_grid(fig, axes):
    """Light layout only — no inflated figure height / giant top band.

    Facet titles sit just above measured Day labels; bbox_inches='tight' on
    save crops the rest. Multi-row grids only need a bit of hspace.
    """
    if fig is None:
        return
    try:
        arr = np.array(axes, dtype=object)
    except Exception:
        return
    visible = _visible_plot_axes(arr)
    has_day = any(getattr(ax, '_chronoroot_has_day_axis', False) for ax in visible)
    if getattr(arr, 'ndim', 1) == 2:
        nrows = max(1, int(arr.shape[0]))
    else:
        n = len(visible)
        nrows = 1 if n <= 1 else max(1, int(math.floor(math.sqrt(n))))
        if visible and all(getattr(ax, '_chronoroot_has_day_axis', False) for ax in visible):
            nrows = 1
    try:
        fig.set_layout_engine('none')
    except Exception:
        pass
    if nrows > 1:
        # Room between rows so lower-row titles don't hit upper-row x labels.
        fig.subplots_adjust(hspace=0.32 if not has_day else 0.36)


def place_elapsed_time_suptitle(fig, title, *, top=None, y=None):
    """Put the figure title just above the highest facet/Day content.

    Only measures content above the axes (Day twin labels, raised facet titles,
    Axes titles). Bottom x tick labels must not drive placement — on interval
    plots those are category Days at the foot of the axes and would pin the
    figure title to the bottom of the SVG.
    """
    if fig is None or not title:
        return
    try:
        fig.set_layout_engine('none')
    except Exception:
        pass
    if top is not None:
        fig.subplots_adjust(top=top)
    if y is not None:
        fig.suptitle(title, y=y)
        return
    renderer = _renderer_for(fig)
    if renderer is None:
        fig.suptitle(title, y=1.02, va='bottom')
        return
    top_disp = None
    for ax in fig.axes:
        if ax is None:
            continue
        day_top = _day_label_top_display_y(ax, renderer)
        if day_top is not None:
            top_disp = day_top if top_disp is None else max(top_disp, day_top)
        # Raised facet titles live in ax.texts (not bottom tick labels).
        for artist in list(getattr(ax, 'texts', []) or []):
            try:
                text = artist.get_text() if hasattr(artist, 'get_text') else ''
                if not (text or '').strip():
                    continue
                y1 = artist.get_window_extent(renderer=renderer).y1
            except Exception:
                continue
            top_disp = y1 if top_disp is None else max(top_disp, y1)
        # Default seaborn facet titles.
        try:
            title_artist = ax.title
            if title_artist is not None and (ax.get_title() or '').strip():
                y1 = title_artist.get_window_extent(renderer=renderer).y1
                top_disp = y1 if top_disp is None else max(top_disp, y1)
        except Exception:
            pass
    fig_h = fig.bbox.height
    if top_disp is None or fig_h <= 0:
        fig.suptitle(title, y=1.02, va='bottom')
        return
    gap_px = _SUPTITLE_GAP_PT * (
        renderer.points_to_pixels(1.0)
        if hasattr(renderer, 'points_to_pixels')
        else renderer.dpi / 72.0
    )
    y_frac = (top_disp + gap_px) / fig_h
    # Guard: never place below mid-figure (bottom tick contamination).
    if y_frac < 0.85:
        y_frac = 1.02
    fig.suptitle(title, y=y_frac, va='bottom')
