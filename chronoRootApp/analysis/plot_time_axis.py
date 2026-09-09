"""Matplotlib helpers for elapsed-time Day labels, midnight dashes, and snapshot axes."""

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
    for ax in list(axes):
        if ax is None:
            continue
        if hasattr(ax, 'get_visible') and not ax.get_visible():
            continue
        if hasattr(ax, 'has_data') and not ax.has_data():
            continue
        visible.append(ax)
    return visible


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
    """Decorate a grid of elapsed-time axes; caption once at figure bottom-right."""
    visible = _visible_plot_axes(axes)
    drawn = False
    for ax in visible:
        ticks = decorate_elapsed_time_axis(
            ax, data, conf, day_axis=day_axis, annotate=False,
            hour_col=hour_col, day_tick_size=day_tick_size,
        )
        if ticks:
            drawn = True
    if drawn and visible:
        _annotate_clock_caption(visible[-1], conf)
    return drawn


def draw_clock_ticks_on_axes(axes, data, conf):
    decorate_elapsed_time_axes(axes, data, conf)


def place_elapsed_time_suptitle(fig, title, *, top=0.82, y=0.98):
    """Put the figure title above Day-axis ticks instead of on top of them."""
    if fig is None or not title:
        return
    try:
        fig.set_layout_engine('none')
    except Exception:
        pass
    fig.subplots_adjust(top=top)
    fig.suptitle(title, y=y)
