"""Dialog to set the analysis period, anchors, and clock ticks."""

from datetime import timedelta

from PyQt5 import QtCore, QtGui, QtWidgets

from analysis.time_windows import (
    collect_acquisition_spans,
    default_duration_hours,
    detect_time_groups,
    elapsed_hours_from_t0,
    format_datetime,
    groups_to_config,
    parse_clock_ticks,
    parse_datetime,
    shared_clock_origin,
    time_period_sources,
)


HELP_TEXT = (
    "Real-time sync uses one shared clock origin (the earliest group start). "
    "Later groups are padded; hours before that origin are dropped. "
    "Anchor sync sets elapsed 0 at a treatment time."
)

ANCHOR_TOOLTIP = (
    "Treatment / application time. Elapsed time is 0 at this instant; "
    "hours before it are negative."
)

PREVIEW_CAPTION = (
    "Elapsed-hour grid shared by every group. "
    "Clock origin is the earliest start; later groups sit to the right (padding). "
    "Acquisition outside each window is cropped."
)

EDITOR_MIN_WIDTH = 220
AXIS_LABEL_H = 18
TABLE_LEFT_PAD = 10
READ_ONLY_EDITOR_STYLE = (
    "QDateEdit, QTimeEdit { background: white; color: black; }"
)


def _format_display_datetime(value):
    ts = parse_datetime(value)
    if ts is None:
        return str(value or '').replace('T', ' ')
    return ts.strftime('%Y-%m-%d %H:%M')


def _overlay_saved_groups(saved, detected):
    """Keep a saved window start and anchor when they still belong to this acquisition.

    Groups are matched by id. The saved start and anchor are copied only when
    the saved acquisition span still overlaps the span just detected for that
    id. The same group numbers show up in every project; without the overlap
    check, the previous project's clock times are painted onto the new one.
    Returns the merged rows and how many saved groups were actually reused.
    """
    saved_by_id = {}
    for group in saved or []:
        try:
            saved_by_id[int(group.get('id') or 0)] = group
        except (TypeError, ValueError):
            continue
    merged = []
    kept = 0
    for det in detected or []:
        row = dict(det)
        try:
            gid = int(det.get('id') or 0)
        except (TypeError, ValueError):
            gid = 0
        old = saved_by_id.get(gid)
        if old is not None:
            old_start = parse_datetime(old.get('spanStart'))
            old_end = parse_datetime(old.get('spanEnd'))
            new_start = parse_datetime(det.get('spanStart'))
            new_end = parse_datetime(det.get('spanEnd'))
            spans_overlap = (
                None not in (old_start, old_end, new_start, new_end)
                and old_start <= new_end
                and new_start <= old_end
            )
            if spans_overlap:
                kept += 1
                if old.get('start'):
                    row['start'] = old['start']
                if old.get('t0'):
                    row['t0'] = old['t0']
        merged.append(row)
    return merged, kept


def _draw_axis_end_labels(painter, width, height, left_text, right_text, left=8):
    painter.setPen(QtGui.QColor(80, 80, 80))
    y = height - AXIS_LABEL_H
    usable = max(width - left - 8, 80)
    half = usable // 2
    painter.drawText(
        QtCore.QRect(left, y, half, AXIS_LABEL_H),
        QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter,
        left_text,
    )
    painter.drawText(
        QtCore.QRect(left + half, y, max(width - left - half - 8, 40), AXIS_LABEL_H),
        QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter,
        right_text,
    )


def _to_qdate(value):
    ts = parse_datetime(value)
    if ts is None:
        return QtCore.QDate.currentDate()
    return QtCore.QDate(int(ts.year), int(ts.month), int(ts.day))


def _to_qtime(value):
    ts = parse_datetime(value)
    if ts is None:
        return QtCore.QTime(0, 0)
    return QtCore.QTime(ts.hour, ts.minute, 0)


def _elapsed_int(value, t0):
    ts = parse_datetime(value)
    if ts is None or parse_datetime(t0) is None:
        return None
    series = elapsed_hours_from_t0([ts], t0)
    if series.isna().iloc[0]:
        return None
    return int(series.iloc[0])


class _DateTimeEditor(QtWidgets.QWidget):
    changed = QtCore.pyqtSignal()

    def __init__(self, value=None, parent=None, read_only=False):
        super().__init__(parent)
        self.setMinimumWidth(EDITOR_MIN_WIDTH)
        self._read_only = False
        self.date_edit = QtWidgets.QDateEdit(_to_qdate(value))
        self.date_edit.setDisplayFormat("yyyy-MM-dd")
        self.date_edit.setCalendarPopup(True)
        self.time_edit = QtWidgets.QTimeEdit(_to_qtime(value))
        self.time_edit.setDisplayFormat("HH:mm")
        self.time_edit.setCalendarPopup(False)
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.date_edit)
        layout.addWidget(self.time_edit)
        self.date_edit.dateChanged.connect(self.changed)
        self.time_edit.timeChanged.connect(self.changed)
        if read_only:
            self.set_read_only(True)

    def set_read_only(self, read_only):
        self._read_only = bool(read_only)
        self.date_edit.setReadOnly(read_only)
        self.time_edit.setReadOnly(read_only)
        self.date_edit.setButtonSymbols(
            QtWidgets.QAbstractSpinBox.NoButtons if read_only
            else QtWidgets.QAbstractSpinBox.UpDownArrows
        )
        self.time_edit.setButtonSymbols(
            QtWidgets.QAbstractSpinBox.NoButtons if read_only
            else QtWidgets.QAbstractSpinBox.UpDownArrows
        )
        self.date_edit.setCalendarPopup(not read_only)
        focus = QtCore.Qt.NoFocus if read_only else QtCore.Qt.StrongFocus
        self.date_edit.setFocusPolicy(focus)
        self.time_edit.setFocusPolicy(focus)
        self.setStyleSheet(READ_ONLY_EDITOR_STYLE if read_only else '')
        palette = self.date_edit.palette()
        palette.setColor(QtGui.QPalette.Base, QtGui.QColor('white'))
        palette.setColor(QtGui.QPalette.Text, QtGui.QColor('black'))
        self.date_edit.setPalette(palette)
        self.time_edit.setPalette(palette)
        self.date_edit.installEventFilter(self)
        self.time_edit.installEventFilter(self)

    def eventFilter(self, obj, event):
        if self._read_only and obj in (self.date_edit, self.time_edit):
            if event.type() in (
                QtCore.QEvent.Wheel,
                QtCore.QEvent.KeyPress,
                QtCore.QEvent.KeyRelease,
            ):
                return True
        return super().eventFilter(obj, event)

    def set_datetime(self, value):
        self.date_edit.blockSignals(True)
        self.time_edit.blockSignals(True)
        if isinstance(value, QtCore.QDateTime):
            self.date_edit.setDate(value.date())
            self.time_edit.setTime(value.time())
        else:
            self.date_edit.setDate(_to_qdate(value))
            self.time_edit.setTime(_to_qtime(value))
        self.date_edit.blockSignals(False)
        self.time_edit.blockSignals(False)

    def to_string(self):
        dt = QtCore.QDateTime(self.date_edit.date(), self.time_edit.time())
        return format_datetime(dt.toPyDateTime())


class _TimelineWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(110)
        self._groups = []
        self._span_start = None
        self._span_end = None
        self._show_anchor = False
        self._empty_text = "No groups to show on the timeline"

    def set_empty_text(self, text):
        self._empty_text = text or "No groups to show on the timeline"
        if not self._groups:
            self.update()

    def set_groups(self, groups, show_anchor=False):
        self._groups = list(groups or [])
        self._show_anchor = bool(show_anchor)
        times = []
        for group in self._groups:
            for key in ('spanStart', 'spanEnd', 'start', 'end', 't0'):
                ts = parse_datetime(group.get(key))
                if ts is not None:
                    times.append(ts)
        if times:
            self._span_start = min(times)
            self._span_end = max(times)
            if self._span_end <= self._span_start:
                self._span_end = self._span_start + timedelta(hours=1)
        else:
            self._span_start = None
            self._span_end = None
        self.update()

    def _x_for_time(self, ts, width):
        if self._span_start is None or ts is None:
            return 8
        parsed = parse_datetime(ts)
        if parsed is None:
            return 8
        total = (self._span_end - self._span_start).total_seconds()
        if total <= 0:
            return 8
        frac = (parsed - self._span_start).total_seconds() / total
        frac = min(max(frac, 0.0), 1.0)
        return int(8 + frac * (width - 16))

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.fillRect(self.rect(), QtGui.QColor(248, 248, 248))
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        width = self.width()
        height = self.height()
        if not self._groups or self._span_start is None:
            painter.setPen(QtGui.QColor(120, 120, 120))
            painter.drawText(
                self.rect(), QtCore.Qt.AlignCenter,
                self._empty_text,
            )
            return

        n = len(self._groups)
        row_h = max(16, min(28, (height - AXIS_LABEL_H - 8) // max(n, 1)))
        colors = [
            QtGui.QColor(70, 130, 180),
            QtGui.QColor(60, 160, 110),
            QtGui.QColor(180, 110, 60),
            QtGui.QColor(130, 90, 170),
        ]
        for i, group in enumerate(self._groups):
            y = 8 + i * row_h
            x1 = self._x_for_time(group.get('spanStart') or group.get('start'), width)
            x2 = self._x_for_time(group.get('spanEnd') or group.get('end') or group.get('start'), width)
            if x2 - x1 < 6:
                x2 = x1 + 6
            rect = QtCore.QRect(x1, y, x2 - x1, row_h - 4)
            painter.setBrush(colors[i % len(colors)])
            painter.setPen(QtCore.Qt.NoPen)
            painter.drawRoundedRect(rect, 3, 3)

            start_x = self._x_for_time(group.get('start'), width)
            painter.setPen(QtGui.QPen(QtGui.QColor(20, 20, 20), 2))
            painter.drawLine(start_x, y, start_x, y + row_h - 4)

            end_x = self._x_for_time(group.get('end'), width)
            painter.setPen(QtGui.QPen(QtGui.QColor(90, 90, 90), 2))
            painter.drawLine(end_x, y, end_x, y + row_h - 4)

            if self._show_anchor:
                t0_x = self._x_for_time(group.get('t0'), width)
                painter.setPen(QtGui.QPen(QtGui.QColor(200, 40, 40), 2))
                painter.drawLine(t0_x, y - 2, t0_x, y + row_h - 2)

            painter.setPen(QtGui.QColor(255, 255, 255))
            label = f"G{group.get('id', i + 1)} n={group.get('n_plants', 0)}"
            painter.drawText(rect.adjusted(4, 0, -4, 0), QtCore.Qt.AlignVCenter, label)

        if self._span_start is not None:
            _draw_axis_end_labels(
                painter, width, height,
                self._span_start.strftime('%Y-%m-%d %H:%M'),
                self._span_end.strftime('%Y-%m-%d %H:%M'),
            )


class _ElapsedTimelineWidget(QtWidgets.QWidget):
    """Read-only Gantt on a shared elapsed-hour axis."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(200)
        self._groups = []
        self._h_min = 0
        self._h_max = 1
        self._show_anchor = False
        self._label_width = 300

    def set_groups(self, groups, show_anchor=False, duration_hours=None):
        self._groups = list(groups or [])
        self._show_anchor = bool(show_anchor)
        starts = [
            int(group['e_start'])
            for group in self._groups
            if group.get('e_start') is not None
        ]
        ends = [
            int(group['e_end'])
            for group in self._groups
            if group.get('e_end') is not None
        ]
        if starts:
            self._h_min = min(starts)
            try:
                duration = int(duration_hours) if duration_hours not in (None, '') else None
            except (TypeError, ValueError):
                duration = None
            self._h_max = self._h_min + duration if duration else (max(ends) if ends else self._h_min + 1)
            if ends:
                self._h_max = max(self._h_max, max(ends))
            if self._show_anchor:
                self._h_min = min(self._h_min, 0)
                self._h_max = max(self._h_max, 0)
            if self._h_max <= self._h_min:
                self._h_max = self._h_min + 1
        else:
            self._h_min = 0
            self._h_max = 1
        self.update()

    def _x_for_hour(self, hour, width):
        left = self._label_width
        if hour is None:
            return left
        total = float(self._h_max - self._h_min)
        if total <= 0:
            return left
        frac = (float(hour) - self._h_min) / total
        frac = min(max(frac, 0.0), 1.0)
        return int(left + frac * max(width - left - 16, 1))

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.fillRect(self.rect(), QtGui.QColor(248, 248, 248))
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        width = self.width()
        height = self.height()
        if not self._groups:
            painter.setPen(QtGui.QColor(120, 120, 120))
            painter.drawText(self.rect(), QtCore.Qt.AlignCenter, "No groups to preview")
            return

        n = len(self._groups)
        row_h = max(36, min(48, (height - AXIS_LABEL_H - 8) // max(n, 1)))
        colors = [
            QtGui.QColor(70, 130, 180),
            QtGui.QColor(60, 160, 110),
            QtGui.QColor(180, 110, 60),
            QtGui.QColor(130, 90, 170),
        ]
        bar_top_pad = 4
        bar_h = row_h - 10
        for i, group in enumerate(self._groups):
            y = 8 + i * row_h
            label_rect = QtCore.QRect(8, y, self._label_width - 16, row_h - 6)
            start_text = group.get('start_label') or ''
            end_text = group.get('end_label') or ''
            group_line = f"G{group.get('id', i + 1)}  n={group.get('n_plants', 0)}"
            time_line = (
                f"{start_text} → {end_text}" if start_text and end_text
                else (start_text or end_text)
            )
            painter.setPen(QtGui.QColor(40, 40, 40))
            painter.drawText(
                label_rect.adjusted(0, 0, 0, -label_rect.height() // 2),
                QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter,
                group_line,
            )
            painter.setPen(QtGui.QColor(70, 70, 70))
            painter.drawText(
                label_rect.adjusted(0, label_rect.height() // 2, 0, 0),
                QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter,
                time_line,
            )

            bar_y = y + bar_top_pad
            x1 = self._x_for_hour(group.get('e_span_start'), width)
            x2 = self._x_for_hour(group.get('e_span_end'), width)
            if (
                group.get('e_span_start') is not None
                and group.get('e_span_end') is not None
            ):
                if x2 - x1 < 6:
                    x2 = x1 + 6
                rect = QtCore.QRect(x1, bar_y, x2 - x1, bar_h)
                painter.setBrush(colors[i % len(colors)])
                painter.setPen(QtCore.Qt.NoPen)
                painter.drawRoundedRect(rect, 3, 3)

            start_x = self._x_for_hour(group.get('e_start'), width)
            painter.setPen(QtGui.QPen(QtGui.QColor(20, 20, 20), 2))
            painter.drawLine(start_x, bar_y, start_x, bar_y + bar_h)

            end_x = self._x_for_hour(group.get('e_end'), width)
            painter.setPen(QtGui.QPen(QtGui.QColor(90, 90, 90), 2))
            painter.drawLine(end_x, bar_y, end_x, bar_y + bar_h)

            if self._show_anchor:
                t0_x = self._x_for_hour(0, width)
                painter.setPen(QtGui.QPen(QtGui.QColor(200, 40, 40), 2))
                painter.drawLine(t0_x, bar_y - 2, t0_x, bar_y + bar_h + 2)

        if self._h_min < 0 < self._h_max:
            zero_x = self._x_for_hour(0, width)
            painter.setPen(QtGui.QColor(80, 80, 80))
            painter.drawText(
                QtCore.QRect(zero_x - 10, height - AXIS_LABEL_H, 20, AXIS_LABEL_H),
                QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter,
                "0",
            )
        _draw_axis_end_labels(
            painter, width, height,
            f"{self._h_min} h",
            f"{self._h_max} h",
            left=self._label_width,
        )


class AlignmentPreviewDialog(QtWidgets.QDialog):
    def __init__(self, groups, show_anchor, duration_hours=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Preview alignment")
        self.setModal(True)
        self.resize(860, 380)
        caption = QtWidgets.QLabel(PREVIEW_CAPTION)
        caption.setWordWrap(True)
        legend = QtWidgets.QLabel(
            "Black = window start.  Gray = window end."
            + ("  Red = anchor (elapsed 0)." if show_anchor else "")
        )
        legend.setStyleSheet("color: #444;")
        timeline = _ElapsedTimelineWidget()
        timeline.set_groups(groups, show_anchor=show_anchor, duration_hours=duration_hours)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(caption)
        layout.addWidget(legend)
        layout.addWidget(timeline, 1)
        layout.addWidget(buttons)


class TimeGroupDetectWorker(QtCore.QObject):
    """Filesystem walk for analysis-period groups; results applied on the UI thread."""

    finished = QtCore.pyqtSignal(object)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, main_folder):
        super().__init__()
        self.main_folder = main_folder or ''

    @QtCore.pyqtSlot()
    def run(self):
        try:
            data = collect_acquisition_spans(self.main_folder)
            if data is None or data.empty:
                self.finished.emit({
                    'groups': [],
                    'duration': None,
                    'sources': [],
                    'empty': True,
                })
                return
            groups = detect_time_groups(data)
            duration = default_duration_hours(groups, data)
            sources = time_period_sources(self.main_folder, spans=data)
            self.finished.emit({
                'groups': groups_to_config(groups),
                'duration': int(duration) if duration is not None else None,
                'sources': list(sources or []),
                'empty': False,
            })
        except Exception as exc:
            self.failed.emit(str(exc))


class TimeWindowDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Set analysis period")
        self.setModal(True)
        self.resize(920, 660)
        self._group_rows = []
        self._updating = False
        self._ticks_touched = False
        self.main_folder = ''
        self._saved_groups = []
        self._saved_sources = []
        self._detected_sources = []
        self._duration_from_conf = False
        self._detect_generation = 0
        self._detect_thread = None
        self._detect_worker = None

        self.clock_radio = QtWidgets.QRadioButton("Synchronize by real time")
        self.anchor_radio = QtWidgets.QRadioButton("Synchronize by anchor time")
        self.clock_radio.setChecked(True)
        self.preview_btn = QtWidgets.QPushButton("Preview alignment")
        self.preview_btn.clicked.connect(self._open_alignment_preview)

        self.duration_edit = QtWidgets.QSpinBox()
        self.duration_edit.setRange(1, 24 * 60)
        self.duration_edit.setValue(40)
        self.duration_edit.setSuffix(" h")
        self.show_ticks_check = QtWidgets.QCheckBox("Mark clock times on figures (dashed lines)")
        self.show_ticks_check.setChecked(True)
        self.ticks_edit = QtWidgets.QLineEdit("00:00")
        self.ticks_edit.setPlaceholderText("HH:MM")
        self.ticks_edit.setMaximumWidth(120)

        top = QtWidgets.QGridLayout()
        top.addWidget(self.clock_radio, 0, 0)
        top.addWidget(self.anchor_radio, 0, 1)
        top.setColumnStretch(2, 1)
        top.addWidget(self.preview_btn, 0, 4)
        duration_row = QtWidgets.QHBoxLayout()
        duration_row.setContentsMargins(0, 0, 0, 0)
        duration_row.addWidget(QtWidgets.QLabel("Duration"))
        duration_row.addWidget(self.duration_edit)
        duration_row.addStretch(1)
        top.addLayout(duration_row, 1, 0)
        top.addWidget(self.show_ticks_check, 1, 1)
        top.addWidget(self.ticks_edit, 1, 3)
        self.help_label = QtWidgets.QLabel(HELP_TEXT)
        self.help_label.setWordWrap(True)
        top.addWidget(self.help_label, 2, 0, 1, 5)

        self.legend_label = QtWidgets.QLabel("")
        self.legend_label.setStyleSheet("color: #444;")
        self.timeline = _TimelineWidget()

        self._table_host = QtWidgets.QWidget()
        self._table_grid = QtWidgets.QGridLayout(self._table_host)
        self._table_grid.setContentsMargins(TABLE_LEFT_PAD, 6, 8, 6)
        self._table_grid.setHorizontalSpacing(10)
        self._table_grid.setVerticalSpacing(8)
        self._table_grid.setColumnStretch(0, 2)
        self._table_grid.setColumnStretch(1, 1)
        self._table_grid.setColumnStretch(2, 1)
        self._table_grid.setColumnStretch(3, 1)
        self._header_group = QtWidgets.QLabel("<b>Group</b>")
        self._header_start = QtWidgets.QLabel("<b>Start</b>")
        self._header_end = QtWidgets.QLabel("<b>End</b>")
        self._header_anchor = QtWidgets.QLabel("<b>Anchor (elapsed = 0)</b>")
        self._header_anchor.setToolTip(ANCHOR_TOOLTIP)
        self.scroll = QtWidgets.QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setWidget(self._table_host)

        self._buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        self._buttons.accepted.connect(self.accept)
        self._buttons.rejected.connect(self.reject)
        self._ok_button = self._buttons.button(QtWidgets.QDialogButtonBox.Ok)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(top)
        layout.addWidget(self.legend_label)
        layout.addWidget(self.timeline)
        layout.addWidget(self.scroll, 1)
        layout.addWidget(self._buttons)

        self._add_table_header()
        self.duration_edit.valueChanged.connect(self._refresh_end_labels)
        self.clock_radio.toggled.connect(self._on_mode_changed)
        self.anchor_radio.toggled.connect(self._on_mode_changed)
        self.show_ticks_check.toggled.connect(self._on_ticks_toggled)
        self._apply_mode_tick_default()
        self._sync_ticks_enabled()
        self._update_legend()
        self._sync_anchor_fields()

    def set_main_folder(self, folder):
        self.main_folder = folder or ''

    def _anchor_mode(self):
        return self.anchor_radio.isChecked()

    def _add_table_header(self):
        self._table_grid.addWidget(self._header_group, 0, 0)
        self._table_grid.addWidget(self._header_start, 0, 1)
        self._table_grid.addWidget(self._header_end, 0, 2)
        self._table_grid.addWidget(self._header_anchor, 0, 3)

    def _apply_mode_tick_default(self):
        self.show_ticks_check.blockSignals(True)
        self.show_ticks_check.setChecked(not self._anchor_mode())
        self.show_ticks_check.blockSignals(False)
        self._sync_ticks_enabled()

    def _on_ticks_toggled(self, _checked):
        if not self._updating:
            self._ticks_touched = True
        self._sync_ticks_enabled()

    def _sync_ticks_enabled(self):
        self.ticks_edit.setEnabled(self.show_ticks_check.isChecked())

    def _update_legend(self):
        if self._anchor_mode():
            self.legend_label.setText(
                "Black line = window start.  Gray line = window end.  "
                "Red line = anchor (elapsed 0)."
            )
        else:
            self.legend_label.setText(
                "Black line = window start.  Gray line = window end."
            )

    def _on_mode_changed(self, checked=True):
        if not checked:
            return
        if self._updating:
            return
        self._apply_shared_clock_t0()
        if not self._ticks_touched:
            self._apply_mode_tick_default()
        self._sync_anchor_fields()
        self._update_legend()
        self._update_timeline()

    def _sync_anchor_fields(self):
        show = self._anchor_mode()
        self._header_anchor.setVisible(show)
        self._table_grid.setColumnStretch(3, 1 if show else 0)
        self._table_grid.setColumnMinimumWidth(3, EDITOR_MIN_WIDTH if show else 0)
        for row in self._group_rows:
            row['t0_edit'].setVisible(show)

    def load_from_conf(self, conf):
        conf = conf or {}
        self._updating = True
        self._ticks_touched = False
        mode = conf.get('timeSyncMode') or 'clock'
        self.clock_radio.setChecked(mode != 'anchor')
        self.anchor_radio.setChecked(mode == 'anchor')
        duration = conf.get('timeDurationHours')
        self._duration_from_conf = duration not in (None, '')
        if self._duration_from_conf:
            try:
                self.duration_edit.setValue(max(1, int(round(float(duration)))))
            except (TypeError, ValueError):
                pass
        ticks = conf.get('figureClockTicks', '00:00')
        if isinstance(ticks, (list, tuple)):
            ticks = ', '.join(str(t) for t in ticks)
        self.ticks_edit.setText(str(ticks or '00:00'))
        if 'showFigureClockTicks' in conf and conf.get('showFigureClockTicks') is not None:
            self.show_ticks_check.setChecked(bool(conf.get('showFigureClockTicks')))
        else:
            self.show_ticks_check.setChecked(mode != 'anchor')
        groups = list(conf.get('timeGroups') or [])
        self._saved_groups = groups
        self._saved_sources = [str(s) for s in list(conf.get('timePeriodSources') or [])]
        self._detected_sources = []
        self._updating = False
        self._sync_ticks_enabled()
        self._update_legend()
        self._sync_anchor_fields()
        # Do not paint the saved groups yet. They may belong to the project
        # that was open before this folder was selected. Detection fills the
        # table, and the saved start/anchor is reused only when it still
        # matches the videos in main_folder.
        if self.main_folder:
            self._start_detect()
        elif groups:
            self._set_groups(groups)
        else:
            self._set_groups([])
            self.timeline.set_empty_text("Select a project to detect time groups")

    def _set_detecting(self, detecting):
        if self._ok_button is not None:
            self._ok_button.setEnabled(not detecting)
        self.preview_btn.setEnabled(not detecting)
        if detecting:
            self.timeline.set_empty_text("Detecting groups…")

    def _stop_detect_thread(self):
        self._detect_generation += 1
        thread = self._detect_thread
        self._detect_thread = None
        self._detect_worker = None
        if thread is not None:
            try:
                thread.quit()
            except RuntimeError:
                pass

    def _start_detect(self):
        if not self.main_folder:
            return
        self._stop_detect_thread()
        generation = self._detect_generation
        # Drop rows already on screen before the scan finishes. A previous
        # project's groups used to stay visible for the whole scan, which is
        # what the dialog showed after a folder change.
        self._detected_sources = []
        self.timeline.set_empty_text("Detecting groups…")
        self._set_groups([])
        self._set_detecting(True)
        worker = TimeGroupDetectWorker(self.main_folder)
        thread = QtCore.QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(
            lambda payload, gen=generation: self._on_detect_finished(gen, payload)
        )
        worker.failed.connect(
            lambda message, gen=generation: self._on_detect_failed(gen, message)
        )
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        self._detect_worker = worker
        self._detect_thread = thread
        thread.start()

    def _on_detect_failed(self, generation, message):
        if generation != self._detect_generation:
            return
        self._set_detecting(False)
        # Leave the table empty and block OK. Accepting here would save an
        # empty period over the one stored for this project.
        if self._ok_button is not None:
            self._ok_button.setEnabled(False)
        self.timeline.set_empty_text("Could not detect time groups")
        QtWidgets.QMessageBox.warning(
            self, "Analysis period", f"Failed to detect time groups:\n{message}",
        )

    def _on_detect_finished(self, generation, payload):
        if generation != self._detect_generation:
            return
        self._set_detecting(False)
        payload = payload or {}
        detected = list(payload.get('groups') or [])
        sources = [str(s) for s in list(payload.get('sources') or [])]
        self._detected_sources = sources
        if payload.get('empty') or not detected:
            self.timeline.set_empty_text(
                "No processed videos or image timestamps were found"
            )
            self._set_groups([])
            QtWidgets.QMessageBox.information(
                self,
                "No acquisitions found",
                "No processed videos or image timestamps were found. "
                "Process a video first (or finish plant analysis so Results_raw files exist).",
            )
            return

        # The saved period is reused only when it was built for this same set
        # of videos and its clock spans still overlap what we just found.
        # Otherwise the table shows the new project's own acquisitions, and
        # the duration follows those acquisitions instead of the previous project.
        same_acquisitions = bool(
            self._saved_groups
            and self._saved_sources
            and sources
            and self._saved_sources == sources
        )
        if same_acquisitions:
            groups, kept_saved_windows = _overlay_saved_groups(self._saved_groups, detected)
        else:
            groups = detected
            kept_saved_windows = 0
        if not kept_saved_windows or not self._duration_from_conf:
            duration = payload.get('duration')
            if duration not in (None, ''):
                self.duration_edit.setValue(max(1, int(duration)))
        self._set_groups(groups)

    def reject(self):
        self._stop_detect_thread()
        super().reject()

    def accept(self):
        self._stop_detect_thread()
        super().accept()

    def closeEvent(self, event):
        self._stop_detect_thread()
        super().closeEvent(event)

    def _clear_rows(self):
        self._group_rows = []
        while self._table_grid.count():
            item = self._table_grid.takeAt(0)
            widget = item.widget()
            if widget is not None and widget not in (
                self._header_group, self._header_start, self._header_end, self._header_anchor,
            ):
                widget.deleteLater()
        self._add_table_header()

    def _set_groups(self, groups):
        self._updating = True
        self._clear_rows()
        for group in groups:
            self._add_group_row(group)
        self._updating = False
        self._sync_anchor_fields()
        self._refresh_end_labels()
        self._update_timeline()

    def _add_group_row(self, group):
        row_index = len(self._group_rows) + 1
        title = QtWidgets.QLabel(
            f"Group {group.get('id', '')}  ·  {group.get('n_plants', 0)} plants\n"
            f"{_format_display_datetime(group.get('spanStart', ''))} → "
            f"{_format_display_datetime(group.get('spanEnd', ''))}"
        )
        title.setWordWrap(True)
        title.setContentsMargins(4, 0, 8, 0)
        start_edit = _DateTimeEditor(group.get('start') or group.get('spanStart'))
        end_edit = _DateTimeEditor(group.get('end') or group.get('spanEnd'), read_only=True)
        t0_edit = _DateTimeEditor(group.get('t0') or group.get('start'))
        t0_edit.setToolTip(ANCHOR_TOOLTIP)
        start_edit.changed.connect(self._refresh_end_labels)
        t0_edit.changed.connect(self._refresh_end_labels)
        self._table_grid.addWidget(title, row_index, 0)
        self._table_grid.addWidget(start_edit, row_index, 1)
        self._table_grid.addWidget(end_edit, row_index, 2)
        self._table_grid.addWidget(t0_edit, row_index, 3)
        self._group_rows.append({
            'id': group.get('id'),
            'n_plants': group.get('n_plants', 0),
            'spanStart': group.get('spanStart') or '',
            'spanEnd': group.get('spanEnd') or '',
            'start_edit': start_edit,
            'end_edit': end_edit,
            't0_edit': t0_edit,
            'title': title,
        })

    def _groups_from_rows(self):
        groups = []
        clock_mode = not self._anchor_mode()
        origin = None
        if clock_mode:
            origin = shared_clock_origin([
                {'start': row['start_edit'].to_string()}
                for row in self._group_rows
            ])
        duration = timedelta(hours=self.duration_edit.value())
        for i, row in enumerate(self._group_rows, start=1):
            start = row['start_edit'].to_string()
            if clock_mode and origin is not None:
                t0 = format_datetime(origin)
            elif clock_mode:
                t0 = start
            else:
                t0 = row['t0_edit'].to_string()
            start_ts = parse_datetime(start)
            end = format_datetime(start_ts + duration) if start_ts is not None else ''
            groups.append({
                'id': int(row.get('id') or i),
                'n_plants': int(row.get('n_plants') or 0),
                'spanStart': row.get('spanStart') or start,
                'spanEnd': row.get('spanEnd') or end,
                'start': start,
                't0': t0,
                'end': end,
            })
        return groups

    def _elapsed_preview_groups(self):
        rows = []
        for group in self._groups_from_rows():
            t0 = group.get('t0')
            e_start = _elapsed_int(group.get('start'), t0)
            e_end = _elapsed_int(group.get('end'), t0)
            e_span_start = _elapsed_int(group.get('spanStart'), t0)
            e_span_end = _elapsed_int(group.get('spanEnd'), t0)
            if e_start is not None and e_span_start is not None:
                e_span_start = max(e_span_start, e_start)
            if e_end is not None and e_span_end is not None:
                e_span_end = min(e_span_end, e_end)
            if (
                e_span_start is not None
                and e_span_end is not None
                and e_span_end < e_span_start
            ):
                e_span_start = None
                e_span_end = None
            rows.append({
                'id': group.get('id'),
                'n_plants': group.get('n_plants'),
                'e_start': e_start,
                'e_end': e_end,
                'e_span_start': e_span_start,
                'e_span_end': e_span_end,
                'start_label': _format_display_datetime(group.get('start')),
                'end_label': _format_display_datetime(group.get('end')),
            })
        return rows

    def _open_alignment_preview(self):
        if not self._group_rows:
            QtWidgets.QMessageBox.information(
                self,
                "Preview alignment",
                "No time groups to preview.",
            )
            return
        dialog = AlignmentPreviewDialog(
            self._elapsed_preview_groups(),
            show_anchor=self._anchor_mode(),
            duration_hours=self.duration_edit.value(),
            parent=self,
        )
        dialog.exec_()

    def _apply_shared_clock_t0(self):
        if self._anchor_mode() or not self._group_rows:
            return
        origin = shared_clock_origin([
            {'start': row['start_edit'].to_string()}
            for row in self._group_rows
        ])
        if origin is None:
            return
        for row in self._group_rows:
            row['t0_edit'].set_datetime(origin)

    def _refresh_end_labels(self):
        if self._updating:
            return
        self._apply_shared_clock_t0()
        duration = timedelta(hours=self.duration_edit.value())
        for row in self._group_rows:
            start_ts = parse_datetime(row['start_edit'].to_string())
            if start_ts is None:
                continue
            row['end_edit'].set_datetime(start_ts + duration)
        self._update_timeline()

    def _update_timeline(self):
        self.timeline.set_groups(self._groups_from_rows(), show_anchor=self._anchor_mode())

    def to_conf(self):
        show_ticks = self.show_ticks_check.isChecked()
        ticks = [
            label for _h, _m, label in parse_clock_ticks({
                'showFigureClockTicks': True,
                'figureClockTicks': self.ticks_edit.text(),
            })
        ]
        if show_ticks and not ticks:
            ticks = ['00:00']
        return {
            'timeSyncMode': 'anchor' if self._anchor_mode() else 'clock',
            'timeDurationHours': int(self.duration_edit.value()),
            'showFigureClockTicks': show_ticks,
            'figureClockTicks': ticks,
            'timeGroups': groups_to_config(self._groups_from_rows()),
            # Fingerprint from the scan that filled this dialog. Re-walking the
            # project here would stamp a different set than the one on screen
            # if files changed mid-edit, and it would also freeze the UI.
            'timePeriodSources': list(self._detected_sources) if self.main_folder else list(self._saved_sources),
        }
