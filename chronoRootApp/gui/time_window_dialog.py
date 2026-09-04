"""Dialog to set the analysis period, anchors, clock ticks, and snapshot hours."""

from datetime import timedelta

from PyQt5 import QtCore, QtGui, QtWidgets

from analysis.time_windows import (
    collect_acquisition_spans,
    default_duration_hours,
    detect_time_groups,
    format_datetime,
    groups_to_config,
    parse_clock_ticks,
    parse_datetime,
    parse_hour_list,
    snapshot_hours,
    time_period_sources,
)


HELP_TEXT = (
    "Acquisitions whose clock times overlap are one group (for example one launch). "
    "Duration is shared by every group and defaults to the shortest group. "
    "Real-time sync uses the same start for the group: later videos are padded, "
    "and hours before start are dropped. "
    "Anchor sync sets elapsed 0 at a treatment time (hours before that stay negative)."
)

ANCHOR_TOOLTIP = (
    "Treatment / application time. Elapsed time is 0 at this instant; "
    "hours before it are negative."
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


class _DateTimeEditor(QtWidgets.QWidget):
    changed = QtCore.pyqtSignal()

    def __init__(self, value=None, parent=None):
        super().__init__(parent)
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
                "Detect groups to show the timeline",
            )
            return

        n = len(self._groups)
        row_h = max(16, min(28, (height - 24) // max(n, 1)))
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

        painter.setPen(QtGui.QColor(80, 80, 80))
        if self._span_start is not None:
            painter.drawText(8, height - 4, self._span_start.strftime('%Y-%m-%d %H:%M'))
            painter.drawText(
                width - 140, height - 4, 132, 12,
                QtCore.Qt.AlignRight, self._span_end.strftime('%Y-%m-%d %H:%M'),
            )


class TimeWindowDialog(QtWidgets.QDialog):
    def __init__(self, parent=None, show_snapshots=True):
        super().__init__(parent)
        self.setWindowTitle("Set analysis period")
        self.setModal(True)
        self.resize(860, 660)
        self.show_snapshots = show_snapshots
        self._group_rows = []
        self._updating = False
        self._ticks_touched = False
        self.main_folder = ''

        self.help_label = QtWidgets.QLabel(HELP_TEXT)
        self.help_label.setWordWrap(True)

        self.clock_radio = QtWidgets.QRadioButton("Synchronize by real time")
        self.anchor_radio = QtWidgets.QRadioButton("Synchronize by anchor time")
        self.clock_radio.setChecked(True)
        self.detect_btn = QtWidgets.QPushButton("Detect groups")
        self.detect_btn.clicked.connect(self.detect_groups)
        mode_row = QtWidgets.QHBoxLayout()
        mode_row.addWidget(self.clock_radio)
        mode_row.addWidget(self.anchor_radio)
        mode_row.addStretch()
        mode_row.addWidget(self.detect_btn)

        self.duration_edit = QtWidgets.QSpinBox()
        self.duration_edit.setRange(1, 24 * 60)
        self.duration_edit.setValue(40)
        self.duration_edit.setSuffix(" h")
        self.show_ticks_check = QtWidgets.QCheckBox("Show clock times on figures")
        self.show_ticks_check.setChecked(True)
        self.ticks_edit = QtWidgets.QLineEdit("00:00")
        self.snapshot_edit = QtWidgets.QLineEdit("0, 24")

        ticks_row = QtWidgets.QHBoxLayout()
        ticks_row.addWidget(self.show_ticks_check)
        ticks_row.addWidget(self.ticks_edit)
        ticks_host = QtWidgets.QWidget()
        ticks_host.setLayout(ticks_row)

        form = QtWidgets.QFormLayout()
        form.addRow("Duration", self.duration_edit)
        form.addRow("Clock times (HH:MM)", ticks_host)
        self.snapshot_label = QtWidgets.QLabel("Snapshot hours (convex / angles)")
        form.addRow(self.snapshot_label, self.snapshot_edit)
        if not show_snapshots:
            self.snapshot_label.hide()
            self.snapshot_edit.hide()

        self.legend_label = QtWidgets.QLabel("")
        self.legend_label.setStyleSheet("color: #444;")

        self.timeline = _TimelineWidget()

        self._rows_host = QtWidgets.QWidget()
        self._rows_layout = QtWidgets.QVBoxLayout(self._rows_host)
        self._rows_layout.setContentsMargins(0, 0, 0, 0)
        self._rows_layout.addStretch()
        self.scroll = QtWidgets.QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setWidget(self._rows_host)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.help_label)
        layout.addLayout(mode_row)
        layout.addLayout(form)
        layout.addWidget(self.legend_label)
        layout.addWidget(self.timeline)
        layout.addWidget(self.scroll, 1)
        layout.addWidget(buttons)

        self.duration_edit.valueChanged.connect(self._refresh_end_labels)
        self.clock_radio.toggled.connect(self._on_mode_changed)
        self.show_ticks_check.toggled.connect(self._on_ticks_toggled)
        self._apply_mode_tick_default()
        self._sync_ticks_enabled()
        self._update_legend()

    def set_main_folder(self, folder):
        self.main_folder = folder or ''

    def _anchor_mode(self):
        return self.anchor_radio.isChecked()

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

    def _on_mode_changed(self):
        if self.clock_radio.isChecked():
            for row in self._group_rows:
                row['t0_edit'].set_datetime(row['start_edit'].to_string())
        if not self._ticks_touched:
            self._apply_mode_tick_default()
        self._sync_anchor_fields()
        self._update_legend()
        self._update_timeline()

    def _sync_anchor_fields(self):
        show = self._anchor_mode()
        for row in self._group_rows:
            row['t0_label'].setVisible(show)
            row['t0_edit'].setVisible(show)

    def load_from_conf(self, conf):
        conf = conf or {}
        self._updating = True
        self._ticks_touched = False
        mode = conf.get('timeSyncMode') or 'clock'
        self.clock_radio.setChecked(mode != 'anchor')
        self.anchor_radio.setChecked(mode == 'anchor')
        duration = conf.get('timeDurationHours')
        if duration not in (None, ''):
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
        hours = snapshot_hours(conf)
        if hours:
            self.snapshot_edit.setText(', '.join(str(h) for h in hours))
        groups = conf.get('timeGroups') or []
        self._updating = False
        if groups:
            self._set_groups(groups)
        elif self.main_folder:
            self.detect_groups()
        self._sync_ticks_enabled()
        self._update_legend()

    def detect_groups(self):
        data = collect_acquisition_spans(self.main_folder)
        if data is None or data.empty:
            QtWidgets.QMessageBox.information(
                self,
                "No acquisitions found",
                "No processed videos or image timestamps were found. "
                "Process a video first (or finish plant analysis so Results_raw files exist).",
            )
            return
        mode = 'anchor' if self._anchor_mode() else 'clock'
        groups = detect_time_groups(data, mode=mode)
        if not groups:
            QtWidgets.QMessageBox.information(self, "No groups", "Could not detect time groups.")
            return
        duration = default_duration_hours(groups, data)
        self.duration_edit.setValue(max(1, int(duration)))
        self._set_groups(groups)

    def _clear_rows(self):
        self._group_rows = []
        while self._rows_layout.count() > 0:
            item = self._rows_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._rows_layout.addStretch()

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
        row = QtWidgets.QWidget()
        grid = QtWidgets.QGridLayout(row)
        grid.setContentsMargins(0, 4, 0, 4)
        title = QtWidgets.QLabel(
            f"Group {group.get('id', '')}  ·  {group.get('n_plants', 0)} plants  ·  "
            f"{group.get('spanStart', '')} → {group.get('spanEnd', '')}"
        )
        title.setWordWrap(True)
        start_edit = _DateTimeEditor(group.get('start') or group.get('spanStart'))
        t0_edit = _DateTimeEditor(group.get('t0') or group.get('start'))
        start_edit.changed.connect(self._refresh_end_labels)
        t0_edit.changed.connect(self._refresh_end_labels)
        end_label = QtWidgets.QLabel("")
        t0_label = QtWidgets.QLabel("Anchor (elapsed = 0)")
        t0_label.setToolTip(ANCHOR_TOOLTIP)
        t0_edit.setToolTip(ANCHOR_TOOLTIP)
        grid.addWidget(title, 0, 0, 1, 4)
        grid.addWidget(QtWidgets.QLabel("Start"), 1, 0)
        grid.addWidget(start_edit, 1, 1)
        grid.addWidget(t0_label, 1, 2)
        grid.addWidget(t0_edit, 1, 3)
        grid.addWidget(QtWidgets.QLabel("End"), 2, 0)
        grid.addWidget(end_label, 2, 1, 1, 3)
        self._rows_layout.insertWidget(self._rows_layout.count() - 1, row)
        self._group_rows.append({
            'id': group.get('id'),
            'n_plants': group.get('n_plants', 0),
            'spanStart': group.get('spanStart') or '',
            'spanEnd': group.get('spanEnd') or '',
            'start_edit': start_edit,
            't0_edit': t0_edit,
            't0_label': t0_label,
            'end_label': end_label,
        })

    def _groups_from_rows(self):
        groups = []
        clock_mode = not self._anchor_mode()
        for i, row in enumerate(self._group_rows, start=1):
            start = row['start_edit'].to_string()
            t0 = start if clock_mode else row['t0_edit'].to_string()
            duration = timedelta(hours=self.duration_edit.value())
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

    def _refresh_end_labels(self):
        if self._updating:
            return
        if not self._anchor_mode():
            for row in self._group_rows:
                row['t0_edit'].set_datetime(row['start_edit'].to_string())
        duration = timedelta(hours=self.duration_edit.value())
        for row in self._group_rows:
            start_ts = parse_datetime(row['start_edit'].to_string())
            if start_ts is None:
                row['end_label'].setText('')
                continue
            end = start_ts + duration
            row['end_label'].setText(end.strftime('%Y-%m-%d %H:%M'))
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
        payload = {
            'timeSyncMode': 'anchor' if self._anchor_mode() else 'clock',
            'timeDurationHours': int(self.duration_edit.value()),
            'showFigureClockTicks': show_ticks,
            'figureClockTicks': ticks,
            'timeGroups': groups_to_config(self._groups_from_rows()),
            'timePeriodSources': time_period_sources(self.main_folder),
        }
        if self.show_snapshots:
            payload['snapshotHours'] = parse_hour_list(self.snapshot_edit.text())
        return payload
