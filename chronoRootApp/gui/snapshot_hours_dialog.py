"""Helper dialog to pick snapshot hours as days, an interval, or extra hours."""

from PyQt5 import QtWidgets

from analysis.time_windows import parse_hour_list

DEFAULT_DURATION_HOURS = 7 * 24


def _duration_hours(raw):
    if raw in (None, ''):
        return DEFAULT_DURATION_HOURS
    try:
        value = int(round(float(raw)))
    except (TypeError, ValueError):
        return DEFAULT_DURATION_HOURS
    return max(1, value)


def interval_hours(step, duration):
    """0, step, 2*step, ... while the value is still inside the window."""
    try:
        step = int(step)
    except (TypeError, ValueError):
        return []
    if step <= 0:
        return []
    duration = int(duration)
    return list(range(0, duration + 1, step))


def format_hour_list(hours):
    return ', '.join(str(int(h)) for h in hours)


class SnapshotHoursDialog(QtWidgets.QDialog):
    def __init__(self, hours=None, duration_hours=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Choose snapshot hours")
        self.setModal(True)
        self.resize(460, 480)
        self.duration_hours = _duration_hours(duration_hours)
        last_day = self.duration_hours // 24

        help_label = QtWidgets.QLabel(
            "Hours are elapsed from each group's t0. "
            "Day checkboxes, an interval fill, and extra hours are merged "
            "into one sorted list."
        )
        help_label.setWordWrap(True)

        self._day_checks = []
        day_box = QtWidgets.QGroupBox("Days")
        day_layout = QtWidgets.QVBoxLayout(day_box)
        day_host = QtWidgets.QWidget()
        day_grid = QtWidgets.QGridLayout(day_host)
        day_grid.setContentsMargins(0, 0, 0, 0)
        for day in range(last_day + 1):
            hour = day * 24
            if hour > self.duration_hours:
                continue
            check = QtWidgets.QCheckBox(f"Day {day}  ({hour} h)")
            check.toggled.connect(self._refresh_preview)
            self._day_checks.append((hour, check))
            row, col = divmod(len(self._day_checks) - 1, 2)
            day_grid.addWidget(check, row, col)
        day_scroll = QtWidgets.QScrollArea()
        day_scroll.setWidgetResizable(True)
        day_scroll.setWidget(day_host)
        day_scroll.setMinimumHeight(120)
        day_layout.addWidget(day_scroll)

        self.every_check = QtWidgets.QCheckBox("Every")
        self.every_spin = QtWidgets.QSpinBox()
        self.every_spin.setRange(1, max(1, self.duration_hours))
        self.every_spin.setValue(min(24, self.duration_hours))
        self.every_spin.setSuffix(" h")
        every_end = QtWidgets.QLabel("from start to finish")
        self.every_check.toggled.connect(self._refresh_preview)
        self.every_spin.valueChanged.connect(self._refresh_preview)

        every_row = QtWidgets.QHBoxLayout()
        every_row.addWidget(self.every_check)
        every_row.addWidget(self.every_spin)
        every_row.addWidget(every_end)
        every_row.addStretch(1)

        extra_label = QtWidgets.QLabel("Extra hours")
        self.extra_edit = QtWidgets.QLineEdit()
        self.extra_edit.setPlaceholderText("e.g. 6, 18")
        self.extra_edit.textChanged.connect(self._refresh_preview)
        extra_row = QtWidgets.QHBoxLayout()
        extra_row.addWidget(extra_label)
        extra_row.addWidget(self.extra_edit, 1)

        self.preview_label = QtWidgets.QLabel("")
        self.preview_label.setWordWrap(True)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(help_label)
        layout.addWidget(day_box, 1)
        layout.addLayout(every_row)
        layout.addLayout(extra_row)
        layout.addWidget(self.preview_label)
        layout.addWidget(buttons)

        self._load_hours(parse_hour_list(hours))
        self._refresh_preview()

    def _load_hours(self, hours):
        remaining = set(int(h) for h in hours)
        for hour, check in self._day_checks:
            on = hour in remaining
            check.setChecked(on)
            if on:
                remaining.discard(hour)
        if remaining:
            self.extra_edit.setText(format_hour_list(sorted(remaining)))

    def selected_hours(self):
        hours = set()
        for hour, check in self._day_checks:
            if check.isChecked():
                hours.add(int(hour))
        if self.every_check.isChecked():
            hours.update(interval_hours(self.every_spin.value(), self.duration_hours))
        hours.update(parse_hour_list(self.extra_edit.text()))
        return sorted(hours)

    def selected_text(self):
        return format_hour_list(self.selected_hours())

    def _refresh_preview(self):
        text = self.selected_text()
        if text:
            self.preview_label.setText(f"Snapshot hours: {text}")
        else:
            self.preview_label.setText("Snapshot hours: (none)")
