"""Choose which plants go into one named report folder."""

from PyQt5 import QtCore, QtWidgets

from analysis.time_windows import report_folder_name
from analysis.utils.fileUtilities import normalize_factor_value
from analysis.utils.report_subset import subset_description
from analysis.utils.report_utils import natural_key as natural_keys


def _unused_factor(values):
    """True when this factor was never set on the plants being reported."""
    cleaned = {normalize_factor_value(value) for value in values}
    return not cleaned or cleaned <= {"unspecified"}


class _FactorGroup(QtWidgets.QGroupBox):
    """Checkboxes for one factor. An unused factor includes every plant."""

    def __init__(self, title):
        super().__init__(title)
        self.unused = False
        self._checks = {}
        self._layout = QtWidgets.QVBoxLayout(self)
        self._note = QtWidgets.QLabel(
            "Not set on these plants. Every plant is included."
        )
        self._note.setWordWrap(True)
        self._note.hide()
        self._layout.addWidget(self._note)
        buttons = QtWidgets.QHBoxLayout()
        self._all = QtWidgets.QPushButton("All")
        self._none = QtWidgets.QPushButton("None")
        self._all.clicked.connect(lambda: self._set_all(True))
        self._none.clicked.connect(lambda: self._set_all(False))
        buttons.addWidget(self._all)
        buttons.addWidget(self._none)
        buttons.addStretch(1)
        self._layout.addLayout(buttons)
        self._checks_host = QtWidgets.QWidget()
        self._checks_layout = QtWidgets.QVBoxLayout(self._checks_host)
        self._checks_layout.setContentsMargins(0, 0, 0, 0)
        self._layout.addWidget(self._checks_host)

    def set_values(self, values):
        while self._checks_layout.count():
            item = self._checks_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._checks = {}
        ordered = sorted({str(value) for value in values if str(value).strip()}, key=natural_keys)
        self.unused = _unused_factor(ordered)
        self._note.setVisible(self.unused)
        self._all.setVisible(not self.unused)
        self._none.setVisible(not self.unused)
        self._checks_host.setVisible(not self.unused)
        if self.unused:
            return
        for value in ordered:
            box = QtWidgets.QCheckBox(value)
            box.setChecked(True)
            self._checks[value] = box
            self._checks_layout.addWidget(box)

    def apply_saved(self, selected):
        """None keeps every box checked. A list checks only those values."""
        if self.unused:
            return
        if selected is None:
            self._set_all(True)
            return
        wanted = {str(value) for value in selected}
        for value, box in self._checks.items():
            box.setChecked(value in wanted)

    def selected_values(self):
        """None when every plant is included for this factor."""
        if self.unused or not self._checks:
            return None
        chosen = [value for value, box in self._checks.items() if box.isChecked()]
        if len(chosen) == len(self._checks):
            return None
        return chosen

    def has_selection(self):
        if self.unused:
            return True
        return any(box.isChecked() for box in self._checks.values())

    def _set_all(self, checked):
        for box in self._checks.values():
            box.setChecked(checked)


class ReportSubsetDialog(QtWidgets.QDialog):
    """Pick a folder name and the factor values included in that report."""

    def __init__(self, choices, reports, labels, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Review and Report")
        self.resize(520, 640)
        self._action = None
        self._loading = False
        self._reports = {item["name"]: item for item in reports}

        self.name_box = QtWidgets.QComboBox()
        self.name_box.setEditable(True)
        self.name_box.addItem("Report")
        for name in sorted(self._reports, key=natural_keys):
            if name != "Report":
                self.name_box.addItem(name)
        self.name_box.setCurrentText("Report")
        self.name_box.activated[str].connect(self._load_named_subset)

        self.saved_list = QtWidgets.QListWidget()
        if self._reports:
            for name in sorted(self._reports, key=natural_keys):
                self._add_saved_item(name)
        else:
            placeholder = QtWidgets.QListWidgetItem("No reports yet")
            placeholder.setFlags(QtCore.Qt.NoItemFlags)
            self.saved_list.addItem(placeholder)
        self.saved_list.currentItemChanged.connect(self._load_current_item)

        self.experiments = _FactorGroup(labels.get("experiments") or "Genotype")
        self.plates = _FactorGroup(labels.get("plates") or "Plate condition")
        self.extras = _FactorGroup(labels.get("extras") or "Extra variable")
        self.experiments.set_values(choices.get("experiments") or [])
        self.plates.set_values(choices.get("plates") or [])
        self.extras.set_values(choices.get("extras") or [])
        for group in (self.experiments, self.plates, self.extras):
            for box in group._checks.values():
                box.toggled.connect(self._refresh_description)

        intro = QtWidgets.QLabel(
            "Choose who is compared in this report. "
            "A genotype includes every plant of that genotype. "
            "Each row is a report folder on disk. "
            "One that stopped early is marked so it can be generated again."
        )
        intro.setWordWrap(True)
        self.description = QtWidgets.QLabel("")
        self.description.setWordWrap(True)

        generate_button = QtWidgets.QPushButton("Generate")
        cancel_button = QtWidgets.QPushButton("Cancel")
        generate_button.clicked.connect(self._generate)
        cancel_button.clicked.connect(self.reject)
        buttons = QtWidgets.QHBoxLayout()
        buttons.addStretch(1)
        buttons.addWidget(generate_button)
        buttons.addWidget(cancel_button)

        factors = QtWidgets.QWidget()
        factor_layout = QtWidgets.QVBoxLayout(factors)
        factor_layout.setContentsMargins(0, 0, 0, 0)
        factor_layout.addWidget(self.experiments)
        factor_layout.addWidget(self.plates)
        factor_layout.addWidget(self.extras)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(factors)

        name_row = QtWidgets.QHBoxLayout()
        name_row.addWidget(QtWidgets.QLabel("Report name"))
        name_row.addWidget(self.name_box, 1)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(intro)
        layout.addLayout(name_row)
        layout.addWidget(self.description)
        layout.addWidget(QtWidgets.QLabel("Reports on disk"))
        layout.addWidget(self.saved_list, 1)
        layout.addWidget(scroll, 3)
        layout.addLayout(buttons)

    def action(self):
        return self._action

    def subset(self):
        return {
            "name": report_folder_name({"reportFolderName": self.name_box.currentText()}),
            "experiments": self.experiments.selected_values(),
            "plates": self.plates.selected_values(),
            "extras": self.extras.selected_values(),
        }

    def _add_saved_item(self, name):
        report = self._reports.get(name) or {}
        text = report.get("description") or "Compares all genotypes."
        item = QtWidgets.QListWidgetItem(f"{name} — {text}")
        item.setData(QtCore.Qt.UserRole, name)
        self.saved_list.addItem(item)

    def _load_current_item(self, current, _previous):
        if current is None:
            return
        self._load_named_subset(current.data(QtCore.Qt.UserRole))

    def _refresh_description(self, *_args):
        if getattr(self, "_loading", False) or not hasattr(self, "description"):
            return
        self.description.setText(subset_description(self.subset()))

    def _load_named_subset(self, name):
        if self._loading:
            return
        if not name or name not in self._reports:
            return
        self._loading = True
        try:
            self.name_box.setCurrentText(name)
            for row in range(self.saved_list.count()):
                item = self.saved_list.item(row)
                if item.data(QtCore.Qt.UserRole) == name:
                    self.saved_list.setCurrentItem(item)
                    break
            saved = self._reports.get(name) or {}
            self.experiments.apply_saved(saved.get("experiments"))
            self.plates.apply_saved(saved.get("plates"))
            self.extras.apply_saved(saved.get("extras"))
            if hasattr(self, "description"):
                self.description.setText(saved.get("description") or subset_description(self.subset()))
        finally:
            self._loading = False

    def _validate(self):
        missing = []
        if not self.experiments.has_selection():
            missing.append(self.experiments.title())
        if not self.plates.has_selection():
            missing.append(self.plates.title())
        if not self.extras.has_selection():
            missing.append(self.extras.title())
        if not missing:
            return True
        QtWidgets.QMessageBox.warning(
            self,
            "Review and Report",
            "Choose at least one value in: " + ", ".join(missing) + ".",
        )
        return False

    def _generate(self):
        if not self._validate():
            return
        self._action = "generate"
        self.accept()
