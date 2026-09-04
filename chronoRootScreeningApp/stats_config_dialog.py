"""Screening report-parameter dialog: chronoRootApp modes plus FPCA, Fourier, measures."""

from PyQt5 import QtGui, QtWidgets

from chrono_root_backend import CHRONOROOT_APP_DIR  # noqa: F401  # puts chronoRootApp on sys.path
from gui.stats_config_dialog import StatsConfigDialog

MODE_LABELS = {
    "statsByGenotype": "Compare groups (all data)",
    "statsGenotypeByPlate": "Compare groups within each plate condition",
    "statsGenotypeByExtra": "Compare groups within each extra variable",
    "statsByPlateCondition": "Compare plate conditions directly",
    "statsByExtraVariable": "Compare extra variable directly",
    "statsPlateWithinGenotype": "Compare plate conditions within each group",
    "statsExtraWithinGenotype": "Compare extra variable within each group",
}

# (objectName, Temporal_Data / germination column, UI label)
MEASURE_WIDGETS = [
    ("measureHypocotyl", "HypocotylLength (mm)", "Hypocotyl length"),
    ("measureMainRoot", "MainRootLength (mm)", "Main root length"),
    ("measureTotalRoot", "TotalLength (mm)", "Total root length"),
    ("measureArea", "Area (mm2)", "Plant area"),
    ("measureDenseRoot", "DenseRootArea (mm2)", "Dense root area"),
    ("measureGermination", "GerminationTime", "Germination time"),
]


class ScreeningStatsConfigDialog(StatsConfigDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Configure Report Parameters")
        self.resize(540, 760)

        self.everyXhourFieldAngles[0].hide()
        self.everyXhourFieldAngles[1].hide()

        for object_name, label in MODE_LABELS.items():
            checkbox = self._mode_checkboxes.get(object_name)
            if checkbox is not None:
                checkbox.setText(label)

        self.doFourier = QtWidgets.QCheckBox("Perform Fourier analysis of growth speed")
        self.doFourier.setObjectName("doFourier")
        self.doFourier.setChecked(True)

        self.fpca_checkbox = QtWidgets.QCheckBox("Perform FPCA analysis")
        self.fpca_checkbox.setObjectName("fpca_checkbox")
        self.fpca_components_edit = QtWidgets.QLineEdit("2")
        self.fpca_components_edit.setObjectName("fpca_components_edit")
        self.fpca_components_edit.setFixedWidth(40)
        self.fpca_components_edit.setValidator(QtGui.QIntValidator(2, 10))
        self.fpca_normalize_checkbox = QtWidgets.QCheckBox("Normalize FPCA data")
        self.fpca_normalize_checkbox.setObjectName("fpca_normalize_checkbox")
        self.fpca_normalize_checkbox.setChecked(False)

        self.fpca_widget = QtWidgets.QWidget()
        fpca_layout = QtWidgets.QHBoxLayout(self.fpca_widget)
        fpca_layout.setContentsMargins(0, 0, 0, 0)
        fpca_layout.addWidget(self.fpca_checkbox)
        fpca_layout.addWidget(QtWidgets.QLabel("Components:"))
        fpca_layout.addWidget(self.fpca_components_edit)
        fpca_layout.addWidget(self.fpca_normalize_checkbox)
        fpca_layout.addStretch()

        self.measures_group = QtWidgets.QGroupBox("Measures")
        measures_layout = QtWidgets.QVBoxLayout()
        self._measure_checkboxes = {}
        for object_name, _column, label in MEASURE_WIDGETS:
            checkbox = QtWidgets.QCheckBox(label)
            checkbox.setObjectName(object_name)
            checkbox.setChecked(True)
            self._measure_checkboxes[object_name] = checkbox
            measures_layout.addWidget(checkbox)
        self.measures_group.setLayout(measures_layout)

        self.genotypeAxisLabelField = QtWidgets.QLineEdit("Group")
        self.genotypeAxisLabelField.setObjectName("genotypeAxisLabelField")
        self.plateConditionAxisLabelField = QtWidgets.QLineEdit("Plate condition")
        self.plateConditionAxisLabelField.setObjectName("plateConditionAxisLabelField")
        self.extraVariableLabelField = QtWidgets.QLineEdit("Run")
        self.extraVariableLabelField.setObjectName("extraVariableLabelField")

        labels_group = QtWidgets.QGroupBox("Axis labels")
        labels_layout = QtWidgets.QFormLayout()
        labels_layout.addRow("Group label", self.genotypeAxisLabelField)
        labels_layout.addRow("Plate condition label", self.plateConditionAxisLabelField)
        labels_layout.addRow("Extra variable label", self.extraVariableLabelField)
        labels_group.setLayout(labels_layout)

        layout = self.layout()
        help_index = layout.indexOf(self.help_label)
        layout.insertWidget(help_index, self.doFourier)
        layout.insertWidget(help_index + 1, self.fpca_widget)
        layout.insertWidget(help_index + 2, self.measures_group)
        layout.insertWidget(help_index + 3, labels_group)

    def register_on_host(self, host):
        super().register_on_host(host)
        host.doFourier = self.doFourier
        host.fpca_checkbox = self.fpca_checkbox
        host.fpca_components_edit = self.fpca_components_edit
        host.fpca_normalize_checkbox = self.fpca_normalize_checkbox
        host.genotypeAxisLabelField = self.genotypeAxisLabelField
        host.plateConditionAxisLabelField = self.plateConditionAxisLabelField
        host.extraVariableLabelField = self.extraVariableLabelField
        for object_name, checkbox in self._measure_checkboxes.items():
            setattr(host, object_name, checkbox)

    def set_defaults(self):
        super().set_defaults()
        self.doFourier.setChecked(True)
        self.fpca_checkbox.setChecked(False)
        self.fpca_components_edit.setText("2")
        self.fpca_normalize_checkbox.setChecked(False)
        self.genotypeAxisLabelField.setText("Group")
        self.plateConditionAxisLabelField.setText("Plate condition")
        self.extraVariableLabelField.setText("Run")
        for checkbox in self._measure_checkboxes.values():
            checkbox.setChecked(True)

    def snapshot_values(self):
        values = super().snapshot_values()
        values.update({
            'doFourier': self.doFourier.isChecked(),
            'fpca_checkbox': self.fpca_checkbox.isChecked(),
            'fpca_components': self.fpca_components_edit.text(),
            'fpca_normalize': self.fpca_normalize_checkbox.isChecked(),
            'genotypeAxisLabel': self.genotypeAxisLabelField.text(),
            'plateConditionAxisLabel': self.plateConditionAxisLabelField.text(),
            'extraVariableLabel': self.extraVariableLabelField.text(),
            'measures': {
                name: checkbox.isChecked()
                for name, checkbox in self._measure_checkboxes.items()
            },
        })
        return values

    def restore_values(self, values):
        super().restore_values(values)
        if not values:
            return
        if 'doFourier' in values:
            self.doFourier.setChecked(bool(values['doFourier']))
        if 'fpca_checkbox' in values:
            self.fpca_checkbox.setChecked(bool(values['fpca_checkbox']))
        if values.get('fpca_components') is not None:
            self.fpca_components_edit.setText(str(values['fpca_components']))
        if 'fpca_normalize' in values:
            self.fpca_normalize_checkbox.setChecked(bool(values['fpca_normalize']))
        if values.get('genotypeAxisLabel') is not None:
            self.genotypeAxisLabelField.setText(str(values['genotypeAxisLabel']))
        if values.get('plateConditionAxisLabel') is not None:
            self.plateConditionAxisLabelField.setText(str(values['plateConditionAxisLabel']))
        if values.get('extraVariableLabel') is not None:
            self.extraVariableLabelField.setText(str(values['extraVariableLabel']))
        for name, checked in (values.get('measures') or {}).items():
            checkbox = self._measure_checkboxes.get(name)
            if checkbox is not None:
                checkbox.setChecked(bool(checked))

    def set_plant_growth_enabled(self, enabled):
        self.fpca_widget.setEnabled(bool(enabled))

    def selected_metric_columns(self):
        columns = []
        for object_name, column, _label in MEASURE_WIDGETS:
            checkbox = self._measure_checkboxes[object_name]
            if checkbox.isChecked():
                columns.append(column)
        return columns
