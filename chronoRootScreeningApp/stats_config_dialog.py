"""Screening statistical analysis dialog: chronoRootApp modes plus measure list."""

from PyQt5 import QtWidgets

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
        self.resize(540, 720)

        self.everyXhourFieldAngles[0].hide()
        self.everyXhourFieldAngles[1].hide()

        for object_name, label in MODE_LABELS.items():
            checkbox = self._mode_checkboxes.get(object_name)
            if checkbox is not None:
                checkbox.setText(label)

        self.doFourier = QtWidgets.QCheckBox("Perform Fourier analysis of growth speed")
        self.doFourier.setObjectName("doFourier")
        self.doFourier.setChecked(True)

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
        layout.insertWidget(help_index + 1, self.measures_group)
        layout.insertWidget(help_index + 2, labels_group)

    def register_on_host(self, host):
        super().register_on_host(host)
        host.doFourier = self.doFourier
        host.genotypeAxisLabelField = self.genotypeAxisLabelField
        host.plateConditionAxisLabelField = self.plateConditionAxisLabelField
        host.extraVariableLabelField = self.extraVariableLabelField
        for object_name, checkbox in self._measure_checkboxes.items():
            setattr(host, object_name, checkbox)

    def set_defaults(self):
        super().set_defaults()
        self.doFourier.setChecked(True)
        self.genotypeAxisLabelField.setText("Group")
        self.plateConditionAxisLabelField.setText("Plate condition")
        self.extraVariableLabelField.setText("Run")
        for checkbox in self._measure_checkboxes.values():
            checkbox.setChecked(True)

    def selected_metric_columns(self):
        columns = []
        for object_name, column, _label in MEASURE_WIDGETS:
            checkbox = self._measure_checkboxes[object_name]
            if checkbox.isChecked():
                columns.append(column)
        return columns
