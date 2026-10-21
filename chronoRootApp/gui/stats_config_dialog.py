"""Modal dialog for statistical analysis configuration."""

from PyQt5 import QtCore, QtWidgets

STATS_MODE_WIDGETS = [
    ("statsByGenotype", "Compare genotypes (all data)"),
    ("statsGenotypeByPlate", "Compare genotypes within each plate condition"),
    ("statsGenotypeByExtra", "Compare genotypes within each extra variable"),
    ("statsByPlateCondition", "Compare plate conditions directly"),
    ("statsByExtraVariable", "Compare extra variable directly"),
    ("statsPlateWithinGenotype", "Compare plate conditions within each genotype"),
    ("statsExtraWithinGenotype", "Compare extra variable within each genotype"),
]

HELP_TEXT = (
    "Hypothesis testing uses the Mann-Whitney U test at each dt interval. "
    "Each plant contributes one mean value per interval. "
    "The dt fields set the length of that interval."
)


class StatsConfigDialog(QtWidgets.QDialog):
  def __init__(self, parent=None):
    super().__init__(parent)
    self.setWindowTitle("Configure Report Parameters")
    self.setModal(True)
    self.resize(520, 560)

    self.exportSvg = QtWidgets.QCheckBox("Also export .svg")
    self.exportSvg.setObjectName("exportSvg")
    self.exportSvg.setToolTip("PNG is always written. This adds an editable .svg copy.")

    self.exportPdf = QtWidgets.QCheckBox("Also export .pdf")
    self.exportPdf.setObjectName("exportPdf")
    self.exportPdf.setToolTip("PNG is always written. This adds a .pdf copy.")

    self.skipIndividualAnglePlots = QtWidgets.QCheckBox("Do not draw individual angle plots")
    self.skipIndividualAnglePlots.setObjectName("skipIndividualAnglePlots")
    self.skipIndividualAnglePlots.setToolTip(
        "Skip the angle overlays and the first-lateral zoom figures so a report "
        "can be remade faster. Angle comparison statistics are still written."
    )

    self.measureRelativeToInitial = QtWidgets.QCheckBox(
        "Measure each plant relative to initial stage"
    )
    self.measureRelativeToInitial.setObjectName("measureRelativeToInitial")
    self.measureRelativeToInitial.setChecked(False)
    self.measureRelativeToInitial.setToolTip(
        "Subtract the first non-NaN value in the analysis window from each "
        "length, area, and lateral-root-count series, per plant. "
        "Speeds, percentages, densities, and angles are unchanged."
    )

    self.everyXhourField = self._labeled_field(
        "Time series stats interval (dt, in hours):", "everyXhourField"
    )
    self.everyXhourFieldFourier = self._labeled_field(
        "Speeds stats interval (dt, in hours):", "everyXhourFieldFourier"
    )
    self.everyXhourFieldAngles = self._labeled_field(
        "First LR Tip Stats interval (dt, in hours):", "everyXhourFieldAngles"
    )

    self.help_label = QtWidgets.QLabel(HELP_TEXT)
    self.help_label.setWordWrap(True)

    self.advanced_group = QtWidgets.QGroupBox("Advanced comparison modes")
    self.advanced_group.setObjectName("advancedComparisonModes")
    self.advanced_group.setCheckable(True)
    self.advanced_group.setChecked(False)
    advanced_layout = QtWidgets.QVBoxLayout()
    self._mode_checkboxes = {}
    for object_name, label in STATS_MODE_WIDGETS:
      checkbox = QtWidgets.QCheckBox(label)
      checkbox.setObjectName(object_name)
      checkbox.setChecked(True)
      self._mode_checkboxes[object_name] = checkbox
      advanced_layout.addWidget(checkbox)
    self.advanced_group.setLayout(advanced_layout)

    buttons = QtWidgets.QDialogButtonBox(
        QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
    )
    buttons.accepted.connect(self.accept)
    buttons.rejected.connect(self.reject)

    layout = QtWidgets.QVBoxLayout(self)
    format_label = QtWidgets.QLabel("Figures are always saved as .png.")
    format_label.setWordWrap(True)
    layout.addWidget(format_label)
    layout.addWidget(self.exportSvg)
    layout.addWidget(self.exportPdf)
    layout.addWidget(self.skipIndividualAnglePlots)
    layout.addWidget(self.measureRelativeToInitial)
    layout.addLayout(self._field_row(self.everyXhourField[0], self.everyXhourField[1]))
    layout.addLayout(self._field_row(self.everyXhourFieldFourier[0], self.everyXhourFieldFourier[1]))
    layout.addLayout(self._field_row(self.everyXhourFieldAngles[0], self.everyXhourFieldAngles[1]))
    layout.addWidget(self.help_label)
    layout.addWidget(self.advanced_group)
    layout.addStretch()
    layout.addWidget(buttons)

  def _labeled_field(self, label_text, object_name):
    label = QtWidgets.QLabel(label_text)
    field = QtWidgets.QLineEdit()
    field.setObjectName(object_name)
    field.setMaximumWidth(80)
    return label, field

  def _field_row(self, label, field):
    row = QtWidgets.QHBoxLayout()
    row.addWidget(label)
    row.addStretch()
    row.addWidget(field)
    return row

  def register_on_host(self, host):
    """Expose dialog widgets on the main window for config save/load."""
    host.exportSvg = self.exportSvg
    host.exportPdf = self.exportPdf
    host.skipIndividualAnglePlots = self.skipIndividualAnglePlots
    host.measureRelativeToInitial = self.measureRelativeToInitial
    host.everyXhourField = self.everyXhourField[1]
    host.everyXhourFieldFourier = self.everyXhourFieldFourier[1]
    host.everyXhourFieldAngles = self.everyXhourFieldAngles[1]
    host.advanced_group = self.advanced_group
    for object_name, checkbox in self._mode_checkboxes.items():
      setattr(host, object_name, checkbox)

  def set_defaults(self):
    self.exportSvg.setChecked(False)
    self.exportPdf.setChecked(False)
    self.skipIndividualAnglePlots.setChecked(False)
    self.measureRelativeToInitial.setChecked(False)
    self.everyXhourField[1].setText("6")
    self.everyXhourFieldFourier[1].setText("6")
    self.everyXhourFieldAngles[1].setText("6")
    self.advanced_group.setChecked(False)
    for checkbox in self._mode_checkboxes.values():
      checkbox.setChecked(True)

  def stats_checkbox_fields(self):
    return list(self._mode_checkboxes.values())

  def snapshot_values(self):
    values = {
        'exportSvg': self.exportSvg.isChecked(),
        'exportPdf': self.exportPdf.isChecked(),
        'skipIndividualAnglePlots': self.skipIndividualAnglePlots.isChecked(),
        'measureRelativeToInitial': self.measureRelativeToInitial.isChecked(),
        'everyXhourField': self.everyXhourField[1].text(),
        'everyXhourFieldFourier': self.everyXhourFieldFourier[1].text(),
        'everyXhourFieldAngles': self.everyXhourFieldAngles[1].text(),
        'advancedComparisonModes': self.advanced_group.isChecked(),
        'modes': {
            name: checkbox.isChecked()
            for name, checkbox in self._mode_checkboxes.items()
        },
    }
    return values

  def restore_values(self, values):
    if not values:
      return
    if 'exportSvg' in values or 'exportPdf' in values:
      self.exportSvg.setChecked(bool(values.get('exportSvg')))
      self.exportPdf.setChecked(bool(values.get('exportPdf')))
    else:
      legacy = str(values.get('figureFormat') or '').strip().lower()
      if legacy and not legacy.startswith('.'):
        legacy = '.' + legacy
      self.exportSvg.setChecked(legacy == '.svg')
      self.exportPdf.setChecked(legacy == '.pdf')
    if 'skipIndividualAnglePlots' in values:
      self.skipIndividualAnglePlots.setChecked(bool(values.get('skipIndividualAnglePlots')))
    self.measureRelativeToInitial.setChecked(bool(values.get('measureRelativeToInitial')))
    if values.get('everyXhourField') is not None:
      self.everyXhourField[1].setText(str(values['everyXhourField']))
    if values.get('everyXhourFieldFourier') is not None:
      self.everyXhourFieldFourier[1].setText(str(values['everyXhourFieldFourier']))
    if values.get('everyXhourFieldAngles') is not None:
      self.everyXhourFieldAngles[1].setText(str(values['everyXhourFieldAngles']))
    for name, checked in (values.get('modes') or {}).items():
      checkbox = self._mode_checkboxes.get(name)
      if checkbox is not None:
        checkbox.setChecked(bool(checked))
    if 'advancedComparisonModes' in values:
      self.advanced_group.setChecked(bool(values['advancedComparisonModes']))
