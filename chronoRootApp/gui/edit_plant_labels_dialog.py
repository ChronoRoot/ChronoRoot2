"""Modal dialog to edit Genotype / plate / extra labels for one plant."""

from PyQt5 import QtWidgets


HELP_TEXT = (
    "Fix mislabeled experimental factors for this plant only. "
    "Changing Genotype moves the plant under Analysis/<new identifier>/. "
    "Plate condition and extra variable only update metadata. "
    "Robot, camera, and plant number cannot be changed here."
)


class EditPlantLabelsDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Edit plant labels")
        self.setModal(True)
        self.resize(460, 280)

        self.help_label = QtWidgets.QLabel(HELP_TEXT)
        self.help_label.setWordWrap(True)

        self.identity_label = QtWidgets.QLabel("")
        self.identity_label.setWordWrap(True)
        font = self.identity_label.font()
        font.setBold(True)
        self.identity_label.setFont(font)

        form = QtWidgets.QFormLayout()
        self.experiment_edit = QtWidgets.QLineEdit()
        self.plate_edit = QtWidgets.QLineEdit()
        self.extra_edit = QtWidgets.QLineEdit()
        form.addRow("Genotype / identifier", self.experiment_edit)
        form.addRow("Plate condition", self.plate_edit)
        form.addRow("Extra variable", self.extra_edit)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.help_label)
        layout.addWidget(self.identity_label)
        layout.addLayout(form)
        layout.addWidget(buttons)

    def populate(self, labels):
        """Fill fields from current_plant_labels() dict."""
        rpi = labels.get("rpi", "")
        cam = labels.get("cam", "")
        plant = labels.get("plant", "")
        self.identity_label.setText(f"Hardware slot: {rpi} / {cam} / {plant}")
        self.experiment_edit.setText(str(labels.get("experiment", "")))
        self.plate_edit.setText(str(labels.get("plate_condition", "")))
        self.extra_edit.setText(str(labels.get("extra_variable", "")))

    def values(self):
        return {
            "experiment": self.experiment_edit.text().strip(),
            "plate_condition": self.plate_edit.text().strip(),
            "extra_variable": self.extra_edit.text().strip(),
        }

    def _on_accept(self):
        if not self.experiment_edit.text().strip():
            QtWidgets.QMessageBox.warning(
                self,
                "Invalid labels",
                "Genotype / identifier cannot be empty.",
            )
            return
        self.accept()
