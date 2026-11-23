"""Window to review and set per-video plate scale (pixel_size).

Follows the same popup pattern as CalibrationHelper / ChronoViewWindow:
non-modal QMainWindow + show(), and the calibration helper is opened the same
way as Open Calibration Helper on the main window.
"""

from PyQt5 import QtCore, QtGui, QtWidgets

from analysis.utils.calibration_registry import (
    STATUS_DEFAULT,
    STATUS_MISMATCH,
    STATUS_MISSING,
    SOURCE_MANUAL,
    apply_scale_to_plants,
    compute_pixel_size,
    list_plate_statuses,
    upsert_calibration,
)


STATUS_LABELS = {
    "ok_manual": "OK (manual)",
    "ok_qr": "OK (QR)",
    "ok_existing": "OK",
    "missing": "Missing",
    "default_fallback": "Default (QR miss)",
    "mismatch": "Mismatch",
}

DIALOG_TITLE = "Review plate calibrations"


class PlatesCalibrationDialog(QtWidgets.QMainWindow):
    def __init__(self, project_dir, parent=None, known_distance_default="10"):
        # parent=None like CalibrationHelper / ChronoViewWindow so this is a
        # free top-level window. Keep host for loading overlays only.
        super().__init__(None)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self._host = parent
        self.project_dir = project_dir
        self.known_distance_default = str(known_distance_default or "10")
        self._rows = []
        self._helper = None
        self._pending_row = None
        self._pending_known = None

        self.setWindowTitle(DIALOG_TITLE)
        self.resize(900, 480)

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)

        help_label = QtWidgets.QLabel(
            "Each video folder has its own plate scale (mm/px). "
            "Camera-to-plate distance can change between videos. "
            "Use Set scale… to measure a known length and apply it to all plants on that plate."
        )
        help_label.setWordWrap(True)
        layout.addWidget(help_label)

        self.table = QtWidgets.QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels([
            "Video", "Robot", "Cam", "Plants", "pixel_size", "Source", "Status",
        ])
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        layout.addWidget(self.table)

        buttons = QtWidgets.QHBoxLayout()
        self.refresh_btn = QtWidgets.QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self.refresh)
        self.set_scale_btn = QtWidgets.QPushButton("Set scale…")
        self.set_scale_btn.clicked.connect(self.set_scale_for_selected)
        self.close_btn = QtWidgets.QPushButton("Close")
        self.close_btn.clicked.connect(self.close)
        buttons.addWidget(self.refresh_btn)
        buttons.addWidget(self.set_scale_btn)
        buttons.addStretch(1)
        buttons.addWidget(self.close_btn)
        layout.addLayout(buttons)

        self.refresh()

    def refresh(self):
        self._rows = list_plate_statuses(self.project_dir)
        self.table.setRowCount(len(self._rows))
        for row_idx, row in enumerate(self._rows):
            video = row["image_path"]
            short = video.rstrip("/").split("/")[-1] if video else ""
            values = [
                short or video,
                row.get("rpi", ""),
                row.get("cam", ""),
                str(row.get("plant_count", 0)),
                "" if row.get("pixel_size") is None else f"{row['pixel_size']:.6g}",
                row.get("source") or "",
                STATUS_LABELS.get(row.get("status"), row.get("status") or ""),
            ]
            for col, text in enumerate(values):
                item = QtWidgets.QTableWidgetItem(text)
                if col == 0:
                    item.setToolTip(video)
                    item.setData(QtCore.Qt.UserRole, row_idx)
                status = row.get("status")
                if status in (STATUS_MISSING, STATUS_DEFAULT, STATUS_MISMATCH):
                    item.setBackground(QtGui.QColor(255, 230, 180))
                self.table.setItem(row_idx, col, item)
        if self._rows:
            self.table.selectRow(0)

    def _selected_row(self):
        indexes = self.table.selectionModel().selectedRows()
        if not indexes:
            return None
        row_idx = indexes[0].row()
        if row_idx < 0 or row_idx >= len(self._rows):
            return None
        return self._rows[row_idx]

    def set_scale_for_selected(self):
        row = self._selected_row()
        if row is None:
            QtWidgets.QMessageBox.information(
                self, DIALOG_TITLE, "Select a plate row first.",
            )
            return
        image_path = row.get("image_path") or ""
        if not image_path:
            QtWidgets.QMessageBox.warning(
                self, DIALOG_TITLE, "This plate has no video path.",
            )
            return

        known_text, ok = QtWidgets.QInputDialog.getText(
            self,
            "Known distance",
            "Known physical length (mm) for the two-point measurement:",
            text=self.known_distance_default,
        )
        if not ok:
            return
        try:
            known = float(known_text)
            if known <= 0:
                raise ValueError
        except ValueError:
            QtWidgets.QMessageBox.warning(
                self, DIALOG_TITLE, "Enter a positive known distance in mm.",
            )
            return

        # Same pattern as run.open_calibration_helper: non-modal QMainWindow,
        # dialog_parent = main window for the loading overlay, show().
        try:
            import calibration_helper
            self._pending_row = row
            self._pending_known = known
            self._helper = calibration_helper.CalibrationHelper(
                image_path, dialog_parent=self._host,
            )
            self._helper.distance_measured.connect(self._on_distance_measured)
            self._helper.show()
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self, DIALOG_TITLE, f"Could not open calibration helper:\n{exc}",
            )

    def _on_distance_measured(self, pixels):
        """Apply scale when two points are chosen; leave the helper open to close."""
        row = self._pending_row
        known = self._pending_known
        if row is None or known is None:
            return
        try:
            pixel_size = compute_pixel_size(known, pixels)
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, DIALOG_TITLE, str(exc))
            return

        upsert_calibration(
            self.project_dir,
            row["image_path"],
            pixel_size=pixel_size,
            source=SOURCE_MANUAL,
            known_distance=known,
            pixel_distance=pixels,
        )
        apply_scale_to_plants(
            row.get("plant_paths") or [],
            pixel_size,
            source=SOURCE_MANUAL,
            known_distance=known,
            pixel_distance=pixels,
        )
        self.refresh()
        if self._helper is not None:
            self._helper.statusBar().showMessage(
                f"Saved {pixel_size:.6g} mm/px — close this window when done"
            )
