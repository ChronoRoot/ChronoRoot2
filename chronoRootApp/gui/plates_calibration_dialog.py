"""Window to review and set per-video plate scale (pixel_size).

Follows the same popup pattern as CalibrationHelper / ChronoViewWindow:
non-modal QMainWindow + show(), and the calibration helper is opened the same
way as Calibrate current video on the main window.
"""

from PyQt5 import QtCore, QtGui, QtWidgets

from analysis.utils.calibration_registry import (
    SOURCE_QR,
    STATUS_DEFAULT,
    STATUS_MISMATCH,
    STATUS_MISSING,
    STATUS_REGISTRY_ONLY,
    apply_scale_to_plants,
    list_plate_statuses,
    try_qr_pixel_size,
    upsert_calibration,
)
from gui.background import run_off_main_thread


STATUS_LABELS = {
    "ok_manual": "OK (manual)",
    "ok_qr": "OK (QR)",
    "ok_existing": "OK",
    "missing": "Missing",
    "default_fallback": "Default (QR miss)",
    "mismatch": "Mismatch",
    "registry_only": "Saved (no plants yet)",
}

DIALOG_TITLE = "Review plate calibrations"


def run_test_qr_flow(
    parent,
    project_dir,
    image_path,
    *,
    plant_paths=None,
    title=DIALOG_TITLE,
    progress_parent=None,
    on_set_scale=None,
    on_saved=None,
):
    """Scan a video for QR scale; offer Save to plate / Set scale on failure.

    Returns True if a scale was saved, False otherwise.
    """
    if not image_path:
        QtWidgets.QMessageBox.warning(parent, title, "No video path selected.")
        return False

    host = progress_parent if progress_parent is not None else parent

    def scan():
        return try_qr_pixel_size(image_path)

    try:
        pixel_size, detail = run_off_main_thread(
            host, "Testing QR detection…", scan,
        )
    except Exception as exc:
        QtWidgets.QMessageBox.critical(parent, title, f"QR test failed:\n{exc}")
        return False

    if pixel_size is None:
        box = QtWidgets.QMessageBox(parent)
        box.setIcon(QtWidgets.QMessageBox.Warning)
        box.setWindowTitle(title)
        box.setText(f"QR test failed.\n\n{detail}")
        set_btn = None
        if on_set_scale is not None:
            set_btn = box.addButton("Set scale…", QtWidgets.QMessageBox.AcceptRole)
        box.addButton(QtWidgets.QMessageBox.Close)
        box.exec_()
        if set_btn is not None and box.clickedButton() == set_btn:
            on_set_scale()
        return False

    box = QtWidgets.QMessageBox(parent)
    box.setIcon(QtWidgets.QMessageBox.Information)
    box.setWindowTitle(title)
    box.setText(
        f"QR scale detected.\n\n"
        f"pixel_size = {pixel_size:.6g} mm/px\n"
        f"{detail}"
    )
    save_btn = box.addButton("Save to plate", QtWidgets.QMessageBox.AcceptRole)
    box.addButton(QtWidgets.QMessageBox.Close)
    box.exec_()
    if box.clickedButton() != save_btn:
        return False

    if not project_dir:
        QtWidgets.QMessageBox.warning(
            parent, title, "Open a valid project folder before saving plate scale.",
        )
        return False

    try:
        upsert_calibration(
            project_dir,
            image_path,
            pixel_size=pixel_size,
            source=SOURCE_QR,
        )
        apply_scale_to_plants(
            plant_paths or [],
            pixel_size,
            source=SOURCE_QR,
        )
    except Exception as exc:
        QtWidgets.QMessageBox.critical(
            parent, title, f"Could not save QR scale:\n{exc}",
        )
        return False

    QtWidgets.QMessageBox.information(
        parent,
        title,
        "QR scale saved for this plate.\n\n"
        "Re-run Process all plants so metrics use the new scale.",
    )
    if on_saved is not None:
        on_saved(pixel_size)
    return True


class PlatesCalibrationDialog(QtWidgets.QMainWindow):
    def __init__(
        self,
        project_dir,
        parent=None,
        known_distance_default="10",
        current_image_path=None,
    ):
        # parent=None like CalibrationHelper / ChronoViewWindow so this is a
        # free top-level window. Keep host for loading overlays only.
        super().__init__(None)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self._host = parent
        self.project_dir = project_dir
        self.known_distance_default = str(known_distance_default or "10")
        self.current_image_path = current_image_path
        self._rows = []
        self._helper = None

        self.setWindowTitle(DIALOG_TITLE)
        self.resize(900, 480)

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)

        help_label = QtWidgets.QLabel(
            "Each video folder has its own plate scale (mm/px). "
            "Use Test QR… or Set scale… on a bad plate, then re-run "
            "Process all plants so metrics use the new scale."
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
        self.test_qr_btn = QtWidgets.QPushButton("Test QR…")
        self.test_qr_btn.setToolTip(
            "Scan the first frames of the selected video for a QR scale marker."
        )
        self.test_qr_btn.clicked.connect(self.test_qr_for_selected)
        self.set_scale_btn = QtWidgets.QPushButton("Set scale…")
        self.set_scale_btn.clicked.connect(self.set_scale_for_selected)
        self.close_btn = QtWidgets.QPushButton("Close")
        self.close_btn.clicked.connect(self.close)
        buttons.addWidget(self.refresh_btn)
        buttons.addWidget(self.test_qr_btn)
        buttons.addWidget(self.set_scale_btn)
        buttons.addStretch(1)
        buttons.addWidget(self.close_btn)
        layout.addLayout(buttons)

        self.refresh()

    def refresh(self):
        self._rows = list_plate_statuses(
            self.project_dir, current_image_path=self.current_image_path,
        )
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
                if status in (
                    STATUS_MISSING, STATUS_DEFAULT, STATUS_MISMATCH, STATUS_REGISTRY_ONLY,
                ):
                    if status == STATUS_REGISTRY_ONLY and row.get("pixel_size") is not None:
                        item.setBackground(QtGui.QColor(220, 240, 220))
                    else:
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

        video = image_path.rstrip("/").split("/")[-1] or image_path
        rpi = row.get("rpi") or ""
        cam = row.get("cam") or ""
        plate_label = video
        if rpi or cam:
            plate_label = f"{video}  ({rpi}/{cam})".strip()

        try:
            import calibration_helper
            self._helper = calibration_helper.CalibrationHelper(
                image_path,
                dialog_parent=self._host,
                known_distance=self.known_distance_default,
                project_dir=self.project_dir,
                image_path=image_path,
                plate_label=plate_label,
                plant_paths=row.get("plant_paths") or [],
            )
            self._helper.scale_saved.connect(self._on_scale_saved)
            self._helper.show()
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self, DIALOG_TITLE, f"Could not open calibration helper:\n{exc}",
            )

    def _on_scale_saved(self, *_args):
        self.refresh()
        QtWidgets.QMessageBox.information(
            self,
            DIALOG_TITLE,
            "Scale saved for this plate.\n\n"
            "Re-run Process all plants so metrics use the new scale.",
        )

    def test_qr_for_selected(self):
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

        run_test_qr_flow(
            self,
            self.project_dir,
            image_path,
            plant_paths=row.get("plant_paths") or [],
            progress_parent=self._host if self._host is not None else self,
            on_set_scale=self.set_scale_for_selected,
            on_saved=lambda *_: self.refresh(),
        )
