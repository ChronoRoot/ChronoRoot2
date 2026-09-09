import os
import sys
import platform
        
# Suppress Qt and OpenGL warnings
os.environ['QT_LOGGING_RULES'] = '*=false'
os.environ['LIBGL_ALWAYS_INDIRECT'] = '1'

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import (
    QTableWidget, QTableWidgetItem, QPushButton, QTreeWidget, QTreeWidgetItem, QLineEdit,
    QSplitter,
)
from PyQt5.QtWidgets import QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QCheckBox

import subprocess
import json
import pathlib
import re
import shutil
import glob
from collections import defaultdict
from PIL import Image
import numpy as np

from analysis.utils.fileUtilities import (
    convertFromPathSafe,
    getImages,
    get_latest_result_dir,
    load_result_metadata,
    normalize_factor_value,
    plant_slot_has_finished_analysis,
    processing_limit_frame_count,
)
from analysis.utils.report_utils import natural_key as natural_keys
from gui.config_store import ConfigStore, PROJECT_CONFIG_NAME
from gui import pipeline_runner
from gui.stats_config_dialog import StatsConfigDialog
from gui.remap_identifiers_dialog import RemapIdentifiersDialog
from gui.time_window_dialog import TimeWindowDialog
from gui.snapshot_hours_dialog import SnapshotHoursDialog
from analysis.time_windows import (
    PERIOD_GATE_MESSAGE,
    analysis_period_is_current,
    report_folder_name,
)
from gui.report_browser import ReportBranch, load_report_catalog
from analysis.utils.remap_identifiers import (
    RemapError,
    apply_experiment_remap,
    collect_experiment_counts,
)
from analysis.utils.plant_inventory import scan_analysis_plants

TAB_HEIGHT = 630
REPORT_PLOT_ROLE = QtCore.Qt.UserRole
REPORT_STATS_ROLE = QtCore.Qt.UserRole + 1

WINDOW_WIDTH = 811
WINDOW_HEIGHT = TAB_HEIGHT + 20


class PlantInventoryWorker(QtCore.QObject):
    """Background scan of Analysis/; results applied on the UI thread."""

    finished = QtCore.pyqtSignal(int, str, object)  # generation, project_path, rows
    failed = QtCore.pyqtSignal(int, str, str)

    def __init__(self, generation, project_path, analysis_path):
        super().__init__()
        self.generation = generation
        self.project_path = project_path
        self.analysis_path = analysis_path

    @QtCore.pyqtSlot()
    def run(self):
        try:
            rows = scan_analysis_plants(self.analysis_path)
            self.finished.emit(self.generation, self.project_path, rows)
        except Exception as exc:
            self.failed.emit(self.generation, self.project_path, str(exc))


class AspectRatioLabel(QLabel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._source_pixmap = None
        self.setAlignment(QtCore.Qt.AlignCenter)

    def sizeHint(self):
        # Do not let pixmap dimensions drive the parent layout height
        return QtCore.QSize(200, 200)

    def minimumSizeHint(self):
        return QtCore.QSize(120, 120)

    def resizeEvent(self, event):
        self._apply_scaled_pixmap()
        super().resizeEvent(event)

    def set_pixmap(self, pixmap, size=None):
        """Keep full-res source; scale to current label size (or optional size)."""
        self._source_pixmap = pixmap
        self._apply_scaled_pixmap(size)

    def clear(self):
        self._source_pixmap = None
        super().clear()

    def _apply_scaled_pixmap(self, size=None):
        if self._source_pixmap is None or self._source_pixmap.isNull():
            return
        if size is None:
            size = self.size()
        if size.width() < 2 or size.height() < 2:
            return
        scaled = self._source_pixmap.scaled(
            size, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation
        )
        self.setPixmap(scaled)

class Ui_ChronoRootAnalysis(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.project_dir = None
        self.selected_plant = None
        self.config_store = ConfigStore()
        # Cache: (project_path, plant_path, overlay_on) -> (pix1, pix2)
        self._overlay_pixmap_cache = {}
        self._plants_loaded_for = None
        self._plants_dirty = True
        self._inventory_generation = 0
        self._last_seen_project = None
        self._inventory_thread = None
        self._inventory_worker = None
        self._prefer_index_after_load = None
        self.setupUi(self)

    def current_project_key(self):
        text = self.projectField.text().strip() if hasattr(self, "projectField") else ""
        if not text:
            return ""
        return os.path.abspath(os.path.expanduser(text))

    def current_plant_path(self):
        """Absolute Results_*/plant path for the plant currently selected in Overlay."""
        if not hasattr(self, 'plant_dropdown') or self.plant_dropdown is None:
            return self.selected_plant
        data = self.plant_dropdown.currentData()
        if data:
            return data
        text = self.plant_dropdown.currentText()
        return text or self.selected_plant

    def _clear_overlay_cache_for_path(self, path):
        if not path:
            return
        for key in list(self._overlay_pixmap_cache.keys()):
            # key is (project, plant_path, overlay_on)
            if len(key) >= 2 and key[1] == path:
                del self._overlay_pixmap_cache[key]

    def inventory_is_stale(self):
        return (
            self._plants_dirty
            or self._plants_loaded_for != self.current_project_key()
        )

    def invalidate_plant_inventory(self):
        """Drop Overview/Overlay plant lists after a project change."""
        self._inventory_generation += 1
        self._plants_dirty = True
        self._plants_loaded_for = None
        self._overlay_pixmap_cache.clear()
        self.selected_plant = None
        self._prefer_index_after_load = None

        if hasattr(self, "table") and self.table is not None:
            self.table.setSortingEnabled(False)
            self.table.clearContents()
            self.table.setRowCount(0)
            self.table.setSortingEnabled(True)

        if hasattr(self, "plant_dropdown") and self.plant_dropdown is not None:
            self.plant_dropdown.blockSignals(True)
            self.plant_dropdown.clear()
            self.plant_dropdown.blockSignals(False)

        if hasattr(self, "image_label1"):
            self.image_label1.clear()
            self.image_label1.set_pixmap(
                QtGui.QPixmap("placeholder_figures/plant_placeholder.png")
            )
        if hasattr(self, "image_label2"):
            self.image_label2.clear()
            self.image_label2.set_pixmap(
                QtGui.QPixmap("placeholder_figures/plant_report_placeholder.png")
            )

        self._set_inventory_loading(False)

        # If Overview/Overlay is visible, reload for the new project immediately
        if hasattr(self, "tab_widget"):
            idx = self.tab_widget.currentIndex()
            if idx in (1, 2):
                QtCore.QTimer.singleShot(0, lambda: self.ensure_plant_inventory(force=True))

    def _set_inventory_loading(self, loading):
        for name in ("refresh_button", "refresh_button_tab3"):
            btn = getattr(self, name, None)
            if btn is not None:
                btn.setEnabled(not loading)
                if loading:
                    btn.setToolTip("Loading plant list…")
                elif name == "refresh_button":
                    btn.setToolTip(
                        "Update the table to show the latest analysis progress and error rates."
                    )
                else:
                    btn.setToolTip(
                        "Refresh the list of plants available for visual inspection."
                    )
        if loading and hasattr(self, "plant_dropdown") and self.plant_dropdown is not None:
            if self.plant_dropdown.count() == 0:
                self.plant_dropdown.blockSignals(True)
                self.plant_dropdown.addItem("Loading…")
                self.plant_dropdown.blockSignals(False)

    def _on_project_path_maybe_changed(self):
        key = self.current_project_key()
        if key == self._last_seen_project:
            return
        self._last_seen_project = key
        self.invalidate_plant_inventory()

    def force_refresh_plant_inventory(self, prefer_index=None):
        """Manual Refresh: always rescan the current project."""
        project = self.current_project_key()
        for key in list(self._overlay_pixmap_cache.keys()):
            if key and key[0] == project:
                del self._overlay_pixmap_cache[key]
        self._plants_dirty = True
        self._prefer_index_after_load = prefer_index
        self.ensure_plant_inventory(force=True)

    def ensure_plant_inventory(self, force=False):
        """Start an async scan if the inventory is stale (or force=True)."""
        if not force and not self.inventory_is_stale():
            return
        project = self.current_project_key()
        generation = self._inventory_generation + 1
        self._inventory_generation = generation
        self._plants_dirty = True

        # Clear UI immediately so old-project rows cannot linger
        if hasattr(self, "table") and self.table is not None:
            self.table.setSortingEnabled(False)
            self.table.clearContents()
            self.table.setRowCount(0)
        if hasattr(self, "plant_dropdown") and self.plant_dropdown is not None:
            self.plant_dropdown.blockSignals(True)
            self.plant_dropdown.clear()
            self.plant_dropdown.blockSignals(False)

        self._set_inventory_loading(True)

        analysis_root = os.path.join(project, "Analysis") if project else ""

        # Drop previous worker thread (stale results ignored via generation)
        if self._inventory_thread is not None:
            try:
                self._inventory_thread.quit()
            except Exception:
                pass
            self._inventory_thread = None
            self._inventory_worker = None

        worker = PlantInventoryWorker(generation, project, analysis_root)
        thread = QtCore.QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._on_inventory_finished)
        worker.failed.connect(self._on_inventory_failed)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        self._inventory_worker = worker
        self._inventory_thread = thread
        thread.start()

    def _on_inventory_failed(self, generation, project_path, message):
        if generation != self._inventory_generation:
            return
        if project_path != self.current_project_key():
            return
        self._set_inventory_loading(False)
        self._plants_dirty = True
        QtWidgets.QMessageBox.warning(
            self, "Plant list", f"Failed to load plants:\n{message}"
        )

    def _on_inventory_finished(self, generation, project_path, rows):
        if generation != self._inventory_generation:
            return
        if project_path != self.current_project_key():
            return
        self.apply_plant_inventory(project_path, rows)

    def apply_plant_inventory(self, project_path, rows):
        """Fill Overview table + Overlay combo on the UI thread."""
        prefer_index = self._prefer_index_after_load
        self._prefer_index_after_load = None

        previous_path = None
        if hasattr(self, "plant_dropdown") and self.plant_dropdown is not None:
            previous_path = self.plant_dropdown.currentData() or self.selected_plant
            self.plant_dropdown.blockSignals(True)
            self.plant_dropdown.clear()

        current_sort_order = self.table.horizontalHeader().sortIndicatorOrder()
        current_sort_column = self.table.horizontalHeader().sortIndicatorSection()
        self.table.setSortingEnabled(False)
        self.table.clearContents()
        self.table.setRowCount(len(rows))

        for row, row_data in enumerate(rows):
            for col, cell_data in enumerate(row_data[:-2]):
                item = QTableWidgetItem(str(cell_data))
                item.path = row_data[-2]
                item.plant_slot = row_data[-1]
                self.table.setItem(row, col, item)

            experiment, rpi, camera, plant = row_data[0], row_data[1], row_data[2], row_data[3]
            active_path = row_data[-2]
            if hasattr(self, "plant_dropdown") and self.plant_dropdown is not None:
                label = f"{experiment} — {rpi} / {camera} / {plant}"
                self.plant_dropdown.addItem(label, active_path)

        self.table.resizeColumnsToContents()
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setSortingEnabled(True)
        self.table.sortItems(current_sort_column, current_sort_order)

        if hasattr(self, "plant_dropdown") and self.plant_dropdown is not None:
            count = self.plant_dropdown.count()
            if count > 0:
                restore = -1
                if previous_path:
                    restore = self.plant_dropdown.findData(previous_path)
                if restore < 0 and prefer_index is not None:
                    restore = min(max(prefer_index, 0), count - 1)
                if restore < 0:
                    restore = 0
                self.plant_dropdown.setCurrentIndex(restore)
                self.selected_plant = self.current_plant_path()
            self.plant_dropdown.blockSignals(False)

        self._plants_loaded_for = project_path
        self._plants_dirty = False
        self._set_inventory_loading(False)

        if hasattr(self, "tab_widget") and self.tab_widget.currentIndex() == 2:
            self.update_image_labels()

    def refresh_table(self, prefer_index=None):
        """Backward-compatible entry: force a rescan (Refresh buttons / remap / remove)."""
        self.force_refresh_plant_inventory(prefer_index=prefer_index)

    def handle_tab_change(self, index):
        # index 0: Plant Analysis
        # index 1: Analysis Overview
        if index == 1:
            self.ensure_plant_inventory()

        # index 2: Plant Overlay
        elif index == 2:
            self.ensure_plant_inventory()
            if not self.inventory_is_stale():
                self.update_image_labels()

        # index 4: Report Viewer
        elif index == 4:
            self.refresh_tab5()
            self.update_report_labels()

    def openFileNameDialog(self):
        options = QtWidgets.QFileDialog.Options() | QtWidgets.QFileDialog.DontUseNativeDialog
        return QtWidgets.QFileDialog.getExistingDirectory(None, "Select Directory", options=options)

    def _config_value(self, data, key, default=None):
        return self.config_store.config_value(data, key, default)

    def _import_roi_selection(self):
        try:
            from analysis.utils.roi_selection import select_roi_and_seed
            return select_roi_and_seed
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self, "Error", f"Failed to import ROI selection:\n{e}",
            )
            return None

    def saveFieldsIntoJson(self):
        self.config_store.save(self)

    def loadJsonIntoFields(self):
        self.config_store.load(self)

    def open_stats_config_dialog(self):
        self.stats_config_dialog.exec_()

    def open_time_window_dialog(self):
        self.saveFieldsIntoJson()
        dialog = TimeWindowDialog(self)
        dialog.set_main_folder(self.projectField.text().strip())
        conf = self.config_store.build_payload(self)
        dialog.load_from_conf(conf)
        if dialog.exec_() != QtWidgets.QDialog.Accepted:
            return
        payload = dialog.to_conf()
        for key, value in payload.items():
            setattr(self, key, value)
        self.saveFieldsIntoJson()

    def _sync_snapshot_hour_fields(self):
        convex_on = self.doConvex.isChecked()
        self.daysConvexField.setEnabled(convex_on)
        self.daysConvexLabel.setEnabled(convex_on)
        self.daysConvexText.setEnabled(convex_on)
        self.daysConvexChooseButton.setEnabled(convex_on)
        self.saveImagesConvex.setEnabled(convex_on)
        angles_on = self.doLateralAngles.isChecked()
        self.daysAnglesField.setEnabled(angles_on)
        self.daysAnglesText.setEnabled(angles_on)
        self.daysAnglesHint.setEnabled(angles_on)
        self.daysAnglesChooseButton.setEnabled(angles_on)

    def open_snapshot_hours_dialog(self, field):
        dialog = SnapshotHoursDialog(
            hours=field.text(),
            duration_hours=getattr(self, 'timeDurationHours', None),
            parent=self,
        )
        if dialog.exec_() != QtWidgets.QDialog.Accepted:
            return
        field.setText(dialog.selected_text())

    def open_remap_identifiers_dialog(self):
        project = self.projectField.text().strip()
        analysis_folder = os.path.join(project, "Analysis") if project else ""
        if not project or not os.path.isdir(analysis_folder):
            QtWidgets.QMessageBox.critical(
                self,
                "Error",
                "Select a project that contains an Analysis folder first.",
            )
            return

        unique_counts = collect_experiment_counts(analysis_folder)
        if not unique_counts:
            QtWidgets.QMessageBox.information(
                self,
                "Remap Identifiers",
                "No plant identifiers found under Analysis.",
            )
            return

        dialog = RemapIdentifiersDialog(self)
        dialog.populate(unique_counts)
        if dialog.exec_() != QtWidgets.QDialog.Accepted:
            return

        mapping = dialog.get_mapping()
        if not mapping:
            return

        # Summarize renames / merges for confirmation
        by_target = defaultdict(list)
        for old, new in mapping.items():
            by_target[new].append(old)
        lines = []
        for new, olds in sorted(by_target.items(), key=lambda x: str(x[0]).lower()):
            if len(olds) == 1 and olds[0] != new:
                lines.append(f"  {olds[0]}  →  {new}")
            else:
                lines.append(f"  merge {', '.join(olds)}  →  {new}")
        confirm = QtWidgets.QMessageBox.question(
            self,
            "Confirm remap",
            "Apply the following identifier changes?\n\n"
            + "\n".join(lines)
            + "\n\nThis renames folders under Analysis and updates metadata.",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No,
        )
        if confirm != QtWidgets.QMessageBox.Yes:
            return

        try:
            apply_experiment_remap(analysis_folder, mapping)
        except RemapError as e:
            QtWidgets.QMessageBox.critical(self, "Remap failed", str(e))
            self.refresh_table()
            return
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self, "Remap failed", f"Unexpected error:\n{e}"
            )
            self.refresh_table()
            return

        self.refresh_table()
        QtWidgets.QMessageBox.information(
            self,
            "Remap complete",
            "Plant identifiers were updated.\n\n"
            "Re-run \"Process all plants\" and \"Generate report\" "
            "so aggregates and figures use the new names.",
        )

    def universal_open(self, path):
        try:
            path = os.path.abspath(os.path.expanduser(path))
            
            # 1. Detect Environment
            is_container = any(k in os.environ for k in ['APPTAINER_CONTAINER', 'SINGULARITY_CONTAINER'])
            is_wsl = False
            if os.path.exists("/proc/version"):
                with open("/proc/version", "r") as f:
                    if "microsoft" in f.read().lower():
                        is_wsl = True

            # --- STRATEGY 1: WSL Interop (Hardcoded Path Logic) ---
            if is_wsl:
                # Inside Apptainer, 'explorer.exe' won't be in PATH. 
                # We must use the absolute path via the bind.
                explorer_cmd = "/mnt/c/Windows/explorer.exe"
                                
                if os.path.exists(explorer_cmd):
                    try:
                        # Attempt to translate path
                        win_path = None
                        print(f"Translating WSL path: {path}")
                        
                        if path.startswith("/mnt/c/"):
                            print("Using manual path conversion for WSL")
                            # Manual fallback for the bind you provided
                            win_path = path.replace("/mnt/c/", "C:\\").replace("/", "\\")
                        else:
                            # Hardcode fallback for standard WSL distro paths
                            wsl_path = "//wsl.localhost/ubuntu/".replace("/", "\\")  
                            win_path = wsl_path + path.lstrip("/").replace("/", "\\")
                            
                        print(f"Converted WSL path to Windows path: {win_path}")
                        if win_path:
                            print(f"Opening path in Windows Explorer: {win_path}")
                            print(f"Using explorer command: {explorer_cmd}")
                            subprocess.Popen(["/bin/sh", "-c", f"{explorer_cmd} '{win_path}'"])
                            return
                    except Exception as e:
                        print(f"WSL Explorer Bridge failed: {e}")

            # --- STRATEGY 2: D-Bus (The "Native Linux" path) ---
            # This works perfectly on Native Ubuntu/Apptainer-on-Ubuntu
            if is_container and shutil.which("dbus-send"):
                try:
                    # We add a timeout so it doesn't hang if no one is listening (like in WSL)
                    subprocess.run([
                        "dbus-send", "--session", "--print-reply", "--dest=org.freedesktop.FileManager1",
                        "--type=method_call", "/org/freedesktop/FileManager1",
                        "org.freedesktop.FileManager1.ShowItems", 
                        f"array:string:file://{path}", "string:''"
                    ], timeout=1, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
                    return
                except:
                    pass # Fall through if D-Bus times out or fails

            # --- STRATEGY 3: Standard OS Openers (Final Fallback) ---
            if platform.system() == "Darwin":
                subprocess.Popen(["open", path])
                return
            elif platform.system() == "Windows":
                os.startfile(path)
                return
            
            # Standard Linux xdg-open
            if shutil.which("xdg-open"):
                subprocess.Popen(["xdg-open", path], stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
                return

        except Exception as e:
            print(f"Universal Open Error: {e}")

        # UI Fallback if everything fails
        QtWidgets.QMessageBox.information(None, 'Manual Action', f'Auto-open failed. Path:\n{path}')
        
    def current_report_dir(self):
        return os.path.join(
            self.projectField.text(),
            report_folder_name({'reportFolderName': getattr(self, 'reportFolderName', 'Report')}),
        )

    def open_report_folder(self):
        self.universal_open(self.current_report_dir())
        
    def open_selected_path_tab3(self):
        self.selected_plant = self.current_plant_path()
        self.universal_open(self.selected_plant)
        
    def open_selected_path(self):
        selected_rows = self.table.selectionModel().selectedRows()
        if selected_rows:
            item = self.table.item(selected_rows[0].row(), 0)
            self.universal_open(item.path)

    def remove_selected_path(self):
        selected_rows = self.table.selectionModel().selectedRows()

        if not selected_rows:
            return

        selected_row = selected_rows[0].row()
        item = self.table.item(selected_row, 0)
        path = getattr(item, 'plant_slot', os.path.dirname(item.path))

        removed_path = os.path.join(self.projectField.text(), "Removed")
        removed_path = os.path.join(
            removed_path,
            os.path.relpath(path, os.path.join(self.projectField.text(), "Analysis")),
        )

        if not os.path.exists(os.path.dirname(removed_path)):
            os.makedirs(os.path.dirname(removed_path))

        # Remove existing destination if it exists
        if os.path.exists(removed_path):
            shutil.rmtree(removed_path)

        # Move to removed folder
        shutil.move(path, removed_path)
        
        self.refresh_table()
        
        return

    def set_default_parameters(self):
        """Set default values for important fields"""
        self.reportEmergenceDistanceField.setText("2")
        self.processingLimitField.setText("0")
        self.reportProcessingLimitField.setText("0")
        self.captureIntervalField.setText("15")
        self.reportCaptureIntervalField.setText("15")
        self.analysisEmergenceDistanceField.setText("2")
        self.numComponentsFPCAField.setText("2")
        self.stats_config_dialog.set_defaults()
        self.timeSyncMode = 'clock'
        self.timeDurationHours = None
        self.reportFolderName = 'Report'
        self.figureClockTicks = ['00:00']
        self.showFigureClockTicks = True
        self.snapshotHours = None
        self.timeGroups = []
        self.timePeriodSources = []

    def validate_numeric_input(self, field):
        """Validate numeric input fields"""
        try:
            text = field.text()
            # Don't allow empty fields
            if text.strip() == "":
                if field in [self.reportEmergenceDistanceField, self.analysisEmergenceDistanceField]:
                    field.setText("2")
                elif field in [self.processingLimitField, self.reportProcessingLimitField]:
                    field.setText("0")
                elif field in [self.captureIntervalField, self.reportCaptureIntervalField]:
                    field.setText("15")
                return
                
            value = float(text)
            if field == self.reportEmergenceDistanceField or field == self.analysisEmergenceDistanceField:
                if value <= 0:
                    field.setText("2")
            elif field in [self.processingLimitField, self.reportProcessingLimitField]:
                if value < 0:
                    field.setText("0")
            elif field in [self.captureIntervalField, self.reportCaptureIntervalField]:
                if value <= 0:
                    field.setText("15")
        except ValueError:
            if field in [self.reportEmergenceDistanceField, self.analysisEmergenceDistanceField]:
                field.setText("2")
            elif field in [self.processingLimitField, self.reportProcessingLimitField]:
                field.setText("0")
            elif field in [self.captureIntervalField, self.reportCaptureIntervalField]:
                field.setText("15")

    def setup_field_validation(self):
        """Set up validation for numeric fields"""
        # Connect validation to editingFinished signal
        self.reportEmergenceDistanceField.editingFinished.connect(
            lambda: self.validate_numeric_input(self.reportEmergenceDistanceField))
        self.analysisEmergenceDistanceField.editingFinished.connect(
            lambda: self.validate_numeric_input(self.analysisEmergenceDistanceField))
        self.processingLimitField.editingFinished.connect(
            lambda: self.validate_numeric_input(self.processingLimitField))
        self.reportProcessingLimitField.editingFinished.connect(
            lambda: self.validate_numeric_input(self.reportProcessingLimitField))
        self.captureIntervalField.editingFinished.connect(
            lambda: self.validate_numeric_input(self.captureIntervalField))
        self.reportCaptureIntervalField.editingFinished.connect(
            lambda: self.validate_numeric_input(self.reportCaptureIntervalField))

    def get_image_paths(self):
        if not os.path.exists(os.path.join(self.selected_plant, "log.txt")):
            return None, None, None, None
        
        metadata = json.load(open(os.path.join(self.selected_plant, "metadata.json"), 'r'))
        bbox = metadata["bounding box"]
        overlayPath = metadata["folders"]["images"] + "/SegMulti/"
        
        experiment = self.selected_plant.split(os.path.sep)[-5]
        rpi = self.selected_plant.split(os.path.sep)[-4]
        camera = self.selected_plant.split(os.path.sep)[-3]
        plant = self.selected_plant.split(os.path.sep)[-2]

        filename = experiment + "_" + rpi + "_" + camera + "_" + plant + ".png"
        image2_path = os.path.join(self.selected_plant, filename)
        
        if not os.path.exists(image2_path):
            image2_path = None

        # list all images in the folder with pathlib, then sort them
        pathlib_dir = pathlib.Path(overlayPath)
        image_files = pathlib_dir.glob('*.png')
        image_files = [str(file) for file in image_files]
        image_files = sorted(image_files, key=lambda x: natural_keys(x))

        if len(image_files) == 0:
            return "Image not found", image2_path, overlayPath, None
        
        overlay = image_files[-1]
        image1_path = metadata["ImagePath"] + '/' + overlay.split(os.path.sep)[-1]

        return image1_path, image2_path, overlay, bbox

    def update_image_labels(self):
        # Lazy import to avoid collisions with pyqt
        import cv2
        
        # Add safety check
        if not hasattr(self, 'plant_dropdown') or self.plant_dropdown is None:
            return

        self.selected_plant = self.current_plant_path()
        if not self.selected_plant:
            return

        overlay_on = self.overlay_checkbox.isChecked()
        project = self.current_project_key()
        cache_key = (project, self.selected_plant, overlay_on)
        cached = self._overlay_pixmap_cache.get(cache_key)
        if cached is not None:
            pix1, pix2 = cached
            self.image_label1.set_pixmap(pix1)
            self.image_label1.show()
            self.image_label2.set_pixmap(pix2)
            self.image_label2.show()
            return

        image1_path, image2_path, overlay, bbox = self.get_image_paths()

        pixmap1 = None
        pixmap2 = None

        # Check if image paths exist
        if image1_path is None:
            self.image_label1.clear()
            pixmap1 = QtGui.QPixmap("placeholder_figures/plant_placeholder.png")
            self.image_label1.set_pixmap(pixmap1)
            self.image_label1.show()
        elif not os.path.exists(image1_path) or not os.path.exists(overlay):
            self.image_label1.clear()
            pixmap1 = QtGui.QPixmap("placeholder_figures/plant_placeholder_2.png")
            self.image_label1.set_pixmap(pixmap1)
            self.image_label1.show()
        else:
            self.image_label1.clear()

            try:
                import plant_viewer

                img = cv2.imread(image1_path)
                if img is None:
                    raise ValueError(f"Could not read image: {image1_path}")

                y1, y2, x1, x2 = bbox
                h, w = img.shape[:2]
                if 0 <= y1 < y2 <= h and 0 <= x1 < x2 <= w:
                    img = img[y1:y2, x1:x2]

                if overlay_on and os.path.exists(overlay):
                    seg = cv2.imread(overlay, cv2.IMREAD_UNCHANGED)
                    if seg is not None:
                        img = plant_viewer._overlay_label_segmentation(img, seg)

                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                if not rgb.flags['C_CONTIGUOUS']:
                    rgb = np.ascontiguousarray(rgb)
                rh, rw, _ = rgb.shape
                qImg = QtGui.QImage(
                    rgb.data.tobytes(), rw, rh, 3 * rw, QtGui.QImage.Format_RGB888
                )
                pixmap1 = QtGui.QPixmap.fromImage(qImg)

                self.image_label1.set_pixmap(pixmap1)
                self.image_label1.show()

            except Exception as e:
                self.image_label1.setText(f"Analysis is not yet finished. \nRefresh to update\nError: {str(e)}")
                self.image_label1.setAlignment(QtCore.Qt.AlignCenter)
                self.image_label1.show()

        # Check if image2_path exists
        if image2_path is not None and os.path.exists(image2_path):
            pixmap2 = QtGui.QPixmap(image2_path)
            self.image_label2.set_pixmap(pixmap2)
            self.image_label2.show()
        else:
            self.image_label2.clear()
            pixmap2 = QtGui.QPixmap("placeholder_figures/plant_report_placeholder.png")
            self.image_label2.set_pixmap(pixmap2)
            self.image_label2.show()

        if pixmap1 is not None and pixmap2 is not None:
            self._overlay_pixmap_cache[cache_key] = (pixmap1, pixmap2)

        return

    def remove_selected_plant(self):
        path = self.current_plant_path()
        if not path:
            return
        idx = self.plant_dropdown.currentIndex()
        plant_slot = os.path.dirname(path) if os.path.basename(path).startswith('Results_') else path

        removed_path = os.path.join(
            self.projectField.text(),
            "Removed",
            os.path.relpath(plant_slot, os.path.join(self.projectField.text(), "Analysis")),
        )

        if not os.path.exists(os.path.dirname(removed_path)):
            os.makedirs(os.path.dirname(removed_path))

        # Open the directory in the file explorer
        if os.name == 'nt':
            os.system(f'move "{plant_slot}" "{removed_path}"')
        elif sys.platform == 'darwin':
            os.system(f'mv "{plant_slot}" "{removed_path}"')
        else:
            os.system(f'mv "{plant_slot}" "{removed_path}"')

        self._clear_overlay_cache_for_path(path)
        # Prefer the same list position (next plant) after the removed entry disappears
        self.refresh_table(prefer_index=idx)

        return
    
    def analysis(self):
        """Run analysis with validation"""

        select_roi_and_seed = self._import_roi_selection()
        if select_roi_and_seed is None:
            return

        # Get and validate video folder
        video_folder = self.videoField.text()
        
        if not video_folder:
            QtWidgets.QMessageBox.warning(None, 'Error', 'Please specify a video folder first!')
            return
            
        if not os.path.exists(video_folder):
            QtWidgets.QMessageBox.warning(None, 'Error', 'Video folder does not exist!\nPlease check the path.')
            return
        
        # Check for PNG images
        images = glob.glob(os.path.join(video_folder, "*.png"))
        
        # Check if there is no images, then look for a file called "segmentation_metadata.json"
        if not images:
            metadata_path = os.path.join(video_folder, 'Segmentation', 'segmentation_metadata.json')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                images = glob.glob(os.path.join(metadata["input_path"], "*.png")) 
                
        if not images:
            QtWidgets.QMessageBox.warning(
                None, 'Error', 
                'No images found in the video folder!\nPlease check the path to the folder where the images are located.'
            )
            return
        
        # Check for segmentation files (required for analysis)
        seg_folder = os.path.join(video_folder, "Segmentation", "Ensemble")
        seg_files = glob.glob(os.path.join(seg_folder, "*.png")) if os.path.exists(seg_folder) else []
        
        if not seg_files:
            QtWidgets.QMessageBox.warning(
                None, 'Error',
                f'Found {len(images)} images but no segmentation files!\n\n'
                'Segmentation is required for analysis.\n'
                'Please ensure the images have been properly segmented first.'
            )
            return
        
        # Validate calibration settings
        if not self.videoHasQRbutton.isChecked():                
            if not self.knownDistanceField.text() or not self.pixelDistanceField.text():
                QtWidgets.QMessageBox.warning(
                    None, 'Error', 
                    'Please provide both known distance and pixel distance for manual calibration,\n'
                    'or enable "Video has QR codes"!'
                )
                return
            
            try:
                known_dist = float(self.knownDistanceField.text())
                pixel_dist = int(self.pixelDistanceField.text())
                if known_dist <= 0 or pixel_dist <= 0:
                    QtWidgets.QMessageBox.warning(None, 'Error', 'Calibration values must be positive numbers!')
                    return
            except ValueError:
                QtWidgets.QMessageBox.warning(
                    None, 'Error', 
                    'Invalid calibration values!\nKnown distance must be a number, pixel distance must be an integer.'
                )
                return
        
        # All validations passed, run analysis
        self.saveFieldsIntoJson()

        config_path = os.path.join(self.projectField.text(), PROJECT_CONFIG_NAME)
        try:
            with open(config_path, 'r') as f:
                conf = json.load(f)
            conf['Experiment'] = conf.get('plantIdentifier', conf.get('Experiment', ''))
            conf['MainFolder'] = self.projectField.text()
            conf['rpi'] = self.rpiField.text()
            conf['cam'] = self.cameraField.text()
            conf['plant'] = self.plantField.text()
        except (OSError, json.JSONDecodeError):
            conf = {}

        if plant_slot_has_finished_analysis(conf):
            reply = QtWidgets.QMessageBox.question(
                None,
                'Plant already analyzed',
                'A finished analysis already exists for this plant slot '
                '(same experiment, robot, camera, and plant number).\n\n'
                'Use "Repeat Analysis" or "Redo Analysis" to re-run.\n\n'
                'Continue anyway and create a new run?',
                QtWidgets.QMessageBox.Cancel | QtWidgets.QMessageBox.Yes,
                QtWidgets.QMessageBox.Cancel,
            )
            if reply != QtWidgets.QMessageBox.Yes:
                return

        image_paths, seg_paths = getImages(conf)
        limit_frames = processing_limit_frame_count(conf)
        if limit_frames is not None:
            image_paths = image_paths[:limit_frames]
            seg_paths = seg_paths[:limit_frames]

        try:
            from analysis.utils.roi_selection import video_path_warnings
            warnings_text = video_path_warnings(conf)
        except Exception:
            warnings_text = ''
        if warnings_text:
            reply = QtWidgets.QMessageBox.question(
                None,
                'Video path warning',
                warnings_text + '\n\nContinue launching this analysis anyway?',
                QtWidgets.QMessageBox.Cancel | QtWidgets.QMessageBox.Yes,
                QtWidgets.QMessageBox.Cancel,
            )
            if reply != QtWidgets.QMessageBox.Yes:
                return

        bbox, seed = select_roi_and_seed(conf, image_paths, seg_paths)
        if seed is None:
            return

        with open(config_path, 'r') as f:
            conf = json.load(f)
        conf['bounding box'] = bbox
        conf['seed'] = seed
        with open(config_path, 'w') as f:
            json.dump(conf, f)

        pipeline_runner.run_analysis(self.projectField.text())
        
    def redoAnalysis(self):
        select_roi_and_seed = self._import_roi_selection()
        if select_roi_and_seed is None:
            return

        self.selected_plant = self.current_plant_path()
        metadata_path = os.path.join(self.selected_plant, "metadata.json")
        if not os.path.exists(metadata_path):
            return

        try:
            with open(metadata_path, 'r') as f:
                conf = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            QtWidgets.QMessageBox.critical(None, 'Error', f'Could not load metadata:\n{e}')
            return

        own_previous_roi = None
        old_bbox = conf.get('bounding box')
        if old_bbox and len(old_bbox) == 4:
            y1, y2, x1, x2 = old_bbox
            own_previous_roi = (x1, y1, x2, y2)

        conf.pop('bounding box', None)
        conf.pop('seed', None)

        image_paths, seg_paths = getImages(conf)
        limit_frames = processing_limit_frame_count(conf)
        if limit_frames is not None:
            image_paths = image_paths[:limit_frames]
            seg_paths = seg_paths[:limit_frames]

        bbox, seed = select_roi_and_seed(
            conf, image_paths, seg_paths, own_previous_roi=own_previous_roi,
        )
        if seed is None:
            return

        conf['bounding box'] = bbox
        conf['seed'] = seed
        with open(metadata_path, 'w') as f:
            json.dump(conf, f)

        pipeline_runner.run_analysis_config(metadata_path)

    def repeatAnalysis(self):
        selected_rows = self.table.selectionModel().selectedRows()

        if not selected_rows:
            return

        selected_row = selected_rows[0].row()
        item = self.table.item(selected_row, 0)
        path = item.path

        metadata_path = os.path.join(path, "metadata.json")
        pipeline_runner.run_analysis_rerun(metadata_path)

    def reviewPlant(self):
        self.selected_plant = self.current_plant_path()
        path = self.selected_plant
        if not path or not os.path.exists(path):
            return

        try:
            import plant_viewer
            
            # Load data using the helper function
            images, segs, bbox, conf = plant_viewer.load_plant_data(path)
            
            # Create and show window
            # We attach it to 'self' so it doesn't get garbage collected
            self.review_window = plant_viewer.ChronoViewWindow(images, segs, bbox, conf, parent=None)
            self.review_window.show()
            
        except FileNotFoundError as e:
            QtWidgets.QMessageBox.warning(self, "Error", str(e))
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to open review tool:\n{str(e)}")

    def preview(self):
        # 1. Save fields
        self.saveFieldsIntoJson()
        
        # 2. Validate folder
        video_folder = self.videoField.text()
        if not video_folder or not os.path.exists(video_folder):
            QtWidgets.QMessageBox.warning(None, 'Error', 'Please specify a valid video folder.')
            return
            
        # 3. Load Config
        config_path = os.path.join(self.projectField.text(), "project_config.json")
        try:
            with open(config_path, 'r') as f:
                conf = json.load(f)
        except Exception as e:
            QtWidgets.QMessageBox.critical(None, 'Error', f'Could not load config:\n{e}')
            return

        # 4. Load Images & Launch
        try:
            import plant_viewer            
            images, segFiles = getImages(conf)
            
            if not images:
                QtWidgets.QMessageBox.warning(None, 'Error', 'No images found.')
                return

            # Launch Window (BBox=None for full image)
            self.preview_window = plant_viewer.ChronoViewWindow(images, segFiles, None, conf, parent=None)
            self.preview_window.show()
            
        except Exception as e:
            QtWidgets.QMessageBox.critical(None, 'Error', f'Failed to launch preview:\n{e}')
        
    def _period_conf(self):
        return {
            'timeGroups': getattr(self, 'timeGroups', []),
            'timePeriodSources': getattr(self, 'timePeriodSources', []),
        }

    def _ensure_analysis_period(self):
        folder = self.projectField.text().strip()
        if not folder:
            QtWidgets.QMessageBox.warning(
                self, 'Error', 'Please select a project directory first.',
            )
            return False
        if analysis_period_is_current(self._period_conf(), folder):
            return True
        reply = QtWidgets.QMessageBox.question(
            self,
            'Analysis period required',
            PERIOD_GATE_MESSAGE + '\n\nOpen Set analysis period now?',
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.Yes,
        )
        if reply != QtWidgets.QMessageBox.Yes:
            return False
        self.open_time_window_dialog()
        if analysis_period_is_current(self._period_conf(), folder):
            return True
        QtWidgets.QMessageBox.warning(
            self,
            'Analysis period required',
            'Process all plants was cancelled. '
            'Set an analysis period for this set of plants first.',
        )
        return False

    def PostProcess(self):
        if not self._ensure_analysis_period():
            return
        self.saveFieldsIntoJson()
        pipeline_runner.run_postprocess(self.projectField.text())
    
    def report(self):
        self.saveFieldsIntoJson()
        pipeline_runner.run_report(self.projectField.text())

    def syncProjectFolderField(self):
        projectFolder = self.projectField.text()
        projectFolder2 = self.reportProjectField.text()

        if self.central_widget.sender() == self.projectField:
            self.reportProjectField.setText(projectFolder)
        elif self.central_widget.sender() == self.reportProjectField:
            self.projectField.setText(projectFolder2)

        self._on_project_path_maybe_changed()
    
    def syncCaptureIntervalField(self):
        captureInterval = self.captureIntervalField.text()
        captureInterval2 = self.reportCaptureIntervalField.text()

        if self.central_widget.sender() == self.captureIntervalField:
            self.reportCaptureIntervalField.setText(captureInterval)
        elif self.central_widget.sender() == self.reportCaptureIntervalField:
            self.captureIntervalField.setText(captureInterval2)

    def syncProcessingLimitField(self):
        processingLimit = self.processingLimitField.text()
        processingLimit2 = self.reportProcessingLimitField.text()

        if self.central_widget.sender() == self.processingLimitField:
            self.reportProcessingLimitField.setText(processingLimit)
        elif self.central_widget.sender() == self.reportProcessingLimitField:
            self.processingLimitField.setText(processingLimit2)
                    
    def setupUi(self, chrono_root_analysis):
        chrono_root_analysis.setObjectName("ChronoRootAnalysis")
        chrono_root_analysis.resize(WINDOW_WIDTH, WINDOW_HEIGHT)
        self.central_widget = QtWidgets.QWidget(chrono_root_analysis)
        self.central_widget.setObjectName("centralwidget")

        self.stats_config_dialog = StatsConfigDialog(self)
        
        self.setup_tabs()
        self.setup_tab1_elements()
        self.setup_tab2_elements()
        self.setup_tab3_elements()
        self.setup_tab4_elements()
        self.setup_tab5_elements()
        
        self.tab6 = QtWidgets.QWidget()
        self.tab6.setObjectName("tab6")
        self.tab_widget.addTab(self.tab6, "About")
        self.setup_tab6_elements()

        self.stats_config_dialog.register_on_host(self)
        self.setup_field_validation()
        self.set_default_parameters()

        chrono_root_analysis.setCentralWidget(self.central_widget)
        self.statusbar = QtWidgets.QStatusBar(chrono_root_analysis)
        self.statusbar.setObjectName("statusbar")
        chrono_root_analysis.setStatusBar(self.statusbar)

        self._apply_window_settings(chrono_root_analysis)
        self.tab_widget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(chrono_root_analysis)

    def _apply_window_settings(self, window):
        window.setWindowTitle("ChronoRootAnalysis")
        fixed_size = QtCore.QSize(810, WINDOW_HEIGHT)
        window.setMinimumSize(fixed_size)
        window.setMaximumSize(fixed_size)

    def setup_tabs(self):
        self.tab_widget = QtWidgets.QTabWidget(self.central_widget)
        self.tab_widget.setGeometry(QtCore.QRect(0, 0, WINDOW_WIDTH, TAB_HEIGHT))
        self.tab_widget.currentChanged.connect(self.handle_tab_change)
        
        font = QtGui.QFont()
        font.setPointSize(9)
        self.tab_widget.setFont(font)
        self.tab_widget.setObjectName("tabWidget")
        
        return

    def read_config_from_file(self):
        options = QtWidgets.QFileDialog.Options() | QtWidgets.QFileDialog.DontUseNativeDialog
        file_filter = "JSON Files (*.json);;All Files (*)"
        json_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            None, "Select Configuration File", "", file_filter, options=options)

        if not json_path:
            return

        self.config_store.apply_file(self, json_path)


    def setup_tab1_elements(self):

        self.tab1 = QtWidgets.QWidget()
        self.tab1.setObjectName("tab1")
        self.tab_widget.addTab(self.tab1, "Plant Analysis")

        self.plantAnalysisSectionLabel = QtWidgets.QLabel(self.tab1)
        self.plantAnalysisSectionLabel.setGeometry(QtCore.QRect(10, 10, 541, 31))
        self.plantAnalysisSectionLabel.setObjectName("plantAnalysisSectionLabel")
        self.plantAnalysisSectionLabel.setText(
            "<html><head/><body><p><span style=\" font-size:10pt; font-weight:600;\">"
            "Individual plant root analysis</span></p></body></html>")

        self.loadProject = QtWidgets.QPushButton(self.tab1)
        self.loadProject.setGeometry(QtCore.QRect(10, 50, 161, 31))
        self.loadProject.setObjectName("loadProject")
        self.loadProject.setText("Select Project Folder")
        self.loadProject.setToolTip(
            "Select the primary directory where experiment results and data are organized.")
        self.loadProject.clicked.connect(lambda: self.projectField.setText(self.openFileNameDialog()))

        self.projectField = QtWidgets.QLineEdit(self.tab1)
        self.projectField.setGeometry(QtCore.QRect(190, 50, 441, 31))
        self.projectField.setObjectName("projectField")
        self.projectField.textChanged.connect(self.syncProjectFolderField)

        self.loadVideo = QtWidgets.QPushButton(self.tab1)
        self.loadVideo.setGeometry(QtCore.QRect(10, 100, 161, 31))
        self.loadVideo.setObjectName("loadVideo")
        self.loadVideo.setText("Select Video Folder")
        self.loadVideo.setToolTip(
            "Select the folder containing the segmented image sequence for analysis. \n"
            "Folder should contain a Segmentation/ subfolder with images. \n"
            "Images can be stored somewhere else as long their path is saved within the "
            "segmentation_metadata.json file and has not been moved.")
        self.loadVideo.clicked.connect(lambda: self.videoField.setText(self.openFileNameDialog()))

        self.videoField = QtWidgets.QLineEdit(self.tab1)
        self.videoField.setGeometry(QtCore.QRect(190, 100, 441, 31))
        self.videoField.setObjectName("videoField")

        self.rpiModuleLabel = QtWidgets.QLabel(self.tab1)
        self.rpiModuleLabel.setGeometry(QtCore.QRect(10, 150, 171, 31))
        self.rpiModuleLabel.setObjectName("rpiModuleLabel")
        self.rpiModuleLabel.setText("Raspberry Module Number")

        self.rpiField = QtWidgets.QLineEdit(self.tab1)
        self.rpiField.setGeometry(QtCore.QRect(190, 150, 81, 31))
        self.rpiField.setObjectName("rpiField")

        self.cameraNumberLabel = QtWidgets.QLabel(self.tab1)
        self.cameraNumberLabel.setGeometry(QtCore.QRect(290, 150, 151, 31))
        self.cameraNumberLabel.setObjectName("cameraNumberLabel")
        self.cameraNumberLabel.setText("Camera Number")

        self.cameraField = QtWidgets.QLineEdit(self.tab1)
        self.cameraField.setGeometry(QtCore.QRect(525, 150, 101, 31))
        self.cameraField.setObjectName("cameraField")

        self.plantNumberLabel = QtWidgets.QLabel(self.tab1)
        self.plantNumberLabel.setGeometry(QtCore.QRect(10, 200, 161, 31))
        self.plantNumberLabel.setObjectName("plantNumberLabel")
        self.plantNumberLabel.setText("Plant Number")

        self.plantField = QtWidgets.QLineEdit(self.tab1)
        self.plantField.setGeometry(QtCore.QRect(190, 200, 81, 31))
        self.plantField.setObjectName("plantField")

        self.genotypeLabel = QtWidgets.QLabel(self.tab1)
        self.genotypeLabel.setGeometry(QtCore.QRect(290, 200, 230, 31))
        self.genotypeLabel.setObjectName("genotypeLabel")
        self.genotypeLabel.setText("Identifier (Genotype, Treatment, etc.)")

        self.plantIdentifier = QtWidgets.QLineEdit(self.tab1)
        self.plantIdentifier.setGeometry(QtCore.QRect(525, 200, 101, 31))
        self.plantIdentifier.setObjectName("plantIdentifier")

        self.plateConditionLabel = QtWidgets.QLabel(self.tab1)
        self.plateConditionLabel.setGeometry(QtCore.QRect(10, 250, 151, 31))
        self.plateConditionLabel.setObjectName("plateConditionLabel")
        self.plateConditionLabel.setText("Plate Growth Condition")

        self.plateConditionName = QtWidgets.QLineEdit(self.tab1)
        self.plateConditionName.setGeometry(QtCore.QRect(190, 250, 121, 31))
        self.plateConditionName.setObjectName("plateConditionName")

        self.plateConditionHintLabel = QtWidgets.QLabel(self.tab1)
        self.plateConditionHintLabel.setGeometry(QtCore.QRect(330, 250, 271, 31))
        self.plateConditionHintLabel.setObjectName("plateConditionHintLabel")
        self.plateConditionHintLabel.setText("(Optional, e.g. \"Control\", \"Treatment\", etc.)")

        self.extraVariableNameLabel = QtWidgets.QLabel(self.tab1)
        self.extraVariableNameLabel.setGeometry(QtCore.QRect(10, 300, 120, 31))
        self.extraVariableNameLabel.setObjectName("extraVariableNameLabel")
        self.extraVariableNameLabel.setText("Extra Variable")

        self.extraField = QtWidgets.QLineEdit(self.tab1)
        self.extraField.setGeometry(QtCore.QRect(190, 300, 121, 31))
        self.extraField.setObjectName("extraField")

        self.extraVariableHintLabel = QtWidgets.QLabel(self.tab1)
        self.extraVariableHintLabel.setGeometry(QtCore.QRect(330, 300, 271, 31))
        self.extraVariableHintLabel.setObjectName("extraVariableHintLabel")
        self.extraVariableHintLabel.setText("(Optional value, e.g. Run 1, Run 2.)")

        self.postprocessParametersSeparator = QtWidgets.QFrame(self.tab1)
        self.postprocessParametersSeparator.setGeometry(QtCore.QRect(0, 350, 651, 16))
        self.postprocessParametersSeparator.setFrameShape(QtWidgets.QFrame.HLine)
        self.postprocessParametersSeparator.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.postprocessParametersSeparator.setObjectName("postprocessParametersSeparator")
        self.postprocessParametersSeparator.lower()

        self.postprocessParametersSectionLabel = QtWidgets.QLabel(self.tab1)
        self.postprocessParametersSectionLabel.setGeometry(QtCore.QRect(10, 360, 541, 31))
        self.postprocessParametersSectionLabel.setObjectName("postprocessParametersSectionLabel")
        self.postprocessParametersSectionLabel.setText(
            "<html><head/><body><p><span style=\" font-size:10pt; font-weight:600;\">"
            "Analysis and postprocessing parameters</span></p></body></html>")

        self.saveImagesButton = QtWidgets.QCheckBox(self.tab1)
        self.saveImagesButton.setGeometry(QtCore.QRect(10, 390, 161, 31))
        self.saveImagesButton.setObjectName("saveImagesButton")
        self.saveImagesButton.setText("Save Cropped Images")
        self.saveImagesButton.setToolTip(
            "Save individual plant crops; required for creating growth time-lapse videos.")

        self.videoHasQRbutton = QtWidgets.QCheckBox(self.tab1)
        self.videoHasQRbutton.setGeometry(QtCore.QRect(10, 420, 161, 31))
        self.videoHasQRbutton.setObjectName("videoHasQRbutton")
        self.videoHasQRbutton.setText("Video has QR codes")
        self.videoHasQRbutton.setToolTip(
            "Enable automatic scale detection using the 1-cm QR code in the images.")

        self.manual_calib_widget = QtWidgets.QWidget(self.tab1)
        self.manual_calib_widget.setGeometry(QtCore.QRect(210, 385, 380, 70))

        self.manual_calib_label = QtWidgets.QLabel(self.manual_calib_widget)
        self.manual_calib_label.setGeometry(QtCore.QRect(0, 0, 200, 31))
        self.manual_calib_label.setText("Manual Calibration Parameters:")

        self.known_dist_label = QtWidgets.QLabel(self.manual_calib_widget)
        self.known_dist_label.setGeometry(QtCore.QRect(220, 0, 100, 31))
        self.known_dist_label.setText("Known (mm):")

        self.knownDistanceField = QtWidgets.QLineEdit(self.manual_calib_widget)
        self.knownDistanceField.setGeometry(QtCore.QRect(320, 0, 51, 31))
        self.knownDistanceField.setPlaceholderText("10")
        self.knownDistanceField.setObjectName("knownDistanceField")

        self.pixel_dist_label = QtWidgets.QLabel(self.manual_calib_widget)
        self.pixel_dist_label.setGeometry(QtCore.QRect(220, 31, 70, 31))
        self.pixel_dist_label.setText("Pixels:")

        self.pixelDistanceField = QtWidgets.QLineEdit(self.manual_calib_widget)
        self.pixelDistanceField.setGeometry(QtCore.QRect(320, 32, 51, 31))
        self.pixelDistanceField.setPlaceholderText("240")
        self.pixelDistanceField.setObjectName("pixelDistanceField")

        self.calibrateBtn = QtWidgets.QPushButton(self.manual_calib_widget)
        self.calibrateBtn.setGeometry(QtCore.QRect(0, 30, 180, 31))
        self.calibrateBtn.setText("Open Calibration Helper")
        self.calibrateBtn.setToolTip(
            "Manually define the physical scale by measuring a known distance in the image.")
        self.calibrateBtn.clicked.connect(self.open_calibration_helper)

        self.processingLimitLabel = QtWidgets.QLabel(self.tab1)
        self.processingLimitLabel.setGeometry(QtCore.QRect(10, 460, 161, 31))
        self.processingLimitLabel.setObjectName("processingLimitLabel")
        self.processingLimitLabel.setText(
            "<html><head/><body><p>Processing limit</p></body></html>")

        self.processingLimitField = QtWidgets.QLineEdit(self.tab1)
        self.processingLimitField.setGeometry(QtCore.QRect(190, 460, 51, 31))
        self.processingLimitField.setObjectName("processingLimitField")
        self.processingLimitField.textChanged.connect(self.syncProcessingLimitField)

        self.processingLimitHintLabel = QtWidgets.QLabel(self.tab1)
        self.processingLimitHintLabel.setGeometry(QtCore.QRect(260, 460, 261, 31))
        self.processingLimitHintLabel.setObjectName("processingLimitHintLabel")
        self.processingLimitHintLabel.setText("(in days, 0 means no limit)")

        self.captureIntervalLabel = QtWidgets.QLabel(self.tab1)
        self.captureIntervalLabel.setGeometry(QtCore.QRect(10, 510, 161, 31))
        self.captureIntervalLabel.setObjectName("captureIntervalLabel")
        self.captureIntervalLabel.setText(
            "<html><head/><body><p>Capture interval</p></body></html>")

        self.captureIntervalField = QtWidgets.QLineEdit(self.tab1)
        self.captureIntervalField.setGeometry(QtCore.QRect(190, 510, 51, 31))
        self.captureIntervalField.setObjectName("captureIntervalField")
        self.captureIntervalField.textChanged.connect(self.syncCaptureIntervalField)

        self.captureIntervalHintLabel = QtWidgets.QLabel(self.tab1)
        self.captureIntervalHintLabel.setGeometry(QtCore.QRect(260, 510, 261, 31))
        self.captureIntervalHintLabel.setObjectName("captureIntervalHintLabel")
        self.captureIntervalHintLabel.setText("(in minutes, usually 15 minutes)")

        self.emergenceDistanceLabel = QtWidgets.QLabel(self.tab1)
        self.emergenceDistanceLabel.setGeometry(QtCore.QRect(10, 560, 161, 31))
        self.emergenceDistanceLabel.setObjectName("emergenceDistanceLabel")
        self.emergenceDistanceLabel.setText(
            "<html><head/><body><p>Emergence distance</p></body></html>")

        self.analysisEmergenceDistanceField = QtWidgets.QLineEdit(self.tab1)
        self.analysisEmergenceDistanceField.setGeometry(QtCore.QRect(190, 560, 51, 31))
        self.analysisEmergenceDistanceField.setObjectName("analysisEmergenceDistanceField")

        self.emergenceDistanceExp = QtWidgets.QLabel(self.tab1)
        self.emergenceDistanceExp.setGeometry(QtCore.QRect(260, 560, 261, 31))
        self.emergenceDistanceExp.setObjectName("emergenceDistanceExp")
        self.emergenceDistanceExp.setText("(in millimeters, default: 2 mm)")

        self.actionButtonsSeparator = QtWidgets.QFrame(self.tab1)
        self.actionButtonsSeparator.setGeometry(QtCore.QRect(640, -30, 20, TAB_HEIGHT))
        self.actionButtonsSeparator.setFrameShape(QtWidgets.QFrame.VLine)
        self.actionButtonsSeparator.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.actionButtonsSeparator.setObjectName("actionButtonsSeparator")
        self.actionButtonsSeparator.lower()

        self.saveButton = QtWidgets.QPushButton(self.tab1)
        self.saveButton.setGeometry(QtCore.QRect(660, 0, 141, 81))
        self.saveButton.setObjectName("saveButton")
        self.saveButton.setText("Save")
        self.saveButton.setToolTip("Save the current parameters to the project configuration.")
        self.saveButton.clicked.connect(self.saveFieldsIntoJson)

        self.previewAnalysisButton = QtWidgets.QPushButton(self.tab1)
        self.previewAnalysisButton.setGeometry(QtCore.QRect(660, 100, 141, 81))
        self.previewAnalysisButton.setObjectName("previewAnalysisButton")
        self.previewAnalysisButton.setText("Preview video")
        self.previewAnalysisButton.setToolTip(
            "<b>Visual Inspection:</b><br>"
            "Open a viewer to check image quality and verify that the root segmentation is accurate.")
        self.previewAnalysisButton.clicked.connect(self.preview)

        self.analysisButton = QtWidgets.QPushButton(self.tab1)
        self.analysisButton.setGeometry(QtCore.QRect(660, 200, 141, 81))
        self.analysisButton.setObjectName("analysisButton")
        self.analysisButton.setText("Analyze Plant")
        self.analysisButton.setToolTip(
            "<b>Initiate Analysis:</b><br>"
            "1. Define the plant analysis area (ROI).<br>"
            "2. Mark the <b>Root Starting Point</b>.<br>"
            "3. Process the growth tracking graph.")
        self.analysisButton.clicked.connect(self.analysis)

        self.tab1TimePeriodButton = QtWidgets.QPushButton(self.tab1)
        self.tab1TimePeriodButton.setGeometry(QtCore.QRect(660, 300, 141, 81))
        self.tab1TimePeriodButton.setObjectName("tab1TimePeriodButton")
        self.tab1TimePeriodButton.setText("Set analysis\nperiod")
        self.tab1TimePeriodButton.setToolTip(
            "Set duration, per-group start/t0, and clock ticks "
            "before processing all plants."
        )
        self.tab1TimePeriodButton.clicked.connect(self.open_time_window_dialog)

        self.PostProcessButton = QtWidgets.QPushButton(self.tab1)
        self.PostProcessButton.setGeometry(QtCore.QRect(660, 400, 141, 81))
        self.PostProcessButton.setObjectName("PostProcessButton")
        self.PostProcessButton.setText("Process\nall plants")
        self.PostProcessButton.setToolTip(
            "Finalize and calculate statistics for all analyzed plants in this project.")
        self.PostProcessButton.clicked.connect(self.PostProcess)

        self.loadConfigFileButton = QtWidgets.QPushButton(self.tab1)
        self.loadConfigFileButton.setGeometry(QtCore.QRect(660, 500, 141, 40))
        self.loadConfigFileButton.setObjectName("loadConfigFileButton")
        self.loadConfigFileButton.setText("Load from file")
        self.loadConfigFileButton.setToolTip("Import settings from an existing configuration file.")
        self.loadConfigFileButton.clicked.connect(self.read_config_from_file)

        self.loadLastConfigButton = QtWidgets.QPushButton(self.tab1)
        self.loadLastConfigButton.setGeometry(QtCore.QRect(660, 540, 141, 41))
        self.loadLastConfigButton.setObjectName("loadLastConfigButton")
        self.loadLastConfigButton.setText("Load previous conf")
        self.loadLastConfigButton.setToolTip("Restore the most recently used settings.")
        self.loadLastConfigButton.clicked.connect(self.loadJsonIntoFields)

        self.videoHasQRbutton.stateChanged.connect(self.toggle_calibration_mode)
        self.toggle_calibration_mode()

        tab1_interactive = [
            self.loadProject, self.projectField, self.loadVideo, self.videoField,
            self.rpiField, self.cameraField, self.plantField, self.plantIdentifier,
            self.plateConditionName, self.extraField, 
            self.saveImagesButton, self.videoHasQRbutton, self.manual_calib_widget,
            self.processingLimitField, self.captureIntervalField, self.analysisEmergenceDistanceField,
            self.saveButton, self.previewAnalysisButton, self.analysisButton,
            self.tab1TimePeriodButton, self.PostProcessButton,
            self.loadConfigFileButton, self.loadLastConfigButton,
        ]
        for widget in tab1_interactive:
            widget.raise_()

        return
    
    def toggle_calibration_mode(self):
        """Toggle between QR and manual calibration modes"""
        has_qr = self.videoHasQRbutton.isChecked()
        self.manual_calib_widget.setVisible(not has_qr)

    def open_calibration_helper(self):
        """Opens calibration helper window"""
        if not self.videoField.text():
            QtWidgets.QMessageBox.warning(None, 'Error', 'Please select a video directory first!')
            return

        try:
            import calibration_helper
            self.calibration_window = calibration_helper.CalibrationHelper(self.videoField.text())
            self.calibration_window.distance_measured.connect(
                lambda pixels: self.pixelDistanceField.setText(str(pixels))
            )
            self.calibration_window.show()
        except Exception as e:
            QtWidgets.QMessageBox.critical(None, "Error", f"Failed to launch calibration helper:\n{e}")

    def setup_tab2_elements(self):
        # Create the table
        self.table = QTableWidget()
        self.table.setColumnCount(9)
        self.table.setHorizontalHeaderLabels([
            "Experiment", "Raspberry", "Camera", "Plant Number",
            "Plate Condition", "Extra Variable",
            "Error Rate", "Status", "Finish Date",
        ])
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self.show_overview_context_menu)

        # Enable sorting
        self.table.setSortingEnabled(True)

        # Create the refresh button
        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.setToolTip(
            "Update the table to show the latest analysis progress and error rates.")
        self.refresh_button.clicked.connect(self.refresh_table)

        self.rerun_analysis_button_tab2 = QPushButton("Repeat Analysis")
        self.rerun_analysis_button_tab2.setToolTip(
            "<b>Quick Re-run:</b> Repeat tracking using the existing ROI and Root Starting Point.")
        self.rerun_analysis_button_tab2.clicked.connect(self.repeatAnalysis)

        self.open_path_button_tab2 = QPushButton("Open Path")
        self.open_path_button_tab2.setToolTip(
            "Open the folder containing the data for the selected plant.")
        self.open_path_button_tab2.clicked.connect(self.open_selected_path)

        self.remove_path_button = QPushButton("Remove Plant")
        self.remove_path_button.setToolTip(
            "Move the selected plant to the 'Removed' folder. This hides it from reports "
            "without deleting the data.")
        self.remove_path_button.clicked.connect(self.remove_selected_path)

        self.remap_identifiers_button = QPushButton("Remap Identifiers")
        self.remap_identifiers_button.setToolTip(
            "Rename or merge plant identifiers (Experiment) across the project. "
            "Useful for fixing typos or assigning genotype names after short codes.")
        self.remap_identifiers_button.clicked.connect(self.open_remap_identifiers_dialog)

        # Set up the layout
        buttons_layout = QHBoxLayout()
        buttons_layout.addWidget(self.refresh_button)
        buttons_layout.addWidget(self.open_path_button_tab2)
        buttons_layout.addWidget(self.remove_path_button)
        buttons_layout.addWidget(self.remap_identifiers_button)
        buttons_layout.addWidget(self.rerun_analysis_button_tab2)

        # Set up the main layout
        layout = QVBoxLayout()
        layout.addWidget(self.table)
        layout.addLayout(buttons_layout)

        # Create and set up the new tab
        self.tab2 = QtWidgets.QWidget()
        self.tab2.setLayout(layout)
        self.tab_widget.addTab(self.tab2, "Analysis Overview")

    def show_overview_context_menu(self, pos):
        index = self.table.indexAt(pos)
        if not index.isValid():
            return
        row = index.row()
        item = self.table.item(row, 0)
        if item is None:
            return

        menu = QtWidgets.QMenu(self)
        action = menu.addAction("See in Plant Overlay")
        chosen = menu.exec_(self.table.viewport().mapToGlobal(pos))
        if chosen != action:
            return

        path = getattr(item, 'path', None)
        if not path:
            return
        self.see_plant_in_overlay(path)

    def see_plant_in_overlay(self, path):
        """Select a plant in the Overlay combo and switch to that tab."""
        if not hasattr(self, 'plant_dropdown'):
            return
        idx = self.plant_dropdown.findData(path)
        if idx < 0:
            QtWidgets.QMessageBox.information(
                self,
                "Plant Overlay",
                "Could not find this plant in the Overlay list. Try Refresh first.",
            )
            return
        self.plant_dropdown.setCurrentIndex(idx)
        self.selected_plant = self.current_plant_path()
        # Plant Overlay is tab index 2
        self.tab_widget.setCurrentIndex(2)

    def setup_tab3_elements(self):
        # Image panels: expand and center content so layout height stays stable
        self.image_label1 = AspectRatioLabel()
        self.image_label2 = AspectRatioLabel()
        for label in (self.image_label1, self.image_label2):
            label.setAlignment(QtCore.Qt.AlignCenter)
            label.setSizePolicy(
                QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding
            )
            label.setMinimumSize(120, 120)

        # Create the checkbox
        self.overlay_checkbox = QCheckBox("Overlay Image")
        self.overlay_checkbox.setToolTip(
            "Show or hide the color-coded tracking mask over the plant image.")

        self.plant_dropdown = QComboBox()

        self.prev_plant_button = QPushButton("Previous")
        self.prev_plant_button.setToolTip("Show the previous plant in the list.")
        self.prev_plant_button.clicked.connect(self.show_previous_plant)

        self.next_plant_button = QPushButton("Next")
        self.next_plant_button.setToolTip("Show the next plant in the list.")
        self.next_plant_button.clicked.connect(self.show_next_plant)

        self.refresh_button_tab3 = QPushButton("Refresh")
        self.refresh_button_tab3.setToolTip(
            "Refresh the list of plants available for visual inspection.")
        self.refresh_button_tab3.clicked.connect(self.refresh_table)

        self.rerun_analysis_button_tab3 = QPushButton("Redo Analysis")
        self.rerun_analysis_button_tab3.setToolTip(
            "<b>Manual Re-run:</b> Restart the analysis to choose a new ROI or Root Starting Point.")
        self.rerun_analysis_button_tab3.clicked.connect(self.redoAnalysis)

        self.remove_path_button_tab3 = QPushButton("Remove Plant")
        self.remove_path_button_tab3.setToolTip(
            "Move the current plant to the 'Removed' folder if it is unsuitable for reporting.")
        self.remove_path_button_tab3.clicked.connect(self.remove_selected_plant)

        # Connect signals
        self.plant_dropdown.currentIndexChanged.connect(self.update_image_labels)
        self.overlay_checkbox.stateChanged.connect(self.update_image_labels)

        self.reviewButton = QPushButton("View full sequence")
        self.reviewButton.setToolTip(
            "Open the sequence viewer to inspect the development of this root system.")
        self.reviewButton.clicked.connect(self.reviewPlant)

        self.open_path_button_tab3 = QPushButton("Open Folder")
        self.open_path_button_tab3.setToolTip(
            "Open the results folder for the plant currently being viewed.")
        self.open_path_button_tab3.clicked.connect(self.open_selected_path_tab3)

        # Navigation row (fixed at bottom, above action buttons — never moves with image height)
        controls_layout = QHBoxLayout()
        controls_layout.addWidget(self.overlay_checkbox)
        controls_layout.addWidget(self.prev_plant_button)
        controls_layout.addWidget(self.plant_dropdown, 1)
        controls_layout.addWidget(self.next_plant_button)

        # Action row — same role as Analysis Overview bottom buttons
        controls_layout2 = QHBoxLayout()
        controls_layout2.addWidget(self.refresh_button_tab3)
        controls_layout2.addWidget(self.open_path_button_tab3)
        controls_layout2.addWidget(self.reviewButton)
        controls_layout2.addWidget(self.rerun_analysis_button_tab3)
        controls_layout2.addWidget(self.remove_path_button_tab3)

        # Two equal columns; each image is centered in its cell
        images_row = QHBoxLayout()
        images_row.setContentsMargins(0, 0, 0, 0)
        images_row.setSpacing(8)
        for label in (self.image_label1, self.image_label2):
            cell = QtWidgets.QWidget()
            cell_layout = QVBoxLayout(cell)
            cell_layout.setContentsMargins(0, 0, 0, 0)
            cell_layout.addWidget(label)
            images_row.addWidget(cell, 1)

        # Stretch images; pin both control rows to the bottom of the tab
        bigLayout = QVBoxLayout()
        bigLayout.setContentsMargins(6, 6, 6, 6)
        bigLayout.setSpacing(6)
        bigLayout.addLayout(images_row, 1)
        bigLayout.addLayout(controls_layout, 0)
        bigLayout.addLayout(controls_layout2, 0)

        self.tab3 = QtWidgets.QWidget()
        self.tab3.setLayout(bigLayout)
        self.tab_widget.addTab(self.tab3, "Plant Overlay")

        self.update_image_labels()

    def show_previous_plant(self):
        if not hasattr(self, 'plant_dropdown') or self.plant_dropdown.count() == 0:
            return
        idx = self.plant_dropdown.currentIndex()
        if idx > 0:
            self.plant_dropdown.setCurrentIndex(idx - 1)

    def show_next_plant(self):
        if not hasattr(self, 'plant_dropdown') or self.plant_dropdown.count() == 0:
            return
        idx = self.plant_dropdown.currentIndex()
        if idx < self.plant_dropdown.count() - 1:
            self.plant_dropdown.setCurrentIndex(idx + 1)

    def setup_tab4_elements(self):
        # Spacing: ROW=31px per control row; SEP=41px separator frame; GAP=10px after separator.
        self.tab4 = QtWidgets.QWidget()
        self.tab4.setObjectName("tab4")
        self.tab_widget.addTab(self.tab4, "Generate Report")

        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)

        self.reportSelectProjectButton = QtWidgets.QPushButton(self.tab4)
        self.reportSelectProjectButton.setGeometry(QtCore.QRect(10, 10, 161, 31))
        self.reportSelectProjectButton.setObjectName("reportSelectProjectButton")
        self.reportSelectProjectButton.setText("Select Project Folder")
        self.reportSelectProjectButton.clicked.connect(lambda: self.projectField.setText(self.openFileNameDialog()))

        self.reportProjectField = QtWidgets.QLineEdit(self.tab4)
        self.reportProjectField.setGeometry(QtCore.QRect(190, 10, 441, 31))
        self.reportProjectField.setObjectName("reportProjectField")
        self.reportProjectField.textChanged.connect(self.syncProjectFolderField)

        self.projectSectionSeparator = QtWidgets.QFrame(self.tab4)
        self.projectSectionSeparator.setGeometry(QtCore.QRect(0, 40, 891, 41))
        self.projectSectionSeparator.setFrameShape(QtWidgets.QFrame.HLine)
        self.projectSectionSeparator.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.projectSectionSeparator.setObjectName("projectSectionSeparator")
        self.projectSectionSeparator.lower()

        self.doFPCA = QtWidgets.QCheckBox(self.tab4)
        self.doFPCA.setGeometry(QtCore.QRect(10, 70, 311, 31))
        self.doFPCA.setFont(font)
        self.doFPCA.setObjectName("doFPCA")
        self.doFPCA.setText("Perform Functional PCA on time series")

        self.normFPCA = QtWidgets.QCheckBox(self.tab4)
        self.normFPCA.setGeometry(QtCore.QRect(340, 70, 221, 31))
        self.normFPCA.setObjectName("normFPCA")
        self.normFPCA.setText("Normalize FPCA Boxplots")

        self.numComponentsFPCAText = QtWidgets.QLabel(self.tab4)
        self.numComponentsFPCAText.setGeometry(QtCore.QRect(570, 70, 200, 31))
        self.numComponentsFPCAText.setObjectName("numComponentsFPCAText")
        self.numComponentsFPCAText.setText("Number of components")

        self.numComponentsFPCAField = QtWidgets.QLineEdit(self.tab4)
        self.numComponentsFPCAField.setGeometry(QtCore.QRect(730, 70, 51, 31))
        self.numComponentsFPCAField.setObjectName("numComponentsFPCAField")

        self.fpcaSectionSeparator = QtWidgets.QFrame(self.tab4)
        self.fpcaSectionSeparator.setGeometry(QtCore.QRect(-40, 90, 891, 41))
        self.fpcaSectionSeparator.setFrameShape(QtWidgets.QFrame.HLine)
        self.fpcaSectionSeparator.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.fpcaSectionSeparator.setObjectName("fpcaSectionSeparator")
        self.fpcaSectionSeparator.lower()

        self.doConvex = QtWidgets.QCheckBox(self.tab4)
        self.doConvex.setGeometry(QtCore.QRect(10, 120, 201, 31))
        self.doConvex.setFont(font)
        self.doConvex.setObjectName("doConvex")
        self.doConvex.setText("Do Convex hull analysis")

        self.saveImagesConvex = QtWidgets.QCheckBox(self.tab4)
        self.saveImagesConvex.setGeometry(QtCore.QRect(370, 120, 311, 31))
        self.saveImagesConvex.setObjectName("saveImagesConvex")
        self.saveImagesConvex.setText("Save images for each snapshot")

        self.daysConvexLabel = QtWidgets.QLabel(self.tab4)
        self.daysConvexLabel.setGeometry(QtCore.QRect(10, 160, 141, 31))
        self.daysConvexLabel.setObjectName("daysConvexLabel")
        self.daysConvexLabel.setText("Snapshot hours")

        self.daysConvexField = QtWidgets.QLineEdit(self.tab4)
        self.daysConvexField.setGeometry(QtCore.QRect(150, 160, 130, 31))
        self.daysConvexField.setObjectName("daysConvexField")
        self.daysConvexField.setPlaceholderText("0, 24")

        self.daysConvexChooseButton = QtWidgets.QPushButton(self.tab4)
        self.daysConvexChooseButton.setGeometry(QtCore.QRect(285, 160, 70, 31))
        self.daysConvexChooseButton.setObjectName("daysConvexChooseButton")
        self.daysConvexChooseButton.setText("Choose")
        self.daysConvexChooseButton.setToolTip(
            "Pick snapshot hours by day, every X hours, or extra elapsed hours."
        )
        self.daysConvexChooseButton.clicked.connect(
            lambda: self.open_snapshot_hours_dialog(self.daysConvexField)
        )

        self.daysConvexText = QtWidgets.QLabel(self.tab4)
        self.daysConvexText.setGeometry(QtCore.QRect(360, 160, 341, 31))
        self.daysConvexText.setObjectName("daysConvexText")
        self.daysConvexText.setText("(elapsed hours, e.g. 0, 24)")

        self.convexSectionSeparator = QtWidgets.QFrame(self.tab4)
        self.convexSectionSeparator.setGeometry(QtCore.QRect(-40, 190, 891, 41))
        self.convexSectionSeparator.setFrameShape(QtWidgets.QFrame.HLine)
        self.convexSectionSeparator.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.convexSectionSeparator.setObjectName("convexSectionSeparator")
        self.convexSectionSeparator.lower()

        self.doFourier = QtWidgets.QCheckBox(self.tab4)
        self.doFourier.setGeometry(QtCore.QRect(10, 225, 451, 31))
        self.doFourier.setFont(font)
        self.doFourier.setObjectName("doFourier")
        self.doFourier.setText("Evaluate Growth Speeds and perform Fourier Analysis")

        self.fourierSectionSeparator = QtWidgets.QFrame(self.tab4)
        self.fourierSectionSeparator.setGeometry(QtCore.QRect(-70, 250, 961, 41))
        self.fourierSectionSeparator.setFrameShape(QtWidgets.QFrame.HLine)
        self.fourierSectionSeparator.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.fourierSectionSeparator.setObjectName("fourierSectionSeparator")
        self.fourierSectionSeparator.lower()

        self.doLateralAngles = QtWidgets.QCheckBox(self.tab4)
        self.doLateralAngles.setGeometry(QtCore.QRect(10, 285, 301, 31))
        self.doLateralAngles.setFont(font)
        self.doLateralAngles.setObjectName("doLateralAngles")
        self.doLateralAngles.setText("Do Lateral Root Angles Analysis")

        self.reportEmergenceDistanceLabel = QtWidgets.QLabel(self.tab4)
        self.reportEmergenceDistanceLabel.setGeometry(QtCore.QRect(370, 285, 131, 31))
        self.reportEmergenceDistanceLabel.setObjectName("reportEmergenceDistanceLabel")
        self.reportEmergenceDistanceLabel.setText("Emergence distance")

        self.reportEmergenceDistanceField = QtWidgets.QLineEdit(self.tab4)
        self.reportEmergenceDistanceField.setGeometry(QtCore.QRect(510, 285, 51, 31))
        self.reportEmergenceDistanceField.setObjectName("emergenceDistanceField")

        self.reportEmergenceDistanceHintLabel = QtWidgets.QLabel(self.tab4)
        self.reportEmergenceDistanceHintLabel.setGeometry(QtCore.QRect(570, 285, 261, 31))
        self.reportEmergenceDistanceHintLabel.setObjectName("reportEmergenceDistanceHintLabel")
        self.reportEmergenceDistanceHintLabel.setText("(in millimeters, default: 2 mm)")

        self.daysAnglesText = QtWidgets.QLabel(self.tab4)
        self.daysAnglesText.setGeometry(QtCore.QRect(10, 330, 141, 31))
        self.daysAnglesText.setObjectName("daysAnglesText")
        self.daysAnglesText.setText("Snapshot hours")

        self.daysAnglesField = QtWidgets.QLineEdit(self.tab4)
        self.daysAnglesField.setGeometry(QtCore.QRect(150, 330, 130, 31))
        self.daysAnglesField.setObjectName("daysAnglesField")
        self.daysAnglesField.setPlaceholderText("0, 24")

        self.daysAnglesChooseButton = QtWidgets.QPushButton(self.tab4)
        self.daysAnglesChooseButton.setGeometry(QtCore.QRect(285, 330, 70, 31))
        self.daysAnglesChooseButton.setObjectName("daysAnglesChooseButton")
        self.daysAnglesChooseButton.setText("Choose")
        self.daysAnglesChooseButton.setToolTip(
            "Pick snapshot hours by day, every X hours, or extra elapsed hours."
        )
        self.daysAnglesChooseButton.clicked.connect(
            lambda: self.open_snapshot_hours_dialog(self.daysAnglesField)
        )

        self.daysAnglesHint = QtWidgets.QLabel(self.tab4)
        self.daysAnglesHint.setGeometry(QtCore.QRect(360, 330, 341, 31))
        self.daysAnglesHint.setObjectName("daysAnglesHint")
        self.daysAnglesHint.setText("(elapsed hours, e.g. 0, 24)")

        self.anglesSectionSeparator = QtWidgets.QFrame(self.tab4)
        self.anglesSectionSeparator.setGeometry(QtCore.QRect(-30, 360, 961, 41))
        self.anglesSectionSeparator.setFrameShape(QtWidgets.QFrame.HLine)
        self.anglesSectionSeparator.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.anglesSectionSeparator.setObjectName("anglesSectionSeparator")
        self.anglesSectionSeparator.lower()

        self.reportProcessingLimitLabel = QtWidgets.QLabel(self.tab4)
        self.reportProcessingLimitLabel.setGeometry(QtCore.QRect(10, 400, 121, 31))
        self.reportProcessingLimitLabel.setObjectName("reportProcessingLimitLabel")
        self.reportProcessingLimitLabel.setText(
            "<html><head/><body><p>Processing limit</p></body></html>")

        self.reportProcessingLimitField = QtWidgets.QLineEdit(self.tab4)
        self.reportProcessingLimitField.setGeometry(QtCore.QRect(140, 400, 51, 31))
        self.reportProcessingLimitField.setObjectName("reportProcessingLimitField")
        self.reportProcessingLimitField.textChanged.connect(self.syncProcessingLimitField)

        self.reportMappingText = QtWidgets.QLabel(self.tab4)
        self.reportMappingText.setGeometry(QtCore.QRect(240, 400, 120, 31))
        self.reportMappingText.setText("Parameter mapping for plots")
        
        self.reportGenotypeAxisLabelText = QtWidgets.QLabel(self.tab4)
        self.reportGenotypeAxisLabelText.setGeometry(QtCore.QRect(490, 400, 120, 31))
        self.reportGenotypeAxisLabelText.setText("Identifier label")
        
        self.reportGenotypeAxisLabelField = QtWidgets.QLineEdit(self.tab4)
        self.reportGenotypeAxisLabelField.setGeometry(QtCore.QRect(630, 400, 120, 31))
        self.reportGenotypeAxisLabelField.setObjectName("reportGenotypeAxisLabelField")
        self.reportGenotypeAxisLabelField.setText("Genotype")

        self.reportPlateConditionAxisLabelText = QtWidgets.QLabel(self.tab4)
        self.reportPlateConditionAxisLabelText.setGeometry(QtCore.QRect(240, 450, 120, 31))
        self.reportPlateConditionAxisLabelText.setText("Plate label")
        
        self.reportPlateConditionAxisLabelField = QtWidgets.QLineEdit(self.tab4)
        self.reportPlateConditionAxisLabelField.setGeometry(QtCore.QRect(330, 450, 120, 31))
        self.reportPlateConditionAxisLabelField.setObjectName("reportPlateConditionAxisLabelField")
        self.reportPlateConditionAxisLabelField.setText("Plate condition")

        self.reportExtraVariableAxisLabelText = QtWidgets.QLabel(self.tab4)
        self.reportExtraVariableAxisLabelText.setGeometry(QtCore.QRect(490, 450, 120, 31))
        self.reportExtraVariableAxisLabelText.setText("Extra variable label")
        
        self.reportExtraVariableAxisLabelField = QtWidgets.QLineEdit(self.tab4)
        self.reportExtraVariableAxisLabelField.setGeometry(QtCore.QRect(630, 450, 120, 31))
        self.reportExtraVariableAxisLabelField.setObjectName("reportExtraVariableAxisLabelField")
        self.reportExtraVariableAxisLabelField.setText("Run")

        self.reportCaptureIntervalLabel = QtWidgets.QLabel(self.tab4)
        self.reportCaptureIntervalLabel.setGeometry(QtCore.QRect(10, 450, 111, 31))
        self.reportCaptureIntervalLabel.setObjectName("reportCaptureIntervalLabel")
        self.reportCaptureIntervalLabel.setText(
            "<html><head/><body><p>Capture interval</p></body></html>")

        self.reportCaptureIntervalField = QtWidgets.QLineEdit(self.tab4)
        self.reportCaptureIntervalField.setGeometry(QtCore.QRect(140, 450, 51, 31))
        self.reportCaptureIntervalField.setObjectName("reportCaptureIntervalField")
        self.reportCaptureIntervalField.textChanged.connect(self.syncCaptureIntervalField)

        self.reportTimeWindowsButton = QtWidgets.QPushButton(self.tab4)
        self.reportTimeWindowsButton.setGeometry(QtCore.QRect(20, 510, 131, 81))
        self.reportTimeWindowsButton.setObjectName("reportTimeWindowsButton")
        self.reportTimeWindowsButton.setText("Set analysis\nperiod")
        self.reportTimeWindowsButton.setToolTip(
            "Set analysis duration, per-group start/t0, and clock ticks."
        )
        self.reportTimeWindowsButton.clicked.connect(self.open_time_window_dialog)

        self.reportPostProcessButton = QtWidgets.QPushButton(self.tab4)
        self.reportPostProcessButton.setGeometry(QtCore.QRect(180, 510, 131, 81))
        self.reportPostProcessButton.setObjectName("reportPostProcessButton")
        self.reportPostProcessButton.setText("Process\nall plants")
        self.reportPostProcessButton.setToolTip(
            "Ensure all plant data is synchronized before generating final figures.")
        self.reportPostProcessButton.clicked.connect(self.PostProcess)
        
        self.reportConfigureStatsButton = QtWidgets.QPushButton(self.tab4)
        self.reportConfigureStatsButton.setGeometry(QtCore.QRect(340, 510, 131, 81))
        self.reportConfigureStatsButton.setObjectName("reportConfigureStatsButton")
        self.reportConfigureStatsButton.setText("Configure \nStatistical \nAnalysis")
        self.reportConfigureStatsButton.setToolTip(
            "Set statistical testing intervals, averaging options, and comparison modes.")
        self.reportConfigureStatsButton.clicked.connect(self.open_stats_config_dialog)

        self.reportGenerateButton = QtWidgets.QPushButton(self.tab4)
        self.reportGenerateButton.setGeometry(QtCore.QRect(500, 510, 131, 81))
        self.reportGenerateButton.setObjectName("reportGenerateButton")
        self.reportGenerateButton.setText("Generate report")
        self.reportGenerateButton.setToolTip(
            "<b>Compile Results:</b><br>"
            "Generate visual charts, CSV data, and perform statistical comparisons between varieties.<br>"
            "Re-applies the current analysis period from hourly files (Postprocess is not required "
            "when the period is already covered)."
        )
        self.reportGenerateButton.clicked.connect(self.report)

        self.reportSaveConfigButton = QtWidgets.QPushButton(self.tab4)
        self.reportSaveConfigButton.setGeometry(QtCore.QRect(650, 510, 141, 40))
        self.reportSaveConfigButton.setObjectName("reportSaveConfigButton")
        self.reportSaveConfigButton.setText("Save")
        self.reportSaveConfigButton.setToolTip("Save current reporting preferences.")
        self.reportSaveConfigButton.clicked.connect(self.saveFieldsIntoJson)

        self.reportLoadConfigButton = QtWidgets.QPushButton(self.tab4)
        self.reportLoadConfigButton.setGeometry(QtCore.QRect(650, 550, 141, 41))
        self.reportLoadConfigButton.setObjectName("reportLoadConfigButton")
        self.reportLoadConfigButton.setText("Load previous")
        self.reportLoadConfigButton.setToolTip(
            "Restore previous reporting and statistical parameters.")
        self.reportLoadConfigButton.clicked.connect(self.loadJsonIntoFields)

        interactive_widgets = [
            self.reportSelectProjectButton, self.reportProjectField,
            self.doFPCA, self.normFPCA, self.numComponentsFPCAField,
            self.doConvex, self.saveImagesConvex, self.daysConvexField,
            self.daysConvexChooseButton,
            self.doFourier, self.doLateralAngles, self.reportEmergenceDistanceField,
            self.daysAnglesField, self.daysAnglesChooseButton, self.reportConfigureStatsButton,
            self.reportTimeWindowsButton,
            self.reportProcessingLimitField, self.reportCaptureIntervalField,
            self.reportGenotypeAxisLabelField, self.reportPlateConditionAxisLabelField,
            self.reportExtraVariableAxisLabelField,
            self.reportSaveConfigButton, self.reportPostProcessButton,
            self.reportGenerateButton, self.reportLoadConfigButton,
        ]
        for widget in interactive_widgets:
            widget.raise_()

        self.doConvex.toggled.connect(self._sync_snapshot_hour_fields)
        self.doLateralAngles.toggled.connect(self._sync_snapshot_hour_fields)
        self._sync_snapshot_hour_fields()

        return

    def _load_report_conf(self):
        if not hasattr(self, 'projectField') or not self.projectField.text():
            return None
        config_path = os.path.join(self.projectField.text(), PROJECT_CONFIG_NAME)
        if not os.path.isfile(config_path):
            return None
        try:
            with open(config_path, 'r') as handle:
                return json.load(handle)
        except Exception:
            return None

    def _append_report_tree_item(self, parent, node):
        if isinstance(node, ReportBranch):
            item = QTreeWidgetItem([node.label])
            item.setFlags(item.flags() & ~QtCore.Qt.ItemIsSelectable)
            if parent is None:
                self.report_tree.addTopLevelItem(item)
            else:
                parent.addChild(item)
            for child in node.children:
                self._append_report_tree_item(item, child)
            item.setExpanded(True)
            return item

        item = QTreeWidgetItem([node.label])
        item.setData(0, REPORT_PLOT_ROLE, node.plot_file)
        item.setData(0, REPORT_STATS_ROLE, node.stats_file or '')
        if parent is None:
            self.report_tree.addTopLevelItem(item)
        else:
            parent.addChild(item)
        return item

    def _show_report_placeholder(self):
        self.report_label_1.clear()
        report_path_1 = os.path.join("placeholder_figures/report_placeholder.png")
        pixmap_1 = QtGui.QPixmap(report_path_1)
        self.report_label_1.set_pixmap(pixmap_1, self.report_label_1.size())
        self.report_label_1.setAlignment(QtCore.Qt.AlignCenter)
        self.report_label_1.show()
        if hasattr(self, 'open_stats_button_tab5'):
            self.open_stats_button_tab5.setEnabled(False)

    def update_report_labels(self):
        if not hasattr(self, 'projectField') or not hasattr(self, 'report_tree'):
            return

        report_path = self.current_report_dir()
        current = self.report_tree.currentItem()
        current_report = current.data(0, REPORT_PLOT_ROLE) if current is not None else None

        if (self.projectField.text() == "" or not os.path.exists(self.projectField.text())
                or not os.path.isdir(report_path) or not current_report):
            self._show_report_placeholder()
            return

        report_path_1 = os.path.join(report_path, current_report)
        if not os.path.isfile(report_path_1):
            self._show_report_placeholder()
            return

        size = self.report_label_1.size()
        if size.width() < 10 or size.height() < 10:
            size = QtCore.QSize(750, 550)
        pixmap_1 = QtGui.QPixmap(report_path_1)
        self.report_label_1.set_pixmap(pixmap_1, size)
        self.report_label_1.setAlignment(QtCore.Qt.AlignCenter)
        self.report_label_1.show()

        stats_file = current.data(0, REPORT_STATS_ROLE) if current is not None else ''
        has_stats = bool(stats_file) and os.path.isfile(os.path.join(report_path, stats_file))
        self.open_stats_button_tab5.setEnabled(has_stats)

    def _filter_report_tree(self, text):
        text = text.strip().lower()

        def visit(item):
            child_match = any(visit(item.child(i)) for i in range(item.childCount()))
            own_match = False
            if item.childCount() == 0:
                own_match = text in item.text(0).lower() or text in (item.data(0, REPORT_PLOT_ROLE) or '').lower()
            visible = child_match or own_match or not text
            item.setHidden(not visible)
            return visible

        for i in range(self.report_tree.topLevelItemCount()):
            visit(self.report_tree.topLevelItem(i))

    def refresh_tab5(self):
        if not hasattr(self, 'report_tree'):
            return

        self.report_tree.clear()
        report_path = self.current_report_dir()
        if not os.path.isdir(report_path):
            self._show_report_placeholder()
            return

        try:
            catalog = load_report_catalog(report_path)
        except Exception as exc:
            print(f'Warning: report catalog failed ({exc}); showing placeholder.')
            catalog = []

        for branch in catalog:
            self._append_report_tree_item(None, branch)

        self.report_tree.resizeColumnToContents(0)

        if self.report_tree.topLevelItemCount() == 0:
            self._show_report_placeholder()
            return

        if hasattr(self, 'report_filter_field'):
            self._filter_report_tree(self.report_filter_field.text())

        first_leaf = None
        iterator = QtWidgets.QTreeWidgetItemIterator(self.report_tree)
        while iterator.value():
            item = iterator.value()
            if item.childCount() == 0 and item.data(0, REPORT_PLOT_ROLE):
                first_leaf = item
                break
            iterator += 1
        if first_leaf is not None:
            self.report_tree.setCurrentItem(first_leaf)
        self.update_report_labels()

    def open_report_stats(self):
        current = self.report_tree.currentItem() if hasattr(self, 'report_tree') else None
        if current is None:
            return
        stats_file = current.data(0, REPORT_STATS_ROLE)
        if not stats_file:
            return
        stats_path = os.path.join(self.projectField.text(), "Report", stats_file)
        if os.path.isfile(stats_path):
            self.universal_open(stats_path)

    def setup_tab5_elements(self):
        self.tab5 = QtWidgets.QWidget()
        self.tab5.setObjectName("tab5")
        self.tab_widget.addTab(self.tab5, "Report")

        self.report_label_1 = AspectRatioLabel()
        self.report_label_1.setObjectName("report_label_1")
        self.report_label_1.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding,
        )
        self.report_label_1.setMinimumSize(200, 200)

        self.report_tree = QTreeWidget(self.tab5)
        self.report_tree.setHeaderHidden(True)
        self.report_tree.setMinimumWidth(200)
        self.report_tree.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.report_tree.setTextElideMode(QtCore.Qt.ElideNone)
        self.report_tree.header().setStretchLastSection(False)
        self.report_tree.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding,
        )
        self.report_tree.currentItemChanged.connect(
            lambda _current, _previous: self.update_report_labels()
        )

        self.report_splitter = QSplitter(QtCore.Qt.Horizontal, self.tab5)
        self.report_splitter.addWidget(self.report_tree)
        self.report_splitter.addWidget(self.report_label_1)
        self.report_splitter.setStretchFactor(0, 1)
        self.report_splitter.setStretchFactor(1, 2)
        self.report_splitter.splitterMoved.connect(
            lambda _pos, _index: self.update_report_labels()
        )

        self.report_filter_field = QLineEdit(self.tab5)
        self.report_filter_field.setPlaceholderText("Filter figures...")
        self.report_filter_field.textChanged.connect(self._filter_report_tree)

        self.refresh_button_tab5 = QPushButton(self.tab5)
        self.refresh_button_tab5.setObjectName("Refresh_5")
        self.refresh_button_tab5.setText("Refresh")
        self.refresh_button_tab5.setToolTip(
            "Update the list of available report figures based on the current project data.")
        self.refresh_button_tab5.clicked.connect(self.refresh_tab5)

        self.open_stats_button_tab5 = QPushButton(self.tab5)
        self.open_stats_button_tab5.setObjectName("Open Stats")
        self.open_stats_button_tab5.setText("Open stats")
        self.open_stats_button_tab5.setToolTip("Open the stats file paired with the selected figure.")
        self.open_stats_button_tab5.clicked.connect(self.open_report_stats)
        self.open_stats_button_tab5.setEnabled(False)

        self.open_path_button_tab5 = QPushButton(self.tab5)
        self.open_path_button_tab5.setObjectName("Open Path")
        self.open_path_button_tab5.setText("Open Report Path")
        self.open_path_button_tab5.setToolTip(
            "Open the directory where generated reports are stored.")
        self.open_path_button_tab5.clicked.connect(self.open_report_folder)

        content_layout = QHBoxLayout()
        content_layout.addWidget(self.report_splitter, 1)

        controls_layout = QHBoxLayout()
        controls_layout.addWidget(self.report_filter_field, 3)
        controls_layout.addWidget(self.open_stats_button_tab5, 1)
        controls_layout.addWidget(self.open_path_button_tab5, 1)
        controls_layout.addWidget(self.refresh_button_tab5, 1)

        layout = QVBoxLayout()
        layout.addLayout(content_layout)
        layout.addLayout(controls_layout)
        self.tab5.setLayout(layout)
    
    def setup_tab6_elements(self):        
        self.tab6.setAutoFillBackground(True)
        self.tab6.setStyleSheet("background-color: white;")
        
        layout = QVBoxLayout(self.tab6)
        layout.setAlignment(QtCore.Qt.AlignCenter)

        # Logo
        self.logo_label = QLabel()
        ico_path = "../logo.ico"
        try:
            with Image.open(ico_path) as img:
                img = img.convert("RGBA").resize((200, 200), Image.Resampling.LANCZOS)
                data = img.tobytes("raw", "RGBA")
                qimg = QtGui.QImage(data, img.size[0], img.size[1], QtGui.QImage.Format_RGBA8888)
                self.logo_label.setPixmap(QtGui.QPixmap.fromImage(qimg))
        except Exception:
            self.logo_label.setPixmap(QtGui.QPixmap(ico_path).scaled(150, 150, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))
        
        self.logo_label.setStyleSheet("background-color: transparent;")
        layout.addWidget(self.logo_label, alignment=QtCore.Qt.AlignCenter)

        # Short Description
        title = QLabel("ChronoRoot")
        title.setStyleSheet("font-size: 28px; font-weight: bold; color: #2c3e50; background-color: transparent;")
        layout.addWidget(title, alignment=QtCore.Qt.AlignCenter)

        description = QLabel("An open-source platform for high-throughput phenotyping of plant root systems.")
        description.setStyleSheet("font-size: 14px; color: #34495e; background-color: transparent; margin-bottom: 5px;")
        layout.addWidget(description, alignment=QtCore.Qt.AlignCenter)
        # Website Link
        web_link = QLabel('<a href="https://chronoroot.github.io/">https://chronoroot.github.io/</a>')
        web_link.setOpenExternalLinks(True)
        web_link.setStyleSheet("font-size: 13px; background-color: transparent; margin-bottom: 20px;")
        layout.addWidget(web_link, alignment=QtCore.Qt.AlignCenter)

        # Update Button
        self.update_btn = QPushButton("Check for Updates")
        self.update_btn.setFixedWidth(250)
        self.update_btn.setCursor(QtCore.Qt.PointingHandCursor)
        self.update_btn.setStyleSheet("""
            QPushButton {
                background-color: #3498db; color: white; border-radius: 5px;
                padding: 10px; font-weight: bold;
            }
            QPushButton:hover { background-color: #2980b9; }
        """)
        self.update_btn.clicked.connect(self.update_software)
        layout.addWidget(self.update_btn, alignment=QtCore.Qt.AlignCenter)
        # Last Commit Info
        self.commit_label = QLabel(f"Last update: {self.get_last_commit_time()}")
        self.commit_label.setStyleSheet("color: #95a5a6; background-color: transparent; margin-top: 15px;")
        layout.addWidget(self.commit_label, alignment=QtCore.Qt.AlignCenter)

    def get_git_hash(self):
        """Returns the current git commit hash (language independent)."""
        try:
            return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
        except:
            return None

    def get_last_commit_time(self):
        """Fetches the date of the last local git commit (YYYY-MM-DD)."""
        try:
            # --date=short is ISO format (2024-05-20), which is universal
            cmd = ["git", "log", "-1", "--format=%cd", "--date=short"]
            return subprocess.check_output(cmd).decode().strip()
        except:
            return "Unknown"

    def update_software(self):
        """Performs a git pull using hash-comparison for language safety."""
        try:
            self.update_btn.setText("Checking...")
            self.update_btn.setEnabled(False)
            QtWidgets.QApplication.processEvents()

            # Record the hash before pulling
            old_hash = self.get_git_hash()
            
            # Perform pull (suppress language-specific text output)
            subprocess.check_call(["git", "pull"], stderr=subprocess.STDOUT)
            
            # Record the hash after pulling
            new_hash = self.get_git_hash()

            if old_hash == new_hash:
                QtWidgets.QMessageBox.information(self, "Update", "ChronoRoot is already up to date!")
            else:
                QtWidgets.QMessageBox.information(self, "Update Success", 
                    "Update downloaded successfully!\nPlease restart the application to apply changes.")
                self.commit_label.setText(f"Last update: {self.get_last_commit_time()}")

        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Update Error", 
                "Failed to update. Make sure you have an internet connection and 'git' is installed.")
        
        finally:
            self.update_btn.setText("Check for Updates")
            self.update_btn.setEnabled(True)

def main():
    app = QtWidgets.QApplication(sys.argv)
    window = Ui_ChronoRootAnalysis()
    window.show()
    
    # Defer table and UI updates until after window is shown
    QtCore.QTimer.singleShot(100, window.ensure_plant_inventory)
    QtCore.QTimer.singleShot(100, window.update_report_labels)
    
    sys.exit(app.exec_())
    
if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()