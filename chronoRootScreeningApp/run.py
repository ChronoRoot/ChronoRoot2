import glob
import os
import sys
import platform
import shutil

# Suppress Qt and OpenGL warnings
os.environ['QT_LOGGING_RULES'] = '*=false'
os.environ['LIBGL_ALWAYS_INDIRECT'] = '1'

import json
import subprocess
from datetime import datetime
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QLabel, QLineEdit, QPushButton, QCheckBox,
                           QFileDialog, QGroupBox, QMessageBox, QScrollArea,
                           QTabWidget, QTableWidget, QTableWidgetItem, QMenu, QComboBox, QDialog,
                           QTreeWidget, QTreeWidgetItem)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QColor, QPixmap, QIntValidator, QDoubleValidator
import ui_errors
import interface_config
import chrono_root_backend  # noqa: F401
from robot_ids import identifier_from_rpi_cam, parse_robot_video_path, resolve_rpi_cam
from stats_config_dialog import ScreeningStatsConfigDialog

# --- CONFIGURATION CONSTANTS ---
APP_NAME = "chronoRootScreening"
GLOBAL_CONFIG_DIR = os.path.expanduser(f"~/.config/{APP_NAME}")
GLOBAL_CONFIG_FILE = os.path.join(GLOBAL_CONFIG_DIR, "mainInterfaceConfig.json")
PROCESS_CONFIG_NAME = "process_config.json"
os.makedirs(GLOBAL_CONFIG_DIR, exist_ok=True)

class GroupEntry(QWidget):
    def __init__(self, index, parent=None):
        super().__init__(parent)
        self.index = index
        layout = QHBoxLayout()
        self.setLayout(layout)
        
        # Group name input
        name_layout = QHBoxLayout()
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText(f'Group {index+1}')
        
        layout.addWidget(QLabel(f'Group {index+1} Name:'))
        layout.addWidget(self.name_edit)
        
        # Add seed count input with label
        layout.addWidget(QLabel('Number of Seeds:'))
        self.seed_count_edit = QLineEdit()
        self.seed_count_edit.setPlaceholderText('Optional')
        self.seed_count_edit.setFixedWidth(100)  # Make it compact
        # Only allow integers to be entered
        self.seed_count_edit.setValidator(QIntValidator(0, 999999))
        layout.addWidget(self.seed_count_edit)
        
        self.delete_btn = QPushButton('Remove')
        layout.addWidget(self.delete_btn)
        
    def get_seed_count(self):
        """Return the seed count if entered, None otherwise"""
        text = self.seed_count_edit.text().strip()
        return int(text) if text else None


class AnalysisTab(QWidget):
    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        self.main_window = main_window
        self.group_entries = []
        self.preview_window = None
        self.calibration_window = None
        self.process_launchers = []
        self.report_launcher = None
        self._auto_identifier = ''
        self._loading_config = False
        self._ui_ready = False
        self.initUI()
    
    def setup_project_fields(self):
        # Project directory
        proj_dir_layout = QHBoxLayout()
        self.proj_dir_edit = QLineEdit()
        self.proj_dir_edit.textChanged.connect(self.on_project_dir_changed)  # Add this connection
        proj_dir_btn = QPushButton('Browse')
        proj_dir_btn.clicked.connect(self.browse_project_dir)
        proj_dir_layout.addWidget(QLabel('Project Directory:'))
        proj_dir_layout.addWidget(self.proj_dir_edit)
        proj_dir_layout.addWidget(proj_dir_btn)
        return proj_dir_layout


    def initUI(self):
        layout = QVBoxLayout()

        config_layout = QHBoxLayout()
        self.export_setup_btn = QPushButton('Save to File...')
        self.export_setup_btn.clicked.connect(self.export_setup_to_file)
        config_layout.addWidget(self.export_setup_btn)

        self.load_last_config_btn = QPushButton('Restore Last Session')
        self.load_last_config_btn.clicked.connect(self.load_last_configuration)
        config_layout.addWidget(self.load_last_config_btn)

        self.load_config_file_btn = QPushButton('Load From File...')
        self.load_config_file_btn.clicked.connect(self.load_configuration_from_file)
        config_layout.addWidget(self.load_config_file_btn)

        config_hint = QLabel(
            'Auto saves automatically when any button is pressed'
        )
        config_hint.setStyleSheet('color: #666; font-size: 9pt;')
        config_layout.addWidget(config_hint)
        
        config_layout.addStretch()
        layout.addLayout(config_layout)
        
        # Project Directory Selection
        proj_group = QGroupBox('Project Settings')
        proj_layout = QVBoxLayout()
        
        # Add project directory fields using the new setup method
        proj_layout.addLayout(self.setup_project_fields())
        
        # Video path
        video_layout = QHBoxLayout()
        self.video_path_edit = QLineEdit()
        self.video_path_edit.textChanged.connect(self.on_video_path_changed)
        video_path_btn = QPushButton('Browse')
        video_path_btn.clicked.connect(self.browse_video_path)
        video_layout.addWidget(QLabel('Video Directory:'))
        video_layout.addWidget(self.video_path_edit)
        video_layout.addWidget(video_path_btn)
        proj_layout.addLayout(video_layout)
        
        # Analysis identifier
        identifier_layout = QHBoxLayout()
        self.identifier_edit = QLineEdit()
        self.identifier_edit.setPlaceholderText('rpi3_cam_0')
        identifier_layout.addWidget(QLabel('Analysis Identifier:'))
        identifier_layout.addWidget(self.identifier_edit)
        proj_layout.addLayout(identifier_layout)

        factor_layout = QHBoxLayout()
        self.plateConditionName = QLineEdit()
        self.plateConditionName.setObjectName("plateConditionName")
        self.plateConditionName.setPlaceholderText('Control')
        factor_layout.addWidget(QLabel('Plate Growth Condition:'))
        factor_layout.addWidget(self.plateConditionName)
        plate_hint = QLabel('(Optional, e.g. "Control", "Treatment")')
        plate_hint.setStyleSheet('color: #666; font-size: 9pt;')
        factor_layout.addWidget(plate_hint)
        proj_layout.addLayout(factor_layout)

        extra_layout = QHBoxLayout()
        self.extraField = QLineEdit()
        self.extraField.setObjectName("extraField")
        self.extraField.setPlaceholderText('Run 1')
        extra_layout.addWidget(QLabel('Extra Variable:'))
        extra_layout.addWidget(self.extraField)
        extra_hint = QLabel('(Optional value, e.g. Run 1, Run 2.)')
        extra_hint.setStyleSheet('color: #666; font-size: 9pt;')
        extra_layout.addWidget(extra_hint)
        proj_layout.addLayout(extra_layout)

        # Time delta field
        time_settings = QHBoxLayout()
        self.time_delta_edit = QLineEdit()
        self.time_delta_edit.setPlaceholderText('15')
        time_settings.addWidget(QLabel('Time between slices (minutes):'))
        time_settings.addWidget(self.time_delta_edit)

        # Time before pictures
        added_time = QHBoxLayout()
        self.add_time_edit = QLineEdit()
        self.add_time_edit.setPlaceholderText('0')
        added_time.addWidget(QLabel('Extra time before first picture (hours):'))
        added_time.addWidget(self.add_time_edit)
        
        # Cut down germination plot time
        germination_time_layout = QHBoxLayout()
        self.germination_time_edit = QLineEdit()
        self.germination_time_edit.setPlaceholderText('0 (Leave 0 for full duration)')
        germination_time_layout.addWidget(QLabel('End germination plot time (hours):'))
        germination_time_layout.addWidget(self.germination_time_edit)
        
        # Put them both in next to each other
        time_layout = QHBoxLayout()
        time_layout.addLayout(time_settings)
        time_layout.addLayout(added_time)
        time_layout.addLayout(germination_time_layout)
        proj_layout.addLayout(time_layout)

        # Calibration Group Box
        calib_group = QGroupBox('Calibration Settings')
        calib_layout = QHBoxLayout()

        # QR Code toggle and calibration options
        self.qr_checkbox = QCheckBox('Video has QR codes for calibration')
        self.qr_checkbox.stateChanged.connect(self.toggle_calibration_mode)
        calib_layout.addWidget(self.qr_checkbox)

        # Manual calibration widget (hidden by default)
        self.manual_calib_widget = QWidget()
        manual_calib_layout = QHBoxLayout()
        
        # Calibration helper button
        self.calibrate_btn = QPushButton('Open Calibration Helper')
        self.calibrate_btn.clicked.connect(self.open_calibration_helper)
        manual_calib_layout.addWidget(self.calibrate_btn)
        
        # Known distance input
        known_dist_layout = QHBoxLayout()
        self.known_dist_edit = QLineEdit()
        self.known_dist_edit.setPlaceholderText('10')
        self.known_dist_edit.setValidator(QDoubleValidator(0.0, 1000.0, 2))
        known_dist_layout.addWidget(QLabel('Known distance (mm):'))
        known_dist_layout.addWidget(self.known_dist_edit)
        manual_calib_layout.addLayout(known_dist_layout)

        # Pixel distance input
        pixel_dist_layout = QHBoxLayout()
        self.pixel_dist_edit = QLineEdit()
        self.pixel_dist_edit.setPlaceholderText('240')
        self.pixel_dist_edit.setValidator(QIntValidator(1, 10000))
        pixel_dist_layout.addWidget(QLabel('Corresponding pixels:'))
        pixel_dist_layout.addWidget(self.pixel_dist_edit)
        manual_calib_layout.addLayout(pixel_dist_layout)

        self.manual_calib_widget.setLayout(manual_calib_layout)
        calib_layout.addWidget(self.manual_calib_widget)
        calib_group.setLayout(calib_layout)
        
        proj_group.setLayout(proj_layout)
        layout.addWidget(proj_group)
        layout.addWidget(calib_group)
        
        # Process customization options
        process_group = QGroupBox('Processing Options')
        process_layout = QHBoxLayout()
        
        # 1 Do germination analysis
        self.germination_checkbox = QCheckBox('Perform germination analysis')
        self.germination_checkbox.setChecked(True)
        process_layout.addWidget(self.germination_checkbox)
        
        # 2 Perform plant growth analysis
        self.plant_growth_checkbox = QCheckBox('Perform plant growth analysis')
        self.plant_growth_checkbox.setChecked(True)
        process_layout.addWidget(self.plant_growth_checkbox)
        self.plant_growth_checkbox.stateChanged.connect(self.toggle_plant_growth_options)
        
        # 3 Store tracking visualization
        self.show_tracking_checkbox = QCheckBox('Store tracking visualization')
        self.show_tracking_checkbox.setChecked(False)
        process_layout.addWidget(self.show_tracking_checkbox)
        
        # 4 Store germination for each video
        self.store_each_video_checkbox = QCheckBox('Store germination plots for each video separately')
        self.store_each_video_checkbox.setChecked(False)
        process_layout.addWidget(self.store_each_video_checkbox)
        
        process_group.setLayout(process_layout)
        layout.addWidget(process_group)
        
        # Group Names
        group_group = QGroupBox('Group Names')
        self.group_layout = QVBoxLayout()
        
        # Scroll area for groups
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_widget = QWidget()
        self.group_scroll_widget = scroll_widget
        self.group_layout = QVBoxLayout(scroll_widget)
        scroll.setWidget(scroll_widget)
        
        # Add Group button
        self.add_group_btn = QPushButton('Add Group')
        self.add_group_btn.clicked.connect(self.add_group)
        self.group_layout.addWidget(self.add_group_btn)
        
        group_group.setLayout(QVBoxLayout())
        group_group.layout().addWidget(scroll)
        layout.addWidget(group_group)

        # Buttons layout
        buttons_layout = QHBoxLayout()
        
        # Preview Video Button
        self.preview_btn = QPushButton('Preview Video')
        self.preview_btn.clicked.connect(self.preview_video)
        buttons_layout.addWidget(self.preview_btn)
        
        # Process Button
        self.process_btn = QPushButton('Process Video')
        self.process_btn.clicked.connect(self.process_video)
        buttons_layout.addWidget(self.process_btn)

        # Generate Report Button
        self.generate_report_btn = QPushButton('Generate Report')
        self.generate_report_btn.clicked.connect(self.generate_report)
        buttons_layout.addWidget(self.generate_report_btn)

        self.configure_stats_btn = QPushButton('Configure Report Parameters')
        self.configure_stats_btn.clicked.connect(self.open_stats_config_dialog)
        buttons_layout.addWidget(self.configure_stats_btn)
        
        # Add name mapping button to the buttons_layout
        self.name_mapping_btn = QPushButton('Edit Name Mapping')
        self.name_mapping_btn.clicked.connect(self.edit_name_mapping)
        buttons_layout.addWidget(self.name_mapping_btn)

        layout.addLayout(buttons_layout)
        
        self.setLayout(layout)
        
        # Add initial groups (no autosave during UI setup)
        for _ in range(3):
            self._append_group_entry()

        # Initialize calibration mode
        self.stats_config_dialog = ScreeningStatsConfigDialog(self)
        self.stats_config_dialog.register_on_host(self)
        self.stats_config_dialog.set_defaults()
        self.toggle_calibration_mode()
        self.toggle_plant_growth_options()
        self._ui_ready = True

    def _autosave_config(self):
        if not self._ui_ready or self._loading_config:
            return
        interface_config.save_interface_config(self, GLOBAL_CONFIG_FILE)

    def _default_setup_directory(self) -> str:
        project_dir = self.proj_dir_edit.text().strip()
        analysis_id = self.identifier_edit.text().strip()
        if project_dir and analysis_id:
            analysis_dir = os.path.join(project_dir, 'analysis', analysis_id)
            if os.path.isdir(analysis_dir):
                return analysis_dir
        if project_dir:
            analysis_root = os.path.join(project_dir, 'analysis')
            if os.path.isdir(analysis_root):
                return analysis_root
            return project_dir
        return os.path.expanduser('~')

    def _default_export_filename(self) -> str:
        analysis_id = self.identifier_edit.text().strip()
        if analysis_id:
            return f'{analysis_id}_setup_config.json'
        return 'analysis_setup_config.json'

    def export_setup_to_file(self):
        start_dir = self._default_setup_directory()
        default_name = self._default_export_filename()
        path, _ = QFileDialog.getSaveFileName(
            self,
            'Export Analysis Setup',
            os.path.join(start_dir, default_name),
            interface_config.SETUP_FILE_FILTER,
        )
        if not path:
            return

        if not path.lower().endswith('.json'):
            path += '.json'

        if not interface_config.is_setup_filename(path):
            ui_errors.show_warning(
                self,
                'Invalid Filename',
                "Setup files must have 'config' in the name.\n\n"
                'Example: my_experiment_setup_config.json or process_config.json',
            )
            return

        if interface_config.save_interface_config(self, path):
            ui_errors.show_information(
                self,
                'Setup Exported',
                f'Setup saved to:\n{path}\n\nYou can reload it with Load Setup From File.',
            )
        else:
            ui_errors.show_critical(self, 'Error', f'Failed to save setup to:\n{path}')

    def load_last_configuration(self):
        if not os.path.exists(GLOBAL_CONFIG_FILE):
            ui_errors.show_warning(
                self, 'No Session Saved',
                f'No saved session found at:\n{GLOBAL_CONFIG_FILE}',
            )
            return

        ok, error = interface_config.load_interface_config(self, GLOBAL_CONFIG_FILE)
        if ok:
            ui_errors.show_information(self, 'Session Restored', 'Your last session configuration has been restored.')
        else:
            ui_errors.show_warning(
                self,
                'Could Not Restore Session',
                f'The saved session file could not be loaded.\n\n{error}',
            )

    def load_configuration_from_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            'Load Analysis Setup',
            self._default_setup_directory(),
            interface_config.SETUP_FILE_FILTER,
        )
        if not path:
            return

        if not interface_config.is_setup_filename(path):
            ui_errors.show_warning(
                self,
                'Wrong File Type',
                "Please choose a setup file with 'config' in the name.\n\n"
                'Example: process_config.json in your analysis folder.',
            )
            return

        ok, error = interface_config.load_interface_config(self, path)
        if ok:
            ui_errors.show_information(self, 'Setup Loaded', f'Analysis setup loaded from:\n{path}')
        else:
            ui_errors.show_warning(self, 'Not a Setup File', error)

    def edit_name_mapping(self):
        """Open dialog to edit name mapping for visualization"""
        if not self.proj_dir_edit.text() or not os.path.exists(self.proj_dir_edit.text()):
            QMessageBox.warning(self, 'Error', 'Please select a valid project directory first!')
            return

        self._autosave_config()
        dialog = NameMappingDialog(self.proj_dir_edit.text(), self)
        dialog.exec_()

    def on_project_dir_changed(self):
        """Handle project directory changes and update all tabs"""
        project_dir = self.proj_dir_edit.text()
        if os.path.exists(project_dir):
            self.main_window.set_project_dir(project_dir)

    def toggle_calibration_mode(self):
        """Toggle between QR and manual calibration modes"""
        has_qr = self.qr_checkbox.isChecked()
        self.manual_calib_widget.setVisible(not has_qr)
    
    def toggle_plant_growth_options(self):
        """Enable or disable plant growth analysis options in the report dialog."""
        enabled = self.plant_growth_checkbox.isChecked()
        if hasattr(self, 'stats_config_dialog'):
            self.stats_config_dialog.set_plant_growth_enabled(enabled)

    def on_video_path_changed(self):
        self._maybe_autofill_identifier()

    def current_rpi_cam(self):
        return resolve_rpi_cam(
            video_dir=self.video_path_edit.text().strip(),
            analysis_id=self.identifier_edit.text().strip(),
        )

    def _maybe_autofill_identifier(self):
        if self._loading_config:
            return
        rpi, cam = parse_robot_video_path(self.video_path_edit.text().strip())
        if not rpi or not cam:
            return
        suggested = identifier_from_rpi_cam(rpi, cam)
        current = self.identifier_edit.text().strip()
        if current and current != self._auto_identifier:
            return
        self._auto_identifier = suggested
        if current != suggested:
            self.identifier_edit.setText(suggested)

    def _remember_auto_identifier(self):
        rpi, cam = parse_robot_video_path(self.video_path_edit.text().strip())
        if not rpi or not cam:
            return
        suggested = identifier_from_rpi_cam(rpi, cam)
        if self.identifier_edit.text().strip() == suggested:
            self._auto_identifier = suggested

    def open_calibration_helper(self):
        """Opens a helper window to assist with manual calibration"""
        video_dir = self.video_path_edit.text().strip()
        if not video_dir:
            ui_errors.show_warning(self, 'Error', 'Please select a video directory first!')
            return
        if not os.path.exists(video_dir):
            ui_errors.show_warning(self, 'Error', 'Video directory does not exist!')
            return

        self._autosave_config()

        try:
            import calibration_helper
        except Exception as e:
            ui_errors.show_critical(self, 'Error', f'Failed to load calibration helper:\n{e}')
            return

        if self.calibration_window is not None:
            same_video = (
                os.path.abspath(self.calibration_window.video_dir)
                == os.path.abspath(video_dir)
            )
            if same_video and self.calibration_window.isVisible():
                self.calibration_window.raise_()
                self.calibration_window.activateWindow()
                return
            self.calibration_window.close()
            self.calibration_window = None

        try:
            self.calibration_window = calibration_helper.CalibrationHelper(video_dir)
            self.calibration_window.distance_measured.connect(self._on_calibration_distance)
            self.calibration_window.destroyed.connect(
                lambda: setattr(self, 'calibration_window', None)
            )
            self.calibration_window.show()
        except Exception as e:
            ui_errors.show_critical(self, 'Error', f'Failed to open calibration helper:\n{e}')

    def _on_calibration_distance(self, pixels: int):
        self.pixel_dist_edit.setText(f"{pixels}")

    def validate_inputs(self):
        """Updated validation to include calibration checks"""
        if not self.basic_validation():
            return False

        # Validate calibration settings
        if not self.qr_checkbox.isChecked():
            if not self.known_dist_edit.text() or not self.pixel_dist_edit.text():
                QMessageBox.warning(self, 'Error', 'Please provide both known distance and pixel distance for manual calibration!')
                return False
            try:
                known_dist = float(self.known_dist_edit.text())
                pixel_dist = int(self.pixel_dist_edit.text())
                if known_dist <= 0 or pixel_dist <= 0:
                    QMessageBox.warning(self, 'Error', 'Calibration values must be positive numbers!')
                    return False
            except ValueError:
                QMessageBox.warning(self, 'Error', 'Invalid calibration values!')
                return False

        return True

    def basic_validation(self):
        """Original validation logic moved to separate method"""
        if not os.path.exists(self.proj_dir_edit.text()):
            QMessageBox.warning(self, 'Error', 'Project directory does not exist!')
            return False
            
        if not os.path.exists(self.video_path_edit.text()):
            QMessageBox.warning(self, 'Error', 'Video directory does not exist!')
            return False
            
        identifier = self.identifier_edit.text().strip()
        if not identifier:
            QMessageBox.warning(self, 'Error', 'Please provide an analysis identifier!')
            return False
            
        if not identifier.replace('_', '').isalnum():
            QMessageBox.warning(self, 'Error', 'Identifier must contain only letters, numbers, and underscores!')
            return False
            
        analysis_dir = os.path.join(self.proj_dir_edit.text(), 'analysis', identifier)
        if os.path.exists(analysis_dir):
            metadata_path = os.path.join(analysis_dir, 'metadata.json')
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    status = metadata.get('status', '')
                except (json.JSONDecodeError, OSError):
                    status = ''
            else:
                status = ''

            if status == 'Complete':
                QMessageBox.warning(self, 'Error', 'Analysis with this identifier already exists!')
                return False
            if status in ('In Progress', 'Failed'):
                reply = QMessageBox.question(
                    self,
                    'Re-run Analysis',
                    f'Analysis "{identifier}" exists with status "{status}".\n'
                    'Re-running will overwrite partial results. Continue?',
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No,
                )
                if reply != QMessageBox.Yes:
                    return False
            else:
                QMessageBox.warning(self, 'Error', 'Analysis with this identifier already exists!')
                return False
            
        group_names = [entry.name_edit.text().strip() for entry in self.group_entries]
        if not all(group_names):
            QMessageBox.warning(self, 'Error', 'All group names must be filled!')
            return False
            
        if len(set(group_names)) != len(group_names):
            QMessageBox.warning(self, 'Error', 'Group names must be unique!')
            return False
            
        return True

    def _append_group_entry(self):
        group_entry = GroupEntry(len(self.group_entries))
        group_entry.delete_btn.clicked.connect(lambda: self.remove_group(group_entry))
        self.group_entries.append(group_entry)
        self.group_layout.insertWidget(len(self.group_entries) - 1, group_entry)

    def add_group(self):
        self._append_group_entry()
        self._autosave_config()
        
    def remove_group(self, group_entry):
        if len(self.group_entries) > 1:
            self.group_entries.remove(group_entry)
            group_entry.deleteLater()
            for i, entry in enumerate(self.group_entries):
                entry.index = i
            self._autosave_config()
        else:
            QMessageBox.warning(self, 'Warning', 'At least one group is required!')
            
    def browse_project_dir(self):
        dir_path = QFileDialog.getExistingDirectory(self, 'Select Project Directory')
        if dir_path:
            self.proj_dir_edit.setText(dir_path)
            self.main_window.set_project_dir(dir_path)
            self._autosave_config()
            
    def _get_time_delta(self):
        try:
            return float(self.time_delta_edit.text() or '15')
        except ValueError:
            return 15.0

    def _validate_video_dataset(self):
        import plant_viewer
        video_folder, segmentation_dir, images, seg_files = plant_viewer.resolve_screening_paths(
            self.video_path_edit.text()
        )
        if not images:
            ui_errors.show_warning(
                self,
                'Error',
                'No images found in the video folder!\nPlease check the path to the folder where the images are located.'
            )
            return None
        if not seg_files:
            ui_errors.show_warning(
                self,
                'Error',
                f'Found {len(images)} images but no segmentation files!\n'
                'The images may not have been properly segmented.'
            )
            return None
        return video_folder, segmentation_dir

    def browse_video_path(self):
        dir_path = QFileDialog.getExistingDirectory(self, 'Select Video Directory')
        if dir_path:
            self.video_path_edit.setText(dir_path)
            self._autosave_config()
            
    def process_video(self):
        if not self.validate_inputs():
            return

        self._autosave_config()

        dataset = self._validate_video_dataset()
        if not dataset:
            return
        video_folder, segmentation_dir = dataset

        project_dir = self.proj_dir_edit.text()
        identifier = self.identifier_edit.text().strip()
        time_delta = self._get_time_delta()
        group_names = [entry.name_edit.text().strip() for entry in self.group_entries]

        try:
            import plant_viewer
        except Exception as e:
            ui_errors.show_critical(self, "Error", f"Failed to load images plant viewer:\n{e}")
            return 
            
        try:
            images, seg_files, _ = plant_viewer.load_screening_sequence(
                video_folder, segmentation_dir, time_delta
            )
        except Exception as e:
            ui_errors.show_critical(self, "Error", f"Failed to load images for ROI selection:\n{e}")
            return

        roi_dialog = plant_viewer.GroupROISelectorWindow(
            images, seg_files, group_names, time_delta=time_delta, parent=self
        )
        if roi_dialog.exec_() != QDialog.Accepted:
            ui_errors.show_warning(self, "Cancelled", "ROI selection was cancelled. Analysis was not started.")
            return

        group_rois = roi_dialog.get_group_rois()
        if not group_rois:
            ui_errors.show_warning(self, "Cancelled", "No group regions were selected.")
            return

        seed_counts = [entry.get_seed_count() for entry in self.group_entries]
        rpi, cam = self.current_rpi_cam()
        config = interface_config.build_interface_config(self)
        config.update({
            'video_dir': video_folder,
            'segmentation_dir': segmentation_dir,
            'project_dir': project_dir,
            'analysis_id': identifier,
            'time_delta': time_delta,
            'group_names': group_names,
            'seed_counts': [count if count is not None else 0 for count in seed_counts],
            'group_rois': {name: list(coords) for name, coords in group_rois.items()},
            'PlateCondition': self.plateConditionName.text().strip(),
            'ExtraVariable': self.extraField.text().strip(),
            'rpi': rpi,
            'cam': cam,
        })

        if not config['has_qr']:
            config['known_distance'] = float(self.known_dist_edit.text())
            config['pixel_distance'] = int(self.pixel_dist_edit.text())

        analysis_dir = os.path.join(project_dir, 'analysis', identifier)
        os.makedirs(analysis_dir, exist_ok=True)
        config_path = os.path.join(analysis_dir, PROCESS_CONFIG_NAME)
        try:
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=4)
        except Exception as e:
            ui_errors.show_critical(self, "Error", f"Failed to write processing config:\n{e}")
            return

        app_dir = os.path.dirname(os.path.abspath(__file__))
        args = ["python", "process_video.py", "--config", config_path]
        self._drop_finished_process_launchers()
        launcher = ui_errors.launch_worker(
            args,
            parent=self,
            working_directory=app_dir,
            started_title="Processing Started",
            started_message=(
                f"Video processing has been started.\n"
                f"Results will be saved in: {os.path.join(project_dir, 'analysis', identifier)}"
            ),
            error_title="Video Processing Error",
        )
        launcher.process.finished.connect(lambda *_args: self._drop_finished_process_launchers())
        self.process_launchers.append(launcher)

    def _drop_finished_process_launchers(self):
        still_running = []
        for launcher in self.process_launchers:
            if launcher.is_running():
                still_running.append(launcher)
            else:
                launcher.deleteLater()
        self.process_launchers = still_running

    def preview_video(self):
        dataset = self._validate_video_dataset()
        if not dataset:
            return

        self._autosave_config()

        video_folder, segmentation_dir = dataset
        time_delta = self._get_time_delta()

        try:
            import plant_viewer
        except Exception as e:
            ui_errors.show_critical(self, "Error", f"Failed to load images plant viewer:\n{e}")
            return 
        
        try:            
            images, seg_files, conf = plant_viewer.load_screening_sequence(
                video_folder, segmentation_dir, time_delta
            )
            self.preview_window = plant_viewer.ChronoViewWindow(
                images, seg_files, None, conf, parent=None
            )
            self.preview_window.show()
        except Exception as e:
            ui_errors.show_critical(self, "Error", f"Failed to launch preview:\n{e}")

    def open_stats_config_dialog(self):
        self.stats_config_dialog.exec_()
        self._autosave_config()

    def _int_field(self, widget, default):
        text = widget.text().strip() if hasattr(widget, 'text') else str(default)
        try:
            return int(float(text)) if text else default
        except ValueError:
            return default

    def _build_report_config(self):
        selected = self.stats_config_dialog.selected_metric_columns()
        if not self.germination_checkbox.isChecked():
            selected = [col for col in selected if col != 'GerminationTime']
        if not self.plant_growth_checkbox.isChecked():
            selected = [col for col in selected if col == 'GerminationTime']

        conf = {
            'MainFolder': self.proj_dir_edit.text(),
            'timeStep': self._get_time_delta(),
            'everyXhourField': self._int_field(self.everyXhourField, 6),
            'everyXhourFieldFourier': self._int_field(self.everyXhourFieldFourier, 6),
            'everyXhourFieldAngles': self._int_field(self.everyXhourFieldAngles, 6),
            'averagePerPlantStats': self.averagePerPlantStats.isChecked(),
            'doFPCA': self.fpca_checkbox.isChecked() and self.plant_growth_checkbox.isChecked(),
            'normFPCA': self.fpca_normalize_checkbox.isChecked(),
            'numComponentsFPCAField': self._int_field(self.fpca_components_edit, 2),
            'doFourier': self.doFourier.isChecked(),
            'doPlantGrowth': self.plant_growth_checkbox.isChecked(),
            'doGermination': self.germination_checkbox.isChecked(),
            'selectedMetrics': selected,
            'includeLateralRootPlots': False,
            'temporalOverviewMetrics': [col for col in selected if col != 'GerminationTime'],
            'fpcaMetrics': [
                col for col in selected
                if col in ('MainRootLength (mm)', 'TotalLength (mm)', 'HypocotylLength (mm)')
            ],
            'genotypeAxisLabel': self.genotypeAxisLabelField.text().strip() or 'Group',
            'plateConditionAxisLabel': self.plateConditionAxisLabelField.text().strip() or 'Plate condition',
            'extraVariableLabel': self.extraVariableLabelField.text().strip() or 'Run',
            'addTimeBeforePhoto': self._int_field(self.add_time_edit, 0),
            'germinationTimeCut': self._int_field(self.germination_time_edit, 0),
            'germinationEachVideo': self.store_each_video_checkbox.isChecked(),
        }
        for name in (
            'statsByGenotype', 'statsGenotypeByPlate', 'statsGenotypeByExtra',
            'statsByPlateCondition', 'statsByExtraVariable',
            'statsPlateWithinGenotype', 'statsExtraWithinGenotype',
        ):
            conf[name] = getattr(self, name).isChecked()
        mapping_file = os.path.join(self.proj_dir_edit.text(), 'name_mapping.json')
        if os.path.exists(mapping_file):
            conf['nameMapping'] = mapping_file
        return conf

    def generate_report(self):
        """Generate report on all completed experiments."""
        if not self.proj_dir_edit.text():
            ui_errors.show_warning(self, 'Error', 'Please select a project directory first!')
            return

        self._autosave_config()

        if self.report_launcher and self.report_launcher.is_running():
            ui_errors.show_warning(self, "Busy", "Report generation is already running.")
            return
        if self.report_launcher is not None:
            self.report_launcher.deleteLater()
            self.report_launcher = None

        project_dir = self.proj_dir_edit.text()
        analysis_dir = os.path.join(project_dir, 'analysis')
        if not os.path.exists(analysis_dir):
            ui_errors.show_warning(self, 'Error', 'No analysis directory found!')
            return

        analyses = [d for d in os.listdir(analysis_dir)
                    if os.path.isdir(os.path.join(analysis_dir, d))]
        if not analyses:
            ui_errors.show_warning(self, 'Error', 'No analyses found to process!')
            return

        conf = self._build_report_config()
        config_path = os.path.join(project_dir, 'report_config.json')
        try:
            with open(config_path, 'w') as handle:
                json.dump(conf, handle, indent=4)
        except OSError as exc:
            ui_errors.show_critical(self, 'Error', f'Failed to write report config:\n{exc}')
            return

        args = ["python", "generate_report.py", "--config", config_path]
        mapping_msg = " with name mapping" if conf.get('nameMapping') else ""
        app_dir = os.path.dirname(os.path.abspath(__file__))
        log_dialog = ui_errors.WorkerLogDialog("Generating report…", parent=self)
        self.report_launcher = ui_errors.launch_worker(
            args,
            parent=self,
            working_directory=app_dir,
            error_title="Report Generation Error",
            log_dialog=log_dialog,
        )
        if self.report_launcher.is_running():
            log_dialog.append_text(
                f"Report generation started{mapping_msg}.\n"
                f"Results will be saved in: {os.path.join(project_dir, 'Report')}\n\n"
            )

class ResultsTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.project_dir = None
        self.sort_column = 0  # Default sort column
        self.sort_order = Qt.AscendingOrder
        self.timer = QTimer()
        self.timer.timeout.connect(self.refresh_results)
        self.initUI()
        
    def initUI(self):
        layout = QVBoxLayout()
        
        # Control buttons layout
        control_layout = QHBoxLayout()
        
        # Refresh button
        self.refresh_btn = QPushButton('Refresh Results')
        self.refresh_btn.clicked.connect(self.refresh_results)
        control_layout.addWidget(self.refresh_btn)
        
        # Auto-refresh toggle
        self.auto_refresh_btn = QPushButton('Auto Refresh: Off')
        self.auto_refresh_btn.setCheckable(True)
        self.auto_refresh_btn.clicked.connect(self.toggle_auto_refresh)
        control_layout.addWidget(self.auto_refresh_btn)
        
        control_layout.addStretch()
        layout.addLayout(control_layout)
        
        # Results table
        self.table = QTableWidget()
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels([
            'Analysis ID', 'Groups', 'Num Groups', 'Start Time', 'Completion Time', 'Status'
        ])
        
        # Enable sorting
        self.table.setSortingEnabled(True)
        self.table.horizontalHeader().sortIndicatorChanged.connect(self.on_sort_changed)
        
        # Enable selection of entire rows
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        
        # Enable context menu
        self.table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self.show_context_menu)
        
        # Make headers stretch
        self.table.horizontalHeader().setStretchLastSection(True)
        
        layout.addWidget(self.table)
        self.setLayout(layout)
        
    def toggle_auto_refresh(self):
        if self.auto_refresh_btn.isChecked():
            self.auto_refresh_btn.setText('Auto Refresh: On')
            self.timer.start(5000)  # Refresh every 5 seconds
        else:
            self.auto_refresh_btn.setText('Auto Refresh: Off')
            self.timer.stop()
    
    def show_context_menu(self, pos):
        item = self.table.itemAt(pos)
        if item is None:
            return
            
        menu = QMenu(self)
        
        # Add actions
        open_folder_action = menu.addAction("Open Results Folder")
        view_metadata_action = menu.addAction("View Metadata")
        
        # Show menu and get selected action
        action = menu.exec_(self.table.viewport().mapToGlobal(pos))
        
        if action == open_folder_action:
            self.open_results_folder(item.row())
        elif action == view_metadata_action:
            self.view_metadata(item.row())
            
    def open_results_folder(self, row):
        analysis_id = self.table.item(row, 0).text()
        folder_path = os.path.join(self.project_dir, 'analysis', analysis_id)
        
        if os.path.exists(folder_path):
            # Open folder in file explorer
            try:
                path = os.path.abspath(os.path.expanduser(folder_path))
                is_container = any(k in os.environ for k in ['APPTAINER_CONTAINER', 'SINGULARITY_CONTAINER'])
                
                # --- STRATEGY 1: D-Bus ---
                if is_container and shutil.which("dbus-send"):
                    try:
                        subprocess.run([
                            "dbus-send", "--session", "--dest=org.freedesktop.FileManager1",
                            "--type=method_call", "/org/freedesktop/FileManager1",
                            "org.freedesktop.FileManager1.ShowItems", 
                            f"array:string:file://{path}", "string:''"
                        ], timeout=2, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
                        return
                    except (subprocess.SubprocessError, OSError):
                        pass

                # --- STRATEGY 2: Standard Openers ---
                cmd = None
                if platform.system() == "Darwin":
                    cmd = "open"
                elif platform.system() == "Windows":
                    os.startfile(path)
                    return
                else:
                    cmd = "xdg-open"
                    
                if cmd and shutil.which(cmd):
                    subprocess.Popen([cmd, path], stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
                    return
                else:
                    print(f"Error opening folder: No suitable opener found for {path}")

            except Exception as e:
                # Final safety net to prevent app crash
                print(f"Error opening folder: {e}")
        
    def view_metadata(self, row):
        analysis_id = self.table.item(row, 0).text()
        metadata_path = os.path.join(self.project_dir, 'analysis', analysis_id, 'metadata.json')
        
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                # Show metadata in a message box
                msg = QMessageBox()
                msg.setWindowTitle(f"Metadata - {analysis_id}")
                msg.setText("\n".join([f"{k}: {v}" for k, v in metadata.items()]))
                msg.exec_()
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Error reading metadata: {str(e)}")
    
    def on_sort_changed(self, logical_index, order):
        self.sort_column = logical_index
        self.sort_order = order
        self.refresh_results()
        
    def set_project_dir(self, dir_path):
        self.project_dir = dir_path
        self.refresh_results()
        
    def create_table_item(self, text, color=None):
        item = QTableWidgetItem(text)
        if color:
            item.setBackground(QColor(color))
        return item
        
    def refresh_results(self):
        if not self.project_dir:
            return
            
        analysis_dir = os.path.join(self.project_dir, 'analysis')
        if not os.path.exists(analysis_dir):
            return
            
        # Store current sort state
        sort_column = self.table.horizontalHeader().sortIndicatorSection()
        sort_order = self.table.horizontalHeader().sortIndicatorOrder()
        
        # Temporarily disable sorting to improve performance
        self.table.setSortingEnabled(False)
        
        # Clear current table
        self.table.setRowCount(0)
        
        # Get all analysis folders
        analyses = [d for d in os.listdir(analysis_dir) 
                   if os.path.isdir(os.path.join(analysis_dir, d))]
        
        self.table.setRowCount(len(analyses))
        
        for row, analysis_id in enumerate(analyses):
            analysis_path = os.path.join(analysis_dir, analysis_id)
            metadata_path = os.path.join(analysis_path, 'metadata.json')
            
            # Initialize empty row
            for col in range(self.table.columnCount()):
                self.table.setItem(row, col, self.create_table_item(""))
                
            # Set analysis ID
            self.table.setItem(row, 0, self.create_table_item(analysis_id))
            
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                        
                    # Fill table row with color coding
                    self.table.setItem(row, 0, self.create_table_item(metadata['analysis_id']))
                    self.table.setItem(row, 1, self.create_table_item(', '.join(metadata['group_names'])))
                    self.table.setItem(row, 2, self.create_table_item(str(metadata['num_groups'])))
                    
                    # Use start_time from metadata
                    if 'start_time' in metadata:
                        self.table.setItem(row, 3, self.create_table_item(metadata['start_time']))
                    else:
                        # Fallback to creation time if metadata is from old version
                        start_time = datetime.fromtimestamp(os.path.getctime(analysis_path)).strftime("%Y-%m-%d %H:%M:%S")
                        self.table.setItem(row, 3, self.create_table_item(start_time))
                    
                    if metadata.get('status') == 'Complete':
                        self.table.setItem(row, 4, self.create_table_item(metadata['completion_time']))
                        self.table.setItem(row, 5, self.create_table_item("Complete", "#90EE90"))  # Light green
                    elif metadata.get('status') == 'In Progress':
                        self.table.setItem(row, 4, self.create_table_item("--"))
                        self.table.setItem(row, 5, self.create_table_item("In Progress", "#FFF68F"))  # Light yellow
                    else:
                        self.table.setItem(row, 4, self.create_table_item("--"))
                        self.table.setItem(row, 5, self.create_table_item("Unknown", "#FFB6C1"))  # Light red
                        
                except Exception as e:
                    self.table.setItem(row, 5, self.create_table_item("Error", "#FFB6C1"))  # Light red
            else:
                # No metadata file exists
                self.table.setItem(row, 5, self.create_table_item("No Metadata", "#FFB6C1"))  # Light red
        
        # Re-enable sorting
        self.table.setSortingEnabled(True)
        
        # Restore sort state
        self.table.sortItems(sort_column, sort_order)
        
        # Resize columns to content
        self.table.resizeColumnsToContents()

class ReportsTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.project_dir = None
        self.report_root = None
        self.current_plot = ''
        self.current_stats = ''
        self.initUI()

    def initUI(self):
        layout = QHBoxLayout()

        side = QVBoxLayout()
        self.refresh_btn = QPushButton('Refresh')
        self.refresh_btn.clicked.connect(self.refresh_images)
        side.addWidget(self.refresh_btn)
        self.open_stats_btn = QPushButton('Open stats')
        self.open_stats_btn.clicked.connect(self.open_stats)
        self.open_stats_btn.setEnabled(False)
        side.addWidget(self.open_stats_btn)
        self.tree = QTreeWidget()
        self.tree.setHeaderLabel('Report figures')
        self.tree.itemClicked.connect(self.on_item_clicked)
        side.addWidget(self.tree)

        right = QVBoxLayout()
        self.path_label = QLabel()
        self.path_label.setWordWrap(True)
        right.addWidget(self.path_label)
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.image_label = QLabel('Select a figure in the report tree')
        self.image_label.setAlignment(Qt.AlignCenter)
        self.scroll_area.setWidget(self.image_label)
        right.addWidget(self.scroll_area)

        layout.addLayout(side, 1)
        layout.addLayout(right, 3)
        self.setLayout(layout)

    def set_project_dir(self, dir_path):
        self.project_dir = dir_path
        self.refresh_images()

    def _populate_tree(self, parent, nodes):
        from gui.report_browser import ReportBranch, ReportLeaf
        for node in nodes:
            item = QTreeWidgetItem(parent, [node.label])
            if isinstance(node, ReportLeaf):
                item.setData(0, Qt.UserRole, {
                    'plot': node.plot_file,
                    'stats': node.stats_file,
                })
            elif isinstance(node, ReportBranch):
                self._populate_tree(item, node.children)

    def refresh_images(self):
        self.tree.clear()
        self.current_plot = ''
        self.current_stats = ''
        self.open_stats_btn.setEnabled(False)
        if not self.project_dir:
            return
        self.report_root = os.path.join(self.project_dir, 'Report')
        if not os.path.isdir(self.report_root):
            self.image_label.setText('No Report folder yet. Generate a report first.')
            self.image_label.setPixmap(QPixmap())
            return
        from gui.report_browser import load_report_catalog
        catalog = load_report_catalog(self.report_root)
        self._populate_tree(self.tree.invisibleRootItem(), catalog)
        self.tree.expandToDepth(1)

    def on_item_clicked(self, item, _column):
        payload = item.data(0, Qt.UserRole)
        if not payload:
            return
        self.current_plot = os.path.join(self.report_root, payload['plot'])
        self.current_stats = os.path.join(self.report_root, payload['stats']) if payload.get('stats') else ''
        self.open_stats_btn.setEnabled(bool(self.current_stats and os.path.isfile(self.current_stats)))
        self.display_current_image()

    def open_stats(self):
        if self.current_stats and os.path.isfile(self.current_stats):
            subprocess.Popen(['xdg-open', self.current_stats])

    def display_current_image(self):
        if not self.current_plot or not os.path.isfile(self.current_plot):
            self.image_label.setText('No image selected')
            self.path_label.setText('')
            return
        self.path_label.setText(self.current_plot)
        pixmap = QPixmap(self.current_plot)
        if pixmap.isNull():
            self.image_label.setText(f'Error loading image: {self.current_plot}')
            return
        scaled = pixmap.scaled(
            self.scroll_area.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.image_label.setPixmap(scaled)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.current_plot:
            self.display_current_image()


class NameMappingDialog(QDialog):
    def __init__(self, project_dir, parent=None):
        super().__init__(parent)
        self.project_dir = project_dir
        self.mapping_file = os.path.join(project_dir, 'name_mapping.json')
        self.mapping = {}
        self.initUI()
        self.load_existing_mapping()
        self.load_group_names()
        
    def initUI(self):
        self.setWindowTitle('Group Name Mapping')
        self.setMinimumWidth(500)
        layout = QVBoxLayout()
        
        # Instructions
        instructions = QLabel(
            "Map original group names to display names for visualization. "
            "Leave blank to use original name."
        )
        instructions.setWordWrap(True)
        layout.addWidget(instructions)
        
        # Scrollable area for mappings
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_content = QWidget()
        self.mapping_layout = QVBoxLayout(scroll_content)
        scroll.setWidget(scroll_content)
        layout.addWidget(scroll)
        
        # Buttons
        buttons = QHBoxLayout()
        self.save_btn = QPushButton('Save Mapping')
        self.save_btn.clicked.connect(self.save_mapping)
        self.cancel_btn = QPushButton('Cancel')
        self.cancel_btn.clicked.connect(self.reject)
        buttons.addWidget(self.save_btn)
        buttons.addWidget(self.cancel_btn)
        layout.addLayout(buttons)
        
        self.setLayout(layout)
    
    def load_existing_mapping(self):
        # Load existing mapping if any
        if os.path.exists(self.mapping_file):
            try:
                with open(self.mapping_file, 'r') as f:
                    self.mapping = json.load(f)
            except (json.JSONDecodeError, OSError):
                self.mapping = {}
    
    def load_group_names(self):
        # Find all unique group names across analyses
        group_names = set()
        analysis_dir = os.path.join(self.project_dir, 'analysis')
        
        if os.path.exists(analysis_dir):
            for analysis_id in os.listdir(analysis_dir):
                group_info_path = os.path.join(analysis_dir, analysis_id, 'group_info.json')
                if os.path.exists(group_info_path):
                    try:
                        with open(group_info_path, 'r') as f:
                            group_info = json.load(f)
                            if 'group_names' in group_info:
                                for name in group_info['group_names']:
                                    group_names.add(str(name).strip())
                    except (json.JSONDecodeError, OSError):
                        pass
        
        # Clear existing layout
        while self.mapping_layout.count():
            item = self.mapping_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        # Create mapping entries
        for name in sorted(group_names):
            row = QHBoxLayout()
            row.addWidget(QLabel(f'Original: {name}'))
            edit = QLineEdit()
            edit.setPlaceholderText(f'Display name for {name}')
            
            # Set existing mapping if any
            if name in self.mapping:
                edit.setText(self.mapping[name])
                
            row.addWidget(edit)
            row.addWidget(QLabel())  # Spacer
            
            # Store the widgets for later retrieval
            setattr(self, f'edit_{name}', edit)
            
            self.mapping_layout.addLayout(row)
            
        # Add stretch to bottom
        self.mapping_layout.addStretch()
    
    def save_mapping(self):
        # Collect mappings from UI
        new_mapping = {}
        for name in self.mapping.keys():
            edit = getattr(self, f'edit_{name}', None)
            if edit and edit.text().strip():
                new_mapping[name] = edit.text().strip()
        
        # Find any new mappings we added during this session
        for child in self.findChildren(QLineEdit):
            if child.text().strip():
                # Extract the original name from placeholder text
                placeholder = child.placeholderText()
                if placeholder.startswith('Display name for '):
                    orig_name = placeholder[17:]  # Length of 'Display name for '
                    new_mapping[orig_name] = child.text().strip()
        
        # Save mapping to file
        try:
            with open(self.mapping_file, 'w') as f:
                json.dump(new_mapping, f, indent=2)
            self.mapping = new_mapping
            QMessageBox.information(self, 'Success', 'Name mapping saved successfully!')
            self.accept()
        except Exception as e:
            QMessageBox.critical(self, 'Error', f'Error saving mapping: {str(e)}')

from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from PIL import Image
import subprocess

class AboutTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Set the background color to white
        self.setAutoFillBackground(True)
        self.setStyleSheet("background-color: white;")
        
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignCenter)

        # Logo Logic (Pillow for high quality)
        self.logo_label = QLabel()
        ico_path = "../logo.ico"
        try:
            with Image.open(ico_path) as img:
                img = img.convert("RGBA").resize((200, 200), Image.Resampling.LANCZOS)
                data = img.tobytes("raw", "RGBA")
                qimg = QImage(data, img.size[0], img.size[1], QImage.Format_RGBA8888)
                self.logo_label.setPixmap(QPixmap.fromImage(qimg))
        except Exception:
            self.logo_label.setPixmap(QPixmap(ico_path).scaled(150, 150, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        
        self.logo_label.setStyleSheet("background-color: transparent;")
        layout.addWidget(self.logo_label, alignment=Qt.AlignCenter)

        # Title
        title = QLabel("ChronoRoot")
        title.setStyleSheet("font-size: 28px; font-weight: bold; color: #2c3e50; background-color: transparent;")
        layout.addWidget(title, alignment=Qt.AlignCenter)

        # Short Description
        description = QLabel("An open-source platform for high-throughput phenotyping of plant root systems.")
        description.setStyleSheet("font-size: 14px; color: #34495e; background-color: transparent; margin-bottom: 5px;")
        layout.addWidget(description, alignment=Qt.AlignCenter)

        # Website Link
        web_link = QLabel('<a href="https://chronoroot.github.io/">https://chronoroot.github.io/</a>')
        web_link.setOpenExternalLinks(True)
        web_link.setStyleSheet("font-size: 13px; background-color: transparent; margin-bottom: 20px;")
        layout.addWidget(web_link, alignment=Qt.AlignCenter)

        # Update Button
        self.update_btn = QPushButton("Check for Updates")
        self.update_btn.setFixedWidth(250)
        self.update_btn.setCursor(Qt.PointingHandCursor)
        self.update_btn.setStyleSheet("""
            QPushButton {
                background-color: #3498db; color: white; border-radius: 5px;
                padding: 10px; font-weight: bold;
            }
            QPushButton:hover { background-color: #2980b9; }
        """)
        self.update_btn.clicked.connect(self.update_software)
        layout.addWidget(self.update_btn, alignment=Qt.AlignCenter)

        # Last Commit Info
        self.commit_label = QLabel(f"Last update: {self.get_last_commit_time()}")
        self.commit_label.setStyleSheet("color: #95a5a6; background-color: transparent; margin-top: 15px;")
        layout.addWidget(self.commit_label, alignment=Qt.AlignCenter)

        self.setLayout(layout)

    def get_git_hash(self):
        """Returns the current git commit hash (language independent)."""
        try:
            return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
        except (subprocess.SubprocessError, OSError):
            return None

    def get_last_commit_time(self):
        """Fetches the ISO date of the last local git commit."""
        try:
            cmd = ["git", "log", "-1", "--format=%cd", "--date=short"]
            return subprocess.check_output(cmd).decode().strip()
        except (subprocess.SubprocessError, OSError):
            return "Unknown"

    def update_software(self):
        """Performs a git pull and compares hashes to detect updates."""
        try:
            self.update_btn.setText("Checking...")
            self.update_btn.setEnabled(False)
            QApplication.processEvents()

            old_hash = self.get_git_hash()
            subprocess.check_call(["git", "pull"], stderr=subprocess.STDOUT)
            new_hash = self.get_git_hash()

            if old_hash == new_hash:
                QMessageBox.information(self, "Update", "ChronoRoot is already up to date!")
            else:
                QMessageBox.information(self, "Update Success", 
                    "Update downloaded successfully!\nPlease restart the application to apply changes.")
                self.commit_label.setText(f"Last update: {self.get_last_commit_time()}")

        except Exception:
            QMessageBox.critical(self, "Update Error", 
                "Failed to update. Please check your internet connection or git installation.")
        
        finally:
            self.update_btn.setText("Check for Updates")
            self.update_btn.setEnabled(True)

        
class ScreeningGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.project_dir = None
        self.initUI()
    
    def initUI(self):
        self.setWindowTitle('ChronoRoot Screening Interface')
        self.setGeometry(100, 100, 1200, 800)
        
        # Create tab widget
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)
        
        # Create and store tabs as instance variables
        self.analysis_tab = AnalysisTab(self)  # passing self as main_window
        self.results_tab = ResultsTab()
        self.reports_tab = ReportsTab()
        self.about_tab = AboutTab()
        
        # Add tabs
        self.tabs.addTab(self.analysis_tab, "Analysis")
        self.tabs.addTab(self.results_tab, "Results")
        self.tabs.addTab(self.reports_tab, "Reports")
        self.tabs.addTab(self.about_tab, "About")
        
    def set_project_dir(self, dir_path):
        """Update project directory for all tabs"""
        self.project_dir = dir_path
        # Update Results tab
        if hasattr(self, 'results_tab'):
            self.results_tab.set_project_dir(dir_path)
            self.results_tab.refresh_results()  # Explicitly refresh the results
        # Update Reports tab
        if hasattr(self, 'reports_tab'):
            self.reports_tab.set_project_dir(dir_path)
            self.reports_tab.refresh_images()  # Explicitly refresh the reports

def main():
    app = QApplication(sys.argv)
    ex = ScreeningGUI()
    ex.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()