import sys
import os
import PIL.Image
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QHBoxLayout, QPushButton, QLabel, QMessageBox,
    QScrollArea, QLineEdit, QFormLayout,
)
from PyQt5.QtCore import Qt, pyqtSignal, QPoint
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QFont
import argparse

from analysis.utils.fileUtilities import getImages


class ZoomableImage(QLabel):
    point_selected = pyqtSignal(tuple)

    def __init__(self):
        super().__init__()
        self.points = []
        self.current_frame = None
        self.zoom_factor = 1.0
        self.overlay_lines = []
        self.setMouseTracking(True)

        self.panning = False
        self.last_pos = None
        self.setAlignment(Qt.AlignCenter)

    def set_overlay_lines(self, lines):
        self.overlay_lines = list(lines or [])
        self.update_display()

    def set_frame(self, frame):
        self.current_frame = frame

        if frame is not None:
            scroll_area = self.parent().parent()
            if scroll_area:
                view_width = scroll_area.viewport().width() - 20
                view_height = scroll_area.viewport().height() - 20

                frame_height, frame_width = frame.shape[:2]
                width_ratio = view_width / frame_width
                height_ratio = view_height / frame_height

                self.zoom_factor = min(width_ratio, height_ratio)

        self.update_display()

    def _draw_outlined_text(self, painter, x, y, lines):
        """High-contrast multi-line label (Analyze Plant style), above (x, y)."""
        if not lines:
            return
        font = QFont()
        font.setBold(True)
        font.setPixelSize(max(18, int(22 * self.zoom_factor)))
        painter.setFont(font)
        metrics = painter.fontMetrics()
        line_h = metrics.height() + max(2, int(4 * self.zoom_factor))
        total_h = line_h * len(lines)
        max_w = max(metrics.horizontalAdvance(line) for line in lines)
        top = y - total_h - max(8, int(10 * self.zoom_factor))
        left = x - max_w // 2
        for i, line in enumerate(lines):
            ty = top + (i + 1) * line_h
            # Black outline then white fill, matching draw_labeled_roi contrast.
            for dx, dy in (
                (-2, 0), (2, 0), (0, -2), (0, 2),
                (-2, -2), (-2, 2), (2, -2), (2, 2),
            ):
                painter.setPen(QColor(0, 0, 0))
                painter.drawText(left + dx, ty + dy, line)
            painter.setPen(QColor(255, 255, 255))
            painter.drawText(left, ty, line)

    def update_display(self):
        if self.current_frame is None:
            return

        height, width, _ = self.current_frame.shape
        bytes_per_line = 3 * width
        q_image = QImage(
            self.current_frame.data, width, height, bytes_per_line, QImage.Format_RGB888,
        )

        pixmap = QPixmap.fromImage(q_image)
        scaled_width = int(width * self.zoom_factor)
        scaled_height = int(height * self.zoom_factor)
        scaled_pixmap = pixmap.scaled(
            scaled_width, scaled_height, Qt.KeepAspectRatio, Qt.SmoothTransformation,
        )

        final_pixmap = QPixmap(scaled_pixmap)
        painter = QPainter(final_pixmap)

        pen = QPen(QColor(0, 255, 0))
        pen.setWidth(max(2, int(2 * self.zoom_factor)))
        painter.setPen(pen)

        point_radius = max(5, int(5 * self.zoom_factor))
        for px, py in self.points:
            scaled_x = int(px * self.zoom_factor)
            scaled_y = int(py * self.zoom_factor)
            painter.drawEllipse(
                scaled_x - point_radius, scaled_y - point_radius,
                point_radius * 2, point_radius * 2,
            )

        if len(self.points) == 2:
            p1 = self.points[0]
            p2 = self.points[1]

            p1_scaled = (int(p1[0] * self.zoom_factor), int(p1[1] * self.zoom_factor))
            p2_scaled = (int(p2[0] * self.zoom_factor), int(p2[1] * self.zoom_factor))
            painter.drawLine(p1_scaled[0], p1_scaled[1], p2_scaled[0], p2_scaled[1])

            mid_x = (p1_scaled[0] + p2_scaled[0]) // 2
            mid_y = (p1_scaled[1] + p2_scaled[1]) // 2
            self._draw_outlined_text(painter, mid_x, mid_y, self.overlay_lines)

        painter.end()
        self.setPixmap(final_pixmap)

    def get_image_coordinates(self, pos):
        if not self.pixmap():
            return None

        scroll_area = self.parent().parent()

        pixmap_width = self.pixmap().width()
        pixmap_height = self.pixmap().height()
        label_width = self.width()
        label_height = self.height()

        x_offset = (label_width - pixmap_width) // 2
        y_offset = (label_height - pixmap_height) // 2

        adjusted_x = pos.x() - x_offset + scroll_area.horizontalScrollBar().value()
        adjusted_y = pos.y() - y_offset + scroll_area.verticalScrollBar().value()

        original_x = int(adjusted_x / self.zoom_factor)
        original_y = int(adjusted_y / self.zoom_factor)

        if 0 <= original_x < self.current_frame.shape[1] and \
           0 <= original_y < self.current_frame.shape[0]:
            return (original_x, original_y)
        return None

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            pos = self.get_image_coordinates(event.pos())
            if pos and len(self.points) < 2:
                self.points.append(pos)
                self.point_selected.emit(pos)
                self.update_display()
        elif event.button() == Qt.RightButton:
            self.panning = True
            self.last_pos = event.pos()
            self.setCursor(Qt.ClosedHandCursor)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.RightButton:
            self.panning = False
            self.setCursor(Qt.ArrowCursor)

    def mouseMoveEvent(self, event):
        if self.panning and self.last_pos:
            delta = event.pos() - self.last_pos
            scrollarea = self.parent().parent()
            vbar = scrollarea.verticalScrollBar()
            hbar = scrollarea.horizontalScrollBar()
            hbar.setValue(hbar.value() - delta.x())
            vbar.setValue(vbar.value() - delta.y())
            self.last_pos = event.pos()

    def wheelEvent(self, event):
        if event.angleDelta().y() > 0:
            self.zoom_factor *= 1.1
        else:
            self.zoom_factor /= 1.1

        self.zoom_factor = max(0.1, min(10.0, self.zoom_factor))
        self.update_display()

    def clear_points(self):
        self.points = []
        self.overlay_lines = []
        self.update_display()


class CalibrationHelper(QMainWindow):
    distance_measured = pyqtSignal(int)
    scale_saved = pyqtSignal(float, float, int)  # pixel_size, known_mm, pixels

    def __init__(
        self,
        video_dir,
        dialog_parent=None,
        known_distance=10,
        project_dir=None,
        image_path=None,
        plate_label=None,
        plant_paths=None,
    ):
        super().__init__()
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.video_dir = os.path.abspath(os.path.expanduser(str(video_dir)))
        self._dialog_parent = dialog_parent
        self.project_dir = (project_dir or "").strip() or None
        self.image_path = os.path.abspath(
            os.path.expanduser(str(image_path or self.video_dir))
        )
        self.plate_label = plate_label or os.path.basename(self.image_path.rstrip(os.sep))
        self.plant_paths = list(plant_paths or [])
        self._known_default = known_distance
        self.initUI()
        self.load_first_frame()

    def initUI(self):
        self.setWindowTitle(f'Calibration Helper — {self.plate_label}')
        self.setGeometry(100, 100, 1000, 800)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        self.plate_header = QLabel(f"Plate / video: {self.plate_label}")
        self.plate_header.setToolTip(self.image_path)
        self.plate_header.setAlignment(Qt.AlignCenter)
        font = self.plate_header.font()
        font.setBold(True)
        self.plate_header.setFont(font)
        layout.addWidget(self.plate_header)

        path_label = QLabel(self.image_path)
        path_label.setWordWrap(True)
        path_label.setStyleSheet("color: #555;")
        path_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(path_label)

        instructions = QLabel(
            "Left click to select two points on a known length.\n"
            "Mouse wheel to zoom. Right click and drag to pan."
        )
        instructions.setAlignment(Qt.AlignCenter)
        layout.addWidget(instructions)

        form = QFormLayout()
        self.known_edit = QLineEdit(str(self._known_default))
        self.known_edit.setPlaceholderText("10")
        self.known_edit.textChanged.connect(self._refresh_measurement_ui)
        self.pixels_label = QLabel("—")
        self.scale_label = QLabel("—")
        form.addRow("Known distance (mm):", self.known_edit)
        form.addRow("Pixels:", self.pixels_label)
        form.addRow("mm/px:", self.scale_label)
        layout.addLayout(form)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)

        self.frame_widget = ZoomableImage()
        self.frame_widget.point_selected.connect(self.on_point_selected)
        scroll.setWidget(self.frame_widget)
        layout.addWidget(scroll)

        button_layout = QHBoxLayout()

        self.clear_btn = QPushButton('Clear Points')
        self.clear_btn.clicked.connect(self._clear_points)
        button_layout.addWidget(self.clear_btn)

        self.save_btn = QPushButton('Save plate scale')
        self.save_btn.setToolTip(
            "Save this scale for the video/plate. "
            "Updates calibrations.json and plant results when a plate was selected."
        )
        self.save_btn.clicked.connect(self.save_plate_scale)
        button_layout.addWidget(self.save_btn)

        self.close_btn = QPushButton('Close')
        self.close_btn.clicked.connect(self.close)
        button_layout.addWidget(self.close_btn)

        layout.addLayout(button_layout)

        self.statusBar().showMessage('Select two points to measure distance')

    def _clear_points(self):
        self.frame_widget.clear_points()
        self._refresh_measurement_ui()

    def _pixel_distance(self):
        points = self.frame_widget.points
        if len(points) != 2:
            return None
        return float(np.sqrt((points[1][0] - points[0][0])**2 +
                             (points[1][1] - points[0][1])**2))

    def _rounded_pixels(self):
        distance = self._pixel_distance()
        if distance is None:
            return 0
        return max(1, int(round(distance)))

    def _known_mm(self):
        try:
            known = float(self.known_edit.text().strip())
            if known <= 0:
                return None
            return known
        except ValueError:
            return None

    def _emit_distance(self):
        pixels = self._rounded_pixels()
        if pixels:
            self.distance_measured.emit(pixels)

    def _refresh_measurement_ui(self):
        pixels = self._rounded_pixels()
        known = self._known_mm()
        if not pixels:
            self.pixels_label.setText("—")
            self.scale_label.setText("—")
            self.frame_widget.set_overlay_lines([])
            return

        self.pixels_label.setText(str(pixels))
        lines = []
        if known is not None:
            pixel_size = known / float(pixels)
            self.scale_label.setText(f"{pixel_size:.6g}")
            lines = [
                f"{known:g} mm",
                f"{pixels} px",
                f"{pixel_size:.6g} mm/px",
            ]
        else:
            self.scale_label.setText("—")
            lines = [f"{pixels} px"]
        self.frame_widget.set_overlay_lines(lines)

    def load_first_frame(self):
        from gui.background import run_off_main_thread

        video_dir = self.video_dir

        def load():
            if not os.path.exists(video_dir):
                return ('missing_dir', None)
            images, _ = getImages({'Images': video_dir})
            if not images:
                return ('no_images', None)
            with PIL.Image.open(images[0]) as src:
                frame = np.array(src.convert("RGB"))
            return ('ok', frame)

        status, frame = run_off_main_thread(
            self._dialog_parent or self,
            "Loading calibration image…",
            load,
        )
        if status == 'missing_dir':
            QMessageBox.critical(self, "Error", "Video directory does not exist!")
            return
        if status == 'no_images':
            QMessageBox.warning(self, "Warning", "No images found in directory!")
            return
        self.frame_widget.set_frame(frame)

    def on_point_selected(self, point):
        points = self.frame_widget.points
        if len(points) == 1:
            self.statusBar().showMessage(f'First point selected at {point}')
            self._refresh_measurement_ui()
        elif len(points) == 2:
            distance = self._pixel_distance()
            self.statusBar().showMessage(
                f'Distance: {distance:.1f} pixels | Point 1: {points[0]} | Point 2: {points[1]}'
            )
            self._refresh_measurement_ui()
            self._emit_distance()

    def save_plate_scale(self):
        pixels = self._rounded_pixels()
        known = self._known_mm()
        if not pixels:
            QMessageBox.warning(self, "Calibration Helper", "Select two points first.")
            return
        if known is None:
            QMessageBox.warning(
                self, "Calibration Helper", "Enter a positive known distance in mm.",
            )
            return

        from analysis.utils.calibration_registry import (
            SOURCE_MANUAL,
            apply_scale_to_plants,
            compute_pixel_size,
            upsert_calibration,
        )

        pixel_size = compute_pixel_size(known, pixels)
        if self.project_dir:
            upsert_calibration(
                self.project_dir,
                self.image_path,
                pixel_size=pixel_size,
                source=SOURCE_MANUAL,
                known_distance=known,
                pixel_distance=pixels,
            )
        updated = 0
        if self.plant_paths:
            updated = apply_scale_to_plants(
                self.plant_paths,
                pixel_size,
                source=SOURCE_MANUAL,
                known_distance=known,
                pixel_distance=pixels,
            )

        self._emit_distance()
        self.scale_saved.emit(float(pixel_size), float(known), int(pixels))
        msg = f"Saved {pixel_size:.6g} mm/px for {self.plate_label}"
        if updated:
            msg += f" ({updated} plant result folder(s) updated)"
        self.statusBar().showMessage(msg)


def main():
    from gui.error_report import install_gui_exception_hook
    install_gui_exception_hook()
    parser = argparse.ArgumentParser(description='Video Calibration Helper')
    parser.add_argument('--video-dir', required=True, help='Directory containing video files')
    args = parser.parse_args()

    app = QApplication(sys.argv)
    ex = CalibrationHelper(args.video_dir)
    ex.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
