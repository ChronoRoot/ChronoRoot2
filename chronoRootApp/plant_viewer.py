import os
import glob
import json
import cv2
import numpy as np
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QSlider, QLabel, QPushButton, QGraphicsView,
                             QGraphicsScene, QGraphicsPixmapItem, QStyle,
                             QDialog, QGraphicsRectItem, QGraphicsTextItem,
                             QRubberBand)
from PyQt5.QtCore import Qt, QTimer, QRect, QRectF, QSize, pyqtSignal, QPointF
from PyQt5.QtGui import QPixmap, QImage, QPainter, QColor, QPen, QFont, QBrush

from analysis.imageUtils.plot import draw_labeled_roi, draw_seed_marker
from analysis.utils.fileUtilities import list_video_pngs
from analysis.utils.metadata_schema import get_bounding_box, video_image_dir
from analysis.time_windows import elapsed_day_clock
from analysis.utils.tiff_stack import (
    TiffStackReader,
    load_seg_frame,
    mask_stack_paths,
    result_frame_names,
)


# --- UTILS ---
def loadPath(path, ext="*.png"):
    return sorted(glob.glob(os.path.join(path, ext)))


def load_plant_data(plant_path):
    """
    Helper function to load data from the plant directory.
    Returns (images, segs, bbox, conf, clock_names) or raises FileNotFoundError.

    clock_names comes from Results_raw.csv when preprocessing has written it,
    in frame order, so the full-sequence viewer does not re-read every image
    name to place the clock.
    """
    json_path = os.path.join(plant_path, 'metadata.json')
    if not os.path.exists(json_path):
        raise FileNotFoundError("metadata.json not found")

    with open(json_path, 'r') as f:
        conf = json.load(f)

    bbox = get_bounding_box(conf)

    images, image_path = list_video_pngs(conf)
    if image_path:
        conf["ImagePath"] = image_path

    if not images:
        local_img_path = os.path.join(plant_path, "Images")
        if os.path.exists(local_img_path):
            images = loadPath(local_img_path, ext="*.png")

    if not images:
        raise FileNotFoundError(f"No images found. Checked: {video_image_dir(conf)}")

    layout = mask_stack_paths(plant_path)
    if layout["kind"] == "tiff" and os.path.isfile(layout["seg_multi"]):
        segs = TiffStackReader(layout["seg_multi"], bgr=True)
    else:
        segs = loadPath(layout["seg_multi"], ext="*.png")

    return images, segs, bbox, conf, result_frame_names(plant_path)


# --- CUSTOM SLIDER FOR CLICK-TO-JUMP ---
class ClickJumpSlider(QSlider):
    def mousePressEvent(self, event):
        val = QStyle.sliderValueFromPosition(
            self.minimum(), self.maximum(),
            event.x(), self.width()
        )
        self.setValue(val)
        super().mousePressEvent(event)


# --- CUSTOM GRAPHICS VIEW ---
class ZoomableGraphicsView(QGraphicsView):
    def __init__(self, parent=None, enable_roi=False, enable_point_pick=False):
        super().__init__(parent)
        self.enable_roi = enable_roi
        self.enable_point_pick = enable_point_pick
        self.setRenderHint(QPainter.Antialiasing)
        self.setRenderHint(QPainter.SmoothPixmapTransform)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setBackgroundBrush(QColor(220, 220, 220))
        self.setDragMode(QGraphicsView.NoDrag)
        self._roi_origin = None
        self._rubber_band = None
        self._panning = False
        self._pan_start = None

    def wheelEvent(self, event):
        zoomInFactor = 1.15
        zoomOutFactor = 1 / zoomInFactor
        if event.angleDelta().y() > 0:
            zoomFactor = zoomInFactor
        else:
            zoomFactor = zoomOutFactor
        self.scale(zoomFactor, zoomFactor)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and event.modifiers() & Qt.ControlModifier:
            self._panning = True
            self._pan_start = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()
            return
        if self.enable_point_pick and event.button() == Qt.LeftButton:
            self.point_selected.emit(self.mapToScene(event.pos()))
            event.accept()
            return
        if self.enable_roi and event.button() == Qt.LeftButton:
            self._roi_origin = event.pos()
            if self._rubber_band is None:
                self._rubber_band = QRubberBand(QRubberBand.Rectangle, self)
            self._rubber_band.setGeometry(QRect(self._roi_origin, QSize()))
            self._rubber_band.show()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._panning and self._pan_start is not None:
            delta = event.pos() - self._pan_start
            self._pan_start = event.pos()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
            event.accept()
            return
        if self.enable_roi and self._roi_origin is not None and self._rubber_band is not None:
            self._rubber_band.setGeometry(QRect(self._roi_origin, event.pos()).normalized())
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._panning and event.button() == Qt.LeftButton:
            self._panning = False
            self._pan_start = None
            self.unsetCursor()
            event.accept()
            return
        if self.enable_roi and event.button() == Qt.LeftButton and self._roi_origin is not None:
            rect = QRect(self._roi_origin, event.pos()).normalized()
            self._rubber_band.hide()
            self._roi_origin = None
            if rect.width() > 3 and rect.height() > 3:
                top_left = self.mapToScene(rect.topLeft())
                bottom_right = self.mapToScene(rect.bottomRight())
                scene_rect = QRectF(top_left, bottom_right).normalized()
                self.roi_selected.emit(scene_rect)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    roi_selected = pyqtSignal(QRectF)
    point_selected = pyqtSignal(QPointF)


def _cv2_to_qpixmap(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, ch = img.shape
    bytesPerLine = 3 * w
    if not img.flags['C_CONTIGUOUS']:
        img = np.ascontiguousarray(img)
    qImg = QImage(img.data.tobytes(), w, h, bytesPerLine, QImage.Format_RGB888)
    return QPixmap.fromImage(qImg)


def _overlay_label_segmentation(img, seg, colors=None):
    """Paint class colors onto labeled pixels only (no alpha blend)."""
    if colors is None:
        colors = {
            1: (0, 0, 255),
            2: (0, 255, 0),
            3: (255, 0, 0),
            4: (0, 255, 255),
            5: (255, 255, 0),
            6: (255, 0, 255),
        }
    if seg is None:
        return img
    if len(seg.shape) == 3 and seg.shape[2] == 4:
        seg = cv2.cvtColor(seg, cv2.COLOR_BGRA2BGR)
    if seg.shape[:2] != img.shape[:2]:
        return img
    out = img.copy()
    if len(seg.shape) == 3:
        labeled = np.any(seg > 0, axis=-1)
        if np.any(labeled):
            out[labeled] = seg[labeled]
        return out
    for val, color in colors.items():
        out[seg == val] = color
    return out


# --- MAIN WINDOW CLASS ---
class ChronoViewWindow(QMainWindow):
    def __init__(self, images, segFiles, bbox, conf, parent=None, clock_names=None):
        super().__init__(parent)
        self.images = images
        self.segFiles = segFiles
        self.bbox = bbox
        self.conf = conf

        n_seg = len(segFiles) if segFiles is not None else 0
        self.n = min(len(images), n_seg) if n_seg else len(images)
        self.idx = 0
        self.playing = False
        self.use_seg = False

        self.timeStep = conf.get('timeStep', 15)
        if conf.get('processingLimit', 0) not in (None, '', 0, '0'):
            try:
                limit_days = int(conf['processingLimit'])
            except (TypeError, ValueError):
                limit_days = 0
            try:
                step = int(self.timeStep)
            except (TypeError, ValueError):
                step = 15
            if limit_days > 0 and step > 0:
                frames_per_day = (24 * 60) // step
                self.n = min(self.n, limit_days * frames_per_day)

        # Preview and analysis playback stop at the processing limit only.
        # The analysis period is applied later, when postprocess writes the
        # frames that reports and overviews are allowed to use.
        self._frame_index = list(range(self.n))
        saved = list(clock_names or self.images)
        self.days, self.hours, self.minutes = elapsed_day_clock(
            saved[:self.n], self.timeStep,
        )

        self.setWindowTitle("ChronoRoot Viewer")
        self.resize(900, 800)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        self.lbl_info = QLabel("Scroll to zoom. Ctrl+drag to pan.")
        self.lbl_info.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.lbl_info)

        self.scene = QGraphicsScene()
        self.view = ZoomableGraphicsView()
        self.view.setScene(self.scene)
        self.pixmap_item = QGraphicsPixmapItem()
        self.scene.addItem(self.pixmap_item)
        layout.addWidget(self.view)

        self.lbl_frame = QLabel()
        self.lbl_frame.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.lbl_frame)

        self.slider = ClickJumpSlider(Qt.Horizontal)
        self.slider.setRange(0, max(0, self.n - 1))
        self.slider.valueChanged.connect(self.set_frame)
        layout.addWidget(self.slider)

        h_layout = QHBoxLayout()
        self.btn_play = QPushButton("Play")
        self.btn_play.clicked.connect(self.toggle_play)

        self.btn_seg = QPushButton("Toggle Segmentation")
        self.btn_seg.clicked.connect(self.toggle_seg)

        for b in [self.btn_play, self.btn_seg]:
            h_layout.addWidget(b)

        layout.addLayout(h_layout)

        self.timer = QTimer()
        self.timer.timeout.connect(self.next_frame)

        self.update_display()

    def showEvent(self, event):
        super().showEvent(event)
        QTimer.singleShot(0, self.fit_image)

    def fit_image(self):
        if self.pixmap_item.pixmap():
            self.view.fitInView(self.pixmap_item, Qt.KeepAspectRatio)

    def cv2_to_qpixmap(self, img):
        return _cv2_to_qpixmap(img)

    def update_display(self):
        if self.idx >= self.n or self.idx >= len(self._frame_index):
            return
        src = self._frame_index[self.idx]
        if src >= len(self.images):
            return

        img = cv2.imread(self.images[src])
        if img is None:
            return

        if self.bbox and len(self.bbox) == 4:
            y1, y2, x1, x2 = self.bbox
            h, w = img.shape[:2]
            if 0 <= y1 < y2 <= h and 0 <= x1 < x2 <= w:
                img = img[y1:y2, x1:x2]

        if self.use_seg and self.segFiles is not None:
            seg = load_seg_frame(self.segFiles, src)
            if seg is not None:
                img = _overlay_label_segmentation(img, seg)

        self.pixmap_item.setPixmap(self.cv2_to_qpixmap(img))
        self.lbl_frame.setText(
            f"Frame {self.idx + 1}/{self.n}  |  Day {self.days[self.idx]}  Time {self.hours[self.idx]:02d}:{self.minutes[self.idx]:02d}"
        )

        self.slider.blockSignals(True)
        self.slider.setValue(self.idx)
        self.slider.blockSignals(False)

    def set_frame(self, val):
        self.idx = val
        self.update_display()

    def next_frame(self):
        self.idx = (self.idx + 1) % self.n
        self.update_display()

    def toggle_play(self):
        self.playing = not self.playing
        if self.playing:
            self.timer.start(50)
            self.btn_play.setText("Pause")
        else:
            self.timer.stop()
            self.btn_play.setText("Play")

    def toggle_seg(self):
        self.use_seg = not self.use_seg
        self.update_display()

    def closeEvent(self, event):
        if isinstance(self.segFiles, TiffStackReader):
            self.segFiles.close()
        super().closeEvent(event)


class PlantROISelectorWindow(QDialog):
    """Single-plant ROI selector; prior ROIs drawn on the frame via plot.draw_labeled_roi."""

    def __init__(
        self,
        images,
        seg_files,
        previous_rois=None,
        time_delta=15,
        plant_label=None,
        own_previous_roi=None,
        parent=None,
    ):
        super().__init__(parent)
        self.images = images
        self.seg_files = seg_files
        self.previous_rois = previous_rois or []
        self.plant_label = plant_label
        self.own_previous_roi = own_previous_roi
        self.time_delta = time_delta
        self.pending_roi = None
        self._selected_roi = None
        self.image_width = 0
        self.image_height = 0
        self.use_seg = False
        self.idx = len(self.images) - 1 if self.images else 0

        self.setWindowTitle("Select Plant Region")
        self.resize(950, 850)
        self.setFocusPolicy(Qt.StrongFocus)

        layout = QVBoxLayout(self)
        if self.plant_label:
            self.lbl_group = QLabel(f"Select ROI for: {self.plant_label}")
            self.lbl_group.setAlignment(Qt.AlignCenter)
            self.lbl_group.setStyleSheet("font-weight: bold; color: #003366;")
            layout.addWidget(self.lbl_group)
        self.lbl_info = QLabel("Drag a rectangle on the image. Press Enter to confirm. Scroll to zoom. Ctrl+drag to pan.")
        self.lbl_info.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.lbl_info)

        self.scene = QGraphicsScene()
        self.view = ZoomableGraphicsView(enable_roi=True)
        self.view.setScene(self.scene)
        self.view.roi_selected.connect(self._on_roi_selected)
        self.pixmap_item = QGraphicsPixmapItem()
        self.scene.addItem(self.pixmap_item)
        layout.addWidget(self.view)

        self.lbl_frame = QLabel()
        self.lbl_frame.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.lbl_frame)

        self.slider = ClickJumpSlider(Qt.Horizontal)
        self.slider.valueChanged.connect(self._set_frame)
        layout.addWidget(self.slider)

        nav_layout = QHBoxLayout()
        self.btn_seg = QPushButton("Toggle Segmentation")
        self.btn_seg.clicked.connect(self._toggle_seg)
        nav_layout.addWidget(self.btn_seg)
        nav_layout.addStretch()
        layout.addLayout(nav_layout)

        action_layout = QHBoxLayout()
        self.btn_confirm = QPushButton("Confirm Selection")
        self.btn_cancel = QPushButton("Cancel Analysis")
        self.btn_confirm.clicked.connect(self._confirm_current)
        self.btn_cancel.clicked.connect(self.reject)
        self.btn_confirm.setEnabled(False)
        action_layout.addWidget(self.btn_confirm)
        action_layout.addWidget(self.btn_cancel)
        layout.addLayout(action_layout)

        self._update_display(refit=True)

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Return, Qt.Key_Enter):
            self._confirm_current()
            event.accept()
            return
        if event.key() == Qt.Key_Escape:
            self._clear_pending_overlay()
            self.btn_confirm.setEnabled(False)
            event.accept()
            return
        super().keyPressEvent(event)

    def get_roi(self):
        return self._selected_roi if self.result() == QDialog.Accepted else None

    def _set_frame(self, val):
        if val != self.idx:
            self._clear_pending_overlay()
            self.btn_confirm.setEnabled(False)
        self.idx = val
        self._update_display(refit=True)

    def _toggle_seg(self):
        self.use_seg = not self.use_seg
        self._update_display(refit=False)

    def _load_frame(self):
        if not self.images:
            return None
        img = cv2.imread(self.images[self.idx])
        if img is None:
            return None
        self.image_height, self.image_width = img.shape[:2]

        for label, x1, y1, x2, y2 in self.previous_rois:
            draw_labeled_roi(img, x1, y1, x2, y2, label, color=(255, 0, 0))

        if self.own_previous_roi:
            x1, y1, x2, y2 = self.own_previous_roi
            label = self.plant_label or "current plant"
            draw_labeled_roi(img, x1, y1, x2, y2, label, color=(0, 0, 255))

        if self.use_seg and self.seg_files and self.idx < len(self.seg_files):
            seg = cv2.imread(self.seg_files[self.idx], cv2.IMREAD_UNCHANGED)
            if seg is not None:
                img = _overlay_label_segmentation(img, seg)

        return img

    def _frame_time_text(self):
        minutes = (self.idx * self.time_delta) % 60
        hours = int((self.idx * self.time_delta / 60) % 24)
        days = int(self.idx * self.time_delta // 1440)
        return f"Frame {self.idx + 1}/{len(self.images)}  |  Day {days}  Time {hours:02d}:{int(minutes):02d}"

    def _update_display(self, refit=False):
        img = self._load_frame()
        if img is None:
            self.lbl_frame.setText("Failed to load frame")
            return

        self.pixmap_item.setPixmap(_cv2_to_qpixmap(img))
        self.scene.setSceneRect(QRectF(self.pixmap_item.pixmap().rect()))
        self.slider.blockSignals(True)
        self.slider.setRange(0, len(self.images) - 1)
        self.slider.setValue(self.idx)
        self.slider.blockSignals(False)

        self.lbl_frame.setText(self._frame_time_text())
        if self.pending_roi:
            self.lbl_info.setText(
                "Press Enter or Confirm to apply this selection. Scroll to zoom. Ctrl+drag to pan."
            )
        else:
            self.lbl_info.setText(
                "Drag a rectangle on the image. Press Enter to confirm. Scroll to zoom. Ctrl+drag to pan."
            )
        if refit:
            QTimer.singleShot(0, self._fit_image)

    def _fit_image(self):
        if self.pixmap_item.pixmap():
            self.view.fitInView(self.pixmap_item, Qt.KeepAspectRatio)

    def _clear_pending_overlay(self):
        if self.pending_roi is None:
            return
        _, rect_item = self.pending_roi
        self.scene.removeItem(rect_item)
        self.pending_roi = None

    def _on_roi_selected(self, scene_rect):
        self._clear_pending_overlay()
        if scene_rect.width() < 1 or scene_rect.height() < 1:
            return
        pen = QPen(QColor(255, 255, 255), 6)
        rect_item = QGraphicsRectItem(scene_rect)
        rect_item.setPen(pen)
        self.scene.addItem(rect_item)
        self.pending_roi = (scene_rect, rect_item)
        self.btn_confirm.setEnabled(True)
        self.lbl_info.setText(
            "Press Enter or Confirm to apply this selection. Scroll to zoom. Ctrl+drag to pan."
        )

    def _confirm_current(self):
        if not self.pending_roi:
            return
        scene_rect, rect_item = self.pending_roi
        x1 = int(max(0, min(scene_rect.left(), self.image_width - 1)))
        y1 = int(max(0, min(scene_rect.top(), self.image_height - 1)))
        x2 = int(max(0, min(scene_rect.right(), self.image_width)))
        y2 = int(max(0, min(scene_rect.bottom(), self.image_height)))
        if x2 <= x1 or y2 <= y1:
            self.lbl_info.setText("Invalid selection. Please draw a larger rectangle.")
            return
        self._selected_roi = (x1, y1, x2, y2)
        self.scene.removeItem(rect_item)
        self.pending_roi = None
        self.accept()


class SeedSelectorWindow(QDialog):
    """Pick root origin inside the selected plant ROI."""

    def __init__(self, images, seg_files, bbox, conf, parent=None):
        super().__init__(parent)
        self.images = images
        self.seg_files = seg_files
        self.bbox = bbox
        self.time_delta = conf.get('timeStep', 15)
        self.use_seg = False
        self.seed_pos = None
        self.idx = len(self.images) - 1 if self.images else 0

        self.setWindowTitle("Select Root Origin")
        self.resize(700, 900)
        self.setFocusPolicy(Qt.StrongFocus)

        layout = QVBoxLayout(self)
        self.lbl_info = QLabel(
            "Click root origin. Enter to confirm. Scroll to zoom. Ctrl+drag to pan."
        )
        self.lbl_info.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.lbl_info)

        self.scene = QGraphicsScene()
        self.view = ZoomableGraphicsView(enable_point_pick=True)
        self.view.setScene(self.scene)
        self.view.point_selected.connect(self._on_point_selected)
        self.pixmap_item = QGraphicsPixmapItem()
        self.scene.addItem(self.pixmap_item)
        layout.addWidget(self.view)

        self.lbl_frame = QLabel()
        self.lbl_frame.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.lbl_frame)

        self.slider = ClickJumpSlider(Qt.Horizontal)
        self.slider.valueChanged.connect(self._set_frame)
        layout.addWidget(self.slider)

        nav_layout = QHBoxLayout()
        self.btn_seg = QPushButton("Toggle Segmentation")
        self.btn_seg.clicked.connect(self._toggle_seg)
        nav_layout.addWidget(self.btn_seg)
        nav_layout.addStretch()
        layout.addLayout(nav_layout)

        action_layout = QHBoxLayout()
        self.btn_confirm = QPushButton("Confirm")
        self.btn_cancel = QPushButton("Cancel Analysis")
        self.btn_confirm.clicked.connect(self._confirm)
        self.btn_cancel.clicked.connect(self.reject)
        self.btn_confirm.setEnabled(False)
        action_layout.addWidget(self.btn_confirm)
        action_layout.addWidget(self.btn_cancel)
        layout.addLayout(action_layout)

        self._update_display(refit=True)

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Return, Qt.Key_Enter):
            self._confirm()
            event.accept()
            return
        if event.key() == Qt.Key_Escape:
            self.reject()
            event.accept()
            return
        super().keyPressEvent(event)

    def get_seed(self):
        if self.result() == QDialog.Accepted and self.seed_pos is not None:
            return [int(self.seed_pos[0]), int(self.seed_pos[1])]
        return None

    def _toggle_seg(self):
        self.use_seg = not self.use_seg
        self._update_display(refit=False)

    def _set_frame(self, val):
        self.idx = val
        self._update_display(refit=True)

    def _load_frame(self):
        if not self.images or len(self.bbox) != 4:
            return None
        y1, y2, x1, x2 = self.bbox
        img = cv2.imread(self.images[self.idx])
        if img is None:
            return None
        h, w = img.shape[:2]
        if not (0 <= y1 < y2 <= h and 0 <= x1 < x2 <= w):
            return None
        img = img[y1:y2, x1:x2].copy()

        if self.use_seg and self.seg_files and self.idx < len(self.seg_files):
            seg = cv2.imread(self.seg_files[self.idx], cv2.IMREAD_UNCHANGED)
            if seg is not None:
                if len(seg.shape) >= 2:
                    seg = seg[y1:y2, x1:x2]
                img = _overlay_label_segmentation(img, seg)

        if self.seed_pos is not None:
            draw_seed_marker(img, int(self.seed_pos[0]), int(self.seed_pos[1]))
        return img

    def _on_point_selected(self, scene_pos):
        h, w = self.bbox[1] - self.bbox[0], self.bbox[3] - self.bbox[2]
        x = max(0, min(scene_pos.x(), w - 1))
        y = max(0, min(scene_pos.y(), h - 1))
        self.seed_pos = (x, y)
        self.btn_confirm.setEnabled(True)
        self._update_display(refit=False)

    def _update_display(self, refit=False):
        img = self._load_frame()
        if img is None:
            self.lbl_frame.setText("Failed to load frame")
            return

        self.pixmap_item.setPixmap(_cv2_to_qpixmap(img))
        self.scene.setSceneRect(QRectF(self.pixmap_item.pixmap().rect()))
        self.slider.blockSignals(True)
        self.slider.setRange(0, len(self.images) - 1)
        self.slider.setValue(self.idx)
        self.slider.blockSignals(False)

        minutes = (self.idx * self.time_delta) % 60
        hours = int((self.idx * self.time_delta / 60) % 24)
        days = int(self.idx * self.time_delta // 1440)
        self.lbl_frame.setText(
            f"Frame {self.idx + 1}/{len(self.images)}  |  Day {days}  Time {hours:02d}:{int(minutes):02d}"
        )
        if self.seed_pos is not None:
            self.lbl_info.setText(
                f"Root at ({int(self.seed_pos[0])}, {int(self.seed_pos[1])}). Enter to confirm. "
                "Scroll to zoom. Ctrl+drag to pan."
            )
        else:
            self.lbl_info.setText(
                "Click root origin. Enter to confirm. Scroll to zoom. Ctrl+drag to pan."
            )
        if refit:
            QTimer.singleShot(0, self._fit_image)

    def _fit_image(self):
        if self.pixmap_item.pixmap():
            self.view.fitInView(self.pixmap_item, Qt.KeepAspectRatio)

    def _confirm(self):
        if self.seed_pos is not None:
            self.accept()


ROI_GROUP_COLORS = [
    QColor(255, 80, 80),
    QColor(80, 180, 80),
    QColor(80, 120, 255),
    QColor(255, 200, 60),
    QColor(200, 80, 255),
    QColor(80, 220, 220),
]

ROI_BOX_PEN_WIDTH = 6
ROI_PENDING_PEN_WIDTH = 6


class GroupROISelectorWindow(QDialog):
    """Modal ROI selector built on the plant viewer controls."""

    def __init__(self, images, seg_files, group_names, time_delta=15, parent=None):
        super().__init__(parent)
        self.images = images
        self.seg_files = seg_files
        self.group_names = group_names
        self.time_delta = time_delta
        self.confirmed_groups = {}
        self.current_group_index = 0
        self.pending_roi = None
        self.roi_overlays = []
        self.image_width = 0
        self.image_height = 0

        self.setWindowTitle("Select Group Regions")
        self.resize(950, 850)
        self.setFocusPolicy(Qt.StrongFocus)

        layout = QVBoxLayout(self)

        self.lbl_group = QLabel()
        self.lbl_group.setAlignment(Qt.AlignCenter)
        self.lbl_group.setStyleSheet("font-size: 14pt; font-weight: bold; color: #003366;")
        layout.addWidget(self.lbl_group)

        self.lbl_info = QLabel()
        self.lbl_info.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.lbl_info)

        self.scene = QGraphicsScene()
        self.view = ZoomableGraphicsView(enable_roi=True)
        self.view.setScene(self.scene)
        self.view.roi_selected.connect(self._on_roi_selected)
        self.pixmap_item = QGraphicsPixmapItem()
        self.scene.addItem(self.pixmap_item)
        layout.addWidget(self.view)

        self.slider = ClickJumpSlider(Qt.Horizontal)
        self.slider.valueChanged.connect(self._set_frame)
        layout.addWidget(self.slider)

        nav_layout = QHBoxLayout()
        self.btn_seg = QPushButton("Toggle Segmentation")
        self.btn_seg.clicked.connect(self._toggle_seg)
        nav_layout.addWidget(self.btn_seg)
        nav_layout.addStretch()
        layout.addLayout(nav_layout)

        action_layout = QHBoxLayout()
        self.btn_previous = QPushButton("Previous")
        self.btn_next = QPushButton("Next")
        self.btn_confirm = QPushButton("Confirm Selection")
        self.btn_cancel = QPushButton("Cancel Analysis")
        self.btn_previous.clicked.connect(self._go_previous)
        self.btn_next.clicked.connect(self._go_next)
        self.btn_confirm.clicked.connect(self._confirm_current)
        self.btn_cancel.clicked.connect(self.reject)
        for btn in [self.btn_previous, self.btn_next, self.btn_confirm, self.btn_cancel]:
            action_layout.addWidget(btn)
        layout.addLayout(action_layout)

        self.use_seg = False
        self.idx = len(self.images) - 1 if self.images else 0
        self._update_group_label()
        self._update_display()

    def keyPressEvent(self, event):
        key = event.key()
        if key in (Qt.Key_Return, Qt.Key_Enter):
            self._confirm_current()
            event.accept()
            return
        if key == Qt.Key_Escape:
            self._clear_pending_overlay()
            self._update_confirm_button()
            event.accept()
            return
        if key == Qt.Key_Left and self.btn_previous.isEnabled():
            self._go_previous()
            event.accept()
            return
        if key == Qt.Key_Right and self.btn_next.isEnabled():
            self._go_next()
            event.accept()
            return
        super().keyPressEvent(event)

    def get_group_rois(self):
        if self.result() == QDialog.Accepted:
            return self.confirmed_groups
        return None

    def _all_groups_confirmed(self):
        return len(self.confirmed_groups) == len(self.group_names)

    def _current_group_name(self):
        return self.group_names[self.current_group_index]

    def _update_group_label(self):
        if self.current_group_index >= len(self.group_names):
            return
        name = self._current_group_name()
        pos = f"({self.current_group_index + 1} of {len(self.group_names)})"
        if name in self.confirmed_groups:
            self.lbl_group.setText(f"Update ROI for: {name}  {pos}")
        else:
            self.lbl_group.setText(f"Select ROI for: {name}  {pos}")
        self._update_confirm_button()
        self._update_nav_buttons()

    def _update_confirm_button(self):
        if self._all_groups_confirmed() and not self.pending_roi:
            self.btn_confirm.setText("Finish")
            self.btn_confirm.setEnabled(True)
        elif self.pending_roi:
            self.btn_confirm.setText("Confirm Selection")
            self.btn_confirm.setEnabled(True)
        else:
            self.btn_confirm.setText("Confirm Selection")
            self.btn_confirm.setEnabled(False)

    def _update_nav_buttons(self):
        self.btn_previous.setEnabled(self.current_group_index > 0)
        current_name = self._current_group_name()
        self.btn_next.setEnabled(
            current_name in self.confirmed_groups
            and self.current_group_index < len(self.group_names) - 1
        )

    def _go_previous(self):
        if self.current_group_index == 0:
            return
        self._clear_pending_overlay()
        self.current_group_index -= 1
        self._redraw_confirmed_overlays()
        self._update_group_label()
        self._refresh_info_line()

    def _go_next(self):
        current_name = self._current_group_name()
        if current_name not in self.confirmed_groups:
            return
        if self.current_group_index >= len(self.group_names) - 1:
            return
        self._clear_pending_overlay()
        self.current_group_index += 1
        self._redraw_confirmed_overlays()
        self._update_group_label()
        self._refresh_info_line()

    def _set_frame(self, val):
        if val != self.idx:
            self._clear_pending_overlay()
            self._update_confirm_button()
        self.idx = val
        self._update_display()

    def _toggle_seg(self):
        self.use_seg = not self.use_seg
        self._update_display()

    def _load_current_image(self):
        if not self.images:
            return None
        img = cv2.imread(self.images[self.idx])
        if img is None:
            return None

        self.image_height, self.image_width = img.shape[:2]

        if self.use_seg and self.seg_files and self.idx < len(self.seg_files):
            seg = cv2.imread(self.seg_files[self.idx], cv2.IMREAD_UNCHANGED)
            if seg is not None:
                img = _overlay_label_segmentation(img, seg)
        return img

    def _clear_roi_overlays(self):
        for overlay in self.roi_overlays:
            for key in ('rect', 'bg', 'label'):
                item = overlay.get(key)
                if item is not None:
                    self.scene.removeItem(item)
        self.roi_overlays = []

    def _add_roi_overlay(self, group_name, x1, y1, x2, y2, color):
        w = x2 - x1
        h = y2 - y1
        rect_item = QGraphicsRectItem(QRectF(x1, y1, w, h))
        rect_item.setPen(QPen(color, ROI_BOX_PEN_WIDTH))
        self.scene.addItem(rect_item)

        label = QGraphicsTextItem(group_name)
        font = QFont()
        font.setBold(True)
        point_size = min(32, max(24, int(min(w, h) / 6)))
        font.setPointSize(point_size)
        label.setFont(font)
        label.setDefaultTextColor(QColor(255, 255, 255))

        text_rect = label.boundingRect()
        while text_rect.width() > w * 0.9 and font.pointSize() > 10:
            font.setPointSize(font.pointSize() - 1)
            label.setFont(font)
            text_rect = label.boundingRect()

        tx = x1 + (w - text_rect.width()) / 2
        ty = y1 + (h - text_rect.height()) / 2

        bg = QGraphicsRectItem(text_rect.adjusted(-4, -2, 4, 2))
        bg.setBrush(QBrush(QColor(0, 0, 0, 160)))
        bg.setPen(QPen(Qt.NoPen))
        bg.setPos(tx - 4, ty - 2)
        label.setPos(tx, ty)

        self.scene.addItem(bg)
        self.scene.addItem(label)

        return {'rect': rect_item, 'bg': bg, 'label': label}

    def _redraw_confirmed_overlays(self):
        self._clear_roi_overlays()
        for idx, group_name in enumerate(self.group_names):
            if group_name not in self.confirmed_groups:
                continue
            x1, y1, x2, y2 = self.confirmed_groups[group_name]
            color = ROI_GROUP_COLORS[idx % len(ROI_GROUP_COLORS)]
            self.roi_overlays.append(
                self._add_roi_overlay(group_name, x1, y1, x2, y2, color)
            )

    def _frame_info_suffix(self):
        if self.pending_roi:
            return "Press Enter or Confirm to apply this selection."
        if self._all_groups_confirmed():
            return "All regions set — press Enter or Finish to complete"
        if self._current_group_name() in self.confirmed_groups:
            return (
                "Region already set — draw a new rectangle to update it, "
                "or press Next to continue."
            )
        return "Drag a rectangle on the image. Use the slider to change frame. Press Enter to confirm."

    def _refresh_info_line(self):
        if not self.images:
            return
        minutes = (self.idx * self.time_delta) % 60
        hours = int((self.idx * self.time_delta / 60) % 24)
        days = int(self.idx * self.time_delta // 1440)
        self.lbl_info.setText(
            f"Frame {self.idx + 1}/{len(self.images)}  |  "
            f"Day {days}  Time {hours:02d}:{int(minutes):02d}  |  "
            f"{self._frame_info_suffix()}"
        )

    def _update_display(self):
        if not self.images:
            self.lbl_info.setText("No images available")
            return

        img = self._load_current_image()
        if img is None:
            self.lbl_info.setText("Failed to load frame")
            return

        pixmap = _cv2_to_qpixmap(img)
        self.pixmap_item.setPixmap(pixmap)
        self.scene.setSceneRect(QRectF(pixmap.rect()))

        self.slider.blockSignals(True)
        self.slider.setRange(0, len(self.images) - 1)
        self.slider.setValue(self.idx)
        self.slider.blockSignals(False)

        self._redraw_confirmed_overlays()
        self._refresh_info_line()
        self._update_nav_buttons()
        QTimer.singleShot(0, self._fit_image)

    def _fit_image(self):
        if self.pixmap_item.pixmap():
            self.view.fitInView(self.pixmap_item, Qt.KeepAspectRatio)

    def _clear_pending_overlay(self):
        if self.pending_roi is None:
            return
        if isinstance(self.pending_roi, tuple):
            _, rect_item = self.pending_roi
            self.scene.removeItem(rect_item)
        elif isinstance(self.pending_roi, QGraphicsRectItem):
            self.scene.removeItem(self.pending_roi)
        self.pending_roi = None

    def _on_roi_selected(self, scene_rect):
        self._clear_pending_overlay()
        if scene_rect.width() < 1 or scene_rect.height() < 1:
            return

        pen = QPen(QColor(255, 255, 255))
        pen.setWidth(ROI_PENDING_PEN_WIDTH)
        rect_item = QGraphicsRectItem(scene_rect)
        rect_item.setPen(pen)
        self.scene.addItem(rect_item)
        self.pending_roi = (scene_rect, rect_item)
        self._update_confirm_button()
        self._refresh_info_line()

    def _confirm_current(self):
        if self._all_groups_confirmed() and not self.pending_roi:
            self.accept()
            return
        if not self.pending_roi:
            return

        scene_rect, rect_item = self.pending_roi
        x1 = int(max(0, min(scene_rect.left(), self.image_width - 1)))
        y1 = int(max(0, min(scene_rect.top(), self.image_height - 1)))
        x2 = int(max(0, min(scene_rect.right(), self.image_width)))
        y2 = int(max(0, min(scene_rect.bottom(), self.image_height)))
        if x2 <= x1 or y2 <= y1:
            self.lbl_info.setText("Invalid selection. Please draw a larger rectangle.")
            return

        group_name = self._current_group_name()
        is_update = group_name in self.confirmed_groups
        self.confirmed_groups[group_name] = (x1, y1, x2, y2)

        self.scene.removeItem(rect_item)
        self.pending_roi = None

        self._redraw_confirmed_overlays()

        if is_update:
            self._update_group_label()
            self._refresh_info_line()
            return

        if self._all_groups_confirmed():
            self.accept()
            return

        self.current_group_index += 1
        self._update_group_label()
        self._refresh_info_line()
