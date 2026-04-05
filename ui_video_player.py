import logging
from pathlib import Path
import threading
import cv2

from PySide6.QtCore import Signal, Qt, Slot, QPoint, QUrl, QMimeData, QThread, QTimer
from PySide6.QtGui import QImage, QPixmap, QMouseEvent, QResizeEvent, QKeyEvent, QWheelEvent, QDrag
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QSlider, QSizePolicy, QFileDialog

import icons
from ui_thumbnails import NavThumbnail

logger = logging.getLogger(__name__)


class VideoWorkerThread(QThread):
    frame_ready = Signal(QImage, int)  # QImage, frame index
    video_loaded = Signal(float, int)  # fps, total_frames

    def __init__(self, parent=None):
        super().__init__(parent)
        self.filepath = None
        self.cap = None
        self.current_filepath = None
        self._is_playing = False
        self._seek_target = -1
        self._stop_requested = False
        self._step_direction = 0
        self.lock = threading.Lock()
        self.condition = threading.Condition(self.lock)

    def load_video(self, filepath):
        with self.lock:
            self.filepath = filepath
            self._is_playing = False
            self._seek_target = 0
            self._step_direction = 0
            self.condition.notify_all()

    def play(self):
        with self.lock:
            self._is_playing = True
            self.condition.notify_all()

    def pause(self):
        with self.lock:
            self._is_playing = False

    def seek(self, frame_idx):
        with self.lock:
            self._seek_target = frame_idx
            self.condition.notify_all()

    def step(self, direction):
        with self.lock:
            self._step_direction = direction
            self._is_playing = False
            self.condition.notify_all()

    def stop(self):
        with self.lock:
            self._stop_requested = True
            self.condition.notify_all()

    def run(self):
        while True:
            with self.lock:
                if self._stop_requested:
                    break

                if not self.filepath:
                    self.condition.wait()
                    continue

                # Setup new video if filepath changed
                if self.cap is None or self.current_filepath != self.filepath:
                    if self.cap:
                        self.cap.release()
                    self.current_filepath = self.filepath
                    self.cap = cv2.VideoCapture(self.filepath)
                    if self.cap.isOpened():
                        fps = self.cap.get(cv2.CAP_PROP_FPS)
                        total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        self.video_loaded.emit(fps if fps > 0 else 30.0, total)
                    else:
                        self.filepath = None
                        continue

            with self.lock:
                if self._stop_requested:
                    break

                is_playing = self._is_playing
                seek_target = self._seek_target
                step_direction = self._step_direction

                if not is_playing and seek_target == -1 and step_direction == 0:
                    self.condition.wait()
                    continue

            # Handle seeking
            if seek_target != -1:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, seek_target)
                with self.lock:
                    self._seek_target = -1

            # Handle stepping backward (requires seek)
            if step_direction == -1:
                current_pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, current_pos - 2))
                with self.lock:
                    self._step_direction = 0
            elif step_direction == 1:
                with self.lock:
                    self._step_direction = 0

            ret, frame = self.cap.read()
            if ret:
                current_idx = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_frame.shape
                bytes_per_line = ch * w
                q_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888).copy()

                self.frame_ready.emit(q_image, current_idx)
            else:
                # Loop video
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            # Sleep to match FPS if playing
            if is_playing and ret:
                fps = self.cap.get(cv2.CAP_PROP_FPS)
                delay = int(1000.0 / (fps if fps > 0 else 30.0))
                QThread.msleep(delay)

        if self.cap:
            self.cap.release()


class OpenCVVideoPlayer(QWidget):
    """
    Video player using OpenCV for reliable cross-platform playback.
    Uses QTimer to drive frame updates.
    """

    closed = Signal()
    next_requested = Signal()
    prev_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.current_filepath = None
        self.video_capture = None
        self.video_fps = 30.0
        self.total_frames = 0
        self.current_frame_idx = 0
        self.is_playing = False
        self.current_frame = None  # Store current frame for extraction

        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(10, 0, 10, 0)
        main_layout.setSpacing(10)

        # Prev Button
        prev_container_layout = QVBoxLayout()
        prev_container_layout.addStretch(1)
        self.prev_btn = NavThumbnail("prev", self)
        self.prev_btn.clicked.connect(self.prev_requested.emit)
        prev_container_layout.addWidget(self.prev_btn)
        prev_container_layout.addStretch(1)
        main_layout.addLayout(prev_container_layout)

        # Center Container
        center_layout = QVBoxLayout()

        # --- Video/Image Display ---
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.video_label.setMinimumSize(200, 200)
        self.video_label.setStyleSheet("background-color: black;")
        self.video_label.setAcceptDrops(False)

        # Tag badge overlay — shown on top of video_label when image is tagged
        self._tag_overlay = QLabel("★", self.video_label)
        self._tag_overlay.setStyleSheet(
            "background-color: rgba(255, 200, 0, 220);"
            "color: #333;"
            "border-radius: 14px;"
            "font-size: 16px;"
            "font-weight: bold;"
            "padding: 2px 6px;"
        )
        self._tag_overlay.setFixedSize(28, 28)
        self._tag_overlay.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._tag_overlay.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self._tag_overlay.setVisible(False)
        self._tag_overlay.raise_()

        center_layout.addWidget(self.video_label, 1)

        # --- Video Controls UI ---
        self.video_controls = QWidget()
        controls_layout = QHBoxLayout(self.video_controls)
        controls_layout.setContentsMargins(0, 5, 0, 5)

        self.play_btn = QPushButton()
        self.play_btn.setIcon(icons.create_icon(icons.SVG_PLAY))
        self.play_btn.setFixedWidth(40)
        self.play_btn.clicked.connect(self._toggle_play_pause)

        self.step_back_btn = QPushButton()
        self.step_back_btn.setIcon(icons.create_icon(icons.SVG_STEP_BACK))
        self.step_back_btn.setFixedWidth(40)
        self.step_back_btn.clicked.connect(lambda: self._step_frame(-1))
        self.step_back_btn.setToolTip("Step Backward 1 Frame")

        self.step_fwd_btn = QPushButton()
        self.step_fwd_btn.setIcon(icons.create_icon(icons.SVG_STEP_FWD))
        self.step_fwd_btn.setFixedWidth(40)
        self.step_fwd_btn.clicked.connect(lambda: self._step_frame(1))
        self.step_fwd_btn.setToolTip("Step Forward 1 Frame")

        self.timeline_slider = QSlider(Qt.Orientation.Horizontal)
        self.timeline_slider.sliderMoved.connect(self._on_slider_moved)
        self.timeline_slider.setRange(0, 1000)

        self.extract_btn = QPushButton("Save Frame")
        self.extract_btn.setIcon(icons.create_icon(icons.SVG_SAVE))
        self.extract_btn.clicked.connect(self._extract_current_frame)

        controls_layout.addWidget(self.play_btn)
        controls_layout.addWidget(self.step_back_btn)
        controls_layout.addWidget(self.step_fwd_btn)
        controls_layout.addWidget(self.timeline_slider)
        controls_layout.addWidget(self.extract_btn)

        center_layout.addWidget(self.video_controls)
        self.video_controls.hide()

        main_layout.addLayout(center_layout, 1)

        # Next Button
        next_container_layout = QVBoxLayout()
        next_container_layout.addStretch(1)
        self.next_btn = NavThumbnail("next", self)
        self.next_btn.clicked.connect(self.next_requested.emit)
        next_container_layout.addWidget(self.next_btn)
        next_container_layout.addStretch(1)
        main_layout.addLayout(next_container_layout)

        # Setup Video Worker Thread
        self.video_worker = VideoWorkerThread()
        self.video_worker.frame_ready.connect(self._on_frame_ready)
        self.video_worker.video_loaded.connect(self._on_video_loaded)
        self.video_worker.start()

    def set_tag_state(self, is_tagged: bool):
        """Show or hide the tag badge overlay."""
        self._tag_overlay.setVisible(is_tagged)
        # Position in top-right corner of the video_label
        self._tag_overlay.move(self.video_label.width() - self._tag_overlay.width() - 8, 8)

    def set_media_data(self, current_path: str, prev_path: str | None, next_path: str | None):
        self.current_filepath = current_path
        self.prev_btn.set_filepath(prev_path)
        self.next_btn.set_filepath(next_path)

        # Stop playback and release any existing capture before loading new media
        self._stop_playback()
        if self.video_capture is not None:
            self.video_capture.release()
            self.video_capture = None

        if not current_path:
            self.video_label.setPixmap(QPixmap())
            self.video_controls.hide()
            return

        if current_path.lower().endswith(".mp4"):
            self.video_controls.show()
            self._load_video(current_path)
        else:
            self.video_controls.hide()
            pixmap = QPixmap(current_path)
            if not pixmap.isNull():
                self._display_pixmap(pixmap, is_video=False)
            else:
                self.video_label.setText("Could not load image.")

    def _load_video(self, filepath: str):
        self.video_worker.load_video(filepath)

    @Slot(float, int)
    def _on_video_loaded(self, fps: float, total_frames: int):
        self.video_fps = fps
        self.total_frames = total_frames
        self.current_frame_idx = 0
        self.is_playing = True
        self.video_worker.play()
        self.play_btn.setIcon(icons.create_icon(icons.SVG_PAUSE))

    def _start_playback(self):
        self.is_playing = True
        self.video_worker.play()
        self.play_btn.setIcon(icons.create_icon(icons.SVG_PAUSE))

    def _stop_playback(self):
        self.is_playing = False
        self.video_worker.pause()
        self.play_btn.setIcon(icons.create_icon(icons.SVG_PLAY))

    def _toggle_play_pause(self):
        if self.is_playing:
            self._stop_playback()
        else:
            self._start_playback()

    @Slot(QImage, int)
    def _on_frame_ready(self, q_image: QImage, frame_idx: int):
        self.current_frame = q_image.copy()
        self.current_frame_idx = frame_idx
        pixmap = QPixmap.fromImage(q_image)
        self._display_pixmap(pixmap, is_video=True)
        self._update_slider()

    def _update_slider(self):
        if self.total_frames > 0:
            progress = int((self.current_frame_idx / self.total_frames) * 1000)
            self.timeline_slider.blockSignals(True)
            self.timeline_slider.setValue(progress)
            self.timeline_slider.blockSignals(False)

    def _on_slider_moved(self, value: int):
        target_frame = int((value / 1000.0) * self.total_frames)
        self.video_worker.seek(target_frame)

    def _step_frame(self, direction: int):
        self._stop_playback()
        self.video_worker.step(direction)

    def _display_pixmap(self, pixmap: QPixmap, is_video: bool = False):
        """Scale and display a pixmap."""
        # Use FastTransformation for 30/60fps video to prevent CPU overload,
        # but keep SmoothTransformation for standard static images
        transform_mode = (
            Qt.TransformationMode.FastTransformation if is_video else Qt.TransformationMode.SmoothTransformation
        )

        scaled = pixmap.scaled(self.video_label.size(), Qt.AspectRatioMode.KeepAspectRatio, transform_mode)
        self.video_label.setPixmap(scaled)

    def _display_frame(self, frame):
        """Convert OpenCV frame to QPixmap and display it."""
        self.current_frame = frame.copy()  # Store original BGR for extraction

        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w

        # .copy() is CRITICAL here so Qt maintains ownership of the memory
        # when the Python numpy array is garbage collected!
        q_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888).copy()
        pixmap = QPixmap.fromImage(q_image)

        # Display as video (FastTransformation for CPU efficiency)
        self._display_pixmap(pixmap, is_video=True)

    def _extract_current_frame(self):
        """Save the current video frame as a PNG file."""
        if self.current_frame is None or not self.current_filepath:
            return

        video_name = Path(self.current_filepath).stem
        default_path = f"{video_name}_frame_{self.current_frame_idx}.png"

        filepath, _ = QFileDialog.getSaveFileName(self, "Save Frame As PNG", default_path, "PNG Images (*.png)")

        if filepath:
            if not filepath.lower().endswith(".png"):
                filepath += ".png"
            # Save frame using Qt native image saving
            self.current_frame.save(filepath, "PNG")
            logger.info(f"Saved frame to: {filepath}")

    def get_current_frame_pixmap(self) -> QPixmap | None:
        """Get the current video frame as a QPixmap for clipboard copying.

        Returns the current frame if viewing a paused/stopped video,
        or None if viewing a static image or no frame is available.
        """
        if self.current_frame is None:
            return None

        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w

        # Create QImage with copy to ensure Qt owns the memory
        q_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888).copy()
        return QPixmap.fromImage(q_image)

    def resizeEvent(self, event: QResizeEvent):
        # Redisplay current content scaled
        if self.current_filepath and not self.current_filepath.lower().endswith(".mp4"):
            pixmap = QPixmap(self.current_filepath)
            if not pixmap.isNull():
                self._display_pixmap(pixmap, is_video=False)
        elif self.current_frame is not None:
            self._display_frame(self.current_frame)

        if self._tag_overlay.isVisible():
            self._tag_overlay.move(self.video_label.width() - self._tag_overlay.width() - 8, 8)

        target_height = int(self.height() * 0.25)
        target_height = max(80, min(300, target_height))

        self.prev_btn.setFixedHeight(target_height)
        self.next_btn.setFixedHeight(target_height)

        super().resizeEvent(event)

    def keyPressEvent(self, event: QKeyEvent):
        key = event.key()
        if key == Qt.Key.Key_Left:
            if self.current_filepath and self.current_filepath.lower().endswith(".mp4"):
                self._step_frame(-1)
            else:
                self.prev_requested.emit()
        elif key == Qt.Key.Key_Right:
            if self.current_filepath and self.current_filepath.lower().endswith(".mp4"):
                self._step_frame(1)
            else:
                self.next_requested.emit()
        elif key == Qt.Key.Key_Space:
            if self.current_filepath and self.current_filepath.lower().endswith(".mp4"):
                self._toggle_play_pause()
        elif key == Qt.Key.Key_Escape:
            self.closed.emit()
        else:
            super().keyPressEvent(event)

    def wheelEvent(self, event: QWheelEvent):
        if self.current_filepath and self.current_filepath.lower().endswith(".mp4"):
            if event.angleDelta().y() > 0:
                self._step_frame(-1)
            else:
                self._step_frame(1)
        else:
            if event.angleDelta().y() > 0:
                self.prev_requested.emit()
            else:
                self.next_requested.emit()

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton and self.current_filepath:
            drag = QDrag(self)
            mime_data = QMimeData()
            urls = [QUrl.fromLocalFile(self.current_filepath)]
            mime_data.setUrls(urls)
            drag.setMimeData(mime_data)

            if self.video_label.pixmap():
                pixmap = self.video_label.pixmap().scaled(100, 100, Qt.AspectRatioMode.KeepAspectRatio)
                drag.setPixmap(pixmap)
                drag.setHotSpot(QPoint(pixmap.width() // 2, pixmap.height() // 2))

            drag.exec(Qt.DropAction.CopyAction)

    def mouseDoubleClickEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self.closed.emit()
            event.accept()
        else:
            super().mouseDoubleClickEvent(event)

    def stop_media(self):
        """Stop video playback when navigating away."""
        self._stop_playback()
        if self.video_capture is not None:
            self.video_capture.release()
            self.video_capture = None

    def cleanup(self):
        """Clean up resources."""
        self.stop_media()
        if self.video_worker.isRunning():
            self.video_worker.stop()
            self.video_worker.wait()


# Alias for backward compatibility
SingleMediaViewer = OpenCVVideoPlayer
