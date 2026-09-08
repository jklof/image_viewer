import logging
from pathlib import Path

import cv2

from PySide6.QtCore import Signal, Qt, Slot, QPoint, QUrl, QMimeData, QRect, QSize, QEvent, QTimer
from PySide6.QtGui import QImage, QPixmap, QMouseEvent, QResizeEvent, QKeyEvent, QWheelEvent, QDrag
from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QSlider, QSizePolicy, QFileDialog, QRubberBand, QFrame, QDialog, QScrollArea, QLineEdit
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput, QVideoSink, QVideoFrame

import icons
from ui_flow_layout import FlowLayout
from loader_manager import get_loader_manager, thumbnail_cache
from ui_thumbnails import NavThumbnail
from metadata_utils import get_image_metadata, ImageMetadata

logger = logging.getLogger(__name__)

SOURCE_THUMB_SIZE = 40
SOURCE_THUMB_COMPACT = 28
# Max lineage badges shown inline in the info overlay; overflow goes to a dialog.
INLINE_SOURCE_CAP = 4


class SourceBadgeWidget(QFrame):
    """Badge for one ComfyUI LoadImage reference: thumb + name + Jump/+Add.

    compact=True renders a smaller single-row chip for the inline overlay;
    the dialog uses the full-size variant.
    """

    jump_requested = Signal(str)
    add_query_requested = Signal(str)

    def __init__(self, raw_filename: str, resolved_path: str | None = None, parent=None, compact: bool = False):
        super().__init__(parent)
        self.raw_filename = raw_filename
        self.resolved_path = resolved_path
        self._thumb_connected = False
        self._press_pos: QPoint | None = None
        self._feedback_timer: QTimer | None = None
        thumb_size = SOURCE_THUMB_COMPACT if compact else SOURCE_THUMB_SIZE
        self._thumb_size = thumb_size
        name_limit = 14 if compact else 20
        btn_size = 20 if compact else 22
        self.setStyleSheet(
            "SourceBadgeWidget { background-color: rgba(30, 30, 30, 200);"
            " border: 1px solid #444; border-radius: 6px; }"
        )
        layout = QHBoxLayout(self)
        layout.setContentsMargins(5, 3, 5, 3)
        layout.setSpacing(6)

        self.thumb_label = QLabel()
        self.thumb_label.setFixedSize(thumb_size, thumb_size)
        self.thumb_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.thumb_label.setStyleSheet("border: none; background-color: #222; border-radius: 4px;")
        layout.addWidget(self.thumb_label)

        if resolved_path:
            cached = thumbnail_cache.get(resolved_path)
            if cached:
                self._apply_pixmap(cached)
            else:
                self.thumb_label.setText("...")
                get_loader_manager().thumbnail_loaded.connect(self._on_thumbnail_ready)
                self._thumb_connected = True
                get_loader_manager().request_thumbnail(resolved_path)
        else:
            self.thumb_label.setText("?")

        name = Path(raw_filename).name
        if len(name) <= name_limit:
            short = name
        else:
            keep = max(1, (name_limit - 3) // 2)
            short = name[:keep] + "..." + name[-keep:]
        name_label = QLabel(short)
        name_label.setToolTip(raw_filename if resolved_path is None else f"{raw_filename}\n{resolved_path}")
        name_label.setStyleSheet("border: none; color: #ddd; font-size: 11px;")
        layout.addWidget(name_label, 1)

        if resolved_path:
            jump_btn = QPushButton("🔍")
            jump_btn.setFixedSize(btn_size, btn_size)
            jump_btn.setToolTip(f"Jump to image:\n{resolved_path}")
            jump_btn.setStyleSheet("border: 1px solid #555; border-radius: 4px;")
            jump_btn.clicked.connect(lambda: self.jump_requested.emit(self.resolved_path))
            layout.addWidget(jump_btn)

            self.add_btn = QPushButton("+")
            self.add_btn.setFixedSize(btn_size, btn_size)
            self.add_btn.setToolTip(f"Add to search query:\n{resolved_path}")
            self.add_btn.setStyleSheet("border: 1px solid #555; border-radius: 4px;")
            self.add_btn.clicked.connect(self._on_add_clicked)
            layout.addWidget(self.add_btn)
        else:
            self.add_btn = None
            self.missing_label = QLabel("(unindexed)")
            self.missing_label.setStyleSheet("border: none; color: #777; font-style: italic; font-size: 10px;")
            layout.addWidget(self.missing_label)

    def _apply_pixmap(self, pixmap: QPixmap):
        scaled = pixmap.scaled(
            self._thumb_size,
            self._thumb_size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.thumb_label.setPixmap(scaled)

    @Slot(str)
    def _on_thumbnail_ready(self, filepath: str):
        if filepath == self.resolved_path:
            cached = thumbnail_cache.get(self.resolved_path)
            if cached:
                self._apply_pixmap(cached)
            self.dispose()  # one-shot: stop listening once our thumb arrived

    def dispose(self):
        """Disconnect the shared thumbnail signal (avoids slot leaks on clear)."""
        if not self._thumb_connected:
            return
        self._thumb_connected = False
        try:
            get_loader_manager().thumbnail_loaded.disconnect(self._on_thumbnail_ready)
        except (RuntimeError, AttributeError):
            pass

    def mousePressEvent(self, event: QMouseEvent):
        # Record press for click detection, and swallow it so it doesn't
        # bubble up to the viewer (which would start a file drag instead).
        if event.button() == Qt.MouseButton.LeftButton:
            self._press_pos = event.pos()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton and self._press_pos is not None:
            start = self._press_pos
            self._press_pos = None
            if (event.pos() - start).manhattanLength() < 6:
                # Plain click on badge chrome (not a button — those consume
                # their own presses): resolved jumps, unindexed copies name.
                if self.resolved_path:
                    self.jump_requested.emit(self.resolved_path)
                else:
                    self._copy_filename_to_clipboard()
                event.accept()
                return
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton and self.resolved_path:
            self.jump_requested.emit(self.resolved_path)
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def wheelEvent(self, event: QWheelEvent):
        # Inside a scrollable dialog the badge spans the viewport width and
        # would otherwise eat the wheel: forward it to the enclosing
        # QScrollArea. Inline (no scroll area) just swallow it so the viewer
        # doesn't switch to the next/previous image.
        p = self.parent()
        while p is not None:
            if isinstance(p, QScrollArea):
                QApplication.sendEvent(p.viewport(), event)
                return
            p = p.parent()
        event.accept()

    @Slot()
    def _on_add_clicked(self):
        self.add_query_requested.emit(self.resolved_path)
        # Inline confirmation: the query builder + status bar live behind the
        # modal dialog, so acknowledge here too.
        if self.add_btn is not None:
            self.add_btn.setText("✓")
            if self._feedback_timer is None:
                self._feedback_timer = QTimer(self)
                self._feedback_timer.setSingleShot(True)
                self._feedback_timer.timeout.connect(self._restore_add_button)
            self._feedback_timer.start(1200)

    @Slot()
    def _restore_add_button(self):
        if self.add_btn is not None:
            self.add_btn.setText("+")

    def _copy_filename_to_clipboard(self):
        """Unindexed badges have no file action; offer the recorded name."""
        try:
            QApplication.clipboard().setText(Path(self.raw_filename).name)
        except Exception:
            return
        if getattr(self, "missing_label", None) is not None:
            self.missing_label.setText("(copied ✓)")
            # Bound-method singleShot: safely dropped if the badge is gone.
            QTimer.singleShot(1200, self._restore_missing_label)

    @Slot()
    def _restore_missing_label(self):
        if getattr(self, "missing_label", None) is not None:
            try:
                self.missing_label.setText("(unindexed)")
            except RuntimeError:
                pass  # badge already destroyed


class SourceLineageDialog(QDialog):
    """Scrollable overflow dialog for workflow sources beyond the inline cap.

    Reuses SourceBadgeWidget rows. Jump closes the dialog (navigation would
    otherwise happen behind the modal); +Add and Add-all keep it open.
    """

    jump_requested = Signal(str)
    add_query_requested = Signal(str)
    add_all_requested = Signal(list)  # all resolved filepaths

    def __init__(self, target_name: str, resolved_items: list, parent=None):
        super().__init__(parent)
        self._badges: list[SourceBadgeWidget] = []
        self.setWindowTitle(f"Workflow sources — {target_name} ({len(resolved_items)})")
        self.setMinimumWidth(520)
        self.setMinimumHeight(300)

        layout = QVBoxLayout(self)

        self.filter_edit = QLineEdit()
        self.filter_edit.setPlaceholderText("Filter sources by name...")
        self.filter_edit.setClearButtonEnabled(True)
        self.filter_edit.textChanged.connect(self._on_filter_changed)
        self.filter_edit.returnPressed.connect(self._on_filter_accepted)
        layout.addWidget(self.filter_edit)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._scroll = scroll
        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setSpacing(6)
        for item in resolved_items:
            badge = SourceBadgeWidget(
                item.get("raw_filename", "?"), item.get("resolved_path"), parent=content
            )
            badge.jump_requested.connect(self._on_badge_jump)
            badge.add_query_requested.connect(self.add_query_requested.emit)
            content_layout.addWidget(badge)
            self._badges.append(badge)
        content_layout.addStretch(1)
        scroll.setWidget(content)
        layout.addWidget(scroll, 1)

        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        resolved_paths = [i.get("resolved_path") for i in resolved_items if i.get("resolved_path")]
        self.add_all_btn = QPushButton(f"Add all resolved ({len(resolved_paths)})")
        self.add_all_btn.setEnabled(bool(resolved_paths))
        self.add_all_btn.setToolTip("Stage every resolved source in the search query")
        self.add_all_btn.clicked.connect(self._on_add_all_clicked)
        btn_row.addWidget(self.add_all_btn)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.reject)
        btn_row.addWidget(close_btn)
        layout.addLayout(btn_row)

    @Slot()
    def _on_add_all_clicked(self):
        paths = [b.resolved_path for b in self._badges if b.resolved_path]
        if not paths:
            return
        self.add_all_requested.emit(list(paths))
        # The query builder lives behind the modal; acknowledge inline.
        self.add_all_btn.setText("Added ✓")
        QTimer.singleShot(1500, self._restore_add_all_button)

    @Slot()
    def _restore_add_all_button(self):
        try:
            n = sum(1 for b in self._badges if b.resolved_path)
            self.add_all_btn.setText(f"Add all resolved ({n})")
        except RuntimeError:
            pass  # dialog already closed

    @Slot(str)
    def _on_filter_changed(self, text: str):
        needle = text.strip().lower()
        for badge in self._badges:
            badge.setVisible(not needle or needle in Path(badge.raw_filename).name.lower())

    @Slot()
    def _on_filter_accepted(self):
        # Enter jumps to the first visible resolved badge.
        for badge in self._badges:
            if badge.isVisible() and badge.resolved_path:
                self._on_badge_jump(badge.resolved_path)
                return

    @Slot(str)
    def _on_badge_jump(self, filepath: str):
        self.jump_requested.emit(filepath)
        self.accept()

    def showEvent(self, event):
        super().showEvent(event)
        # Focus the filter box so typing works immediately; PgUp/PgDn still
        # reach the dialog handler (QLineEdit ignores them), arrows/Home/End
        # edit text while the filter has focus.
        QTimer.singleShot(0, self.filter_edit.setFocus)

    def keyPressEvent(self, event: QKeyEvent):
        # Scroll the badge list; anything else keeps default dialog behavior
        # (Escape closes, Enter/Space activates the focused button).
        bar = self._scroll.verticalScrollBar()
        key = event.key()
        if key == Qt.Key.Key_Up:
            bar.setValue(bar.value() - bar.singleStep())
            event.accept()
        elif key == Qt.Key.Key_Down:
            bar.setValue(bar.value() + bar.singleStep())
            event.accept()
        elif key == Qt.Key.Key_PageUp:
            bar.setValue(bar.value() - bar.pageStep())
            event.accept()
        elif key == Qt.Key.Key_PageDown:
            bar.setValue(bar.value() + bar.pageStep())
            event.accept()
        elif key == Qt.Key.Key_Home:
            bar.setValue(bar.minimum())
            event.accept()
        elif key == Qt.Key.Key_End:
            bar.setValue(bar.maximum())
            event.accept()
        else:
            super().keyPressEvent(event)

    def done(self, result: int):
        for badge in self._badges:
            badge.dispose()
        self._badges = []
        super().done(result)


class CroppableLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.rubber_band = QRubberBand(QRubberBand.Shape.Rectangle, self)
        self.rubber_band.hide()
        self.origin = QPoint()
        self._is_selecting = False
        self._shift_pressed = False

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                # Start crop selection if Shift is held
                self.origin = event.pos()
                self._is_selecting = False
                self._shift_pressed = True
                event.accept()
                return
            else:
                # Without Shift, clear existing selection and let the press
                # bubble up (Ctrl+Left starts a file drag in the viewer).
                self.clear_selection()
                self._shift_pressed = False
                event.ignore()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        if (event.buttons() & Qt.MouseButton.LeftButton) and self._shift_pressed:
            # Only start drawing rubber band if moved past a threshold
            if not self._is_selecting and (event.pos() - self.origin).manhattanLength() > 5:
                self._is_selecting = True
                self.rubber_band.setGeometry(QRect(self.origin, QSize()))
                self.rubber_band.show()

            if self._is_selecting:
                self.rubber_band.setGeometry(QRect(self.origin, event.pos()).normalized())
                event.accept()
                return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            if self._shift_pressed:
                self._is_selecting = False
                self._shift_pressed = False
                event.accept()
                return
            else:
                event.ignore()
                return
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            event.ignore()  # Bubble up to SingleMediaViewer
        else:
            super().mouseDoubleClickEvent(event)

    def has_selection(self) -> bool:
        return self.rubber_band.isVisible() and self.rubber_band.geometry().width() > 5

    def clear_selection(self):
        self.rubber_band.hide()
        self._is_selecting = False
        self._shift_pressed = False


class SingleMediaViewer(QWidget):
    """
    Single-view media inspector.

    Video playback (with sound) is driven by QtMultimedia (QMediaPlayer +
    QAudioOutput + QVideoSink), which owns the single A/V clock during
    playback. QMediaPlayer has no frame-step API, so exact single-frame
    stepping while paused is served by a small on-demand OpenCV grabber
    (one seek + one decode per discrete user action, no background thread).
    """

    closed = Signal()
    next_requested = Signal()
    prev_requested = Signal()
    source_jump_requested = Signal(str)        # resolved filepath to navigate to
    source_add_query_requested = Signal(str)   # resolved filepath to stage in query builder
    source_add_all_requested = Signal(list)    # all resolved filepaths to stage at once

    # Approximate nudge used only when the OpenCV grabber is unavailable.
    _FALLBACK_STEP_MS = 33

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.current_filepath = None
        self.current_frame = None  # Store current QImage for crops/frame saves
        self._cached_image_pixmap = QPixmap()
        self._dup_count: int = 1
        self._metadata: ImageMetadata | None = None
        self._is_video = False
        self._resolved_items: list = []  # last backend lineage results
        self._more_btn: QPushButton | None = None  # overflow button, if shown

        # --- QtMultimedia pipeline (audio + video, single clock) ---
        self.player = QMediaPlayer(self)
        self.audio_output = QAudioOutput(self)
        self.player.setAudioOutput(self.audio_output)
        self.audio_output.setVolume(1.0)

        self.video_sink = QVideoSink(self)
        self.player.setVideoSink(self.video_sink)
        self.player.setLoops(QMediaPlayer.Loops.Infinite)

        self.video_sink.videoFrameChanged.connect(self._on_video_frame_changed)
        self.player.positionChanged.connect(self._on_player_position_changed)
        self.player.playbackStateChanged.connect(self._on_playback_state_changed)
        self.player.errorOccurred.connect(self._on_player_error)

        # --- On-demand OpenCV step grabber (paused stepping only) ---
        self._step_cap: cv2.VideoCapture | None = None
        self._step_fps: float = 30.0
        self._step_total: int = 0
        # Exact frame index of current_frame when known (paused + grabbed).
        # None while playing or when position is only known approximately.
        self._step_idx: int | None = None

        self._init_ui()

    def _init_ui(self):
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
        self.video_label = CroppableLabel()
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

        # Info panel overlay — shows metadata.
        # NOTE: this container must NOT carry WA_TransparentForMouseEvents:
        # the flag disables hit-testing for the whole subtree, which would
        # make every child (source badges, buttons) unclickable. Only the
        # static text label below is click-through.
        self._info_panel = QWidget(self.video_label)
        self._info_panel.setVisible(False)

        info_panel_layout = QVBoxLayout(self._info_panel)
        info_panel_layout.setContentsMargins(12, 8, 12, 8)
        info_panel_layout.setSpacing(4)

        self._info_label = QLabel()
        self._info_label.setWordWrap(True)
        self._info_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self._info_label.setStyleSheet(
            "color: #f0f0f0; font-size: 12px; background: transparent;"
        )
        # Static text stays click-through so crop/drag gestures on the image
        # underneath keep working; interactive children are unaffected.
        self._info_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        info_panel_layout.addWidget(self._info_label)

        # ComfyUI source lineage badges (populated async via set_resolved_sources).
        # Header row: count label + always-available "show all" button so the
        # dialog can be opened even when few enough sources fit inline.
        self._sources_header_row = QWidget()
        self._sources_header_row.setStyleSheet("border: none; background: transparent;")
        header_row_layout = QHBoxLayout(self._sources_header_row)
        header_row_layout.setContentsMargins(0, 0, 0, 0)
        header_row_layout.setSpacing(6)
        self._sources_header = QLabel("Sources referenced by this workflow:")
        self._sources_header.setStyleSheet("border: none; color: #aaa; font-size: 11px;")
        header_row_layout.addWidget(self._sources_header)
        header_row_layout.addStretch(1)
        self._sources_all_btn = QPushButton("⧉ all")
        self._sources_all_btn.setFixedHeight(22)
        self._sources_all_btn.setToolTip("Show all workflow sources")
        self._sources_all_btn.setStyleSheet(
            "color: #aaa; font-size: 11px; border: 1px solid #555; border-radius: 4px; padding: 1px 8px;"
        )
        self._sources_all_btn.clicked.connect(self._on_more_clicked)
        # Same wheel-bubble problem as the overflow button (see eventFilter).
        self._sources_all_btn.installEventFilter(self)
        header_row_layout.addWidget(self._sources_all_btn)
        self._sources_header_row.setVisible(False)
        info_panel_layout.addWidget(self._sources_header_row)

        self._sources_container = QWidget()
        self._sources_container.setStyleSheet("border: none; background: transparent;")
        self._sources_layout = FlowLayout(self._sources_container, spacing=6)
        self._sources_container.setVisible(False)
        info_panel_layout.addWidget(self._sources_container)

        self._info_panel.setStyleSheet(
            "background-color: rgba(10, 10, 10, 175); border-radius: 8px;"
        )
        self._info_panel.raise_()

        # Duplicate count overlay
        self._dup_overlay = QLabel(self.video_label)
        self._dup_overlay.setStyleSheet(
            "background-color: rgba(50, 130, 220, 210);"
            "color: white;"
            "border-radius: 10px;"
            "font-size: 11px;"
            "font-weight: bold;"
            "padding: 2px 7px;"
        )
        self._dup_overlay.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self._dup_overlay.setVisible(False)

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
        self.step_back_btn.setToolTip("Step Backward 1 Frame (exact, pauses)")

        self.step_fwd_btn = QPushButton()
        self.step_fwd_btn.setIcon(icons.create_icon(icons.SVG_STEP_FWD))
        self.step_fwd_btn.setFixedWidth(40)
        self.step_fwd_btn.clicked.connect(lambda: self._step_frame(1))
        self.step_fwd_btn.setToolTip("Step Forward 1 Frame (exact, pauses)")

        self.timeline_slider = QSlider(Qt.Orientation.Horizontal)
        self.timeline_slider.sliderMoved.connect(self._on_slider_moved)
        self.timeline_slider.sliderReleased.connect(self._on_slider_released)
        self.timeline_slider.setRange(0, 1000)

        self.volume_btn = QPushButton()
        self.volume_btn.setIcon(icons.create_icon(icons.SVG_VOLUME_UP))
        self.volume_btn.setFixedWidth(40)
        self.volume_btn.setToolTip("Mute / Unmute")
        self.volume_btn.clicked.connect(self._toggle_mute)

        self.volume_slider = QSlider(Qt.Orientation.Horizontal)
        self.volume_slider.setRange(0, 100)
        self.volume_slider.setValue(100)
        self.volume_slider.setFixedWidth(80)
        self.volume_slider.setToolTip("Volume")
        self.volume_slider.valueChanged.connect(self._on_volume_changed)

        self.extract_btn = QPushButton("Save Frame")
        self.extract_btn.setIcon(icons.create_icon(icons.SVG_SAVE))
        self.extract_btn.clicked.connect(self._extract_current_frame)

        controls_layout.addWidget(self.play_btn)
        controls_layout.addWidget(self.step_back_btn)
        controls_layout.addWidget(self.step_fwd_btn)
        controls_layout.addWidget(self.timeline_slider)
        controls_layout.addWidget(self.volume_btn)
        controls_layout.addWidget(self.volume_slider)
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

    def set_tag_state(self, is_tagged: bool):
        """Show or hide the tag badge overlay."""
        self._tag_overlay.setVisible(is_tagged)
        self._reposition_overlays()

    def set_media_data(self, current_path: str, prev_path: str | None, next_path: str | None, *, dup_count: int = 1):
        self.current_filepath = current_path
        self.prev_btn.set_filepath(prev_path)
        self.next_btn.set_filepath(next_path)

        # Stop playback before loading new media
        self._stop_playback()
        self._close_step_grabber()
        # Drop previous lineage badges; fresh ones arrive async (if any).
        self.clear_sources()

        if not current_path:
            self.video_label.setPixmap(QPixmap())
            self.video_controls.hide()
            self._info_panel.setVisible(False)
            return

        self._is_video = current_path.lower().endswith(".mp4")
        self._dup_count = dup_count
        self.set_dup_state(dup_count)
        self._metadata = get_image_metadata(current_path)
        self._update_info_panel()

        if self._is_video:
            self.video_controls.show()
            self._cached_image_pixmap = QPixmap()
            self._open_step_grabber(current_path)
            self.player.setSource(QUrl.fromLocalFile(current_path))
            self._start_playback()
        else:
            self.video_controls.hide()
            pixmap = QPixmap(current_path)
            self._cached_image_pixmap = pixmap
            if not pixmap.isNull():
                self._display_pixmap(pixmap, is_video=False)
            else:
                self.video_label.setText("Could not load image.")
            self.video_label.clear_selection()  # Clear selection when loading new image

    # --- Playback (QtMultimedia owns the A/V clock) ---

    def _start_playback(self):
        self._step_idx = None  # position is player-driven from here
        self.player.play()

    def _stop_playback(self):
        self.player.pause()
        self.play_btn.setIcon(icons.create_icon(icons.SVG_PLAY))

    def _toggle_play_pause(self):
        if self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self._stop_playback()
        else:
            self._start_playback()

    @Slot(QVideoFrame)
    def _on_video_frame_changed(self, frame: QVideoFrame):
        if not frame.isValid():
            return
        if (
            self.player.playbackState() != QMediaPlayer.PlaybackState.PlayingState
            and self._step_idx is not None
        ):
            # An exact OpenCV grab is on screen (paused stepping); ignore the
            # async frame produced by the trailing setPosition() so it cannot
            # overwrite the source-resolution image with a playback-res one.
            return
        q_image = frame.toImage()
        if q_image.isNull():
            return
        self.current_frame = q_image.copy()
        pixmap = QPixmap.fromImage(q_image)
        self._display_pixmap(pixmap, is_video=True)

    @Slot(int)
    def _on_player_position_changed(self, pos_ms: int):
        duration = self.player.duration()
        if duration > 0:
            progress = int((pos_ms / duration) * 1000)
            self.timeline_slider.blockSignals(True)
            self.timeline_slider.setValue(progress)
            self.timeline_slider.blockSignals(False)

    @Slot(QMediaPlayer.PlaybackState)
    def _on_playback_state_changed(self, state):
        if state == QMediaPlayer.PlaybackState.PlayingState:
            self.play_btn.setIcon(icons.create_icon(icons.SVG_PAUSE))
        else:
            self.play_btn.setIcon(icons.create_icon(icons.SVG_PLAY))

    @Slot(QMediaPlayer.Error, str)
    def _on_player_error(self, error: QMediaPlayer.Error, error_string: str):
        if error == QMediaPlayer.Error.NoError:
            return
        logger.error(f"Video playback failed for {self.current_filepath}: {error_string}")
        self._stop_playback()
        self.video_label.setText(f"Could not play video.\n{error_string or 'Missing codec?'}")

    def _on_slider_moved(self, value: int):
        # While playing, scrub approximately via the player clock.
        # While paused, wait for release so we can grab the exact frame once.
        if self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            duration = self.player.duration()
            if duration > 0:
                self.player.setPosition(int((value / 1000.0) * duration))

    def _on_slider_released(self):
        if not self._is_video:
            return
        if self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            return
        # Paused: resolve the slider position to the exact nearest frame.
        value = self.timeline_slider.value()
        if self._step_cap is not None and self._step_total > 0:
            target_idx = max(0, min(self._step_total - 1, int(round((value / 1000.0) * self._step_total))))
            self._grab_exact_frame(target_idx)
        else:
            duration = self.player.duration()
            if duration > 0:
                self.player.setPosition(int((value / 1000.0) * duration))

    # --- Exact paused stepping (OpenCV on-demand grabber) ---

    def _open_step_grabber(self, filepath: str):
        """Open a dedicated OpenCV capture used only for paused stepping."""
        self._close_step_grabber()
        try:
            cap = cv2.VideoCapture(filepath)
            if not cap.isOpened():
                cap.release()
                logger.warning(f"Step grabber could not open {filepath}; stepping will be approximate.")
                return
            fps = cap.get(cv2.CAP_PROP_FPS)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self._step_cap = cap
            self._step_fps = fps if fps and fps > 0 else 30.0
            self._step_total = total if total and total > 0 else 0
            self._step_idx = None
        except Exception as e:
            logger.warning(f"Step grabber failed for {filepath}: {e}; stepping will be approximate.")
            self._step_cap = None

    def _close_step_grabber(self):
        if self._step_cap is not None:
            try:
                self._step_cap.release()
            except Exception:
                pass
            self._step_cap = None
        self._step_total = 0
        self._step_idx = None

    def _step_frame(self, direction: int):
        """Step exactly one frame forward (+1) or backward (-1), pausing first."""
        if not self._is_video:
            return
        self.player.pause()

        if self._step_cap is None or self._step_total <= 0:
            self._fallback_nudge(direction)
            return

        if self._step_idx is None:
            # Derive the base index from the player clock, then step exactly.
            base = int(round((self.player.position() / 1000.0) * self._step_fps))
        else:
            base = self._step_idx
        target_idx = max(0, min(self._step_total - 1, base + direction))
        if not self._grab_exact_frame(target_idx):
            self._fallback_nudge(direction)

    def _grab_exact_frame(self, idx: int) -> bool:
        """Seek the step grabber to idx and display that exact frame."""
        if self._step_cap is None:
            return False
        try:
            self._step_cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = self._step_cap.read()
            if not ret:
                return False
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            q_image = QImage(rgb_frame.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()
            self.current_frame = q_image
            self._step_idx = idx
            self._display_pixmap(QPixmap.fromImage(q_image), is_video=True)
            self._update_slider_from_index(idx)
            # Keep the player clock on the stepped frame so resume is seamless.
            if self._step_fps > 0:
                self.player.setPosition(int((idx / self._step_fps) * 1000))
            return True
        except Exception as e:
            logger.warning(f"Exact frame grab failed at index {idx}: {e}")
            return False

    def _fallback_nudge(self, direction: int):
        """Approximate step used only when the OpenCV grabber is unavailable."""
        self._step_idx = None
        new_pos = self.player.position() + direction * self._FALLBACK_STEP_MS
        self.player.setPosition(max(0, min(self.player.duration(), new_pos)))

    def _update_slider_from_index(self, idx: int):
        if self._step_total > 0:
            progress = int((idx / self._step_total) * 1000)
            self.timeline_slider.blockSignals(True)
            self.timeline_slider.setValue(progress)
            self.timeline_slider.blockSignals(False)

    # --- Volume ---

    def _on_volume_changed(self, value: int):
        self.audio_output.setVolume(value / 100.0)
        if value == 0:
            self.volume_btn.setIcon(icons.create_icon(icons.SVG_VOLUME_OFF))
        else:
            self.volume_btn.setIcon(icons.create_icon(icons.SVG_VOLUME_UP))

    def _toggle_mute(self):
        is_muted = self.audio_output.isMuted()
        self.audio_output.setMuted(not is_muted)
        self.volume_btn.setIcon(
            icons.create_icon(icons.SVG_VOLUME_UP if is_muted else icons.SVG_VOLUME_OFF)
        )

    def _display_pixmap(self, pixmap: QPixmap, is_video: bool = False):
        """Scale and display a pixmap."""
        # Use FastTransformation for 30/60fps video to prevent CPU overload,
        # but keep SmoothTransformation for standard static images
        transform_mode = (
            Qt.TransformationMode.FastTransformation if is_video else Qt.TransformationMode.SmoothTransformation
        )

        scaled = pixmap.scaled(self.video_label.size(), Qt.AspectRatioMode.KeepAspectRatio, transform_mode)
        self.video_label.setPixmap(scaled)

    def get_cropped_image(self) -> QImage | None:
        """Get the currently selected crop area as a QImage.

        Returns:
            QImage: The cropped image in original resolution, or None if no selection.
        """
        if not self.video_label.has_selection():
            return None

        # Get the original full-res image
        if self.current_filepath and not self._is_video:
            orig_img = QImage(self.current_filepath)
            if orig_img.isNull():
                return None
        else:
            # For video, use the current frame (exact source frame when stepped,
            # playback-resolution frame while playing)
            if self.current_frame is None:
                return None
            orig_img = self.current_frame

        # Get the displayed scaled pixmap
        scaled_pixmap = self.video_label.pixmap()
        if scaled_pixmap is None or scaled_pixmap.isNull():
            return None

        # Calculate letterbox offset
        offset_x = (self.video_label.width() - scaled_pixmap.width()) // 2
        offset_y = (self.video_label.height() - scaled_pixmap.height()) // 2

        # Get the selection box
        sel = self.video_label.rubber_band.geometry()

        # Calculate scale factors
        scale_x = orig_img.width() / scaled_pixmap.width()
        scale_y = orig_img.height() / scaled_pixmap.height()

        # Translate the crop to original image coordinates
        orig_x = int((sel.x() - offset_x) * scale_x)
        orig_y = int((sel.y() - offset_y) * scale_y)
        orig_w = int(sel.width() * scale_x)
        orig_h = int(sel.height() * scale_y)

        # Clamp to image boundaries
        crop_rect = QRect(orig_x, orig_y, orig_w, orig_h).intersected(orig_img.rect())

        return orig_img.copy(crop_rect)

    def _extract_current_frame(self):
        """Save the current video frame as a PNG file."""
        if self.current_frame is None or not self.current_filepath:
            return

        video_name = Path(self.current_filepath).stem
        if self._step_idx is not None:
            default_path = f"{video_name}_frame_{self._step_idx}.png"
        else:
            default_path = f"{video_name}_frame_{self.player.position()}ms.png"

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

        # current_frame is always QImage
        return QPixmap.fromImage(self.current_frame)

    def set_dup_state(self, dup_count: int):
        self._dup_count = dup_count
        if dup_count > 1:
            self._dup_overlay.setText(f"×{dup_count} copies")
            self._dup_overlay.adjustSize()
            self._dup_overlay.setVisible(True)
        else:
            self._dup_overlay.setVisible(False)
        self._reposition_overlays()

    def _build_info_text(self, metadata: ImageMetadata) -> str:
        lines = []
        lines.append(f"<b>{metadata.filename}</b>")
        if metadata.width and metadata.height:
            lines.append(f"{metadata.width} × {metadata.height} px")
        if metadata.has_comfy_workflow:
            lines.append("<span style='color:#7ec8e3;'>⚡ ComfyUI Workflow</span>")
            if metadata.comfy_model:
                lines.append(f"Model: {metadata.comfy_model}")
            if metadata.comfy_sampler:
                parts = [metadata.comfy_sampler]
                if metadata.comfy_steps:
                    parts.append(f"{metadata.comfy_steps} steps")
                if metadata.comfy_cfg:
                    parts.append(f"CFG {metadata.comfy_cfg:.1f}")
                if metadata.comfy_scheduler:
                    parts.append(metadata.comfy_scheduler)
                lines.append(f"Sampler: {', '.join(parts)}")
            if metadata.comfy_positive_prompt:
                truncated = metadata.comfy_positive_prompt[:200]
                if len(metadata.comfy_positive_prompt) > 200:
                    truncated += "…"
                lines.append(f"Prompt: <i>{truncated}</i>")
        return "<br>".join(lines)

    def get_current_metadata(self) -> ImageMetadata | None:
        """Return the metadata extracted for the currently displayed media."""
        return self._metadata

    def clear_sources(self):
        """Remove all lineage badges, disconnecting their thumbnail slots."""
        while self._sources_layout.count():
            item = self._sources_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                if isinstance(widget, SourceBadgeWidget):
                    widget.dispose()
                widget.deleteLater()
        self._resolved_items = []
        self._more_btn = None
        self._sources_header_row.setVisible(False)
        self._sources_container.setVisible(False)
        self._reposition_overlays()

    def set_resolved_sources(self, resolved_items: list):
        """Render ComfyUI source badges from backend results.

        Each item: {"raw_filename": str, "resolved_path": str | None}.
        The overlay stays compact no matter the graph size:
        - resolved sources get up to INLINE_SOURCE_CAP small badges;
        - unindexed names never get inline badges, only a count;
        - anything beyond that collapses into a single overflow button
          opening the full dialog.
        When nothing resolved, the whole section is one summary line.
        """
        self.clear_sources()
        items = list(resolved_items or [])
        self._resolved_items = items
        if not items:
            return
        resolved = [d for d in items if d.get("resolved_path")]
        n_unindexed = len(items) - len(resolved)

        for item in resolved[:INLINE_SOURCE_CAP]:
            badge = SourceBadgeWidget(
                item.get("raw_filename", "?"), item.get("resolved_path"),
                parent=self._sources_container, compact=True,
            )
            badge.jump_requested.connect(self.source_jump_requested.emit)
            badge.add_query_requested.connect(self.source_add_query_requested.emit)
            self._sources_layout.addWidget(badge)

        n_hidden = len(resolved) - min(len(resolved), INLINE_SOURCE_CAP) + n_unindexed
        if n_hidden > 0:
            if not resolved:
                # Nothing actionable inline: one summary line for the section.
                label = f"Sources ({len(items)}): none in library — show all"
            else:
                parts = []
                n_hidden_resolved = len(resolved) - INLINE_SOURCE_CAP
                if n_hidden_resolved > 0:
                    parts.append(f"{n_hidden_resolved} more")
                if n_unindexed > 0:
                    parts.append(f"{n_unindexed} unindexed")
                label = "+" + ", ".join(parts)
            more_btn = QPushButton(label)
            more_btn.setToolTip("Show all workflow sources")
            more_btn.setStyleSheet(
                "background-color: rgba(30, 30, 30, 200); color: #ddd;"
                " border: 1px solid #444; border-radius: 6px; padding: 6px 10px;"
            )
            more_btn.clicked.connect(self._on_more_clicked)
            # A button ignores wheel, which would otherwise bubble to the
            # viewer and switch images; swallow them via event filter.
            more_btn.installEventFilter(self)
            self._more_btn = more_btn
            self._sources_layout.addWidget(more_btn)
        self._sources_header.setText(f"Sources ({len(items)}):")
        self._sources_header_row.setVisible(True)
        self._sources_container.setVisible(True)
        self._reposition_overlays()

    def _on_more_clicked(self):
        """Open the overflow dialog with every resolved source."""
        items = list(self._resolved_items or [])
        if not items:
            return
        target = Path(self.current_filepath).name if self.current_filepath else "?"
        dlg = SourceLineageDialog(target, items, self)
        dlg.jump_requested.connect(self.source_jump_requested.emit)
        dlg.add_query_requested.connect(self.source_add_query_requested.emit)
        dlg.add_all_requested.connect(self.source_add_all_requested.emit)
        dlg.exec()

    def _update_info_panel(self):
        if self._metadata is None:
            self._info_panel.setVisible(False)
            return
        html = self._build_info_text(self._metadata)
        self._info_label.setText(html)
        self._reposition_overlays()
        self._info_panel.setVisible(True)

    def _reposition_overlays(self):
        margin = 8
        # Tag overlay: top-right
        self._tag_overlay.move(
            self.video_label.width() - self._tag_overlay.width() - margin,
            margin
        )
        # Dup overlay: top-left
        if self._dup_overlay.isVisible():
            self._dup_overlay.move(margin, margin)
        # Info panel: bottom, full width minus margins, auto height
        panel_width = self.video_label.width() - (margin * 2)
        self._info_label.setFixedWidth(panel_width - 24)  # account for panel margins
        self._info_panel.adjustSize()
        self._info_panel.setFixedWidth(panel_width)
        self._info_panel.move(
            margin,
            self.video_label.height() - self._info_panel.height() - margin
        )
        self._info_panel.raise_()

    def resizeEvent(self, event: QResizeEvent):
        # Redisplay current content scaled
        if self.current_filepath and not self._is_video:
            if not self._cached_image_pixmap.isNull():
                self._display_pixmap(self._cached_image_pixmap, is_video=False)
        elif self.current_frame is not None:
            pixmap = QPixmap.fromImage(self.current_frame)
            self._display_pixmap(pixmap, is_video=True)

        self._reposition_overlays()

        target_height = int(self.height() * 0.25)
        target_height = max(80, min(300, target_height))

        self.prev_btn.setFixedHeight(target_height)
        self.next_btn.setFixedHeight(target_height)

        super().resizeEvent(event)

    def keyPressEvent(self, event: QKeyEvent):
        key = event.key()
        if key == Qt.Key.Key_Left:
            if self._is_video:
                self._step_frame(-1)
            else:
                self.prev_requested.emit()
        elif key == Qt.Key.Key_Right:
            if self._is_video:
                self._step_frame(1)
            else:
                self.next_requested.emit()
        elif key == Qt.Key.Key_Space:
            if self._is_video:
                self._toggle_play_pause()
        elif key == Qt.Key.Key_Escape:
            self.closed.emit()
        else:
            super().keyPressEvent(event)

    def wheelEvent(self, event: QWheelEvent):
        if self._is_video:
            if event.angleDelta().y() > 0:
                self._step_frame(-1)
            else:
                self._step_frame(1)
        else:
            if event.angleDelta().y() > 0:
                self.prev_requested.emit()
            else:
                self.next_requested.emit()

    def eventFilter(self, watched, event):
        # Swallow wheel over the overflow / show-all buttons (plain buttons
        # ignore wheel, which would bubble to the viewer and switch images).
        if event.type() == QEvent.Type.Wheel and watched in (self._more_btn, self._sources_all_btn):
            event.accept()
            return True
        return super().eventFilter(watched, event)

    def mousePressEvent(self, event: QMouseEvent):
        # File drag requires Ctrl+Left: a plain left press starts drags far
        # too eagerly and hijacks clicks meant for badges, crop selection,
        # and other interactions on the view.
        if (
            event.button() == Qt.MouseButton.LeftButton
            and self.current_filepath
            and event.modifiers() & Qt.KeyboardModifier.ControlModifier
        ):
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
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self.closed.emit()
            event.accept()
        else:
            super().mouseDoubleClickEvent(event)

    def stop_media(self):
        """Stop video playback when navigating away."""
        self.player.stop()
        self._close_step_grabber()

    def cleanup(self):
        """Clean up resources."""
        self.stop_media()


# Alias for backward compatibility
OpenCVVideoPlayer = SingleMediaViewer
