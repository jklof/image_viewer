import logging
from pathlib import Path
import threading

import cv2

from PySide6.QtCore import Signal, Qt, Slot, QPoint, QModelIndex, QSize, QMimeData, QUrl, QRect, QTimer, QThread
from PySide6.QtGui import (
    QAction,
    QPixmap,
    QKeyEvent,
    QResizeEvent,
    QWheelEvent,
    QColor,
    QDrag,
    QMouseEvent,
    QPainter,
    QPen,
    QBrush,
    QPolygon,
    QRegion,
    QIcon,  # Ensure QIcon is imported
    QImage,
)
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QSizePolicy,
    QStatusBar,
    QMessageBox,
    QListView,
    QMenu,
    QStackedWidget,
    QAbstractItemView,
    QSplitter,
    QProgressBar,
    QSlider,
    QFileDialog,
)

from query_builder import UniversalQueryBuilder
from qt_visualizer import QtVisualizer
from loading_spinner import PulsingSpinner
from ui_components import SearchResultDelegate, FILEPATH_ROLE, TAGS_ROLE, SmoothListView
from virtual_model import ImageResultModel
from loader_manager import get_loader_manager, thumbnail_cache
from preferences_dialog import PreferencesDialog
from constants import ITEM_WIDTH, ITEM_HEIGHT

import icons
from ui_video_player import OpenCVVideoPlayer, SingleMediaViewer

logger = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    composite_search_triggered = Signal(list)
    visualization_triggered = Signal()
    random_order_triggered = Signal()
    sort_by_date_triggered = Signal()
    sync_triggered = Signal()
    sync_cancel_triggered = Signal()
    preferences_saved = Signal(bool)
    closing = Signal()
    toggle_tags_requested = Signal(list)
    untag_all_requested = Signal()
    move_tagged_requested = Signal()
    delete_tagged_requested = Signal()

    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Image Explorer")
        self._set_initial_size()
        self.current_single_view_index = -1
        self._init_ui()
        self._connect_ui_signals()

    def _set_initial_size(self):
        screen = QApplication.primaryScreen()
        if screen:
            screen_geometry = screen.availableGeometry()
            self.setGeometry(50, 50, int(screen_geometry.width() * 0.8), int(screen_geometry.height() * 0.8))
        else:
            self.setGeometry(50, 50, 1200, 800)

    def _init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.query_builder = UniversalQueryBuilder()
        main_splitter.addWidget(self.query_builder)

        content_container = QWidget()
        content_layout = QVBoxLayout(content_container)
        content_layout.setContentsMargins(10, 10, 10, 10)
        content_layout.setSpacing(10)

        action_bar_layout = QHBoxLayout()
        action_bar_layout.setSpacing(10)

        ACTION_BAR_BUTTON_HEIGHT = 35

        # --- Group 1: Left-aligned action buttons ---
        self.random_order_btn = QPushButton("Random Order")
        self.random_order_btn.setFixedHeight(ACTION_BAR_BUTTON_HEIGHT)
        self.random_order_btn.setToolTip("Display all images in a new random order.")
        self.random_order_btn.setIcon(icons.create_icon(icons.SVG_SHUFFLE))

        self.sort_by_date_btn = QPushButton("Sort by Date")
        self.sort_by_date_btn.setFixedHeight(ACTION_BAR_BUTTON_HEIGHT)
        self.sort_by_date_btn.setToolTip("Sort all images by modification date (newest first).")
        self.sort_by_date_btn.setIcon(icons.create_icon(icons.SVG_CALENDAR))

        self.toggle_view_btn = QPushButton("Single View")
        self.toggle_view_btn.setFixedHeight(ACTION_BAR_BUTTON_HEIGHT)
        self.toggle_view_btn.setToolTip("Toggle between grid and single image view.")
        self.toggle_view_btn.setEnabled(False)
        self.toggle_view_btn.setIcon(icons.create_icon(icons.SVG_IMAGE))

        self.visualize_btn = QPushButton("Visualize Embeddings")
        self.visualize_btn.setFixedHeight(ACTION_BAR_BUTTON_HEIGHT)
        self.visualize_btn.setToolTip("View a 2D visualization of all image embeddings.")
        self.visualize_btn.setIcon(icons.create_icon(icons.SVG_CHART))

        self.show_tagged_only_btn = QPushButton("Show Tagged Only")
        self.show_tagged_only_btn.setFixedHeight(ACTION_BAR_BUTTON_HEIGHT)
        self.show_tagged_only_btn.setToolTip("Show only tagged images.")
        self.show_tagged_only_btn.setIcon(icons.create_icon(icons.SVG_TAG))
        self.show_tagged_only_btn.setCheckable(True)

        self.batch_actions_btn = QPushButton("Batch Actions")
        self.batch_actions_btn.setFixedHeight(ACTION_BAR_BUTTON_HEIGHT)
        self.batch_actions_btn.setToolTip("Batch operations for tagged images.")
        self.batch_actions_btn.setIcon(icons.create_icon(icons.SVG_BATCH))

        batch_menu = QMenu(self)
        untag_all_action = QAction("Untag All", self)
        untag_all_action.triggered.connect(self.untag_all_requested.emit)
        batch_menu.addAction(untag_all_action)

        move_tagged_action = QAction("Move Tagged Files", self)
        move_tagged_action.triggered.connect(self.move_tagged_requested.emit)
        batch_menu.addAction(move_tagged_action)

        delete_tagged_action = QAction("Delete Tagged Files", self)
        delete_tagged_action.triggered.connect(self.delete_tagged_requested.emit)
        batch_menu.addAction(delete_tagged_action)

        self.batch_actions_btn.setMenu(batch_menu)

        action_bar_layout.addWidget(self.random_order_btn)
        action_bar_layout.addWidget(self.sort_by_date_btn)
        action_bar_layout.addWidget(self.toggle_view_btn)
        action_bar_layout.addWidget(self.visualize_btn)
        action_bar_layout.addWidget(self.show_tagged_only_btn)
        action_bar_layout.addWidget(self.batch_actions_btn)

        action_bar_layout.addStretch()

        # --- Group 2: Right-aligned sync/settings controls ---
        self.settings_btn = QPushButton("Settings")
        self.settings_btn.setFixedHeight(ACTION_BAR_BUTTON_HEIGHT)
        self.settings_btn.setToolTip("Configure directories, database location, and model.")
        self.settings_btn.setIcon(icons.create_icon(icons.SVG_SETTINGS))
        action_bar_layout.addWidget(self.settings_btn)

        self.start_sync_btn = QPushButton("Synchronize")
        self.start_sync_btn.setFixedHeight(ACTION_BAR_BUTTON_HEIGHT)
        self.start_sync_btn.setToolTip("Scan configured directories and update the database.")
        self.start_sync_btn.setIcon(icons.create_icon(icons.SVG_SYNC))
        action_bar_layout.addWidget(self.start_sync_btn)

        content_layout.addLayout(action_bar_layout)

        self.content_stack = QStackedWidget()
        content_layout.addWidget(self.content_stack)

        self.loading_overlay_widget = QWidget()
        loading_layout = QVBoxLayout(self.loading_overlay_widget)
        loading_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.loading_spinner = PulsingSpinner(self.loading_overlay_widget)
        self.loading_message_label = QLabel()
        self.loading_message_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.loading_message_label.setStyleSheet("font-size: 16px; color: #aaa;")
        loading_layout.addWidget(self.loading_spinner)
        loading_layout.addWidget(self.loading_message_label)

        self.results_view = SmoothListView()
        self.results_view.setViewMode(QListView.ViewMode.IconMode)
        self.results_view.setResizeMode(QListView.ResizeMode.Adjust)
        self.results_view.setMovement(QListView.Movement.Static)
        self.results_view.setSpacing(10)
        # Add a little padding to the grid size to ensure spacing works visually
        self.results_view.setGridSize(QSize(ITEM_WIDTH + 10, ITEM_HEIGHT + 10))

        self.results_view.setDragDropMode(QAbstractItemView.DragDropMode.DragOnly)
        self.results_view.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.results_view.setUniformItemSizes(True)
        self.results_view.setBatchSize(20)
        self.results_view.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.results_model = ImageResultModel(self)
        self.results_view.setModel(self.results_model)
        self.results_view.setItemDelegate(SearchResultDelegate(self))
        self.results_view.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)

        self.visualizer_widget = QtVisualizer()
        self.single_image_view_widget = SingleMediaViewer()
        self.single_image_view_widget.video_label.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)

        self.content_stack.addWidget(self.loading_overlay_widget)
        self.content_stack.addWidget(self.results_view)
        self.content_stack.addWidget(self.visualizer_widget)
        self.content_stack.addWidget(self.single_image_view_widget)

        main_splitter.addWidget(content_container)
        main_splitter.setSizes([350, 1000])
        main_layout.addWidget(main_splitter)

        self.setStatusBar(QStatusBar(self))
        self.statusBar().setStyleSheet("font-size: 14px; padding-left: 5px;")

        self.sync_status_label = QLabel("")
        self.sync_status_label.setStyleSheet("padding-right: 15px; color: #aaa;")
        self.sync_progress_bar = QProgressBar()
        self.sync_progress_bar.setFixedWidth(300)
        self.sync_progress_bar.setTextVisible(False)
        self.sync_cancel_btn = QPushButton("Cancel")
        self.sync_cancel_btn.setFixedWidth(100)
        self.sync_cancel_btn.setIcon(icons.create_icon(icons.SVG_CANCEL))

        self.statusBar().addPermanentWidget(self.sync_status_label)
        self.statusBar().addPermanentWidget(self.sync_progress_bar)
        self.statusBar().addPermanentWidget(self.sync_cancel_btn)

        self.sync_status_label.hide()
        self.sync_progress_bar.hide()
        self.sync_cancel_btn.hide()

        self.content_stack.setCurrentWidget(self.loading_overlay_widget)
        self.loading_message_label.setText("Initializing, please wait...")
        self.loading_spinner.start_animation(QColor(85, 170, 255))
        self.set_controls_enabled(False)
        self.set_sync_controls_enabled(False)

    def _connect_ui_signals(self):
        self.query_builder.search_triggered.connect(self.composite_search_triggered.emit)
        self.visualize_btn.clicked.connect(self.visualization_triggered.emit)
        self.results_view.customContextMenuRequested.connect(self.on_results_context_menu)
        self.results_view.doubleClicked.connect(self._on_image_double_clicked)
        self.single_image_view_widget.video_label.customContextMenuRequested.connect(self.on_single_view_context_menu)
        self.single_image_view_widget.closed.connect(self._return_to_grid_view)
        self.single_image_view_widget.next_requested.connect(self._navigate_next)
        self.single_image_view_widget.prev_requested.connect(self._navigate_prev)
        self.visualizer_widget.image_selected.connect(self._on_visualizer_image_selected)
        self.random_order_btn.clicked.connect(self.random_order_triggered.emit)
        self.sort_by_date_btn.clicked.connect(self.sort_by_date_triggered.emit)
        self.toggle_view_btn.clicked.connect(self._on_toggle_view_clicked)
        self.start_sync_btn.clicked.connect(self.sync_triggered.emit)
        self.sync_cancel_btn.clicked.connect(self.sync_cancel_triggered.emit)
        self.settings_btn.clicked.connect(self._on_settings_clicked)
        self.results_model.dataChanged.connect(self._on_model_data_changed)

    @Slot()
    def _on_settings_clicked(self):
        dlg = PreferencesDialog(self)
        dlg.preferences_saved.connect(self.preferences_saved.emit)
        dlg.exec()

    def _update_toggle_view_button_state(self):
        is_enabled = self.results_model.rowCount() > 0
        self.toggle_view_btn.setEnabled(is_enabled)

        if not is_enabled:
            self.toggle_view_btn.setText("Single View")
            self.toggle_view_btn.setIcon(icons.create_icon(icons.SVG_IMAGE))
            return

        current_widget = self.content_stack.currentWidget()
        # If we are in Single or Visualizer view, button goes back to Grid
        if current_widget is self.single_image_view_widget or current_widget is self.visualizer_widget:
            self.toggle_view_btn.setText("Grid View")
            self.toggle_view_btn.setIcon(icons.create_icon(icons.SVG_GRID))
        else:
            self.toggle_view_btn.setText("Single View")
            self.toggle_view_btn.setIcon(icons.create_icon(icons.SVG_IMAGE))

    @Slot(str)
    def _on_visualizer_image_selected(self, image_path: str):
        self.query_builder.add_image_element(image_path)

    def set_controls_enabled(self, enabled: bool):
        self.visualize_btn.setEnabled(enabled)
        self.query_builder.set_interactive_controls_enabled(enabled)
        self.random_order_btn.setEnabled(enabled)
        self.sort_by_date_btn.setEnabled(enabled)
        self.settings_btn.setEnabled(enabled)
        self.batch_actions_btn.setEnabled(enabled)
        if not enabled:
            self.toggle_view_btn.setEnabled(False)
        else:
            self._update_toggle_view_button_state()

    # --- Slots for Controller to manage Sync UI ---
    def show_sync_active_view(self):
        self.sync_cancel_btn.setEnabled(True)
        self.settings_btn.setEnabled(False)  # Disable settings during sync
        self.sync_status_label.show()
        self.sync_progress_bar.show()
        self.sync_cancel_btn.show()

    def show_sync_idle_view(self):
        self.settings_btn.setEnabled(True)
        self.sync_status_label.hide()
        self.sync_progress_bar.hide()
        self.sync_cancel_btn.hide()

    def set_sync_controls_enabled(self, enabled: bool):
        self.start_sync_btn.setEnabled(enabled)

    def set_sync_cancel_button_enabled(self, enabled: bool):
        self.sync_cancel_btn.setEnabled(enabled)

    def show_sync_prompt_view(self):
        self.loading_spinner.stop_animation()
        self.loading_message_label.setText("No images found.\n\nPlease Synchronize to build your database.")
        self.loading_message_label.setStyleSheet("font-size: 18px; color: #ccc;")
        self.content_stack.setCurrentWidget(self.loading_overlay_widget)

        self.set_controls_enabled(False)
        self.set_sync_controls_enabled(True)
        self.settings_btn.setEnabled(True)  # Allow settings even if empty
        self.show_sync_idle_view()
        self.update_status_bar("Ready. Please run a sync to begin.")

    def show_no_tags_view(self):
        self.loading_spinner.stop_animation()
        self.loading_message_label.setText("No tags selected.\n\nTag images to see them here.")
        self.loading_message_label.setStyleSheet("font-size: 18px; color: #ccc;")
        self.content_stack.setCurrentWidget(self.loading_overlay_widget)

        self.set_controls_enabled(True)
        self.set_sync_controls_enabled(True)
        self.settings_btn.setEnabled(True)
        self.show_sync_idle_view()
        self.update_status_bar("No tagged images.")

    @Slot(str)
    def update_sync_status(self, message: str):
        self.sync_status_label.setText(message)

    @Slot(str, int, int)
    def update_sync_progress(self, stage: str, value: int, total: int):
        self.sync_progress_bar.setRange(0, total)
        self.sync_progress_bar.setValue(value)
        if total > 0 and value > 0:
            self.sync_status_label.setText(f"{stage.capitalize()}: {value}/{total}")
        else:
            self.sync_status_label.setText(stage.capitalize())

    def update_status_bar(self, message: str):
        self.statusBar().showMessage(message)

    @Slot(str)
    def show_loading_state(self, message: str):
        self.loading_message_label.setStyleSheet("font-size: 16px; color: #aaa;")
        self.loading_message_label.setText(message)
        self.loading_spinner.start_animation(QColor(85, 170, 255))
        self.content_stack.setCurrentWidget(self.loading_overlay_widget)

    def show_results_view(self):
        self.loading_spinner.stop_animation()
        self.content_stack.setCurrentWidget(self.results_view)
        self._update_toggle_view_button_state()

    def show_visualizer_view(self):
        self.loading_spinner.stop_animation()
        self.content_stack.setCurrentWidget(self.visualizer_widget)
        self._update_toggle_view_button_state()

    def show_critical_error_state(self):
        self.loading_message_label.setText("A critical error occurred.")
        self.loading_spinner.start_animation(QColor(220, 50, 50))
        self.content_stack.setCurrentWidget(self.loading_overlay_widget)

    def set_results_data(self, results: list):
        self.results_model.set_results(results)
        if results:
            self.results_view.scrollToTop()
        else:
            # If results are empty, ensure we're not stuck in single view
            self.content_stack.setCurrentWidget(self.results_view)
        self.current_single_view_index = -1
        self._update_toggle_view_button_state()

    def clear_results(self):
        self.results_model.clear()
        self.current_single_view_index = -1
        self._update_toggle_view_button_state()

    def show_critical_error(self, title: str, message: str):
        QMessageBox.critical(self, title, message)

    def _create_context_menu(self, filepath: str) -> QMenu:
        context_menu = QMenu(self)
        add_pos_action = QAction("Add to Query (+)", self)
        add_pos_action.triggered.connect(lambda: self.query_builder.add_image_element(filepath))
        context_menu.addAction(add_pos_action)

        toggle_tag_action = QAction("Toggle Tag (T)", self)
        toggle_tag_action.triggered.connect(self._toggle_tag_for_selection)
        context_menu.addAction(toggle_tag_action)

        context_menu.addSeparator()
        copy_path_action = QAction("Copy Full Path", self)
        copy_path_action.triggered.connect(lambda: QApplication.clipboard().setText(filepath))
        context_menu.addAction(copy_path_action)
        copy_image_action = QAction("Copy Image", self)
        copy_image_action.triggered.connect(lambda: self.copy_image_to_clipboard(filepath))
        context_menu.addAction(copy_image_action)
        return context_menu

    @Slot(QPoint)
    def on_results_context_menu(self, pos: QPoint):
        index = self.results_view.indexAt(pos)
        if not index.isValid():
            return
        filepath = index.data(FILEPATH_ROLE)
        context_menu = self._create_context_menu(filepath)
        context_menu.exec(self.results_view.viewport().mapToGlobal(pos))

    @Slot(QPoint)
    def on_single_view_context_menu(self, pos: QPoint):
        if self.current_single_view_index < 0:
            return
        _, filepath, _ = self.results_model.results_data[self.current_single_view_index]
        context_menu = self._create_context_menu(filepath)
        context_menu.exec(self.single_image_view_widget.video_label.mapToGlobal(pos))

    @Slot(QModelIndex)
    def _on_image_double_clicked(self, index: QModelIndex):
        if not index.isValid():
            return
        self.current_single_view_index = index.row()
        self._show_current_single_image()

    def _update_single_image_view_pixmap(self):
        total_count = self.results_model.rowCount()
        if not (0 <= self.current_single_view_index < total_count):
            return

        _, current_filepath, _ = self.results_model.results_data[self.current_single_view_index]

        prev_filepath = None
        if self.current_single_view_index > 0:
            _, prev_filepath, _ = self.results_model.results_data[self.current_single_view_index - 1]

        next_filepath = None
        if self.current_single_view_index < total_count - 1:
            _, next_filepath, _ = self.results_model.results_data[self.current_single_view_index + 1]

        self.single_image_view_widget.set_media_data(current_filepath, prev_filepath, next_filepath)

        status = f"Viewing image {self.current_single_view_index + 1} of {total_count} | {Path(current_filepath).name}"
        self.update_status_bar(status)

    def _update_single_view_tag_state(self):
        """Sync the tag badge overlay in single view with the model's current tag state."""
        if not (0 <= self.current_single_view_index < self.results_model.rowCount()):
            self.single_image_view_widget.set_tag_state(False)
            return
        _, _, tags = self.results_model.results_data[self.current_single_view_index]
        self.single_image_view_widget.set_tag_state("marked" in tags)

    def _show_current_single_image(self):
        self._update_single_image_view_pixmap()
        self._update_single_view_tag_state()
        self.content_stack.setCurrentWidget(self.single_image_view_widget)
        self.single_image_view_widget.setFocus()
        self._update_toggle_view_button_state()

    @Slot()
    def _return_to_grid_view(self):
        self.content_stack.setCurrentWidget(self.results_view)
        self.update_status_bar("Ready")
        self._update_toggle_view_button_state()

    @Slot()
    def _on_toggle_view_clicked(self):
        if self.results_model.rowCount() == 0:
            return

        current_widget = self.content_stack.currentWidget()
        if current_widget is self.single_image_view_widget or current_widget is self.visualizer_widget:
            self._return_to_grid_view()
        else:
            if self.current_single_view_index == -1:
                self.current_single_view_index = 0

            self._show_current_single_image()

    @Slot()
    def _navigate_next(self):
        if self.current_single_view_index < self.results_model.rowCount() - 1:
            self.current_single_view_index += 1
            self._update_single_image_view_pixmap()
            self._update_single_view_tag_state()

    @Slot()
    def _navigate_prev(self):
        if self.current_single_view_index > 0:
            self.current_single_view_index -= 1
            self._update_single_image_view_pixmap()
            self._update_single_view_tag_state()

    def resizeEvent(self, event: QResizeEvent):
        if self.content_stack.currentWidget() is self.single_image_view_widget:
            self.single_image_view_widget.resizeEvent(event)
        super().resizeEvent(event)

    def copy_image_to_clipboard(self, filepath: str):
        # Check if we're in single view with a video - use current frame if available
        if self.content_stack.currentWidget() is self.single_image_view_widget:
            video_pixmap = self.single_image_view_widget.get_current_frame_pixmap()
            if video_pixmap is not None:
                QApplication.clipboard().setPixmap(video_pixmap)
                return

        # Fall back to loading from file path for static images
        pixmap = QPixmap(filepath)
        if not pixmap.isNull():
            QApplication.clipboard().setPixmap(pixmap)

    def _toggle_tag_for_selection(self):
        """Toggle tag for currently selected items."""
        # When in single view, use the currently displayed image index directly.
        # The grid selection is stale/empty in this mode.
        if (
            self.content_stack.currentWidget() is self.single_image_view_widget
            and 0 <= self.current_single_view_index < self.results_model.rowCount()
        ):
            index = self.results_model.createIndex(self.current_single_view_index, 0)
            self.toggle_tags_requested.emit([index])
            return

        selected_indexes = self.results_view.selectionModel().selectedIndexes()
        if selected_indexes:
            self.toggle_tags_requested.emit(selected_indexes)

    @Slot(QModelIndex, QModelIndex, list)
    def _on_model_data_changed(self, top_left, bottom_right, roles):
        """Refresh the single-view tag overlay when the underlying model changes."""
        if self.content_stack.currentWidget() is not self.single_image_view_widget:
            return
        if TAGS_ROLE in roles or not roles:
            self._update_single_view_tag_state()

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() == Qt.Key.Key_T:
            self._toggle_tag_for_selection()
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event):
        self.single_image_view_widget.cleanup()
        self.closing.emit()
        event.accept()
