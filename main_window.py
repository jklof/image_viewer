import logging
from pathlib import Path
import typing

from PySide6.QtCore import Signal, Qt, Slot, QPoint, QModelIndex, QSize, QMimeData, QUrl
from PySide6.QtGui import (
    QAction,
    QPixmap,
    QKeyEvent,
    QResizeEvent,
    QWheelEvent,
    QColor,
    QDrag,
    QMouseEvent,
)
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QStatusBar,
    QMessageBox,
    QListView,
    QMenu,
    QStackedWidget,
    QAbstractItemView,
    QSplitter,
    QProgressBar,
)

from query_builder import UniversalQueryBuilder
from qt_visualizer import QtVisualizer
from loading_spinner import PulsingSpinner
from ui_components import SearchResultDelegate, FILEPATH_ROLE
from virtual_model import ImageResultModel

if typing.TYPE_CHECKING:
    from app_controller import AppController

logger = logging.getLogger(__name__)


class SingleImageViewer(QWidget):
    closed = Signal()
    next_requested = Signal()
    prev_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.current_filepath = None  # Store the path for drag-and-drop

        layout = QVBoxLayout(self)
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.image_label, 1)

    def set_image(self, filepath: str):
        """Stores the filepath and loads the pixmap for display."""
        self.current_filepath = filepath
        if not filepath:
            self.image_label.setPixmap(QPixmap())  # Clear the image
            return

        pixmap = QPixmap(filepath)
        scaled_pixmap = pixmap.scaled(
            self.image_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.image_label.setPixmap(scaled_pixmap)

    def keyPressEvent(self, event: QKeyEvent):
        key = event.key()
        if key == Qt.Key.Key_Left:
            self.prev_requested.emit()
        elif key == Qt.Key.Key_Right:
            self.next_requested.emit()
        elif key == Qt.Key.Key_Escape:
            self.closed.emit()
        else:
            super().keyPressEvent(event)

    def wheelEvent(self, event: QWheelEvent):
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

            pixmap = self.image_label.pixmap().scaled(100, 100, Qt.AspectRatioMode.KeepAspectRatio)
            drag.setPixmap(pixmap)

            # --- FIX: Set the hot spot to the CENTER of the thumbnail. ---
            # This makes the drag operation feel natural, as if holding the image from its middle.
            drag.setHotSpot(QPoint(pixmap.width() // 2, pixmap.height() // 2))

            drag.exec(Qt.DropAction.CopyAction)


class MainWindow(QMainWindow):
    composite_search_triggered = Signal(list)
    visualization_triggered = Signal()
    random_order_triggered = Signal()
    sync_triggered = Signal()
    sync_cancel_triggered = Signal()
    closing = Signal()

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
        # --- Menu Bar ---
        menu_bar = self.menuBar()
        database_menu = menu_bar.addMenu("&Database")
        self.sync_action = QAction("Synchronize Image Collection...", self)
        database_menu.addAction(self.sync_action)

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

        self.random_order_btn = QPushButton("Random Order")
        self.random_order_btn.setMinimumHeight(35)
        self.random_order_btn.setToolTip("Display all images in a new random order.")

        self.toggle_view_btn = QPushButton("Single View")
        self.toggle_view_btn.setMinimumHeight(35)
        self.toggle_view_btn.setToolTip("Toggle between grid and single image view.")
        self.toggle_view_btn.setEnabled(False)

        self.visualize_btn = QPushButton("Visualize Embeddings")
        self.visualize_btn.setMinimumHeight(35)
        self.visualize_btn.setToolTip("View a 2D visualization of all image embeddings.")
        self.visualize_btn.setStyleSheet(
            """
            QPushButton { background-color: #55aaff; color: white; border-radius: 8px; }
            QPushButton:disabled { background-color: #404040; color: #808080; border: 1px solid #505050; }
            """
        )
        action_bar_layout.addStretch()
        action_bar_layout.addWidget(self.random_order_btn)
        action_bar_layout.addWidget(self.toggle_view_btn)
        action_bar_layout.addWidget(self.visualize_btn)

        # --- Sync Stack (Morphing Button/Status Bar) ---
        self.sync_stack = QStackedWidget()
        self.sync_stack.setMinimumWidth(250)

        # Page 0: Idle Button
        self.start_sync_btn = QPushButton("Synchronize")
        self.start_sync_btn.setToolTip("Scan configured directories and update the database.")
        self.sync_stack.addWidget(self.start_sync_btn)

        # Page 1: Active Status Bar
        sync_status_bar = QWidget()
        status_layout = QHBoxLayout(sync_status_bar)
        status_layout.setContentsMargins(5, 0, 5, 0)
        self.sync_status_label = QLabel("Sync active...")
        self.sync_progress_bar = QProgressBar()
        self.sync_progress_bar.setTextVisible(False)
        self.sync_cancel_btn = QPushButton("Cancel")
        status_layout.addWidget(self.sync_status_label, 1)
        status_layout.addWidget(self.sync_progress_bar, 2)
        status_layout.addWidget(self.sync_cancel_btn)
        self.sync_stack.addWidget(sync_status_bar)

        # --- FIX: Constrain the height of the sync stack to match other buttons ---
        reference_height = self.random_order_btn.sizeHint().height()
        self.sync_stack.setMaximumHeight(reference_height)

        action_bar_layout.addWidget(self.sync_stack)
        # --- End Sync Stack ---

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

        self.results_view = QListView()
        self.results_view.setViewMode(QListView.ViewMode.IconMode)
        self.results_view.setResizeMode(QListView.ResizeMode.Adjust)
        self.results_view.setMovement(QListView.Movement.Static)
        self.results_view.setSpacing(10)
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
        self.single_image_view_widget = SingleImageViewer()
        self.single_image_view_widget.image_label.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)

        self.content_stack.addWidget(self.loading_overlay_widget)
        self.content_stack.addWidget(self.results_view)
        self.content_stack.addWidget(self.visualizer_widget)
        self.content_stack.addWidget(self.single_image_view_widget)

        main_splitter.addWidget(content_container)
        main_splitter.setSizes([350, 1000])
        main_layout.addWidget(main_splitter)

        self.setStatusBar(QStatusBar(self))
        self.statusBar().setStyleSheet("font-size: 14px; padding-left: 5px;")

        self.content_stack.setCurrentWidget(self.loading_overlay_widget)
        self.loading_message_label.setText("Initializing, please wait...")
        self.loading_spinner.start_animation(QColor(85, 170, 255))
        self.set_controls_enabled(False)

    def _connect_ui_signals(self):
        self.query_builder.search_triggered.connect(self.composite_search_triggered.emit)
        self.visualize_btn.clicked.connect(self.visualization_triggered.emit)
        self.results_view.customContextMenuRequested.connect(self.on_results_context_menu)
        self.results_view.doubleClicked.connect(self._on_image_double_clicked)
        self.single_image_view_widget.image_label.customContextMenuRequested.connect(self.on_single_view_context_menu)
        self.single_image_view_widget.closed.connect(self._return_to_grid_view)
        self.single_image_view_widget.next_requested.connect(self._navigate_next)
        self.single_image_view_widget.prev_requested.connect(self._navigate_prev)
        self.visualizer_widget.image_selected.connect(self._on_visualizer_image_selected)
        self.random_order_btn.clicked.connect(self.random_order_triggered.emit)
        self.toggle_view_btn.clicked.connect(self._on_toggle_view_clicked)
        self.sync_action.triggered.connect(self.sync_triggered.emit)
        self.start_sync_btn.clicked.connect(self.sync_triggered.emit)
        self.sync_cancel_btn.clicked.connect(self.sync_cancel_triggered.emit)

    def _update_toggle_view_button_state(self):
        is_enabled = self.results_model.rowCount() > 0
        self.toggle_view_btn.setEnabled(is_enabled)

        if not is_enabled:
            self.toggle_view_btn.setText("Single View")
            return

        current_widget = self.content_stack.currentWidget()
        if current_widget is self.single_image_view_widget or current_widget is self.visualizer_widget:
            self.toggle_view_btn.setText("Grid View")
        else:
            self.toggle_view_btn.setText("Single View")

    @Slot(str)
    def _on_visualizer_image_selected(self, image_path: str):
        self.query_builder.add_image_element(image_path)

    def set_controls_enabled(self, enabled: bool):
        self.visualize_btn.setEnabled(enabled)
        self.query_builder.setEnabled(enabled)
        self.random_order_btn.setEnabled(enabled)
        self.set_sync_controls_enabled(enabled)
        if not enabled:
            self.toggle_view_btn.setEnabled(False)
        else:
            self._update_toggle_view_button_state()

    # --- New Public Slots for Controller to manage Sync UI ---
    def show_sync_active_view(self):
        self.sync_stack.setCurrentIndex(1)
        self.sync_cancel_btn.setEnabled(True)

    def show_sync_idle_view(self):
        self.sync_stack.setCurrentIndex(0)

    def set_sync_controls_enabled(self, enabled: bool):
        self.sync_action.setEnabled(enabled)
        self.start_sync_btn.setEnabled(enabled)

    def set_sync_cancel_button_enabled(self, enabled: bool):
        self.sync_cancel_btn.setEnabled(enabled)

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
        _, filepath = self.results_model.results_data[self.current_single_view_index]
        context_menu = self._create_context_menu(filepath)
        context_menu.exec(self.single_image_view_widget.image_label.mapToGlobal(pos))

    @Slot(QModelIndex)
    def _on_image_double_clicked(self, index: QModelIndex):
        if not index.isValid():
            return
        self.current_single_view_index = index.row()
        self._show_current_single_image()

    def _update_single_image_view_pixmap(self):
        if not (0 <= self.current_single_view_index < self.results_model.rowCount()):
            return
        _, filepath = self.results_model.results_data[self.current_single_view_index]

        self.single_image_view_widget.set_image(filepath)

        status = f"Viewing image {self.current_single_view_index + 1} of {self.results_model.rowCount()} | {Path(filepath).name}"
        self.update_status_bar(status)

    def _show_current_single_image(self):
        self._update_single_image_view_pixmap()
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

    @Slot()
    def _navigate_prev(self):
        if self.current_single_view_index > 0:
            self.current_single_view_index -= 1
            self._update_single_image_view_pixmap()

    def resizeEvent(self, event: QResizeEvent):
        if self.content_stack.currentWidget() is self.single_image_view_widget:
            self._update_single_image_view_pixmap()
        super().resizeEvent(event)

    def copy_image_to_clipboard(self, filepath: str):
        pixmap = QPixmap(filepath)
        if not pixmap.isNull():
            QApplication.clipboard().setPixmap(pixmap)

    def closeEvent(self, event):
        self.closing.emit()
        event.accept()
