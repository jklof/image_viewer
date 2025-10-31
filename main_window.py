import logging
import sys
import typing
from pathlib import Path

from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QFileDialog,
    QLabel,
    QStatusBar,
    QMessageBox,
    QListView,
    QMenu,
    QStackedWidget,
)
from PySide6.QtGui import QAction, QPixmap, QKeyEvent, QResizeEvent, QWheelEvent
from PySide6.QtCore import Signal, Qt, Slot, QPoint, QModelIndex

from ui_components import SearchResultDelegate, FILEPATH_ROLE
from qt_visualizer import QtVisualizer
from virtual_model import ImageResultModel

# Forward-declaring the controller type for type hinting to avoid circular imports
if typing.TYPE_CHECKING:
    from app_controller import AppController

logger = logging.getLogger(__name__)


# Custom widget for the single image viewer
class SingleImageViewer(QWidget):
    # Signals to notify the MainWindow of user actions
    closed = Signal()
    next_requested = Signal()
    prev_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        layout = QVBoxLayout(self)
        self.back_button = QPushButton("← Back to Grid View (Esc)")
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout.addWidget(self.back_button)
        layout.addWidget(self.image_label, 1)

        self.back_button.clicked.connect(self.closed.emit)

    def setPixmap(self, pixmap: QPixmap):
        self.image_label.setPixmap(pixmap)

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


class MainWindow(QMainWindow):
    """
    The main window (View) for the application.
    """

    # --- Signals to Controller ---
    text_search_triggered = Signal(str)
    image_search_triggered = Signal(str)
    visualization_triggered = Signal()
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
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        controls_layout = QHBoxLayout()
        self.search_bar = QLineEdit()
        self.search_bar.setPlaceholderText("Enter text to re-order all images by similarity...")
        self.text_search_btn = QPushButton("Order by Text")
        self.image_search_btn = QPushButton("Order by Image")
        self.selected_image_label = QLabel("No image selected.")
        self.selected_image_label.setWordWrap(True)
        self.visualize_btn = QPushButton("Visualize Embeddings")

        controls_layout.addWidget(self.search_bar)
        controls_layout.addWidget(self.text_search_btn)
        controls_layout.addWidget(self.image_search_btn)
        controls_layout.addWidget(self.selected_image_label)
        controls_layout.addStretch()
        controls_layout.addWidget(self.visualize_btn)
        main_layout.addLayout(controls_layout)

        self.content_stack = QStackedWidget()
        self.init_label = QLabel("Initializing, please wait...")
        self.init_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.loading_label = QLabel()
        self.loading_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.results_view = QListView()
        self.results_view.setViewMode(QListView.ViewMode.IconMode)
        self.results_view.setResizeMode(QListView.ResizeMode.Adjust)
        self.results_view.setMovement(QListView.Movement.Static)
        self.results_view.setSpacing(10)
        self.results_model = ImageResultModel(self)
        self.results_view.setModel(self.results_model)
        self.results_view.setItemDelegate(SearchResultDelegate(self))
        self.results_view.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)

        self.visualizer_widget = QtVisualizer()

        self.single_image_view_widget = SingleImageViewer()
        self.single_image_view_widget.image_label.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)

        for label in [self.init_label, self.loading_label]:
            label.setStyleSheet("font-size: 16px; color: #aaa;")

        self.content_stack.addWidget(self.init_label)
        self.content_stack.addWidget(self.results_view)
        self.content_stack.addWidget(self.loading_label)
        self.content_stack.addWidget(self.visualizer_widget)
        self.content_stack.addWidget(self.single_image_view_widget)

        main_layout.addWidget(self.content_stack)

        self.setStatusBar(QStatusBar(self))
        self.content_stack.setCurrentWidget(self.init_label)
        self.set_controls_enabled(False)

    def _connect_ui_signals(self):
        self.search_bar.returnPressed.connect(self._on_text_search_button_clicked)
        self.text_search_btn.clicked.connect(self._on_text_search_button_clicked)
        self.image_search_btn.clicked.connect(self._on_select_image_for_search)
        self.visualize_btn.clicked.connect(self.visualization_triggered.emit)
        self.results_view.customContextMenuRequested.connect(self.on_results_context_menu)

        self.results_view.doubleClicked.connect(self._on_image_double_clicked)

        self.single_image_view_widget.image_label.customContextMenuRequested.connect(self.on_single_view_context_menu)
        self.single_image_view_widget.closed.connect(self._return_to_grid_view)
        self.single_image_view_widget.next_requested.connect(self._navigate_next)
        self.single_image_view_widget.prev_requested.connect(self._navigate_prev)

        # --- Link visualizer double-click to the main image search trigger ---
        self.visualizer_widget.image_search_requested.connect(self.image_search_triggered)

    def set_controls_enabled(self, enabled: bool):
        self.search_bar.setEnabled(enabled)
        self.text_search_btn.setEnabled(enabled)
        self.image_search_btn.setEnabled(enabled)
        self.visualize_btn.setEnabled(enabled)

    def update_status_bar(self, message: str):
        self.statusBar().showMessage(message)

    def show_loading_state(self, message: str):
        self.loading_label.setText(message)
        self.content_stack.setCurrentWidget(self.loading_label)

    def show_results_view(self):
        self.content_stack.setCurrentWidget(self.results_view)

    def show_visualizer_view(self):
        self.content_stack.setCurrentWidget(self.visualizer_widget)

    def show_initialization_failed_state(self):
        self.init_label.setText("Initialization Failed. Please restart.")
        self.content_stack.setCurrentWidget(self.init_label)

    def set_results_data(self, results: list):
        self.results_model.set_results(results)
        if results:
            self.results_view.scrollToTop()

    def clear_results(self):
        self.results_model.clear()

    def update_selected_image_label(self, filepath: str):
        self.selected_image_label.setText(f"Selected: ...{Path(filepath).name}")

    def show_critical_error(self, title: str, message: str):
        QMessageBox.critical(self, title, message)

    @Slot()
    def _on_text_search_button_clicked(self):
        query = self.search_bar.text()
        self.text_search_triggered.emit(query)

    @Slot()
    def _on_select_image_for_search(self):
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Select Image to Order By", "", "Image Files (*.png *.jpg *.jpeg)"
        )
        if filepath:
            self.image_search_triggered.emit(filepath)

    def _create_context_menu(self, filepath: str) -> QMenu:
        context_menu = QMenu(self)
        reorder_action = QAction("Order by Similarity to This", self)
        reorder_action.triggered.connect(lambda: self.image_search_triggered.emit(filepath))
        context_menu.addAction(reorder_action)
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
        self._update_single_image_view()
        self.content_stack.setCurrentWidget(self.single_image_view_widget)
        self.set_controls_enabled(False)
        self.single_image_view_widget.setFocus()

    def _update_single_image_view(self):
        if not (0 <= self.current_single_view_index < self.results_model.rowCount()):
            return
        _, filepath = self.results_model.results_data[self.current_single_view_index]
        pixmap = QPixmap(filepath)
        scaled_pixmap = pixmap.scaled(
            self.single_image_view_widget.image_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.single_image_view_widget.setPixmap(scaled_pixmap)

        status = f"Viewing image {self.current_single_view_index + 1} of {self.results_model.rowCount()} | {Path(filepath).name}"
        self.update_status_bar(status)

    @Slot()
    def _return_to_grid_view(self):
        self.content_stack.setCurrentWidget(self.results_view)
        self.set_controls_enabled(True)
        self.update_status_bar("Ready")
        self.current_single_view_index = -1

    @Slot()
    def _navigate_next(self):
        if self.current_single_view_index < self.results_model.rowCount() - 1:
            self.current_single_view_index += 1
            self._update_single_image_view()

    @Slot()
    def _navigate_prev(self):
        if self.current_single_view_index > 0:
            self.current_single_view_index -= 1
            self._update_single_image_view()

    def resizeEvent(self, event: QResizeEvent):
        if self.content_stack.currentWidget() is self.single_image_view_widget:
            self._update_single_image_view()
        super().resizeEvent(event)

    def copy_image_to_clipboard(self, filepath: str):
        pixmap = QPixmap(filepath)
        if not pixmap.isNull():
            QApplication.clipboard().setPixmap(pixmap)

    def closeEvent(self, event):
        self.closing.emit()
        event.accept()
