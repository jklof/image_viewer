import logging
from pathlib import Path
from typing import List, Dict

from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtGui import (
    QPixmap,
    QPainter,
    QColor,
    QBrush,
    QDragEnterEvent,
    QDropEvent,
)
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QFileDialog,
    QLabel,
    QSlider,
    QScrollArea,
    QFrame,
    QSizePolicy,
    QStyle,
    QStyleOptionSlider,
)

logger = logging.getLogger(__name__)

THUMBNAIL_SIZE = 60
SLIDER_MIN = -100
SLIDER_MAX = 100
SLIDER_DEFAULT = 100


def truncate_text(text: str, max_length: int) -> str:
    """Shortens text to a max length, showing the start and end."""
    if len(text) <= max_length:
        return text

    if max_length < 5:
        return text[:max_length]

    chars_to_show = max_length - 3
    start_len = chars_to_show // 2 + chars_to_show % 2
    end_len = chars_to_show // 2

    return f"{text[:start_len]}...{text[-end_len:]}"


class WeightSlider(QSlider):
    """A custom-painted QSlider that visually represents the -1 to +1 range."""

    def __init__(self, orientation):
        super().__init__(orientation)
        self.setMinimum(SLIDER_MIN)
        self.setMaximum(SLIDER_MAX)
        self.setValue(SLIDER_DEFAULT)
        self.setToolTip(
            "Adjust the weight of this query element.\nNegative (red) values will search for opposite concepts."
        )

    def paintEvent(self, event):
        painter = QPainter(self)
        opt = QStyleOptionSlider()
        self.initStyleOption(opt)

        # --- 1. Let the style draw the entire default slider first ---
        # This draws the handle and the default track. This call is robust and avoids the enum error.
        self.style().drawComplexControl(QStyle.CC_Slider, opt, painter, self)

        # --- 2. Paint our custom colors ON TOP of the default track ---
        groove_rect = self.style().subControlRect(QStyle.CC_Slider, opt, QStyle.SC_SliderGroove, self)
        handle_rect = self.style().subControlRect(QStyle.CC_Slider, opt, QStyle.SC_SliderHandle, self)

        # Ensure the track is thin
        groove_rect.setHeight(8)
        groove_rect.moveTop((self.height() - groove_rect.height()) // 2)

        handle_x = handle_rect.center().x()

        # Define the green (left) and red (right) sections
        green_part = groove_rect.adjusted(0, 0, 0, 0)
        green_part.setRight(handle_x)

        red_part = groove_rect.adjusted(0, 0, 0, 0)
        red_part.setLeft(handle_x)

        # Paint the two sections over the default groove
        painter.setPen(Qt.PenStyle.NoPen)
        if green_part.width() > 0:
            painter.setBrush(QColor("#2f9e44"))  # Green
            painter.drawRect(green_part)

        if red_part.width() > 0:
            painter.setBrush(QColor("#c92a2a"))  # Red
            painter.drawRect(red_part)

        # The handle was already drawn by drawComplexControl, so it remains visible on top.


class QueryElementWidget(QFrame):
    """A widget representing a single text or image query element in the list."""

    removed = Signal(object)
    weight_changed = Signal()

    def __init__(self, element_type: str, value: str):
        super().__init__()
        self.element_type = element_type
        self.value = value

        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setObjectName("QueryElementFrame")
        self.setStyleSheet("#QueryElementFrame { border: 1px solid #444; border-radius: 5px; }")

        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(8)

        if self.element_type == "image":
            self.thumbnail_label = QLabel()
            self.thumbnail_label.setFixedSize(THUMBNAIL_SIZE, THUMBNAIL_SIZE)
            self.thumbnail_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.thumbnail_label.setStyleSheet("border: 1px solid #555; border-radius: 4px;")
            pixmap = QPixmap(self.value)
            scaled = pixmap.scaled(
                THUMBNAIL_SIZE,
                THUMBNAIL_SIZE,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self.thumbnail_label.setPixmap(scaled)
            main_layout.addWidget(self.thumbnail_label)

            full_filename = Path(self.value).name
            short_filename = truncate_text(full_filename, 20)
            self.image_filename_label = QLabel(short_filename)
            self.image_filename_label.setToolTip(full_filename)
            main_layout.addWidget(self.image_filename_label, 1)

        else:  # text
            self.text_label = QLabel(f'"{self.value}"')
            self.text_label.setWordWrap(True)
            main_layout.addWidget(self.text_label, 1)

        slider_layout = QHBoxLayout()
        self.weight_slider = WeightSlider(Qt.Orientation.Horizontal)
        self.weight_slider.valueChanged.connect(self._update_weight_label)
        self.weight_label = QLabel()
        self.weight_label.setFixedWidth(40)
        self.weight_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self._update_weight_label()
        self.remove_btn = QPushButton("✕")
        self.remove_btn.setFixedSize(24, 24)
        self.remove_btn.setToolTip("Remove this element from the query.")
        self.remove_btn.clicked.connect(lambda: self.removed.emit(self))
        slider_layout.addWidget(self.weight_slider)
        slider_layout.addWidget(self.weight_label)
        slider_layout.addWidget(self.remove_btn)
        main_layout.addLayout(slider_layout, 2)

    def _update_weight_label(self):
        self.weight_label.setText(f"{self.get_weight():+.1f}")
        self.weight_changed.emit()

    def get_weight(self) -> float:
        return self.weight_slider.value() / 100.0

    def get_data(self) -> Dict:
        return {"type": self.element_type, "value": self.value, "weight": self.get_weight()}


class UniversalQueryBuilder(QWidget):
    search_triggered = Signal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(350)
        self.setMaximumWidth(500)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        self.setAcceptDrops(True)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(5, 5, 5, 5)
        input_frame = QFrame()
        input_frame.setFrameShape(QFrame.Shape.StyledPanel)
        input_layout = QVBoxLayout(input_frame)
        self.text_input = QLineEdit()
        self.text_input.setPlaceholderText("Type text query...")
        self.text_input.returnPressed.connect(self.add_text_element)
        input_layout.addWidget(self.text_input)
        input_buttons_layout = QHBoxLayout()
        self.add_text_btn = QPushButton("+ Add Text")
        self.add_text_btn.clicked.connect(self.add_text_element)
        self.add_image_btn = QPushButton("+ Add Image")
        self.add_image_btn.clicked.connect(self.add_image_element_from_dialog)
        input_buttons_layout.addWidget(self.add_text_btn)
        input_buttons_layout.addWidget(self.add_image_btn)
        input_layout.addLayout(input_buttons_layout)
        main_layout.addWidget(input_frame)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll_area.setStyleSheet("QScrollArea { border: 1px solid #444; }")
        self.element_list_widget = QWidget()
        self.element_list_layout = QVBoxLayout(self.element_list_widget)
        self.element_list_layout.setContentsMargins(5, 5, 5, 5)
        self.element_list_layout.setSpacing(5)
        self.element_list_layout.addStretch()
        scroll_area.setWidget(self.element_list_widget)
        main_layout.addWidget(scroll_area, 1)

        action_frame = QFrame()
        action_frame.setFrameShape(QFrame.Shape.StyledPanel)
        action_layout = QHBoxLayout(action_frame)
        self.clear_all_btn = QPushButton("Clear All")
        self.clear_all_btn.clicked.connect(self.clear_all_elements)
        self.search_btn = QPushButton("Search")
        self.search_btn.setStyleSheet("background-color: #55aaff; color: white;")
        self.search_btn.clicked.connect(self._on_search_clicked)
        self.search_btn.setMinimumHeight(40)
        self.search_btn.setEnabled(False)
        action_layout.addWidget(self.clear_all_btn)
        action_layout.addWidget(self.search_btn, 1)
        main_layout.addWidget(action_frame)

    @Slot()
    def add_text_element(self):
        text = self.text_input.text().strip()
        if not text:
            return
        element = QueryElementWidget("text", text)
        self._add_element_widget(element)
        self.text_input.clear()

    @Slot()
    def add_image_element_from_dialog(self):
        filepath, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Image Files (*.png *.jpg *.jpeg)")
        if filepath:
            self.add_image_element(filepath)

    def add_image_element(self, filepath: str):
        element = QueryElementWidget("image", filepath)
        self._add_element_widget(element)

    def _add_element_widget(self, element_widget: QueryElementWidget):
        element_widget.removed.connect(self.remove_element)
        element_widget.weight_changed.connect(self._check_search_button_state)
        self.element_list_layout.insertWidget(self.element_list_layout.count() - 1, element_widget)
        self._check_search_button_state()

    @Slot(object)
    def remove_element(self, element_widget: QueryElementWidget):
        element_widget.deleteLater()
        self._check_search_button_state()

    @Slot()
    def clear_all_elements(self):
        while (item := self.element_list_layout.takeAt(0)) is not None:
            if item.widget():
                item.widget().deleteLater()
            else:
                self.element_list_layout.addStretch()
                break
        self._check_search_button_state()

    def _get_all_elements_data(self) -> List[Dict]:
        data = []
        for i in range(self.element_list_layout.count()):
            item = self.element_list_layout.itemAt(i)
            if item and isinstance(item.widget(), QueryElementWidget):
                data.append(item.widget().get_data())
        return data

    @Slot()
    def _on_search_clicked(self):
        query_data = self._get_all_elements_data()
        if query_data:
            self.search_triggered.emit(query_data)

    def _check_search_button_state(self):
        has_elements = any(
            isinstance(self.element_list_layout.itemAt(i).widget(), QueryElementWidget)
            for i in range(self.element_list_layout.count())
        )
        self.search_btn.setEnabled(has_elements)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                if url.toLocalFile().lower().endswith((".png", ".jpg", ".jpeg")):
                    event.acceptProposedAction()
                    return
        event.ignore()

    def dropEvent(self, event: QDropEvent):
        for url in event.mimeData().urls():
            filepath = url.toLocalFile()
            if filepath.lower().endswith((".png", ".jpg *.jpeg")):
                self.add_image_element(filepath)
