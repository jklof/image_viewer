# ui_components.py

from pathlib import Path
from PySide6.QtWidgets import QWidget, QLabel, QVBoxLayout, QFrame, QMenu, QApplication
from PySide6.QtGui import QPixmap, QAction
from PySide6.QtCore import Qt, QSize, Signal
from PIL import Image

THUMBNAIL_SIZE = 200

class SearchResultWidget(QWidget):
    """A widget to display a single image search result with a context menu."""

    find_similar_requested = Signal(str)

    def __init__(self, score: float, filepath: str):
        super().__init__()
        self.filepath = filepath
        self.score = score

        layout = QVBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)

        # --- Thumbnail ---
        self.thumbnail_label = QLabel()
        self.thumbnail_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.thumbnail_label.setFixedSize(QSize(THUMBNAIL_SIZE, THUMBNAIL_SIZE))
        self.thumbnail_label.setStyleSheet("background-color: #2c313c; border-radius: 5px;")
        
        try:
            with Image.open(filepath) as img:
                img.thumbnail((THUMBNAIL_SIZE, THUMBNAIL_SIZE), Image.Resampling.LANCZOS)
                q_img = img.toqpixmap()
                self.thumbnail_label.setPixmap(q_img)
        except Exception as e:
            self.thumbnail_label.setText(f"Error:\nCould not load thumbnail.")
            print(f"Error loading thumbnail for {filepath}: {e}")

        # --- Info Labels ---
        score_text = f"Similarity: {self.score:.4f}"
        self.score_label = QLabel(score_text)
        self.score_label.setAlignment(Qt.AlignmentFlag.AlignLeft)

        self.filepath_label = QLabel(self.filepath)
        self.filepath_label.setWordWrap(True)
        self.filepath_label.setAlignment(Qt.AlignmentFlag.AlignLeft)

        layout.addWidget(self.thumbnail_label)
        layout.addWidget(self.score_label)
        layout.addWidget(self.filepath_label)
        
        frame = QFrame()
        frame.setLayout(layout)
        frame.setFrameShape(QFrame.Shape.StyledPanel)
        
        main_layout = QVBoxLayout(self)
        main_layout.addWidget(frame)

    def contextMenuEvent(self, event):
        """This event handler is called when the widget is right-clicked."""
        context_menu = QMenu(self)
        
        # --- Search Action ---
        find_similar_action = QAction("Find Similar", self)
        find_similar_action.triggered.connect(self.emit_find_similar_signal)
        context_menu.addAction(find_similar_action)
        
        # Add a separator for visual grouping
        context_menu.addSeparator()

        # --- NEW: Clipboard Actions ---
        copy_filename_action = QAction("Copy Filename", self)
        copy_filename_action.triggered.connect(self.copy_filename_to_clipboard)
        context_menu.addAction(copy_filename_action)

        copy_image_action = QAction("Copy Image", self)
        copy_image_action.triggered.connect(self.copy_image_to_clipboard)
        context_menu.addAction(copy_image_action)
        
        # Show the menu at the cursor's position
        context_menu.exec(event.globalPos())

    def emit_find_similar_signal(self):
        """Emits the signal with this widget's specific filepath."""
        self.find_similar_requested.emit(self.filepath)

    def copy_filename_to_clipboard(self):
        """Copies just the filename part of the path to the system clipboard."""
        clipboard = QApplication.clipboard()
        filename = Path(self.filepath).name
        clipboard.setText(filename)
        print(f"Copied filename to clipboard: {filename}")

    def copy_image_to_clipboard(self):
        """Copies the full-resolution image to the system clipboard."""
        try:
            clipboard = QApplication.clipboard()
            # Load the full image, not the thumbnail, into a QPixmap
            pixmap = QPixmap(self.filepath)
            if pixmap.isNull():
                print(f"Error: Could not load image for clipboard: {self.filepath}")
                return
            clipboard.setPixmap(pixmap)
            print(f"Copied image to clipboard: {self.filepath}")
        except Exception as e:
            print(f"An error occurred while copying image to clipboard: {e}")