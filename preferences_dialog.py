import logging
from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QListWidget,
    QFileDialog,
    QComboBox,
    QDialogButtonBox,
    QMessageBox,
    QGroupBox,
    QWidget,
)
from PySide6.QtCore import Qt
from config_utils import load_config, save_config

logger = logging.getLogger(__name__)

KNOWN_MODELS = [
    "laion/CLIP-ViT-L-14-laion2B-s32B-b82K",  # Default, good balance
    "openai/clip-vit-base-patch32",          # Faster, less accurate
    "openai/clip-vit-large-patch14",         # Standard OpenAI Large
    "laion/CLIP-ViT-H-14-laion2B-s32B-b79K", # Very heavy, high accuracy
]

class PreferencesDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Preferences")
        self.setMinimumWidth(500)
        self.setMinimumHeight(600)
        
        self.current_config = load_config()
        self._init_ui()
        self._load_values()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        # --- Scan Directories Section ---
        dir_group = QGroupBox("Image Scan Directories")
        dir_layout = QVBoxLayout(dir_group)
        
        self.dir_list = QListWidget()
        self.dir_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        dir_layout.addWidget(self.dir_list)

        btn_layout = QHBoxLayout()
        self.add_dir_btn = QPushButton("Add Folder...")
        self.add_dir_btn.clicked.connect(self._add_directory)
        self.remove_dir_btn = QPushButton("Remove Selected")
        self.remove_dir_btn.clicked.connect(self._remove_directory)
        btn_layout.addWidget(self.add_dir_btn)
        btn_layout.addWidget(self.remove_dir_btn)
        dir_layout.addLayout(btn_layout)
        
        layout.addWidget(dir_group)

        # --- Database Section ---
        db_group = QGroupBox("Database Storage")
        db_layout = QVBoxLayout(db_group)
        
        db_input_layout = QHBoxLayout()
        self.db_path_input = QLineEdit()
        self.db_path_input.setPlaceholderText("path/to/images.db")
        self.browse_db_btn = QPushButton("Browse...")
        self.browse_db_btn.clicked.connect(self._browse_db_path)
        
        db_input_layout.addWidget(self.db_path_input)
        db_input_layout.addWidget(self.browse_db_btn)
        
        db_layout.addLayout(db_input_layout)
        db_layout.addWidget(QLabel("Note: Changing this requires an application restart."))
        layout.addWidget(db_group)

        # --- Model Section ---
        model_group = QGroupBox("AI Model (CLIP)")
        model_layout = QVBoxLayout(model_group)
        
        self.model_combo = QComboBox()
        self.model_combo.setEditable(True) # Allow custom HuggingFace strings
        self.model_combo.addItems(KNOWN_MODELS)
        
        model_layout.addWidget(self.model_combo)
        model_layout.addWidget(QLabel("Note: Changing model requires a full re-sync and restart."))
        layout.addWidget(model_group)

        # --- Dialog Buttons ---
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        button_box.accepted.connect(self._save_and_accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def _load_values(self):
        # Directories
        dirs = self.current_config.get("directories", [])
        self.dir_list.addItems(dirs)

        # DB Path
        self.db_path_input.setText(self.current_config.get("database_path", "images.db"))

        # Model
        current_model = self.current_config.get("model_id", KNOWN_MODELS[0])
        self.model_combo.setCurrentText(current_model)

    def _add_directory(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Directory to Scan")
        if directory:
            # Check duplicates
            existing_items = [self.dir_list.item(i).text() for i in range(self.dir_list.count())]
            if directory not in existing_items:
                self.dir_list.addItem(directory)

    def _remove_directory(self):
        for item in self.dir_list.selectedItems():
            self.dir_list.takeItem(self.dir_list.row(item))

    def _browse_db_path(self):
        filepath, _ = QFileDialog.getSaveFileName(self, "Select Database File", self.db_path_input.text(), "SQLite DB (*.db);;All Files (*)")
        if filepath:
            self.db_path_input.setText(filepath)

    def _save_and_accept(self):
        new_dirs = [self.dir_list.item(i).text() for i in range(self.dir_list.count())]
        new_db_path = self.db_path_input.text().strip()
        new_model = self.model_combo.currentText().strip()

        if not new_db_path:
            QMessageBox.warning(self, "Invalid Input", "Database path cannot be empty.")
            return
        if not new_model:
            QMessageBox.warning(self, "Invalid Input", "Model ID cannot be empty.")
            return

        # Detect changes that require restart
        restart_needed = False
        if new_db_path != self.current_config.get("database_path"):
            restart_needed = True
        if new_model != self.current_config.get("model_id"):
            restart_needed = True

        # Update config dictionary
        self.current_config["directories"] = new_dirs
        self.current_config["database_path"] = new_db_path
        self.current_config["model_id"] = new_model

        save_config(self.current_config)
        
        if restart_needed:
            QMessageBox.information(self, "Restart Required", "You have changed settings (Database or Model) that require an application restart to take effect.")
        
        self.accept()