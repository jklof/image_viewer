from PySide6.QtCore import Qt

# --- UI Sizing ---
THUMBNAIL_SIZE = 150
ITEM_WIDTH = 180
ITEM_HEIGHT = 210

# --- Custom Model/View Roles ---
# These are used to retrieve specific data from the model.
FILEPATH_ROLE = Qt.ItemDataRole.UserRole + 1
SCORE_ROLE = Qt.ItemDataRole.UserRole + 2
