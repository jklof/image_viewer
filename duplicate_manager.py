import logging
import os
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTreeWidget, QTreeWidgetItem, QDialogButtonBox, QMessageBox,
)

logger = logging.getLogger(__name__)


def _fmt_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.0f} {unit}"
        n /= 1024
    return f"{n:.1f} GB"


def _fmt_mtime(ts: float) -> str:
    if not ts:
        return "\u2014"
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")


class DuplicateManagerDialog(QDialog):
    """
    Shows all filepaths that share a single SHA256.
    The user checks rows to DELETE and must leave at least one unchecked (kept).
    Returns the list of filepaths to delete via .filepaths_to_delete after exec().
    """

    def __init__(self, duplicate_info: list[dict], parent=None):
        """
        Args:
            duplicate_info: list of dicts with keys filepath, size, mtime.
                            Comes from ImageDatabase.get_duplicate_paths().
        """
        super().__init__(parent)
        self.duplicate_info = duplicate_info
        self.filepaths_to_delete: list[str] = []

        self.setWindowTitle(f"Manage duplicates \u2014 {len(duplicate_info)} copies")
        self.setMinimumWidth(700)
        self.setMinimumHeight(340)
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        info_label = QLabel(
            "Check the files you want to <b>delete</b>. "
            "At least one copy must remain unchecked."
        )
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        self.tree = QTreeWidget()
        self.tree.setColumnCount(5)
        self.tree.setHeaderLabels(["Delete?", "Directory", "Filename", "Size", "Modified"])
        self.tree.setRootIsDecorated(False)
        self.tree.setAlternatingRowColors(True)
        self.tree.itemChanged.connect(self._on_item_changed)

        for info in self.duplicate_info:
            p = Path(info["filepath"])
            item = QTreeWidgetItem([
                "",                          # checkbox column
                str(p.parent),
                p.name,
                _fmt_size(info["size"]),
                _fmt_mtime(info["mtime"]),
            ])
            item.setCheckState(0, Qt.CheckState.Unchecked)
            item.setData(0, Qt.ItemDataRole.UserRole, info["filepath"])
            self.tree.addTopLevelItem(item)

        self.tree.resizeColumnToContents(1)
        self.tree.resizeColumnToContents(2)
        self.tree.resizeColumnToContents(3)
        self.tree.resizeColumnToContents(4)
        layout.addWidget(self.tree)

        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: #aaa; font-size: 12px;")
        layout.addWidget(self.status_label)

        btn_box = QDialogButtonBox()
        self.delete_btn = btn_box.addButton(
            "Delete selected", QDialogButtonBox.ButtonRole.AcceptRole
        )
        self.delete_btn.setEnabled(False)
        cancel_btn = btn_box.addButton(QDialogButtonBox.StandardButton.Cancel)
        btn_box.accepted.connect(self._on_accept)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

        self._refresh_status()

    def _checked_items(self) -> list[QTreeWidgetItem]:
        return [
            self.tree.topLevelItem(i)
            for i in range(self.tree.topLevelItemCount())
            if self.tree.topLevelItem(i).checkState(0) == Qt.CheckState.Checked
        ]

    def _on_item_changed(self, item, column):
        if column == 0:
            self._refresh_status()

    def _refresh_status(self):
        total = self.tree.topLevelItemCount()
        checked = len(self._checked_items())
        kept = total - checked
        self.delete_btn.setEnabled(checked > 0 and kept >= 1)
        if checked == 0:
            self.status_label.setText("Select at least one file to delete.")
        elif kept == 0:
            self.status_label.setText("You must keep at least one copy.")
            self.delete_btn.setEnabled(False)
        else:
            self.status_label.setText(
                f"{checked} file(s) will be deleted, {kept} will be kept."
            )

    def _on_accept(self):
        items = self._checked_items()
        if not items:
            return
        kept = self.tree.topLevelItemCount() - len(items)
        if kept < 1:
            QMessageBox.warning(self, "Cannot delete all", "You must keep at least one copy.")
            return

        paths = [it.data(0, Qt.ItemDataRole.UserRole) for it in items]
        filenames = "\n".join(f"  \u2022 {Path(p).name}" for p in paths)
        reply = QMessageBox.warning(
            self,
            "Confirm deletion",
            f"Permanently delete {len(paths)} file(s)?\n\n{filenames}\n\nThis cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.filepaths_to_delete = paths
            self.accept()