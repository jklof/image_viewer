"""
FlowLayout implementation for PySide6 that supports wrapping logic and stretches.
"""
from PySide6.QtCore import Qt, QPoint, QRect, QSize
from PySide6.QtWidgets import QLayout, QSizePolicy, QSpacerItem


class FlowLayout(QLayout):
    def __init__(self, parent=None, margin=0, spacing=-1):
        super().__init__(parent)
        self.itemList = []
        self._spacing = spacing
        if margin >= 0:
            self.setContentsMargins(margin, margin, margin, margin)

    def __del__(self):
        item = self.takeAt(0)
        while item:
            item = self.takeAt(0)

    def addItem(self, item):
        self.itemList.append(item)

    def addSpacing(self, size):
        self.addItem(QSpacerItem(size, 0, QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Minimum))

    def addStretch(self, stretch=0):
        # We model a stretch as an item with Expanding size policy
        self.addItem(QSpacerItem(0, 0, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum))

    def count(self):
        return len(self.itemList)

    def itemAt(self, index):
        if 0 <= index < len(self.itemList):
            return self.itemList[index]
        return None

    def takeAt(self, index):
        if 0 <= index < len(self.itemList):
            return self.itemList.pop(index)
        return None

    def expandingDirections(self):
        return Qt.Orientation(0)

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, width):
        return self._doLayout(QRect(0, 0, width, 0), True)

    def setGeometry(self, rect):
        super().setGeometry(rect)
        self._doLayout(rect, False)

    def sizeHint(self):
        return self.minimumSize()

    def minimumSize(self):
        size = QSize()
        for item in self.itemList:
            size = size.expandedTo(item.minimumSize())
        margins = self.contentsMargins()
        size += QSize(margins.left() + margins.right(), margins.top() + margins.bottom())
        return size

    def spacing(self):
        if self._spacing >= 0:
            return self._spacing
        return 10

    def _doLayout(self, rect, testOnly):
        x = rect.x()
        y = rect.y()
        lineHeight = 0
        spacing = self.spacing()

        lines = []
        current_line = []
        current_line_width = 0

        for item in self.itemList:
            # Skip hidden widgets. Spacers/Layouts are always included.
            widget = item.widget()
            if widget and widget.isHidden():
                continue

            # Spacers have 0 width in sizeHint() usually if not Fixed, but expanding
            item_width = item.sizeHint().width()
            
            spaceX = spacing if current_line else 0
            nextX = x + spaceX + item_width

            if current_line and nextX > rect.right():
                # wrap
                lines.append((current_line, current_line_width, lineHeight))
                current_line = []
                x = rect.x()
                y = y + lineHeight + spacing
                nextX = x + item_width
                lineHeight = 0
                current_line_width = 0
                spaceX = 0

            current_line.append((item, item_width, spaceX))
            x += item_width + spaceX
            current_line_width += item_width + spaceX
            
            item_height = item.sizeHint().height()
            # If it's a spacer, it might have 0 height, don't let it shrink line height if others have height
            # But wait, QSpacerItem sizeHint() height may be large if it's vertical expanding natively.
            # In our addStretch, it is QSizePolicy.Policy.Minimum vertical policy.
            lineHeight = max(lineHeight, item_height)

        if current_line:
            lines.append((current_line, current_line_width, lineHeight))

        current_y = rect.y()
        for line_items, line_width, lh in lines:
            current_x = rect.x()
            
            # Find expanding items
            expanding_items = []
            for item, i_w, sp in line_items:
                if item.expandingDirections() & Qt.Orientation.Horizontal:
                    expanding_items.append(item)

            extra_space = max(0, rect.width() - line_width)
            stretch_per_item = extra_space // len(expanding_items) if expanding_items else 0

            for item, i_w, sp in line_items:
                current_x += sp
                
                final_w = i_w
                if item in expanding_items:
                    final_w += stretch_per_item
                    
                if not testOnly:
                    # Centering vertically within the line
                    item_height = item.sizeHint().height()
                    if item.widget():
                        y_offset = max(0, (lh - item_height) // 2)
                        item.setGeometry(QRect(QPoint(current_x, current_y + y_offset), QSize(final_w, item_height)))
                    else:
                        item.setGeometry(QRect(QPoint(current_x, current_y), QSize(final_w, lh)))
                    
                current_x += final_w
                
            current_y += lh + spacing

        return current_y - rect.y()
