from PyQt5.QtCore import QAbstractListModel, QModelIndex, Qt, QVariant
from PyQt5.QtGui import QColor, QIcon, QPixmap

from models.label import Label
from models.polygon import Polygon


class EditorList(QAbstractListModel):
    def __init__(self, data: list[Label | Polygon] | None = None, parent=None):
        super().__init__(parent)
        self._data: list[Label | Polygon] | None = data

    @property
    def list(self):
        return self._data

    def data(self, index: QModelIndex, role=Qt.DisplayRole):
        if not index.isValid():
            return QVariant()

        row = index.row()
        if role == Qt.DisplayRole:
            return self._data[row].attributes.name
        elif role == Qt.DecorationRole:
            pixmap = QPixmap(10, 10)
            pixmap.fill(QColor(self._data[row].attributes.color))
            return QIcon(pixmap)
        return QVariant()

    def append(self, item: Label | Polygon):
        if self._data is not None:
            start = len(self._data)
            self.beginInsertRows(QModelIndex(), start, start)
            self._data.append(item)
            self.endInsertRows()

    def remove(self, row: int):
        if self._data is not None and 0 <= row < len(self._data):
            self.beginRemoveRows(QModelIndex(), row, row)
            self._data.pop(row).destroy()
            self.endRemoveRows()

    def clear(self):
        if self._data is not None:
            self.beginRemoveRows(QModelIndex(), 0, len(self._data) - 1)
            for item in self._data:
                item.destroy()
            self._data.clear()
            self.endRemoveRows()

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        if self._data is not None and not parent.isValid():
            return len(self._data)
        return 0

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return 1

    def setItemData(self, index: QModelIndex, value: Label | Polygon, role=Qt.EditRole):
        if (
            self._data is not None
            and index.isValid()
            and -1 < index.row() < len(self._data)
        ):
            self.dataChanged.emit(index, index)
            self._data[index.row()] = value
            return True
        else:
            return False
