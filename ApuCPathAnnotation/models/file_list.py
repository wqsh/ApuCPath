from PyQt5.QtCore import QAbstractListModel, QModelIndex, Qt, QVariant
from PyQt5.QtWidgets import QMessageBox

from models.file import File


class FileList(QAbstractListModel):
    def __init__(self, data: list[File] | None = None, parent=None):
        super().__init__(parent)
        self._data: list[File] | None = data

    @property
    def list(self):
        return self._data

    def data(self, index: QModelIndex, role=Qt.DisplayRole):
        if not index.isValid():
            return QVariant()

        row = index.row()
        if role == Qt.DisplayRole:
            return self._data[row].name
        return QVariant()

    def append(self, item: File):
        if self._data is not None:
            start = len(self._data)
            self.beginInsertRows(QModelIndex(), start, start)
            self._data.append(item)
            self.endInsertRows()

    def removeRow(self, row: int, parent: QModelIndex = None):
        if self._data is not None and 0 <= row < len(self._data):
            res = QMessageBox.question(
                None,
                "Delete Confirmation",
                f"You will lose any unsaved or unexported annotations.\nAre you sure you want to remove the file <{self._data[row].name}>?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if res == QMessageBox.Yes:
                for label in self._data[row].labels:
                    label.destroy()
                self.beginRemoveRows(QModelIndex(), row, row)
                self._data.pop(row)
                self.endRemoveRows()

    def resetInternalData(self):
        if self._data is not None:
            self.beginRemoveRows(QModelIndex(), 0, len(self._data) - 1)
            self._data.clear()
            self.endRemoveRows()

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        if self._data is not None and not parent.isValid():
            return len(self._data)
        return 0

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return 1

    def setItemData(
        self, index: QModelIndex, value: resetInternalData, role=Qt.EditRole
    ):
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
