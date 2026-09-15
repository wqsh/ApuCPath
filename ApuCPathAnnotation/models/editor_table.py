from PyQt5.QtCore import (
    QAbstractTableModel,
    QModelIndex,
    Qt,
    QVariant,
)

from models.label import LabelAttributes
from models.polygon import PolygonAttributes


class EditorTable(QAbstractTableModel):

    def __init__(
        self, data: LabelAttributes | PolygonAttributes | None = None, parent=None
    ):
        super().__init__(parent)
        self.beginResetModel()
        self._data = data
        self.endResetModel()

    @property
    def value(self):
        return self._data

    def data(self, index: QModelIndex, role=Qt.DisplayRole):
        if (
            self._data is not None
            and index.isValid()
            and (role == Qt.DisplayRole or role == Qt.EditRole)
        ):
            if index.row() < len(self._data):
                return list(self._data.list)[index.row()][index.column()]
            else:
                return ""
        else:
            return QVariant()

    def setData(self, index: QModelIndex, value, role=Qt.EditRole):
        if self._data is not None and index.isValid() and role == Qt.EditRole:
            setattr(self._data, self._data.list[index.row()][0], value)
            self.dataChanged.emit(index, index)
            return True
        return False

    def flags(self, index: QModelIndex):
        if index.isValid() and index.column() == 1:
            return Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsEditable
        else:
            return Qt.ItemIsEnabled | Qt.ItemIsSelectable

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal:
                if section == 0:
                    return "Key"
                elif section == 1:
                    return "Value"
                else:
                    return ""
            else:
                return ""
        else:
            return QVariant()

    def rowCount(self, index: QModelIndex = QModelIndex()) -> int:
        if self._data is not None and not index.isValid():
            return len(self._data)
        else:
            return 0

    def columnCount(self, index: QModelIndex = QModelIndex) -> int:
        return 2
