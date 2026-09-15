from PyQt5.QtCore import QObject, pyqtSignal, QPointF
from PyQt5.QtGui import QPolygonF

from models.attributes import Attributes


class PolygonAttributes(Attributes):
    def __init__(self, **kwargs):
        super().__init__()
        self.name = kwargs.get("name", "")
        self.color = kwargs.get("color", "black")


class Polygon(QObject):
    on_selected = pyqtSignal(bool)
    on_destroy = pyqtSignal()
    on_attributes_changed = pyqtSignal()
    on_appended = pyqtSignal(QPointF)
    on_removed = pyqtSignal(int)
    on_modified = pyqtSignal(int, QPointF)
    on_inserted = pyqtSignal(int, QPointF)

    def __init__(self, attributes: PolygonAttributes, points: QPolygonF, parent=None):
        super().__init__(parent)
        self._points: QPolygonF = points
        self._attributes: PolygonAttributes = attributes
        attributes.changed.connect(self.on_attributes_changed)
        self._selected: bool = False

    @property
    def points(self) -> QPolygonF:
        return self._points

    @property
    def attributes(self) -> PolygonAttributes:
        return self._attributes

    @attributes.setter
    def attributes(self, value: PolygonAttributes):
        self._attributes = value
        self.on_attributes_changed.emit()

    @property
    def json(self):
        return {
            **self._attributes.json,
            "points": [[point.x(), point.y()] for point in self._points],
        }

    def append(self, point: QPointF):
        self._points.append(point)
        self.on_appended.emit(point)

    def remove(self, index: int):
        self._points.remove(index)
        self.on_removed.emit(index)

    def modify(self, index: int, point: QPointF):
        if self._points[index] != point:
            self._points[index] = point
            self.on_modified.emit(index, point)

    def insert(self, index: int, point: QPointF):
        self._points.insert(index, point)
        self.on_inserted.emit(index, point)

    def select(self, value: bool):
        if self._selected != value:
            self._selected = value
            self.on_selected.emit(value)

    def destroy(self):
        self.on_destroy.emit()
