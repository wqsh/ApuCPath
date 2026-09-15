from typing import Callable

from PyQt5.QtCore import pyqtSignal, QObject, QPointF, QEvent
from PyQt5.QtGui import QColor, QPolygonF, QCursor
from PyQt5.QtWidgets import (
    QMessageBox,
    QMainWindow,
)
from shapely.geometry import Polygon as ShapelyPolygon

from models.polygon import Polygon, PolygonAttributes
from views.main_window import Ui_MainWindow
from views.polygon_widget import PolygonWidget


class MouseMoveFilter(QObject):
    def __init__(self, callback: Callable):
        super().__init__()
        self._callback = callback

    def eventFilter(self, obj, event):
        if event.type() == QEvent.MouseMove:
            self._callback(event.pos())
        return super().eventFilter(obj, event)


class DrawingController(QObject):
    on_completed = pyqtSignal(QPolygonF)

    def __init__(self, window: QMainWindow, ui: Ui_MainWindow):
        super().__init__(window)
        self._ui = ui
        self._color: QColor | None = None
        self._polygon: Polygon | None = None
        self._mouse_move_filter = MouseMoveFilter(self._update_vertex)
        ui.viewer.viewport().installEventFilter(self._mouse_move_filter)

    def start(self, color: QColor):
        self._ui.btn_new_polygon.setChecked(True)
        self._color = color
        self._polygon = Polygon(
            PolygonAttributes(color=color),
            QPolygonF([self._ui.viewer.mapToScene(QCursor.pos())]),
        )
        polygon = PolygonWidget(self._ui.viewer.scene(), self._polygon)
        polygon.vertices[0].signals.on_clicked.connect(self._append_vertex)
        polygon.vertices[0].signals.on_double_clicked.connect(self._complete)
        polygon.vertices[0].signals.on_right_clicked.disconnect()
        polygon.vertices[0].signals.on_right_clicked.connect(self._undo_vertex)
        polygon.vertices[0].setZValue(4999)
        polygon.setSelected(True)

    def _update_vertex(self, pos: QPointF):
        if self._polygon:
            self._polygon.modify(0, self._ui.viewer.mapToScene(pos))

    def _append_vertex(self, point: QPointF):
        if self._polygon:
            if (
                len(self._polygon.points) < 3
                or ShapelyPolygon((p.x(), p.y()) for p in self._polygon.points).is_valid
            ):
                self._polygon.append(point)
            else:
                QMessageBox.warning(
                    None,
                    "Warning",
                    "This is not a simple polygon",
                    QMessageBox.Ok,
                )

    def _undo_vertex(self):
        if self._polygon and len(self._polygon.points) > 1:
            self._polygon.remove(len(self._polygon.points) - 1)

    def cancel(self):
        self._ui.btn_new_polygon.setChecked(False)
        if self._polygon is not None:
            self._polygon.destroy()
            self._polygon = None

    def _complete(self):
        if self._polygon:
            if len(self._polygon.points) > 3:
                # 验证简单多边形
                if ShapelyPolygon(
                    (p.x(), p.y()) for p in self._polygon.points[1:]
                ).is_valid:
                    self.on_completed.emit(self._polygon.points[1:])
                    self.cancel()
                    self.start(self._color)
                    self._ui.btn_new_polygon.click()
            else:
                QMessageBox.warning(
                    None,
                    "Warning",
                    "At least 3 vertices are required",
                    QMessageBox.Ok,
                )
