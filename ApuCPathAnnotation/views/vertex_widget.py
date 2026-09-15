from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import QGraphicsEllipseItem, QGraphicsItem
from PyQt5.QtCore import Qt, pyqtSignal, QPointF, QObject


class VertexWidgetSignals(QObject):
    on_position_changed = pyqtSignal(QPointF)
    on_clicked = pyqtSignal(QPointF)
    on_double_clicked = pyqtSignal()
    on_right_clicked = pyqtSignal(object)
    on_handled_changed = pyqtSignal(bool)


class VertexWidget(QGraphicsEllipseItem):

    def __init__(self, pos: QPointF, color: QColor):
        super().__init__(-5, -5, 10, 10)
        self.signals = VertexWidgetSignals()

        self.setPos(pos)
        self.set_color(color)
        self.setFlag(QGraphicsEllipseItem.ItemIsSelectable, False)
        self.setFlag(QGraphicsEllipseItem.ItemIsMovable)
        self.setFlag(QGraphicsEllipseItem.ItemSendsGeometryChanges)
        self.setFlag(QGraphicsEllipseItem.ItemIgnoresTransformations)
        self._is_double_clicking = False

        self.setZValue(4000)

    def set_color(self, color: QColor):
        self.setPen(color)
        self.setBrush(color)

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemPositionChange:
            self.signals.on_position_changed.emit(value)
        return super().itemChange(change, value)

    def mouseMoveEvent(self, event):
        event.accept()
        super().mouseMoveEvent(event)

    def mousePressEvent(self, event):
        self.signals.on_handled_changed.emit(True)
        if not self._is_double_clicking:
            self.setFlag(QGraphicsItem.ItemIsMovable, False)
            if event.button() == Qt.LeftButton:
                self.signals.on_clicked.emit(event.scenePos())
            elif event.button() == Qt.RightButton:
                self.signals.on_handled_changed.emit(False)
                self.signals.on_right_clicked.emit(self)
            self.setFlag(QGraphicsItem.ItemIsMovable, True)
        else:
            self._is_double_clicking = False
        event.accept()
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        event.accept()
        super().mouseReleaseEvent(event)
        self.signals.on_handled_changed.emit(False)

    def mouseDoubleClickEvent(self, event):
        self._is_double_clicking = True
        if event.button() == Qt.LeftButton:
            self.signals.on_double_clicked.emit()
        event.accept()
        super().mouseDoubleClickEvent(event)
