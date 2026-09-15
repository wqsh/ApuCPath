import math
from PyQt5.QtWidgets import (
    QGraphicsPolygonItem,
    QGraphicsItem,
    QStyle,
    QMessageBox,
    QGraphicsScene,
)
from PyQt5.QtCore import Qt, QPointF, pyqtSignal, QObject
from PyQt5.QtGui import QPolygonF, QColor, QPen, QMouseEvent

from models.polygon import Polygon
from views.vertex_widget import VertexWidget


class PolygonWidgetSignals(QObject):
    on_first_vertex_clicked = pyqtSignal(QPointF)
    on_first_vertex_double_clicked = pyqtSignal()
    on_first_vertex_right_clicked = pyqtSignal(object)


class PolygonWidget(QGraphicsPolygonItem):

    def __init__(self, scene: QGraphicsScene, polygon: Polygon):
        super().__init__()
        self.signals = PolygonWidgetSignals()
        self._scene = scene
        self._polygon = polygon
        self._vertices: list[VertexWidget] = []
        self._last_pos = QPointF()

        # 延迟加载相关
        self._vertices_created = False
        self._pending_points = list(polygon.points)  # 暂存初始顶点坐标

        # 先直接设置多边形形状（不创建顶点控件）
        points = QPolygonF()
        for p in polygon.points:
            points.append(p)
        self.setPolygon(points)

        self._update_attributes()
        # 注意：不再调用 for point in polygon.points: self._append(point)

        self._select(False)
        self.setFlag(QGraphicsItem.ItemIsSelectable)
        self.setFlag(QGraphicsItem.ItemSendsGeometryChanges)
        self.setZValue(3000)

        polygon.on_attributes_changed.connect(self._update_attributes)
        polygon.on_appended.connect(self._append)
        polygon.on_removed.connect(self._remove)
        polygon.on_modified.connect(self._modify)
        polygon.on_inserted.connect(self._insert)
        polygon.on_selected.connect(self._select)
        polygon.on_destroy.connect(self._destroy)

        scene.addItem(self)

    @property
    def vertices(self) -> list[VertexWidget]:
        """对外暴露顶点列表，首次访问时自动创建所有顶点控件"""
        self._ensure_vertices_created()
        return self._vertices

    def _ensure_vertices_created(self):
        """创建所有顶点控件（仅一次）"""
        if self._vertices_created:
            return
        for point in self._pending_points:
            self._append_vertex_only(point)
        self._pending_points = None
        self._vertices_created = True

    def _append_vertex_only(self, point: QPointF):
        """仅创建顶点控件并加入场景（不改变多边形形状）"""
        vertex = self._vertex(point)
        self._vertices.append(vertex)
        self._scene.addItem(vertex)

    def _update_attributes(self):
        color = QColor(self._polygon.attributes.color)
        pen = QPen(color, 1)
        pen.setCosmetic(True)
        self.setPen(pen)

        # 如果顶点已创建，更新颜色
        if self._vertices_created:
            for vertex in self._vertices:
                vertex.set_color(color)
        color.setAlpha(40)
        self.setBrush(color)

    def _vertex(self, point: QPointF) -> VertexWidget:
        vertex = VertexWidget(point, QColor(self._polygon.attributes.color))
        # vertex.signals.on_handled_changed.connect(self._vertex_handled)
        vertex.signals.on_right_clicked.connect(lambda: self._remove_vertex(vertex))
        vertex.signals.on_position_changed.connect(
            lambda p: self._polygon.modify(self._vertices.index(vertex), p)
        )
        return vertex

    def _append(self, point: QPointF):
        """添加新顶点（来自模型信号）"""
        points = self.polygon()
        points.append(point)
        self.setPolygon(points)

        if self._vertices_created:
            vertex = self._vertex(point)
            self._vertices.append(vertex)
            self._scene.addItem(vertex)
        else:
            # 尚未创建顶点控件，暂存到待创建列表
            self._pending_points.append(point)

    def _remove(self, index: int):
        points = self.polygon()
        points.remove(index)
        self.setPolygon(points)

        if self._vertices_created:
            self._scene.removeItem(self._vertices.pop(index))
        else:
            self._pending_points.pop(index)

    def _modify(self, index: int, point: QPointF):
        points = self.polygon()
        points[index] = point
        self.setPolygon(points)
        if self._vertices_created:
            self._vertices[index].setPos(point)
        else:
            self._pending_points[index] = point

    def _insert(self, index: int, point: QPointF):
        points = self.polygon()
        points.insert(index, point)
        self.setPolygon(points)

        if self._vertices_created:
            vertex = self._vertex(point)
            self._vertices.insert(index, vertex)
            self._scene.addItem(vertex)
        else:
            self._pending_points.insert(index, point)

    def _select(self, selected: bool):
        if selected:
            self._ensure_vertices_created()   # 选中时创建顶点控件
        self._scene.clearSelection()
        self.setSelected(selected)
        if self._vertices_created:
            for vertex in self._vertices:
                vertex.setVisible(selected)

    def _remove_vertex(self, vertex: VertexWidget):
        if self.polygon().count() > 3:
            self._polygon.remove(self._vertices.index(vertex))
        else:
            QMessageBox.warning(
                self._scene.views()[0], "提示", "至少需要3个顶点", QMessageBox.Ok
            )

    def _vertex_handled(self, handled: bool):
        self.setFlag(QGraphicsItem.ItemIsMovable, not handled)

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemPositionChange:
            for i in range(len(self._polygon.points)):
                self._polygon.modify(
                    i, self._polygon.points[i] + value - self._last_pos
                )
            self._last_pos = value
            return self.pos()
        elif change == QGraphicsItem.ItemSelectedChange:
            self._polygon.select(value)
        return super().itemChange(change, value)

    def paint(self, painter, option=None, *args):
        option.state = QStyle.State_None
        super().paint(painter, option, *args)

    def mousePressEvent(self, event):
        self._last_pos = QPointF()
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.LeftButton:
            point = event.scenePos()
            for i in range(len(self.polygon())):
                next_index = (i + 1) % len(self.polygon())
                ax = self.mapToScene(self.polygon()[i]).x()
                ay = self.mapToScene(self.polygon()[i]).y()
                bx = self.mapToScene(self.polygon()[next_index]).x()
                by = self.mapToScene(self.polygon()[next_index]).y()
                dx, dy = bx - ax, by - ay

                length = math.sqrt(dx**2 + dy**2) - 20
                factor = self._scene.views()[0].transform().m11()
                unit_dx, unit_dy = dx / length / factor * 4, dy / length / factor * 4
                perp_dx, perp_dy = -unit_dy * 5 / 2, unit_dx * 5 / 2

                if QPolygonF(
                    [
                        QPointF(ax + perp_dx, ay + perp_dy),
                        QPointF(ax - perp_dx, ay - perp_dy),
                        QPointF(bx - perp_dx, by - perp_dy),
                        QPointF(bx + perp_dx, by + perp_dy),
                    ]
                ).containsPoint(point, Qt.WindingFill):
                    self._polygon.insert(next_index, point)
                    event.accept()
                    break
        super().mouseDoubleClickEvent(event)

    def _destroy(self):
        if self._vertices_created:
            for vertex in self._vertices:
                self._scene.removeItem(vertex)
        self._scene.removeItem(self)