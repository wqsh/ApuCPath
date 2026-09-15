from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import QColor
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt, QRectF, QPoint, QPointF, pyqtSignal, QEvent, QSize
from PyQt5.QtGui import QImage, QPixmap, QPainterPath, QMouseEvent, QPainter, QPen


# from QTPackage.Control.Code.MSDataControl import CDataControlLocInfo
from QTPackage.MSSystemGlobal import CURSOR_STATE

class CControlGraphicsViewShowImage(QGraphicsView):
    # 这里只重写类方法，不能添加原有不存在的类方法！！！
    # 定义使用到的鼠标操作信号
    leftMouseButtonPressed = pyqtSignal(float, float)
    leftMouseButtonReleased = pyqtSignal(float, float)
    middleMouseButtonPressed = pyqtSignal(float, float)
    middleMouseButtonReleased = pyqtSignal(float, float)
    rightMouseButtonPressed = pyqtSignal(float, float)
    rightMouseButtonReleased = pyqtSignal(float, float)
    leftMouseButtonDoubleClicked = pyqtSignal(float, float)

    moveMouseGraphicsView2WindowInterface = pyqtSignal(int, int)
    pressMouseGraphicsView2WindowInterface = pyqtSignal()

    rightMouseButtonDoubleClicked = pyqtSignal(float, float)
    viewChanged = pyqtSignal()
    mousePositionOnImageChanged = pyqtSignal(QPoint)
    roiSelected = pyqtSignal(int)

    def __init__(self, *__args):
        QGraphicsView.__init__(self, *__args)
        self.scene = QGraphicsScene()
        self.setScene(self.scene)
        # 放大或缩小的因子
        self.wheelZoomFactor = 1.25
        self.zoomStack = []  # 这个是用来记录图像调整前的在UI上显示的情况的
        # 设置图像展示方式
        self.aspectRatioMode = Qt.AspectRatioMode.KeepAspectRatio

    def mouseMoveEvent(self, event):
        if self.sceneRect().width() < 1e-6:
            self.moveMouseGraphicsView2WindowInterface.emit(-1, -1)
        else:
            scene_point = self.mapToScene(event.pos())
            self.moveMouseGraphicsView2WindowInterface.emit(int(scene_point.y()), int(scene_point.x()))
        QGraphicsView.mouseMoveEvent(self, event)

    def mousePressEvent(self, event) -> None:

        self.pressMouseGraphicsView2WindowInterface.emit()
        QGraphicsView.mousePressEvent(self, event)


