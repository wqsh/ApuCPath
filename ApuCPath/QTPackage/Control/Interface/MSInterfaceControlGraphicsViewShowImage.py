from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import QColor
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt, QRectF, QPoint, QPointF, pyqtSignal, QEvent, QSize
from PyQt5.QtGui import QImage, QPixmap, QPainterPath, QMouseEvent, QPainter, QPen, QCursor


from QTPackage.Control.Code.MSDataControl import CDataControlLocInfo, CDataControlSize
from QTPackage.Control.Code.MSControlGraphicsViewShowImage import CControlGraphicsViewShowImage
from QTPackage.Control.Code.MSOperatorControl import COperatorDataControlSize

from ApuCellGlobal.GeneralFunction.MSFunctionIO import CFunctionFileIO
from ApuCellGlobal.GeneralSystem.MSSystemEnum import EnumVisualFileType, LIST_VISUAL_FILE_TYPE_SUFFIX

import os

class CInterfaceInstanceGraphicsViewShowImage():

    def __init__(self, instance_ControlGraphicsViewShowImage:CControlGraphicsViewShowImage):
        self.GraphicsView_show_image = instance_ControlGraphicsViewShowImage
        self.GraphicsViewScene = QGraphicsScene()
        self.GraphicsView_show_image.setScene(self.GraphicsViewScene)
        # 放大或缩小的因子
        self.wheelZoomFactor = 1.25
        self.zoomStack = []  # 这个是用来记录图像调整前的在UI上显示的情况的
        # 设置图像展示方式
        self.aspectRatioMode = Qt.AspectRatioMode.KeepAspectRatio
        # 显示的图片
        self.__cur_image = None
        self.__cur_pixmap = None
        self.__zoomCur = None

        self.draw_broad_size:CDataControlSize = None
        self.scene_size:CDataControlSize = None

    # 这里只是方便直接调用类QGraphicsView的方法
    def sceneRect(self):

        return self.GraphicsView_show_image.sceneRect()

    def setSceneRect(self, *__args):
        return self.GraphicsView_show_image.setSceneRect(*__args)

    def fitInView(self, *__args):
        return self.GraphicsView_show_image.fitInView(*__args)

    def mapToScene(self,*__args):
        """
        mapToScene(self, point: QPoint) -> QPointF
        mapToScene(self, rect: QRect) -> QPolygonF
        mapToScene(self, polygon: QPolygon) -> QPolygonF
        mapToScene(self, path: QPainterPath) -> QPainterPath
        mapToScene(self, ax: int, ay: int) -> QPointF
        mapToScene(self, ax: int, ay: int, w: int, h: int) -> QPolygonF
        """
        return self.GraphicsView_show_image.mapToScene(*__args)

    # 上方只是方便直接调用类QGraphicsView的方法

    def GraphicsViewShowImage(self, path_img:str=None)->bool:
        if not path_img is None:
            self.__captainLoadImage(path_img)
            self.GraphicsViewScene.addPixmap(self.__cur_pixmap)
            self.setSceneRect(QRectF(self.__cur_pixmap.rect()))  # Set scene size to image size.
            self.UpdateGraphicsView()

    def __captainLoadImage(self, path_img:str):
        if self.HasImage():
            self.ClearImage()

        file_suffix = os.path.splitext(path_img)[-1]
        if (file_suffix == LIST_VISUAL_FILE_TYPE_SUFFIX[EnumVisualFileType.SVS.value]):
            return
            # img_array = CFunctionFileIO.ReadImage(path_img)
            # if len(img_array.shape) == 2:
            #     return
            # h, w, c = img_array.shape
            # image = QImage(img_array.copy(), w, h, w*3, QImage.Format_RGB888)
        elif ((file_suffix == LIST_VISUAL_FILE_TYPE_SUFFIX[EnumVisualFileType.PNG.value]) or
            (file_suffix == LIST_VISUAL_FILE_TYPE_SUFFIX[EnumVisualFileType.JPG.value]) or
            (file_suffix == LIST_VISUAL_FILE_TYPE_SUFFIX[EnumVisualFileType.TIFF.value]) or
            (file_suffix == LIST_VISUAL_FILE_TYPE_SUFFIX[EnumVisualFileType.TIF.value])):
            img_array = CFunctionFileIO.ReadImage(path_img)
            if len(img_array.shape) == 2:
                return
            h, w, c = img_array.shape
            image = QImage(img_array.copy(), w, h, w*3, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(image)
            self.__cur_pixmap = pixmap
            self.__cur_image = image
        else:
            return


    def ImageRecover(self):

        tmp = self.__zoomCur
        self.__zoomCur = None
        del tmp
        self.setSceneRect(QRectF(self.__cur_pixmap.rect()))  # Set scene size to image size.
        self.fitInView(self.sceneRect(), self.aspectRatioMode)
        self.setSceneRect(QRectF(self.__cur_pixmap.rect()))
        self.fitInView(self.sceneRect(), self.aspectRatioMode)
        # self.UpdateGraphicsView()

    def HasImage(self):

        if self.__cur_pixmap is None:
            return False
        else:
            return True

    def ZoomIn(self):
        if self.__zoomCur is None:
            self.__zoomCur = self.sceneRect()
        zoomRect = self.__zoomCur
        zoomRect.setWidth(zoomRect.width() / self.wheelZoomFactor)
        zoomRect.setHeight(zoomRect.height() / self.wheelZoomFactor)
        # 获取当前鼠标点的中心位置，并将视角移动至当前鼠标所在位置的中心
        scene_point = self.GraphicsView_show_image.mapToScene(self.GraphicsView_show_image.mapFromGlobal(QCursor.pos()))
        zoomRect.moveCenter(scene_point)
        self.__zoomCur = zoomRect.intersected(self.sceneRect())

        self.UpdateGraphicsView()

    def ZoomOut(self):

        if self.__zoomCur is None:
            self.__zoomCur = self.sceneRect()
        zoomRect = self.__zoomCur
        center = zoomRect.center()
        zoomRect.setWidth(zoomRect.width() * self.wheelZoomFactor)
        zoomRect.setHeight(zoomRect.height() * self.wheelZoomFactor)
        zoomRect.moveCenter(center)
        self.__zoomCur = zoomRect.intersected(self.sceneRect())
        self.UpdateGraphicsView()

    def ClearImage(self):

        if self.HasImage():
            tmp = self.__cur_pixmap
            self.__cur_pixmap = None
            del tmp
            tmp = self.__cur_image
            self.__cur_image = None
            del tmp
        self.GraphicsViewScene.clear()
        self.fitInView(self.sceneRect())

        # if self.draw_broad_size is None:
        #     self.draw_broad_size = CDataControlSize(self.GraphicsView_show_image.geometry().width(), self.GraphicsView_show_image.geometry().height())
        # else:
        #     self.setSceneRect(0, 0, self.draw_broad_size.width - 2, self.draw_broad_size.height - 2)
        #     COperatorDataControlSize.Update(self.draw_broad_size, self.draw_broad_size.width - 2, self.draw_broad_size.height - 2)
        # self.UpdateGraphicsView()

        # self.fitInView(self.sceneRect())

    def GetRGBValue(self, y, x):

        pixel = self.__cur_image.pixel(x, y)
        color = QColor(pixel)
        rgb = color.red(), color.green(), color.blue()
        return rgb

    def GetCurImageSize(self):
        w, h = int(self.GraphicsViewScene.width()), int(self.GraphicsViewScene.height())
        return h, w

    def UpdateGraphicsView(self):

        if self.__zoomCur is None:
            self.fitInView(self.sceneRect(), self.aspectRatioMode)
        else:
            self.fitInView(self.__zoomCur, self.aspectRatioMode)
        #
        # if self.__image is None:
        #     if self.draw_broad_size is None:
        #         self.draw_broad_size = CDataControlSize(self.GraphicView_show_image.geometry().width(), self.GraphicView_show_image.geometry().height())
        #     else:
        #         COperatorDataControlSize.Update(self.draw_broad_size, self.GraphicView_show_image.geometry().width(), self.GraphicView_show_image.geometry().height())
        # else:
        #     if self.scene_size is None:
        #         self.scene_size = CDataControlSize(self.GraphicView_show_image.sceneRect().width(), self.GraphicView_show_image.sceneRect().height())
        #     else:
        #         COperatorDataControlSize.Update(self.scene_size, self.GraphicView_show_image.sceneRect().width(), self.GraphicView_show_image.sceneRect().height())