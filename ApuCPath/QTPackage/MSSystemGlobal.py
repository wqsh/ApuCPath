from enum import Enum, unique
from PyQt5.QtGui import QImage, QPixmap, QPainterPath, QMouseEvent, QMoveEvent, QCursor
from QTPackage.Window.IconCode.MSSystemIcon import CSystemPathIconNormal, CSystemPathIconImageView
from PyQt5.QtCore import Qt
@unique
class EnumCursorState(Enum):

    Arrow = 1
    ZOOM_IN = 2
    ZOOM_OUT = 3
    CENTER = 4
    NORMAL = 5
    CROSS = 6

CURSOR_STATE = EnumCursorState.Arrow

class CInterfaceCursor:

    def __init__(self):

        for attribute in dir(CSystemPathIconImageView):
            if attribute.startswith('_'):
                continue
            path = getattr(CSystemPathIconImageView, attribute)
            cursor = QCursor(QPixmap(path).scaled(50, 50), 24, 24)
            setattr(self, attribute, cursor)
        setattr(self, EnumCursorState.Arrow.name, Qt.ArrowCursor)
        # setattr(self, EnumCursorState.Cross.name, Qt.CrossCursor)

    def GetCursor(self, attribute_name):

        return getattr(self, attribute_name)

class CSystemWindowSelectForDataset:
    PATH_IMAGE = ""
    PATH_MASK = ""