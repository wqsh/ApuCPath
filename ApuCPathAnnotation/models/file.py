import json
from os import path

from PyQt5.QtCore import QObject, pyqtSignal, QPointF
from PyQt5.QtGui import QPolygonF
from PyQt5.QtWidgets import QMessageBox

from models.label import Label, LabelAttributes
from models.polygon import Polygon, PolygonAttributes

from enum import unique, Enum

@unique
class EnumTypeImportImage(Enum):

    tif = 0
    tiff = 1
    svs = 2

class File(QObject):
    on_annotation_updated = pyqtSignal()
    on_image_updated = pyqtSignal()
    on_mask_updated = pyqtSignal()

    def __init__(
        self, image_path: str = "", annotation_path: str = "", mask_path: str = ""
    ):
        super().__init__()
        self._name: str | None = None
        self._image_path: str = image_path
        self._annotation_path: str = annotation_path
        self._mask_path: str = mask_path
        self._show_mask: bool = False
        self._labels: list[Label] = []
        self.annotation_path = annotation_path
        if self._name is None:
            self._name = path.splitext(path.basename(image_path))[0]

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, name: str):
        self._name = name

    @property
    def image_path(self) -> str:
        return self._image_path

    @image_path.setter
    def image_path(self, image_path: str):
        self._image_path = image_path
        self.on_image_updated.emit()

    @property
    def annotation_path(self) -> str:
        return self._annotation_path

    @annotation_path.setter
    def annotation_path(self, annotation_path: str):
        if self.annotation_path != "":
            for label in self._labels:
                label.destroy()
        if annotation_path == "":
            self._annotation_path = ""
            return
        if not path.exists(annotation_path):
            QMessageBox.warning(
                None,
                "Warning",
                "The annotation file doesn't exist.",
                QMessageBox.Ok,
            )
            return
        self._annotation_path = annotation_path
        with open(annotation_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if self._name is None:
                self._name = getattr(data, "name", None)
            self._labels[:] = [
                Label(
                    LabelAttributes(**label),
                    [
                        Polygon(
                            PolygonAttributes(**polygon),
                            QPolygonF(
                                [QPointF(*point) for point in polygon.get("points", [])]
                            ),
                        )
                        for polygon in label.get("polygons", [])
                    ],
                )
                for label in data.get("labels", [])
            ]
        self._annotation_path = annotation_path
        self.on_annotation_updated.emit()

    @property
    def mask_path(self) -> str:
        return self._mask_path

    @mask_path.setter
    def mask_path(self, mask_path: str):
        self._mask_path = mask_path
        self.on_mask_updated.emit()

    @property
    def labels(self) -> list[Label]:
        return self._labels

    @labels.setter
    def labels(self, labels: list[Label]):
        self._labels = labels

    @property
    def json(self):
        return {
            "name": self.name,
            "labels": [label.json for label in self._labels],
        }
