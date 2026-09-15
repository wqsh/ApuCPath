import random

from PyQt5.QtCore import QObject
from PyQt5.QtGui import QColor

from models.attributes import Attributes
from models.polygon import Polygon


class LabelAttributes(Attributes):
    def __init__(self, **kwargs):
        super().__init__()
        self.name = kwargs.get("name", "")
        self.color = kwargs.get(
            "color",
            QColor(
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255),
            )
            .name()
            .upper(),
        )


class Label(QObject):
    def __init__(
        self,
        attributes: LabelAttributes,
        polygons: list[Polygon] | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self._attributes: LabelAttributes = attributes
        self._polygons: list[Polygon] = polygons if polygons is not None else []

    @property
    def attributes(self) -> LabelAttributes:
        return self._attributes

    @attributes.setter
    def attributes(self, value: LabelAttributes):
        self._attributes = value

    @property
    def polygons(self):
        return self._polygons

    @property
    def json(self):
        return {
            **self.attributes.json,
            "polygons": [polygon.json for polygon in self._polygons],
        }

    def destroy(self):
        for polygon in self._polygons:
            polygon.destroy()
