import os
import json

class Attributes():
    def __init__(self):
        pass

    def __len__(self):
        return len(self.__dict__)

    @property
    def json(self):
        return self.__dict__

    @property
    def list(self):
        return list(self.__dict__.items())


class PolygonAttributes(Attributes):
    def __init__(self, **kwargs):
        super(Attributes).__init__()
        self.name = kwargs.get("name", "")
        self.color = kwargs.get("color", "blue")

class Polygon:
    def __init__(self, attributes: PolygonAttributes, points:list[list[int, int]]):
        self._points:list[list[int, int]] = points
        self._attributes:PolygonAttributes = attributes

    @property
    def attributes(self) -> PolygonAttributes:
        return self._attributes

    @attributes.setter
    def attributes(self, value: PolygonAttributes):
        self._attributes = value

    @property
    def json(self):
        return {
            **self._attributes.json,
            "points": [[point[0], point[1]] for point in self._points],
        }

class LabelAttributes(Attributes):
    def __init__(self, **kwargs):
        super(Attributes).__init__()
        self.name = kwargs.get("name", "")
        self.color = kwargs.get("color", "#00FF00")

class Label:

    def __init__(self, attributes: LabelAttributes, polygons: list[Polygon] = None):
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

    def AddPolygon(self, polygon:Polygon):

        self._polygons.append(polygon)

class File:

    def __init__(self, image_path: str = "", annotation_path: str = "", mask_path: str = "" ):
        self._name: str = None
        self._image_path: str = image_path
        self._annotation_path: str = annotation_path
        self._mask_path: str = mask_path
        self._show_mask: bool = False
        self._labels: list[Label] = []
        self.annotation_path = annotation_path
        if self._name is None:
            self._name = os.path.splitext(os.path.basename(image_path))[0]

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

    @property
    def annotation_path(self) -> str:
        return self._annotation_path

    @annotation_path.setter
    def annotation_path(self, annotation_path: str):
        if annotation_path == "":
            self._annotation_path = ""
            return
        if not os.path.exists(annotation_path):
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
                            [point for point in polygon.get("points", [])]
                        )
                        for polygon in label.get("polygons", [])
                    ],
                )
                for label in data.get("labels", [])
            ]
        self._annotation_path = annotation_path

    @property
    def mask_path(self) -> str:
        return self._mask_path

    @mask_path.setter
    def mask_path(self, mask_path: str):
        self._mask_path = mask_path

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

class CDataImageLabelFile(File):

    def __init__(self, image_path: str = "", annotation_path: str = "", mask_path: str = ""):
        super(CDataImageLabelFile, self).__init__(image_path, annotation_path, mask_path)


if __name__ == '__main__':

    path_img = r"E:\20250103_ApuCPath\Project\out-label\tumor_091.tif"
    annotation_path = r"E:\20250103_ApuCPath\Project\out-label\label.json"
    test_output_json = r"E:\20250103_ApuCPath\Project\out-label\out.json"
    file_data = File(path_img, annotation_path, None)
    with open(test_output_json, 'w', encoding="utf-8") as f:
        json.dump(file_data.json, f, indent=4)

    # image_label_file = CDataImageLabelFile("", "", mask_path="")
