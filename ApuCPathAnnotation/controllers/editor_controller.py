from PyQt5.QtCore import QObject, QModelIndex, pyqtSignal
from PyQt5.QtGui import QColor, QPixmap, QIcon, QPolygonF
from PyQt5.QtWidgets import (
    QMainWindow,
    QPushButton,
    QMessageBox,
)
from models.editor_list import EditorList
from models.editor_table import EditorTable
from models.file import File
from models.label import Label, LabelAttributes
from models.polygon import Polygon, PolygonAttributes
from views.main_window import Ui_MainWindow
from views.polygon_widget import PolygonWidget


def _color_dot(color: QColor):
    pixmap = QPixmap(10, 10)
    pixmap.fill(color)
    return QIcon(pixmap)


class EditorController(QObject):
    start_drawing = pyqtSignal(QColor)
    cancel_painting = pyqtSignal()
    add_polygon = pyqtSignal(Polygon)

    def __init__(self, window: QMainWindow, ui: Ui_MainWindow):
        super().__init__(window)
        self._ui = ui
        self._file: File | None = None
        self._new_label_cnt = 0
        self._new_polygon_cnt = 0

        self._label_list_model: EditorList = EditorList()
        ui.label_list.setModel(self._label_list_model)
        self._label_attribute_model: EditorTable = EditorTable()
        ui.label_attribute.setModel(self._label_attribute_model)
        self._polygon_list_model: EditorList = EditorList()
        ui.polygon_list.setModel(self._polygon_list_model)
        self._polygon_attribute_model: EditorTable = EditorTable()
        ui.polygon_attribute.setModel(self._polygon_attribute_model)

        self._btn_new_polygon: QPushButton = ui.btn_new_polygon

        ui.btn_new_label.clicked.connect(self._append_label)
        ui.btn_del_label.clicked.connect(self._remove_label)
        ui.btn_clear_label.clicked.connect(self._clear_label)

        ui.btn_locate.clicked.connect(self._locate_current_polygon)
        ui.btn_new_polygon.clicked.connect(self._new_polygon)
        ui.btn_del_polygon.clicked.connect(self._remove_polygon)
        ui.btn_clear_polygon.clicked.connect(self._clear_polygon)

        self._ui.LineEdit_Label_Count.setReadOnly(True)
        self._ui.LineEdit_Polygon_Count.setReadOnly(True)

    def init(self, file: File):
        self._file = file
        self._ui.label_list.clearSelection()
        self._label_list_model = (
            EditorList(file.labels) if file is not None else EditorList()
        )
        self._ui.label_list.setModel(self._label_list_model)
        self._label_attribute_model: EditorTable = EditorTable()
        self._ui.label_attribute.setModel(self._label_attribute_model)
        self._polygon_list_model: EditorList = EditorList()
        self._ui.polygon_list.setModel(self._polygon_list_model)
        self._polygon_attribute_model: EditorTable = EditorTable()
        self._ui.polygon_attribute.setModel(self._polygon_attribute_model)
        self._ui.label_list.selectionModel().currentChanged.connect(self._switch_label)
        self._ui.polygon_list.selectionModel().currentChanged.connect(
            self._switch_polygon
        )
        if file is not None:
            for label in file.labels:
                for polygon in label.polygons:
                    polygon.on_selected.connect(
                        lambda value, l=label, p=polygon: (
                            self._switch(l, p) if value else None
                        )
                    )
                    PolygonWidget(self._ui.viewer.scene(), polygon)
        self._update_count()


    def _append_label(self):
        if self._label_list_model.list is not None:
            self._new_label_cnt += 1
            # label = Label(LabelAttributes(name=f"New Label {self._new_label_cnt}"), [])
            label = Label(LabelAttributes(name=f"Pos"), [])
            self._label_list_model.append(label)
            self._ui.label_list.setCurrentIndex(
                self._label_list_model.index(self._label_list_model.rowCount() - 1)
            )
        self._update_count()

    def _remove_label(self):
        if self._label_list_model.list is not None:
            row = self._ui.label_list.currentIndex().row()
            if row > -1:
                res = QMessageBox.question(
                    None,
                    "Delete Confirmation",
                    f"Are you sure you want to delete label <{self._label_attribute_model.value.name}> and all the polygons associated with this label?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No,
                )
                if res == QMessageBox.Yes:
                    self._label_list_model.remove(row)
                    self._ui.label_list.selectionModel().clearSelection()
                    self._ui.label_list.setCurrentIndex(QModelIndex())
        self._update_count()

    def _clear_label(self):
        if self._label_list_model.list is not None:
            res = QMessageBox.question(
                None,
                "Clear Confirmation",
                f"Are you sure you want to delete all the labels and polygons?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if res == QMessageBox.Yes:
                self._ui.label_list.clearSelection()
                self._label_list_model.clear()
        self._update_count()

    def _new_polygon(self, state):
        if self._polygon_list_model is not None and state:
            if self._ui.label_list.currentIndex().row() > -1:
                self._ui.polygon_list.clearSelection()
                self.start_drawing.emit(QColor(self._label_attribute_model.value.color))
            else:
                self._btn_new_polygon.setChecked(False)
                self._btn_new_polygon.clicked.emit()
        else:
            self.cancel_painting.emit()
        self._update_count()

    def _remove_polygon(self):
        if self._polygon_list_model is not None:
            row = self._ui.polygon_list.currentIndex().row()
            if row > -1:
                res = QMessageBox.question(
                    None,
                    "Delete Confirmation",
                    f"Are you sure you want to delete the polygon <{self._polygon_attribute_model.value.name}> associated with label <{self._label_attribute_model.value.name}> ?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No,
                )
                if res == QMessageBox.Yes:
                    self._polygon_list_model.remove(row)
                    self._ui.polygon_list.selectionModel().clearSelection()
                    self._ui.polygon_list.setCurrentIndex(QModelIndex())
        self._update_count()

    def _clear_polygon(self):
        if self._polygon_list_model is not None:
            res = QMessageBox.question(
                None,
                "Clear Confirmation",
                f"Are you sure you want to delete all the polygons associated with label <{self._label_attribute_model.value.name}> ?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if res == QMessageBox.Yes:
                self._ui.polygon_list.clearSelection()
                self._polygon_list_model.clear()
        self._update_count()

    def append_polygon(self, points: QPolygonF):
        if self._polygon_list_model is not None:
            self._new_polygon_cnt += 1
            polygon = Polygon(
                PolygonAttributes(
                    name=f"New Polygon {self._new_polygon_cnt}",
                    color=self._label_attribute_model.value.color,
                ),
                points,
            )
            self._polygon_list_model.append(polygon)

            label = self._label_list_model.list[
                self._ui.label_list.selectionModel().currentIndex().row()
            ]
            polygon.on_selected.connect(
                lambda value, l=label, p=polygon: (
                    self._switch(l, p) if value else None
                )
            )
            PolygonWidget(self._ui.viewer.scene(), polygon)
        self._update_count()

    def _switch(self, label: Label, polygon: Polygon):
        label_index = self._label_list_model.index(
            self._label_list_model.list.index(label)
        )
        self._ui.label_list.setCurrentIndex(label_index)
        self._switch_label(label_index)
        self._ui.polygon_list.setCurrentIndex(
            self._polygon_list_model.index(self._polygon_list_model.list.index(polygon))
        )
        self._update_count()

    def _switch_label(self, index: QModelIndex, _previous: QModelIndex = None):
        if index.row() > -1:
            self._label_attribute_model = EditorTable(
                self._label_list_model.list[index.row()].attributes
            )
            self._label_attribute_model.dataChanged.connect(
                self._label_list_model.dataChanged
            )
            self._polygon_list_model = EditorList(
                self._label_list_model.list[index.row()].polygons
            )
        else:
            self._label_attribute_model = EditorTable()
            self._polygon_list_model = EditorList()

        self._ui.label_attribute.setModel(self._label_attribute_model)

        self._ui.polygon_list.clearSelection()
        self._ui.polygon_list.selectionModel().currentChanged.disconnect(
            self._switch_polygon
        )
        self._ui.polygon_list.setModel(self._polygon_list_model)
        self._ui.polygon_list.selectionModel().currentChanged.connect(
            self._switch_polygon
        )

        self.cancel_painting.emit()
        self._update_count()

    def _switch_polygon(self, index: QModelIndex, _previous: QModelIndex = None):
        row = self._ui.label_list.currentIndex().row()

        if row > -1 and index.row() > -1:
            self._polygon_attribute_model = EditorTable(
                self._polygon_list_model.list[index.row()].attributes
            )
            self._polygon_attribute_model.dataChanged.connect(
                self._polygon_list_model.dataChanged
            )
            self._polygon_list_model.list[index.row()].select(True)
        else:
            self._polygon_attribute_model = EditorTable()
            self._ui.viewer.scene().clearSelection()

        self._ui.polygon_attribute.setModel(self._polygon_attribute_model)
        self._update_count()

    def _locate_current_polygon(self):
        pass


    def _update_count(self):

        self._ui.LineEdit_Polygon_Count.setText(str(self._ui.polygon_list.model().rowCount()))
        self._ui.LineEdit_Label_Count.setText(str(self._ui.label_list.model().rowCount()))
