import json
from os import path
from typing import Literal
import imagesize
import os

from PyQt5.QtCore import QObject, QModelIndex, pyqtSignal, Qt, QThread
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import (
    QMainWindow,
    QFileDialog,
    QMessageBox,
    QWidget,
    QToolButton,
    QProgressDialog,
    QTableWidgetItem,
)

from models.file import File
from models.file_list import FileList
from models.overview_builder import OverviewBuilder
from models.tiff_writer import TiffWriter
from views.main_window import Ui_MainWindow

from models.file import EnumTypeImportImage


class FileController(QObject):
    annotation_changed = pyqtSignal(object)
    image_changed = pyqtSignal(str)
    mask_changed = pyqtSignal(str)

    def __init__(self, window: QMainWindow, ui: Ui_MainWindow):
        super().__init__(window)
        self._ui = ui
        self._file_list_model: FileList = FileList([])
        ui.file_list.setModel(self._file_list_model)

        self._config_processor = 0
        self._config_block_size = 0
        self._config_task_size = 0

        self._writer_list: list[QThread] = []
        self._dialog: QProgressDialog | None = None

        # Signals
        ui.input_info_name.textChanged.connect(self._update_info_name)
        ui.checkbox_info_mask.clicked.connect(self._change_info_mask_visible)
        ui.btn_info_image.clicked.connect(self._select_info_image)
        ui.btn_info_annotation.clicked.connect(self._select_info_annotation)
        ui.btn_info_mask.clicked.connect(self._select_info_mask)
        ui.btn_info_fix.clicked.connect(self._fix_tiff)
        ui.btn_info_save.clicked.connect(self._save_info)

        ui.file_list.rowsInserted = self._add_btn_to_file_list_item
        ui.file_list.selectionModel().currentChanged.connect(self._switch_file)

        ui.btn_import_original_image.clicked.connect(self._select_import_original_image)
        ui.CheckBox_Import_Folder.clicked.connect(self._select_image_folder)
        ui.btn_import_annotation.clicked.connect(self._select_import_annotation)
        ui.btn_import_mask.clicked.connect(self._select_import_mask)
        ui.btn_import.clicked.connect(self._import)

        ui.btn_export_annotation.clicked.connect(self._select_export_annotation)
        ui.btn_export_outline_image.clicked.connect(self._select_export_outline_image)
        ui.btn_export_fill_image.clicked.connect(self._select_export_fill_image)
        ui.btn_export_mask_image.clicked.connect(self._select_export_mask_image)
        ui.btn_export.clicked.connect(self._export)

    def _add_btn_to_file_list_item(self, _parent, start, end):
        for row in range(start, end + 1):
            index = self._file_list_model.index(row, 0)
            container = QWidget(self._ui.file_list)
            container.setFixedSize(280, self._ui.file_list.sizeHintForRow(row))
            btn = QToolButton(container)
            btn.setIcon(QIcon("resources/icons/delete.svg"))
            btn.setFixedSize(20, 20)
            btn.move(250, (container.height() - btn.height()) // 2)
            btn.clicked.connect(
                lambda checked, f=self._file_list_model.list[
                    row
                ]: self._file_list_model.removeRow(self._file_list_model.list.index(f)),
            )
            self._ui.file_list.setIndexWidget(index, container)

    @property
    def _info(self):
        return self._file_list_model.list[self._ui.file_list.currentIndex().row()]

    # Info
    def _update_info_name(self, text: str):
        self._info.name = text

    def _select_info_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            None,
            "Open Image File",
            "",
            "TIFF Files (*.tif *.tiff)",
            options=QFileDialog.Options(),
        )
        if file_path:
            self._info.image_path = file_path
            self._ui.input_info_image.setText(self._info.image_path)
            self.image_changed.emit(self._info.image_path)

    def _select_image_folder(self):

        if self._ui.CheckBox_Import_Folder.checkState():

            self._ui.import_mask.setDisabled(True)
            self._ui.checkbox_import_mask.setDisabled(True)
            self._ui.checkbox_import_mask.setChecked(False)
            self._ui.input_import_mask.setDisabled(True)
            self._ui.input_import_mask.setText("")

            self._ui.import_annotation.setDisabled(True)
            self._ui.checkbox_import_annotation.setDisabled(True)
            self._ui.checkbox_import_annotation.setChecked(False)
            self._ui.input_import_annotation.setDisabled(True)
            self._ui.input_import_annotation.setText("")

        else:

            self._ui.import_mask.setEnabled(False)
            self._ui.checkbox_import_mask.setDisabled(False)
            self._ui.checkbox_import_mask.setChecked(False)
            self._ui.input_import_mask.setDisabled(False)
            self._ui.input_import_mask.setText("")

            self._ui.import_annotation.setDisabled(False)
            self._ui.checkbox_import_annotation.setDisabled(False)
            self._ui.checkbox_import_annotation.setChecked(False)
            self._ui.input_import_annotation.setDisabled(False)
            self._ui.input_import_annotation.setText("")

    def _select_info_annotation(self):
        file_path, _ = QFileDialog.getOpenFileName(
            None,
            "Open Annotation File",
            "",
            "JSON Files (*.json)",
            options=QFileDialog.Options(),
        )
        if file_path:
            self._info.annotation_path = file_path
            self._ui.input_info_annotation.setText(self._info.annotation_path)
            self._ui.btn_info_save.setEnabled(True)
            self.annotation_changed.emit(self._info)

    def _select_info_mask(self):
        file_path, _ = QFileDialog.getOpenFileName(
            None,
            "Open Mask File",
            "",
            "TIFF Files (*.tif *.tiff)",
            options=QFileDialog.Options(),
        )
        if file_path:
            self._info.mask_path = file_path
            self._ui.input_info_mask.setText(self._info.mask_path)
            self._ui.checkbox_info_mask.setEnabled(True)
            self._ui.checkbox_info_mask.setChecked(True)
            self.mask_changed.emit(self._info.mask_path)

    def _change_info_mask_visible(self, visible: bool):
        if visible:
            self._info.mask_path = self._ui.input_info_mask.text()
        else:
            self._info.mask_path = ""
        self.mask_changed.emit(self._info.mask_path)

    def _fix_tiff(self):
        self._build_overview([self._info.image_path, "", self._info.mask_path])

    def _build_overview(self, path_list: list[str]):
        if len(self._writer_list) > 0:
            QMessageBox.warning(
                None,
                "Warning",
                "Currently writing, please try again later",
                QMessageBox.Ok,
            )
            return
        print("start build overview ...")
        self._dialog = QProgressDialog("processing...", None, 0, 100)
        self._dialog.setWindowFlags(
            Qt.Dialog
            | Qt.CustomizeWindowHint
            | Qt.WindowTitleHint
            | Qt.MSWindowsFixedSizeDialogHint
            | Qt.WindowStaysOnTopHint
        )
        self._dialog.setWindowTitle("Build Overviews")
        self._dialog.setCancelButton(None)
        self._dialog.setAutoClose(False)
        self._dialog.setAutoReset(False)
        self._dialog.setValue(0)
        count = [0, 0, 0]
        total = 0

        for index, file_path in enumerate(path_list):
            print("Start Build Overview ...")
            if path.isfile(file_path):
                print("Build Overview file_path " + file_path)
                overview_builder = OverviewBuilder(file_path)
                self._writer_list.append(overview_builder)
                total = total + 1 if index == 2 else 3

                def on_writer_progress(progress, i=index):
                    count[i] = progress * 1 if index == 2 else 3
                    self._dialog.setValue(int((count[0] + count[1] + count[2]) / total))

                def on_writer_completed(writer, i=index):
                    count[i] = 100 * 1 if index == 2 else 3
                    self._dialog.setValue(int((count[0] + count[1] + count[2]) / total))
                    self._writer_list.remove(writer)
                    if len(self._writer_list) == 0:
                        self._dialog.close()
                        QMessageBox.information(
                            None,
                            "Success",
                            "File processing completed",
                            QMessageBox.Ok,
                        )

                overview_builder.progress.connect(on_writer_progress)
                overview_builder.completed.connect(on_writer_completed)
                overview_builder.start()

        self._dialog.show()

    def _save_info(self):
        res = QMessageBox.question(
            None,
            "Overwrite Confirmation",
            f"Are you sure you want to overwrite the annotation file?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if res == QMessageBox.Yes:
            with open(self._info.annotation_path, "w", encoding="utf-8") as file:
                json.dump(self._info.json, file, indent=4)

    # List
    def _switch_file(self, index: QModelIndex, previous: QModelIndex = None):
        row = index.row()
        if previous.row() > -1:
            for label in self._file_list_model.list[previous.row()].labels:
                label.destroy()
        if row > -1:
            self._ui.input_info_name.setText(self._info.name)
            self._ui.input_info_image.setText(self._info.image_path)
            self._ui.input_info_annotation.setText(self._info.annotation_path)
            self._ui.input_info_mask.setText(self._info.mask_path)
            self._ui.input_info_name.setReadOnly(False)
            self._ui.btn_info_image.setEnabled(True)
            self._ui.btn_info_annotation.setEnabled(True)
            self._ui.btn_info_mask.setEnabled(True)
            self._ui.btn_info_fix.setEnabled(True)
            self._ui.btn_info_save.setEnabled(self._info.annotation_path != "")
            self._ui.checkbox_info_mask.setEnabled(self._info.mask_path != "")
            self._ui.checkbox_info_mask.setChecked(self._info.mask_path != "")

            self._ui.checkbox_export_annotation.setEnabled(True)
            self._ui.checkbox_export_annotation.setChecked(False)
            self._ui.checkbox_export_outline_image.setEnabled(True)
            self._ui.checkbox_export_outline_image.setChecked(False)
            self._ui.checkbox_export_fill_image.setEnabled(True)
            self._ui.checkbox_export_fill_image.setChecked(False)
            self._ui.checkbox_export_mask_image.setEnabled(True)
            self._ui.checkbox_export_mask_image.setChecked(False)
            self._ui.input_export_annotation.setText("")
            self._ui.input_export_outline_image.setText("")
            self._ui.input_export_fill_image.setText("")
            self._ui.input_export_mask_image.setText("")
            self._ui.input_export_line_width.setValue(1)
            self._ui.input_export_line_width.setReadOnly(False)
            self._ui.input_export_opacity.setValue(30)
            self._ui.input_export_opacity.setReadOnly(False)
            self._ui.btn_export_annotation.setEnabled(True)
            self._ui.btn_export_outline_image.setEnabled(True)
            self._ui.btn_export_fill_image.setEnabled(True)
            self._ui.btn_export_mask_image.setEnabled(True)
            self._ui.btn_export.setEnabled(True)

            self.annotation_changed.emit(self._info)
            self.image_changed.emit(self._info.image_path)
            self.mask_changed.emit(self._info.mask_path)

            self._ui.TableWidget_ImageInfo.setItem(1, 1, QTableWidgetItem(os.path.basename(self._info.image_path)))
            self._ui.TableWidget_ImageInfo.setItem(2, 1, QTableWidgetItem(os.path.dirname(self._info.image_path)))
            w, h = imagesize.get(self._info.image_path)
            self._ui.TableWidget_ImageInfo.setItem(3, 1, QTableWidgetItem(str(h)))
            self._ui.TableWidget_ImageInfo.setItem(4, 1, QTableWidgetItem(str(w)))

        else:
            self._ui.input_info_name.setText("")
            self._ui.input_info_name.setReadOnly(True)
            self._ui.input_info_image.setText("")
            self._ui.input_info_annotation.setText("")
            self._ui.input_info_mask.setText("")
            self._ui.btn_info_image.setEnabled(False)
            self._ui.btn_info_annotation.setEnabled(False)
            self._ui.btn_info_mask.setEnabled(False)
            self._ui.btn_info_save.setEnabled(False)
            self._ui.btn_info_fix.setEnabled(False)
            self._ui.checkbox_info_mask.setChecked(False)
            self._ui.checkbox_info_mask.setEnabled(False)

            self._ui.checkbox_export_annotation.setEnabled(False)
            self._ui.checkbox_export_annotation.setChecked(False)
            self._ui.checkbox_export_outline_image.setEnabled(False)
            self._ui.checkbox_export_outline_image.setChecked(False)
            self._ui.checkbox_export_fill_image.setEnabled(False)
            self._ui.checkbox_export_fill_image.setChecked(False)
            self._ui.checkbox_export_mask_image.setEnabled(False)
            self._ui.checkbox_export_mask_image.setChecked(False)
            self._ui.input_export_annotation.setText("")
            self._ui.input_export_outline_image.setText("")
            self._ui.input_export_fill_image.setText("")
            self._ui.input_export_mask_image.setText("")
            self._ui.input_export_line_width.setValue(1)
            self._ui.input_export_line_width.setReadOnly(True)
            self._ui.input_export_opacity.setValue(30)
            self._ui.input_export_opacity.setReadOnly(True)
            self._ui.btn_export_annotation.setEnabled(False)
            self._ui.btn_export_outline_image.setEnabled(False)
            self._ui.btn_export_fill_image.setEnabled(False)
            self._ui.btn_export_mask_image.setEnabled(False)
            self._ui.btn_export.setEnabled(False)

            self.annotation_changed.emit(None)
            self.image_changed.emit("")
            self.mask_changed.emit("")

            self._ui.TableWidget_ImageInfo.setItem(1, 1, QTableWidgetItem(""))
            self._ui.TableWidget_ImageInfo.setItem(2, 1, QTableWidgetItem(""))
            self._ui.TableWidget_ImageInfo.setItem(3, 1, QTableWidgetItem(""))
            self._ui.TableWidget_ImageInfo.setItem(4, 1, QTableWidgetItem(""))

    # Import
    def _select_import_original_image(self):

        if self._ui.CheckBox_Import_Folder.checkState():

            folder = QFileDialog.getExistingDirectory(None, "Select image File Folder")
            if folder:
                self._ui.input_import_original_image.setText(folder)

        else:

            file_path, _ = QFileDialog.getOpenFileName(
                None,
                "Open Image File",
                "",
                f"TIFF Files (*.{EnumTypeImportImage.tif.name} *.{EnumTypeImportImage.tiff.name} *.{EnumTypeImportImage.svs.name})",
                options=QFileDialog.Options(),
            )
            if file_path:
                self._ui.input_import_original_image.setText(file_path)
                annotation_path = f"{path.splitext(file_path)[0]}.json"
                if (
                    path.isfile(annotation_path)
                    and self._ui.input_import_annotation.text() == ""
                ):
                    self._ui.input_import_annotation.setText(annotation_path)
                    self._ui.checkbox_import_annotation.setChecked(True)
                mask_path = (
                    f"{path.splitext(file_path)[0]}_mask{path.splitext(file_path)[1]}"
                )
                if path.isfile(mask_path) and self._ui.input_import_mask.text() == "":
                    self._ui.input_import_mask.setText(mask_path)
                    self._ui.checkbox_import_mask.setChecked(True)

    def _select_import_annotation(self):
        file_path, _ = QFileDialog.getOpenFileName(
            None,
            "Open Annotation File",
            "",
            "JSON Files (*.json)",
        )
        if file_path:
            self._ui.input_import_annotation.setText(file_path)
            self._ui.checkbox_import_annotation.setChecked(True)

    def _select_import_mask(self):
        file_path, _ = QFileDialog.getOpenFileName(
            None,
            "Open Mask File",
            "",
            "TIFF Files (*.tif *.tiff)",
            options=QFileDialog.Options(),
        )
        if file_path:
            self._ui.input_import_mask.setText(file_path)
            self._ui.checkbox_import_mask.setChecked(True)

    def _import(self):
        image_path = self._ui.input_import_original_image.text()
        annotation_path = self._ui.input_import_annotation.text()
        mask_path = self._ui.input_import_mask.text()

        if self._ui.CheckBox_Import_Folder.checkState():

            if not path.isdir(image_path):
                QMessageBox.warning(
                    None,
                    "Warning",
                    "The original image folder does not exist.",
                    QMessageBox.Ok,
                )
                return

        else:
            if not path.isfile(image_path):
                QMessageBox.warning(
                    None,
                    "Warning",
                    "The original image file does not exist.",
                    QMessageBox.Ok,
                )
                return

        if self._ui.checkbox_import_annotation.isChecked():
            if not path.isfile(self._ui.input_import_annotation.text()):
                QMessageBox.warning(
                    None,
                    "Warning",
                    "The annotation file does not exist.",
                    QMessageBox.Ok,
                )
                return
        else:
            annotation_path = ""

        if self._ui.checkbox_import_mask.isChecked():
            if not path.isfile(self._ui.input_import_mask.text()):
                QMessageBox.warning(
                    None,
                    "Warning",
                    "The mask file does not exist.",
                    QMessageBox.Ok,
                )
                return
        else:
            mask_path = ""
        self._ui.input_import_original_image.setText("")
        self._ui.input_import_annotation.setText("")
        self._ui.input_import_mask.setText("")
        self._ui.checkbox_import_annotation.setChecked(False)
        self._ui.checkbox_import_mask.setChecked(False)

        if self._ui.CheckBox_Import_Folder.checkState():
            list_file = os.listdir(image_path)
            for file_name in list_file:
                if (file_name.endswith("tif") or  file_name.endswith("tiff")):
                    self._file_list_model.append(File(os.path.join(image_path, file_name), annotation_path, mask_path))
        else:
            self._file_list_model.append(File(image_path, annotation_path, mask_path))

    # Export
    def _select_export_annotation(self):
        file_path, _ = QFileDialog.getSaveFileName(
            None,
            "Save Annotation File",
            os.path.join(os.getcwd(), os.path.splitext(os.path.basename(self._ui.input_info_image.text()))[0] + '.json'),
            "JSON Files (*.json)",
            options=QFileDialog.Options(),
        )
        if file_path:
            self._ui.input_export_annotation.setText(file_path)
            self._ui.checkbox_export_annotation.setChecked(True)
            outline_image_path = f"{path.splitext(file_path)[0]}_outline.tif"
            if self._ui.input_export_outline_image.text() == "":
                self._ui.input_export_outline_image.setText(outline_image_path)
                self._ui.checkbox_export_outline_image.setChecked(True)
            fill_image_path = f"{path.splitext(file_path)[0]}_fill.tif"
            if self._ui.input_export_fill_image.text() == "":
                self._ui.input_export_fill_image.setText(fill_image_path)
                self._ui.checkbox_export_fill_image.setChecked(True)
            mask_image_path = f"{path.splitext(file_path)[0]}_mask.tif"
            if self._ui.input_export_mask_image.text() == "":
                self._ui.input_export_mask_image.setText(mask_image_path)
                self._ui.checkbox_export_mask_image.setChecked(True)

    def _select_export_outline_image(self):
        file_path, _ = QFileDialog.getSaveFileName(
            None,
            "Save Outline Image File",
            "",
            "TIFF Files (*.tif *.tiff)",
            options=QFileDialog.Options(),
        )
        if file_path:
            self._ui.input_export_outline_image.setText(file_path)
            self._ui.checkbox_export_outline_image.setChecked(True)

    def _select_export_fill_image(self):
        file_path, _ = QFileDialog.getSaveFileName(
            None,
            "Save Fill Image File",
            "",
            "TIFF Files (*.tif *.tiff)",
            options=QFileDialog.Options(),
        )
        if file_path:
            self._ui.input_export_fill_image.setText(file_path)
            self._ui.checkbox_export_fill_image.setChecked(True)

    def _select_export_mask_image(self):
        file_path, _ = QFileDialog.getSaveFileName(
            None,
            "Save Mask Image File",
            "",
            "TIFF Files (*.tif *.tiff)",
            options=QFileDialog.Options(),
        )
        if file_path:
            self._ui.input_export_mask_image.setText(file_path)
            self._ui.checkbox_export_mask_image.setChecked(True)

    def _export(self):
        if len(self._writer_list) > 0:
            QMessageBox.warning(
                None,
                "Warning",
                "Currently writing, please try again later",
                QMessageBox.Ok,
            )
            return
        if self._ui.checkbox_export_annotation.isChecked():
            with open(
                self._ui.input_export_annotation.text(), "w", encoding="utf-8"
            ) as file:
                json.dump(self._info.json, file, indent=4)
        total = 0
        count = {
            "outline": 0,
            "fill": 0,
            "mask": 0,
        }
        path_list = {
            "outline": self._ui.input_export_outline_image.text(),
            "fill": self._ui.input_export_fill_image.text(),
            "mask": self._ui.input_export_mask_image.text(),
        }

        self._dialog = QProgressDialog("processing...", None, 0, 100)
        self._dialog.setWindowFlags(
            Qt.Dialog
            | Qt.CustomizeWindowHint
            | Qt.WindowTitleHint
            | Qt.MSWindowsFixedSizeDialogHint
            | Qt.WindowStaysOnTopHint
        )
        self._dialog.setWindowTitle("Export")
        self._dialog.setCancelButton(None)
        self._dialog.setAutoClose(False)
        self._dialog.setAutoReset(False)
        self._dialog.setValue(0)

        def create_writer(mode: Literal["outline", "fill", "mask"], value):
            print("TIFF Writer type: " + mode)
            nonlocal total
            total = total + 3
            tiff_writer = TiffWriter(
                self._info.image_path,
                path_list[mode],
                [polygon for label in self._info.labels for polygon in label.polygons],
                mode,
                value,
            )
            self._writer_list.append(tiff_writer)

            def on_writer_progress(progress):
                count[mode] = progress * 3
                self._dialog.setValue(
                    int((count["outline"] + count["fill"] + count["mask"]) / total)
                )

            def on_writer_completed(writer):
                count[mode] = 100 * 3
                self._dialog.setValue(
                    int((count["outline"] + count["fill"] + count["mask"]) / total)
                )
                self._writer_list.remove(writer)
                if len(self._writer_list) == 0:
                    self._dialog.close()
                    self._build_overview(list(path_list.values()))

            tiff_writer.progress.connect(on_writer_progress)
            tiff_writer.completed.connect(on_writer_completed)
            tiff_writer.start()
        print("Start TIFF Writer ... ")
        if self._ui.checkbox_export_outline_image.isChecked():
            create_writer("outline", self._ui.input_export_line_width.value())
        if self._ui.checkbox_export_fill_image.isChecked():
            create_writer("fill", self._ui.input_export_opacity.value())
        if self._ui.checkbox_export_mask_image.isChecked():
            create_writer("mask", None)

        self._dialog.show()

        if (not self._ui.checkbox_export_outline_image.isChecked()) and (not self._ui.checkbox_export_fill_image.isChecked()) and (not self._ui.checkbox_export_mask_image.isChecked()):
            self._dialog.setValue(100)
            self._dialog.close()
            QMessageBox.information(
                None,
                "Success",
                "File processing completed",
                QMessageBox.Ok,
            )

        self._ui.checkbox_export_annotation.setChecked(False)
        self._ui.checkbox_export_outline_image.setChecked(False)
        self._ui.checkbox_export_fill_image.setChecked(False)
        self._ui.checkbox_export_mask_image.setChecked(False)
        self._ui.input_export_annotation.setText("")
        self._ui.input_export_outline_image.setText("")
        self._ui.input_export_fill_image.setText("")
        self._ui.input_export_mask_image.setText("")

    def stop_all_writers(self):
        """停止所有正在运行的写入线程，并等待它们结束"""
        for writer in self._writer_list[:]:  # 遍历副本
            writer.requestInterruption()  # 请求中断
            writer.quit()  # 退出 QThread 的事件循环（虽然没用到，但安全）
            writer.wait(5000)  # 等待最多5秒
        self._writer_list.clear()
