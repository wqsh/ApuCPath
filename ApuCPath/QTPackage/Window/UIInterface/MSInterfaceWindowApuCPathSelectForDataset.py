from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

from QTPackage.Window.UICode.MSWindowApuCPathSelectForDataset import  Ui_WindowApuCPathSelectForDataset

from ApuCellGlobal.GeneralSystem.MSSystemEnum import LIST_VISUAL_FILE_TYPE_SUFFIX, LIST_VISUAL_FILE_IMAGE_IDX, EnumVisualFileType

from QTPackage.Window.MSSystemWindowGlobal import EnumSelectForDatasetFileType, CSystemWindowSelectForDataset, EnumUIDatasetType
import os


class CInterfaceWindowApuCPathSelectForDataset(QMainWindow, Ui_WindowApuCPathSelectForDataset):

    saveSelectDataset = pyqtSignal(int)

    def __init__(self, select_dataset_type_idx:int):
        super(CInterfaceWindowApuCPathSelectForDataset, self).__init__()
        self.__select_dataset_type:EnumUIDatasetType
        if select_dataset_type_idx == EnumUIDatasetType.Train.value:
            self.__select_dataset_type = EnumUIDatasetType.Train
        elif select_dataset_type_idx == EnumUIDatasetType.Assist.value:
            self.__select_dataset_type = EnumUIDatasetType.Assist
        elif select_dataset_type_idx == EnumUIDatasetType.Predict.value:
            self.__select_dataset_type = EnumUIDatasetType.Predict
        elif select_dataset_type_idx == EnumUIDatasetType.Valid.value:
            self.__select_dataset_type = EnumUIDatasetType.Valid
        self.__file_type:EnumSelectForDatasetFileType = EnumSelectForDatasetFileType.ImageFile
        self.cwd = os.getcwd()
        self.setupUi(self)
        self.__captainInitControl()

    def __captainInitControl(self):

        self.PushButton_ImageFile.setCheckable(True)
        self.PushButton_ImageFolder.setCheckable(True)

        self.UpdateFileType()

        self.LineEdit_ImageFile_Image.setReadOnly(True)
        self.LineEdit_ImageFile_Mask.setReadOnly(True)
        self.ComboBox_ImageFile_DatasetType.addItems([item.name for item in EnumUIDatasetType])
        self.ComboBox_ImageFile_DatasetType.setCurrentIndex(self.__select_dataset_type.value)
        self.ComboBox_ImageFile_DatasetType.setDisabled(True)
        if self.__select_dataset_type == EnumUIDatasetType.Predict:
            self.Frame_ImageFile_Mask.setHidden(True)

        self.LineEdit_ImageFolder_Image.setReadOnly(True)
        self.LineEdit_ImageFolder_Mask.setReadOnly(True)
        self.ComboBox_ImageFolder_DatasetType.addItems([item.name for item in EnumUIDatasetType])
        self.ComboBox_ImageFolder_DatasetType.setCurrentIndex(self.__select_dataset_type.value)
        self.ComboBox_ImageFolder_DatasetType.setDisabled(True)
        if self.__select_dataset_type == EnumUIDatasetType.Predict:
            self.Frame_ImageFolder_Mask.setHidden(True)

    def UpdateFileType(self):

        self.PushButton_ImageFile.setChecked(False)
        self.PushButton_ImageFolder.setChecked(False)

        if self.__file_type == EnumSelectForDatasetFileType.ImageFile:
            self.PushButton_ImageFile.setChecked(True)
        elif self.__file_type == EnumSelectForDatasetFileType.ImageFolder:
            self.PushButton_ImageFolder.setChecked(True)
        else:
            self.__file_type = EnumSelectForDatasetFileType.ImageFile
            self.PushButton_ImageFile_Image.setChecked(True)

        self.stackedWidget.setCurrentIndex(self.__file_type.value)

    @pyqtSlot()
    def on_PushButton_ImageFile_clicked(self):

        self.__file_type = EnumSelectForDatasetFileType.ImageFile
        self.UpdateFileType()

    @pyqtSlot()
    def on_PushButton_ImageFolder_clicked(self):

        self.__file_type = EnumSelectForDatasetFileType.ImageFolder
        self.UpdateFileType()

    @pyqtSlot()
    def on_PushButton_ImageFile_Image_clicked(self):

        img_select_filter = "Image Files("
        for item in LIST_VISUAL_FILE_IMAGE_IDX:
            img_select_filter += f"*{LIST_VISUAL_FILE_TYPE_SUFFIX[item]} "
        img_select_filter += ")"
        path_cur_file, file_type = QFileDialog.getOpenFileName(self, 'Choose the file', self.cwd, img_select_filter)
        path_img = path_cur_file
        if len(path_img) == 0:
            pass
        else:
            self.LineEdit_ImageFile_Image.setText(path_img)

    @pyqtSlot()
    def on_PushButton_ImageFile_Mask_clicked(self):

        img_select_filter = "Image Files("
        # for item in LIST_VISUAL_FILE_IMAGE_IDX:
        #     img_select_filter += f"*{LIST_VISUAL_FILE_TYPE_SUFFIX[item]} "
        img_select_filter += f"*{LIST_VISUAL_FILE_TYPE_SUFFIX[EnumVisualFileType.JSON.value]}"
        img_select_filter += ")"
        path_cur_file, file_type = QFileDialog.getOpenFileName(self, 'Choose the file', self.cwd, img_select_filter)
        path_img = path_cur_file
        if len(path_img) == 0:
            pass
        else:
            self.LineEdit_ImageFile_Mask.setText(path_img)

    @pyqtSlot()
    def on_PushButton_ImageFolder_Image_clicked(self):

        dir_choose = QFileDialog.getExistingDirectory(self, 'Directory Select', os.getcwd())
        if len(dir_choose) == 0:
            pass
        else:
            self.LineEdit_ImageFolder_Image.setText(dir_choose)

    @pyqtSlot()
    def on_PushButton_ImageFolder_Mask_clicked(self):

        dir_choose = QFileDialog.getExistingDirectory(self, 'Directory Select', os.getcwd())
        if len(dir_choose) == 0:
            pass
        else:
            self.LineEdit_ImageFolder_Mask.setText(dir_choose)


    @pyqtSlot()
    def on_PushButton_OK_clicked(self):
        if self.__file_type == EnumSelectForDatasetFileType.ImageFile:
            CSystemWindowSelectForDataset.FILE_TYPE = self.__file_type
            CSystemWindowSelectForDataset.PATH_IMAGE = self.LineEdit_ImageFile_Image.text()
            CSystemWindowSelectForDataset.PATH_MASK = self.LineEdit_ImageFile_Mask.text()
        elif self.__file_type == EnumSelectForDatasetFileType.ImageFolder:
            CSystemWindowSelectForDataset.FILE_TYPE = self.__file_type
            CSystemWindowSelectForDataset.PATH_IMAGE = self.LineEdit_ImageFolder_Image.text()
            CSystemWindowSelectForDataset.PATH_MASK = self.LineEdit_ImageFolder_Mask.text()
        else:
            CSystemWindowSelectForDataset.FILE_TYPE = None
        self.saveSelectDataset.emit(self.__file_type.value)
        self.close()

    @pyqtSlot()
    def on_PushButton_Cancel_clicked(self):

        self.close()