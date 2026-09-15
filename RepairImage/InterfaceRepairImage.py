from PyQt5.QtWidgets import QMainWindow, QFileDialog
from PyQt5.QtCore import pyqtSlot

from WindowRepairImage import Ui_WindowRepairImage

from ApuCellGlobal.GeneralFunction.MSFunctionIO import CFunctionFileIO

import os

class RepariImageWindow(QMainWindow, Ui_WindowRepairImage):

    def __init__(self):
        super().__init__()
        # loadUi("VistaNaviUI.ui", self)
        self.setupUi(self)
        self.ProgressBar.setValue(0)

    @pyqtSlot()
    def on_PushButton_PathImage_clicked(self):
        fileName, _ = QFileDialog.getOpenFileName(self, "Select File", "", "tif, tiff file (*tif *tiff)")
        if os.path.isfile(fileName):
            self.LineEdit_PathImage.setText(fileName)
        else:
            return

    @pyqtSlot()
    def on_PushButton_PathExport_clicked(self):

        fileName, _ = QFileDialog.getSaveFileName(self, "Select File", "", "tif, tiff file (*tif *tiff)")
        if os.path.isdir(fileName):
            return
        else:
            if fileName.endswith(".tif") or fileName.endswith(".tiff"):
                self.LineEdit_PathExport.setText(fileName)
            else:
                self.LineEdit_PathExport.setText(fileName + ".tiff")

    @pyqtSlot()
    def on_PushButton_Start_clicked(self):

        self.ProgressBar.setValue(0)

        self.ProgressBar.setValue(20)
        try:
            img = CFunctionFileIO.ReadImage(self.LineEdit_PathImage.text())
        except Exception as msg:
            img = CFunctionFileIO.ReadTIFByGDAl(self.LineEdit_PathImage.text())
        try:
            self.ProgressBar.setValue(40)
            if self.CheckBox_Compress.checkState():
                CFunctionFileIO.WriteTIFByGDALCompress(os.path.dirname(self.LineEdit_PathExport.text()), os.path.basename(self.LineEdit_PathExport.text()), img)
            else:
                CFunctionFileIO.WriteTIFByGDAL(os.path.dirname(self.LineEdit_PathExport.text()), os.path.basename(self.LineEdit_PathExport.text()), img)
            self.ProgressBar.setValue(80)
            self.ProgressBar.setValue(100)
        except Exception as msg:
            print(msg)
            self.ProgressBar.setValue(70)
