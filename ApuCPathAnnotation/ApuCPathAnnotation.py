import sys

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QCursor, QIcon
from PyQt5.QtWidgets import QApplication, QMainWindow, QGraphicsScene
from osgeo import gdal

from controllers.drawing_controller import DrawingController
from controllers.editor_controller import EditorController
from controllers.file_controller import FileController
from controllers.image_controller import ImageController
from views.main_window import Ui_MainWindow

# 自定义主窗口类
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.file_controller = None   # 稍后赋值

    def closeEvent(self, event):
        if self.file_controller:
            self.file_controller.stop_all_writers()   # 停止所有后台线程
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    gdal.UseExceptions()

    # Views
    main_window = MainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(main_window)
    with open("ui/main_window.qss", "r") as file:
        app.setStyleSheet(file.read())
    ui.viewer.setScene(QGraphicsScene())
    ui.viewer.viewport().setCursor(QCursor(Qt.CrossCursor))

    # Controllers
    file_controller = FileController(main_window, ui)
    editor_controller = EditorController(main_window, ui)
    image_controller = ImageController(main_window, ui)
    drawing_controller = DrawingController(main_window, ui)

    # Signals
    file_controller.annotation_changed.connect(editor_controller.init)
    file_controller.image_changed.connect(image_controller.load_image)
    file_controller.mask_changed.connect(image_controller.load_mask)
    editor_controller.start_drawing.connect(drawing_controller.start)
    editor_controller.cancel_painting.connect(drawing_controller.cancel)

    drawing_controller.on_completed.connect(editor_controller.append_polygon)

    main_window.show()
    sys.exit(app.exec())
