import sys

from PyQt5.QtWidgets import QApplication

from InterfaceRepairImage import RepariImageWindow


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = RepariImageWindow()
    window.show()
    sys.exit(app.exec_())