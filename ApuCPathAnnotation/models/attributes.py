from PyQt5.QtCore import QObject, pyqtSignal


class Attributes(QObject):
    changed = pyqtSignal()

    def __init__(self, parent=None):
        QObject.__init__(self, parent)

    def __len__(self):
        return len(self.__dict__)

    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        self.changed.emit()

    @property
    def json(self):
        return self.__dict__

    @property
    def list(self):
        return list(self.__dict__.items())
