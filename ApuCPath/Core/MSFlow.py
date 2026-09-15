import sys

from PyQt5.QtWidgets import *

from QTPackage.Window.UIInterface.MSInterfaceWindowApuCPathMain import CInterfaceWindowApuCPathMain

class CFlow:

    def __init__(self, flow_enum:int=0):
        self.__FlowEnum:int = flow_enum

    def GetFlowEnum(self):

        return self.__FlowEnum

class CFlow1(CFlow):

    def __init__(self, flow_enum:int=1):
        super(CFlow1, self).__init__(flow_enum)

    def run(self, argv):
        # 开始
        app = QApplication(argv)
        window = CInterfaceWindowApuCPathMain()
        window.show()
        sys.exit(app.exec_())