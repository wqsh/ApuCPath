# -*- coding: utf-8 -*-
from ApuCellGlobal.MSApuCellSystemGlobal import CSystemGlobal
CSystemGlobal.SetNameSoftware("ApuCPath")
from Core.MSFlow import CFlow1
import sys

class CStaff:
    def __init__(self):
        pass

    def start(self, argv):

        self.__captainRunFlow(argv)

    def __captainRunFlow(self, argv):

        flow1 = CFlow1()
        flow1.run(argv)
