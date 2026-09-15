from ApuCellGlobal.MSApuCellExternalPackge import *
from ApuCellGlobal.GeneralSystem.MSLogging import CLogging

import ctypes
import gc
import sys

class CFunctionMemory:
    @staticmethod
    def GetCurMemoryInfo():
        memory = psutil.virtual_memory()
        CLogging.LogToUser("[Memory report] Total memory %.3f GB. "%(memory.total / 1024 / 1024 / 1024))
        CLogging.LogToUser("[Memory report] Available memory %.3f GB. "%(memory.available / 1024 / 1024 / 1024))
        CLogging.LogToUser("[Memory report] Used memory %.3f GB. "%(memory.used / 1024 / 1024 / 1024))
        CLogging.LogToUser("[Memory report] Percent %.2f%%. "%memory.percent)

    @staticmethod
    def DeleteVariableMemory(variable):
        # 慎用这个，这个会把所有作用域上引用该变量的都释放，即引用次数直接为0,没有被改写还有可能被访问到不会触发异常
        if isinstance(variable, list):
            for item in variable:
                ctypes.pythonapi.Py_DecRef(ctypes.py_object(item))
        else:
            ctypes.pythonapi.Py_DecRef(ctypes.py_object(variable))
            # del variable
        gc.collect()

    @staticmethod
    def CleanMemory():

        gc.collect()
