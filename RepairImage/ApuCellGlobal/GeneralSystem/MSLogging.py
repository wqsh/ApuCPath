# -*- mode: python ; coding: utf-8 -*-
import os.path

from ApuCellGlobal.MSApuCellExternalPackge import *

from ApuCellGlobal.GeneralSystem.MSSystem import PATH_LOG, PATH_ERROR, NAME_SOFTWARE_LOG, FOLDER_LOG, NAME_SOFTWARE, PATH_STATE
from ApuCellGlobal.GeneralSystem.MSLoggingInfo import ERROR_TO_USER



LIST_INFO_TYPE_LOG = ["[Info] ", "[Warn] ", "[Error] ", "[State] "]
@unique
class EnumInfoType(Enum):

    Info = 0
    Warn =1
    Error = 2
    State = 3


def Assert(value:bool):
    if not value:
        tb_stack = traceback.extract_stack()
        info = "[Error] "
        for s in tb_stack:
            info += s.filename + ' ' + str(s.lineno) + ' ' + s.name + '\n'
        CLogging.LogError(info)
        assert value

class CLogging:
    loc_time = time.localtime()
    str_loc_time = "%4d-%02d-%02d" % (loc_time.tm_year, loc_time.tm_mon, loc_time.tm_mday)

    __path_log = os.path.join(os.path.join(os.getcwd(), FOLDER_LOG), PATH_LOG + str_loc_time)
    __path_error = os.path.join(os.path.join(os.getcwd(), FOLDER_LOG), PATH_ERROR + str_loc_time)
    __path_state = os.path.join(os.path.join(os.getcwd(), FOLDER_LOG), PATH_STATE + str_loc_time)

    NAME_SOFTWARE = NAME_SOFTWARE
    NAME_SOFTWARE_LOG = NAME_SOFTWARE_LOG

    CHECK_RUN_STATE:bool = False

    if not os.path.exists(os.path.join(os.getcwd(), FOLDER_LOG)):
        os.makedirs(os.path.join(os.getcwd(), FOLDER_LOG))

    @staticmethod
    def GetLocTimeStr():
        loc_time = time.localtime()
        str_loc_time = "%4d-%02d-%02d %02d:%02d:%02d "%(loc_time.tm_year, loc_time.tm_mon, loc_time.tm_mday, loc_time.tm_hour, loc_time.tm_min, loc_time.tm_sec)
        return str_loc_time

    @staticmethod
    def GetLocTimeStrForName():
        loc_time = time.localtime()
        str_loc_time = "%4d%02d%02d%02d%02d%02d"%(loc_time.tm_year, loc_time.tm_mon, loc_time.tm_mday, loc_time.tm_hour, loc_time.tm_min, loc_time.tm_sec)
        return str_loc_time

    @staticmethod
    def LogForStart(str_name:str=""):
        strInfo = "=========== Start " + str_name + " ==========="
        CLogging.LogToUser("\n")
        CLogging.LogToUser(strInfo)

    @staticmethod
    def LogForEnd(str_name:str=""):
        strInfo = "=========== End " + str_name + " ==========="
        CLogging.LogToUser(strInfo)
        CLogging.LogToUser("\n")

    @staticmethod
    def LogForLackPath(path:str, info_type:int=1):

        warn_str_name = ERROR_TO_USER.LACK_FILE + path
        if info_type == 1:
            CLogging.LogError(warn_str_name)
        elif info_type == 2:
            CLogging.LogWarning(warn_str_name)
        else:
            CLogging.LogError(warn_str_name)

    # 也许向某个日志文件里写东西
    @staticmethod
    def LogToUser(strInfo:str=None, log2file:bool=True, log2user:bool=True, log_type:int=0):
        if strInfo == None:
            Info = ""
        else:
            Info = LIST_INFO_TYPE_LOG[log_type] + NAME_SOFTWARE_LOG + CLogging.GetLocTimeStr() + strInfo
        if log2user:
            print(Info)
            sys.stdout.flush()
        if log2file:
            try:
                f_w = open(CLogging.__path_log, 'a', encoding='utf8')
                f_w.write(Info + '\n')
                f_w.close()
            except IOError:
                print(f"{PATH_LOG}.log is opened! Please close it and run the program again!")
                sys.exit(0)

    @staticmethod
    def LogError(info, log2user:bool=True):

        tb_stack = traceback.extract_stack()
        stack_info = CLogging.GetLocTimeStr() + "\n"
        for i in range(len(tb_stack) - 1):
            s = tb_stack[i]
            stack_info += LIST_INFO_TYPE_LOG[EnumInfoType.Error.value] + s.filename + ' ' + str(s.lineno) + ' ' + s.name + '\n'
        print(stack_info)

        CLogging.__captainExportToError(info)
        CLogging.__captainExportToError(stack_info)

        CLogging.__captainExportToLog(info)
        CLogging.__captainExportToLog(stack_info)

        if log2user:
            CLogging.LogToUser(info, log_type=EnumInfoType.Error.value)

        sys.exit("Error")

    @staticmethod
    def LogWarning(info, log2user:bool=True, raise_state:bool=False):
        if log2user:
            if raise_state:
                CLogging.LogToUser(info, log_type=EnumInfoType.Warn.value, log2user=False)
                raise Exception(info)
            else:
                CLogging.LogToUser(info, log_type=EnumInfoType.Warn.value, log2user=True)
        CLogging.__captainExportToWarn(info)

    @staticmethod
    def LogException(msg:Exception):

        stack_info = str(msg) + "\n"
        tmp_trace_back = msg.__traceback__
        while tmp_trace_back is not None:
            stack_info += tmp_trace_back.tb_frame.f_globals["__file__"] + ' ' + str(tmp_trace_back.tb_lineno) + '\n'
            tmp_trace_back = tmp_trace_back.tb_next

        CLogging.__captainExportToError(stack_info)
        CLogging.LogToUser(stack_info, True, True, EnumInfoType.Error.value)

    @staticmethod
    def __captainExportToLog(info:str):
        try:
            f_w = open(PATH_LOG, 'a', encoding='utf8')
            f_w.write(info + '\n')
            f_w.close()
        except IOError:
            print(f"{CLogging.__path_log} is opened! Please close it and run the program again!")
            sys.exit(0)

    @staticmethod
    def __captainExportToWarn(info: str):
        try:
            f_w = open(CLogging.__path_log, 'a', encoding='utf8')
            f_w.write(info + '\n')
            f_w.close()
        except IOError:
            print(f"{CLogging.__path_log} is opened! Please close it and run the program again!")
            sys.exit(0)

    @staticmethod
    def __captainExportToError(info: str):
        try:
            f_w = open(CLogging.__path_error, 'a', encoding='utf8')
            f_w.write(info + '\n')
            f_w.close()
        except IOError:
            print(f"{CLogging.__path_error} is opened! Please close it and run the program again!")
            sys.exit(0)

    @staticmethod
    def CheckState():
        if not CLogging.CHECK_RUN_STATE:
            return
        tb_stack = traceback.extract_stack()
        stack_info = CLogging.GetLocTimeStr() + "\n"
        for i in range(len(tb_stack) - 1):
            s = tb_stack[i]
            stack_info += LIST_INFO_TYPE_LOG[EnumInfoType.State.value] + s.filename + ' ' + str(s.lineno) + ' ' + s.name + '\n'
        print(stack_info)
        CLogging.__captainExportCheckState(stack_info)

    @staticmethod
    def __captainExportCheckState(info:str):

        try:
            f_w = open(CLogging.__path_state, 'a', encoding='utf8')
            f_w.write(info + '\n')
            f_w.close()
        except IOError:
            print(f"{CLogging.__path_state} is opened! Please close it and run the program again!")
            sys.exit(0)


class CLogProgressPercent:

    def __init__(self, start_num:int, end_num:int, point_step:float, title_info:str):

        point_value = 0.0
        self.__title_info:str = "[Progress] " + title_info
        flag = False
        if point_step < 0.01:
            point_step = 0.01
        if end_num < start_num:
            CLogging.LogWarning("[Progress] end_num must be greater than start_num . ")
            end_num = start_num

        delta_num = end_num - start_num
        self.__progress_point_num:list=[start_num + delta_num * point_value]
        self.__end_num:int = end_num
        while point_value < 1.0:
            point_value += point_step
            if point_value > 1.0:
                self.__progress_point_num.append(self.__end_num)
            else:
                self.__progress_point_num.append(start_num + point_value * delta_num)
        self.step_point = point_step
        self.__cur_idx:int = 1

        CLogging.LogToUser(self.__title_info + " Progress: 0.00 %")
        if flag:
            CLogging.LogToUser(self.__title_info + " Progress: 100.00 %")

    def ShowProgressPercent(self, cur_num):
        if cur_num >= self.__end_num:
            CLogging.LogToUser(self.__title_info + " Progress: 100.00 %")
        else:
            while cur_num > self.__progress_point_num[self.__cur_idx]:
                CLogging.LogToUser(self.__title_info + " Progress: %.2f %%" % (self.step_point * self.__cur_idx * 100))
                self.__cur_idx += 1
                if self.__cur_idx > len(self.__progress_point_num):
                    CLogging.LogToUser(self.__title_info + " Progress: 100.00 %", log2file=False)
                    break

class CLoggingTime:

    __StartTime = None
    __ListTimePoint = []
    @staticmethod
    def SetStartTime():

        CLoggingTime.__StartTime = time.time()

    @staticmethod
    def UseTime():
        cur_time = time.time()
        CLogging.LogToUser("Use time : %.2f s. " % (cur_time - CLoggingTime.__StartTime))

    @staticmethod
    def AddTimePoint(name:str="", show_log:bool=True):

        time_point = time.time()
        CLoggingTime.__ListTimePoint.append([time_point, name])
        if show_log:
            CLogging.LogToUser(name)

    @staticmethod
    def PrintTimePoint():
        for idx, item in enumerate(CLoggingTime.__ListTimePoint):
            if idx == 0:
                used_time = item[0] - CLoggingTime.__StartTime
            elif idx == len(CLoggingTime.__ListTimePoint) - 1:
                used_time = time.time() - item[0]
            else:
                used_time = CLoggingTime.__ListTimePoint[idx][0] - CLoggingTime.__ListTimePoint[idx - 1][0]
            CLogging.LogToUser(item[1] + "Use time : %.2f s. "%(used_time))
        CLoggingTime.UseTime()