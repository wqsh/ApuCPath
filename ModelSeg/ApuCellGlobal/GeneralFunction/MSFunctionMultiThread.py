from ApuCellGlobal.GeneralSystem.MSLogging import CLogProgressPercent

from threading import Thread, Lock

THREAD_NUM = 8

class CFunctionMultiThread():


    def __init__(self, data_list:list=[], thread_num:int=1, progress_log_info:str="", show_progress_log_info:bool=True, show_thread_idx:bool=False):

        self.thread_num = thread_num
        self.__lock = Lock()
        self.count_idx = 0
        self.data_list = data_list
        self.progress = None
        self.show_progress_log_info = show_progress_log_info
        self.progress_log_info = progress_log_info
        self.show_thread_idx = show_thread_idx

    def RunMultiThread(self):
        thread_list = []
        if self.show_progress_log_info:
            self.progress = CLogProgressPercent(0, len(self.data_list) - 1, 0.05, self.progress_log_info)
        for i in range(self.thread_num):
            if self.show_thread_idx:
                run_thread = Thread(target=self.__captainRunUserFunction, args=(i, ))
            else:
                run_thread = Thread(target=self.__captainRunUserFunction, args=(None, ))
            thread_list.append(run_thread)
        for run_thread in thread_list:
            run_thread.start()
        for run_thread in thread_list:
            run_thread.join()

    def __captainRunUserFunction(self, thread_idx:int=None):

        while True:

            self.__lock.acquire()
            idx = self.count_idx
            self.count_idx += 1
            if not self.progress is None:
                if idx + 1 < len(self.data_list):
                    self.progress.ShowProgressPercent(self.count_idx)
            self.__lock.release()

            if idx >= len(self.data_list):
                break
            if not thread_idx is None:
                print("Thread idx : %d"%thread_idx)
            self.UserFunction(idx)



    def UserFunction(self, idx:int):

        pass



