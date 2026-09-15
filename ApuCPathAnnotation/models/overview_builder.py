import time

import psutil
from PyQt5.QtCore import pyqtSignal, QThread
from osgeo import gdal
# import multiprocessing as mp
import threading
import queue

# [CHANGED] 新增全局锁，用于保护 GDAL 配置
_gdal_config_lock = threading.Lock()

class OverviewBuilder(QThread):
    progress = pyqtSignal(int)
    completed = pyqtSignal(object)

    def __init__(
        self,
        file_path,
        parent=None,
    ):
        super().__init__(parent)
        self._tiff_path = file_path
        # self.queue = mp.Queue()
        # self.process = mp.Process(target=self.task, args=(file_path, self.queue))
        self.queue = queue.Queue()
        self.thread = threading.Thread(target=self.task, args=(file_path, self.queue))

    def run(self):
        start_time = time.time()
        # self.process.start()
        # while self.process.is_alive():
        self.thread.start()
        while self.thread.is_alive() and not self.isInterruptionRequested():
            while not self.queue.empty():
                try:
                    msg = self.queue.get(timeout=1)
                    self.progress.emit(msg)
                except Exception:
                    pass
            # 避免忙等，让出 CPU
            self.msleep(10)
        # self.process.join()
        # 如果是因为中断而退出，则等待子线程结束
        if self.thread.is_alive():
            self.thread.join(timeout=5)  # 等待最多5秒
        self.completed.emit(self)
        end_time = time.time()
        print(f"Overview Build Time: {end_time - start_time}s")

    @staticmethod
    def task(file_path, queue):
        gdal.UseExceptions()
        dataset = gdal.Open(file_path, gdal.GA_Update)

        # 获取波段数和颜色解释
        bands = dataset.RasterCount
        color_interp = [dataset.GetRasterBand(i + 1).GetColorInterpretation() for i in range(bands)]
        is_rgb = (bands == 3 and color_interp == [gdal.GCI_RedBand, gdal.GCI_GreenBand, gdal.GCI_BlueBand])
        is_rgba = (bands == 4 and color_interp[:3] == [gdal.GCI_RedBand, gdal.GCI_GreenBand, gdal.GCI_BlueBand] and color_interp[3] == gdal.GCI_AlphaBand)
        # ========== 根据图像类型配置概览压缩 ==========
        with _gdal_config_lock:  # <--- 加锁开始
            if bands == 1:
                # 单波段灰度图：使用无损压缩（DEFLATE），不用 JPEG
                gdal.SetConfigOption("COMPRESS_OVERVIEW", "DEFLATE")
                gdal.SetConfigOption("ZLEVEL", "6")
                # 清除可能遗留的 PHOTOMETRIC_OVERVIEW 配置
                gdal.SetConfigOption("PHOTOMETRIC_OVERVIEW", "")
            elif is_rgb:
                # 标准三波段 RGB：JPEG + YCBCR（有损，高压缩比）
                gdal.SetConfigOption("COMPRESS_OVERVIEW", "JPEG")
                gdal.SetConfigOption("JPEG_QUALITY_OVERVIEW", "80")
                gdal.SetConfigOption("PHOTOMETRIC_OVERVIEW", "YCBCR")
            elif is_rgba:
                # RGBA 四波段：使用 DEFLATE 无损压缩，且只对 RGB 波段生成概览
                gdal.SetConfigOption("COMPRESS_OVERVIEW", "DEFLATE")
                gdal.SetConfigOption("ZLEVEL", "6")
                gdal.SetConfigOption("PHOTOMETRIC_OVERVIEW", "")
                # 注意：下面 BuildOverviews 时会指定 bands=[1,2,3]，跳过 Alpha 波段
            else:
                # 其他情况（2、5+ 波段等）：DEFLATE 无损压缩
                gdal.SetConfigOption("COMPRESS_OVERVIEW", "DEFLATE")
                gdal.SetConfigOption("ZLEVEL", "6")
                gdal.SetConfigOption("PHOTOMETRIC_OVERVIEW", "")

            # ========== 性能配置 ==========
            gdal.SetConfigOption(
                "GDAL_NUM_THREADS", str((psutil.cpu_count(logical=False) - 1) // 3)
            )
            gdal.SetConfigOption(
                "GDAL_CACHEMAX",
                str((psutil.virtual_memory().available * 0.2) // (1024 * 1024)),
            )
            gdal.SetConfigOption("BLOCKXSIZE_OVERVIEW", "512")
            gdal.SetConfigOption("BLOCKYSIZE_OVERVIEW", "512")
            gdal.SetConfigOption("JPEGTABLESMODE", "1")
            gdal.SetConfigOption("BIGTIFF_OVERVIEW", "IF_SAFER")
            gdal.SetConfigOption("COPY_SRC_OVERVIEWS", "NO")

            def update_progress(complete, _message, _data):
                queue.put(int(complete * 100))
                return 1

            queue.put(0)
            # ========== 生成概览（所有波段） ==========
            if is_rgba:
                # 对 RGBA 图像，只对波段 1,2,3 生成概览（避免 Alpha 波段只读错误）
                dataset.BuildOverviews("NEAREST", [2, 4, 8, 16, 32, 64, 128],
                                       callback=update_progress, bands=[1, 2, 3])
            else:
                # 正常对所有波段生成概览
                dataset.BuildOverviews("NEAREST", [2, 4, 8, 16, 32, 64, 128],
                                       callback=update_progress)
        dataset.FlushCache()
        dataset = None
        del dataset

    def __del__(self):
        if hasattr(self, 'thread') and self.thread.is_alive():
            self.thread.join(timeout=1)