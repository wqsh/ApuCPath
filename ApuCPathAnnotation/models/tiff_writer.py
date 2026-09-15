import math
from typing import Literal

import psutil
from osgeo import gdal
import time
import cv2
import numpy as np
from PyQt5.QtCore import pyqtSignal, QThread
from PyQt5.QtGui import QColor
from models.polygon import Polygon
# import multiprocessing as mp
import threading
import queue

class TiffWriter(QThread):
    progress = pyqtSignal(int)
    completed = pyqtSignal(object)

    def __init__(
        self,
        origin_path: str,
        target_path: str,
        polygons: list[Polygon],
        mode: Literal["outline", "fill", "mask"],
        value,
    ):
        super().__init__()

        # self.queue = mp.Queue()
        # self.process = mp.Process(
        self.queue = queue.Queue()
        self.thread = threading.Thread(
            target=self.task,
            args=(
                origin_path,
                target_path,
                [
                    [
                        [[p.x(), p.y()] for p in polygon.points],
                        QColor(polygon.attributes.color).getRgb()[:3][::-1],
                        (
                            min(p.x() for p in polygon.points),
                            max(p.x() for p in polygon.points),
                            min(p.y() for p in polygon.points),
                            max(p.y() for p in polygon.points),
                        ),
                    ]
                    for polygon in polygons
                ],
                mode,
                value,
                self.queue,
            ),
        )

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
        end_time = time.time()
        print(f"TIFF Write Time: {end_time - start_time}s")
        self.completed.emit(self)

    @staticmethod
    def task(
        origin_path: str,
        target_path: str,
        polygons,
        mode: Literal["outline", "fill", "mask"],
        value,
        queue: queue.Queue,
    ):
        gdal.UseExceptions()
        gdal.SetConfigOption(
            "GDAL_NUM_THREADS", str((psutil.cpu_count(logical=False) - 1) // 3)
        )
        gdal.SetConfigOption(
            "GDAL_CACHEMAX",
            str((psutil.virtual_memory().available * 0.2) // (1024 * 1024)),
        )
        task_size = 4096

        band = 1 if mode == "mask" else 3
        origin = gdal.OpenShared(origin_path, gdal.GA_ReadOnly)
        width = origin.RasterXSize
        height = origin.RasterYSize
        target = gdal.GetDriverByName("GTiff").Create(
            target_path,
            width,
            height,
            band,
            gdal.GDT_Byte,
            options=[
                "BIGTIFF=YES",
                "COMPRESS=JPEG",
                f"PHOTOMETRIC={'MINISBLACK' if band == 1 else 'YCBCR'}",
                "JPEG_QUALITY=80",
                "INTERLEAVE=PIXEL",
                "JPEGTABLESMODE=1",
                "TILED=YES",
                f"BLOCKXSIZE=512",
                f"BLOCKYSIZE=512",
                f"NUM_THREADS={(psutil.cpu_count(logical=False) - 1) // 3}",
            ],
        )
        total = math.ceil(height / task_size) * math.ceil(width / task_size)
        count = 0
        for y in range(0, height, task_size):
            for x in range(0, width, task_size):
                tile_width = min(task_size, width - x)
                tile_height = min(task_size, height - y)
                data = np.zeros((tile_height, tile_width, band), dtype=np.uint8)
                if band == 3:
                    for band_idx in range(3):
                        data[:, :, band_idx] = origin.GetRasterBand(
                            band_idx + 1
                        ).ReadAsArray(x, y, tile_width, tile_height)
                if band == 3:
                    data = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)
                for polygon in polygons:
                    points, color, bbox = polygon
                    if not (
                        x > bbox[1]
                        or x + tile_width < bbox[0]
                        or y > bbox[3]
                        or y + tile_height < bbox[2]
                    ):
                        if mode == "outline":
                            cv2.polylines(
                                data,
                                [np.array(points, dtype=np.int32) - [x, y]],
                                True,
                                color,
                                value,
                                cv2.LINE_4,
                            )
                        elif mode == "fill":
                            overlay = data.copy()
                            cv2.fillPoly(
                                overlay,
                                [np.array(points, dtype=np.int32) - [x, y]],
                                color,
                            )
                            alpha = value / 100
                            cv2.addWeighted(overlay, alpha, data, 1 - alpha, 0, data)
                        elif mode == "mask":
                            cv2.fillPoly(
                                data, [np.array(points, dtype=np.int32) - [x, y]], [255]
                            )
                if band == 3:
                    data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
                for band_idx in range(band):
                    target.GetRasterBand(band_idx + 1).WriteArray(
                        data[:, :, band_idx], x, y
                    )
                target.FlushCache()
                count = count + 1
                queue.put(int(count * 100 / total))
        origin = None
        target = None
        del origin, target

    def __del__(self):
        if hasattr(self, 'thread') and self.thread.is_alive():
            self.thread.join(timeout=1)
