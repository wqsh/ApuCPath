import gc
import os
import time
from concurrent.futures.thread import ThreadPoolExecutor
from concurrent.futures import as_completed

import numpy as np
from PyQt5.QtCore import QObject, QRectF, pyqtSignal, Qt, QEvent, QMetaMethod
from PyQt5.QtGui import QWheelEvent, QImage, QPixmap, QMouseEvent
from PyQt5.QtWidgets import (
    QGraphicsView,
    QMainWindow,
    QGraphicsPixmapItem,
    QLabel,
)

from osgeo import gdal

from views.main_window import Ui_MainWindow
from models.file import EnumTypeImportImage

FOLDER_DLL = "DLL"
OPENSLIDE_PATH = os.path.join(os.getcwd(), FOLDER_DLL)
if os.name == 'nt':
    os.add_dll_directory(OPENSLIDE_PATH)
elif os.name == 'posix':
    ld_path = os.getenv('LD_LIBRARY_PATH')
    if ld_path is None:
        os.environ["LD_LIBRARY_PATH"] = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), FOLDER_DLL)
    else:
        os.environ["LD_LIBRARY_PATH"] += ':' + os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), FOLDER_DLL)
import openslide
from PIL import Image

class ImageController(QObject):
    _item_added = pyqtSignal(object, object)

    def __init__(self, window: QMainWindow, ui: Ui_MainWindow):
        super().__init__(window)
        self._ui = ui
        self._image_item: QGraphicsPixmapItem | None = None
        self._mask_item: QGraphicsPixmapItem | None = None
        self._factor = 1
        self._min_factor = 1
        self._rendering_cnt = 0
        self._render_executor = ThreadPoolExecutor(max_workers=1)
        self._current_image_type:EnumTypeImportImage | None = None
        self._current_image_file_path:str | None = None
        self._item_added.connect(self._add_to_scene)

        ui.viewer.wheelEvent = self._wheel_event
        ui.viewer.horizontalScrollBar().valueChanged.connect(self._render)
        ui.viewer.verticalScrollBar().valueChanged.connect(self._render)

        # Overlay
        self._overlay = QLabel("", window)
        self._overlay.setStyleSheet(
            """
               background-color: rgba(100, 100, 100, 200);
               color: #FFFFFF;
               padding: 5px;
               font-size: 16px;
               line-height: 20px;
            """
        )
        self._overlay.setParent(ui.viewer.viewport())
        self._overlay.adjustSize()
        self._overlay.move(
            self._ui.viewer.viewport().width() - self._overlay.width(), 0
        )
        self._overlay.hide()

        self._ui.viewer.viewport().setCursor(Qt.CrossCursor)
        self._ui.viewer.mouseMoveEvent = self._update_overlay

        def _mouse_press_event(event: QMouseEvent):
            QGraphicsView.mousePressEvent(self._ui.viewer, event)
            self._ui.viewer.viewport().setCursor(Qt.CrossCursor)

        def _mouse_release_event(event: QMouseEvent):
            QGraphicsView.mouseReleaseEvent(self._ui.viewer, event)
            self._ui.viewer.viewport().setCursor(Qt.CrossCursor)

        def _mouse_leave_event(event: QEvent):
            QGraphicsView.leaveEvent(self._ui.viewer, event)
            self._overlay.hide()

        self._ui.viewer.mousePressEvent = _mouse_press_event
        self._ui.viewer.mouseReleaseEvent = _mouse_release_event
        self._ui.viewer.leaveEvent = _mouse_leave_event

    def _update_overlay(self, event: QMouseEvent):
        scene_pos = self._ui.viewer.mapToScene(event.pos())
        x, y = int(scene_pos.x()), int(scene_pos.y())
        if self._image_item is not None:
            self._overlay.setText(f"X:{x} Y:{y}")
            self._overlay.adjustSize()
            self._overlay.move(
                self._ui.viewer.viewport().width() - self._overlay.width(), 0
            )
            self._overlay.show()
        else:
            self._overlay.hide()
        QGraphicsView.mouseMoveEvent(self._ui.viewer, event)

    def load_image(self, file_path: str):
        if file_path == "":
            self._remove_image()
            return
        file_name, file_suffix = os.path.splitext(file_path)
        file_suffix = file_suffix[1:]
        if ((file_suffix == EnumTypeImportImage.tif.name) or (file_suffix == EnumTypeImportImage.tiff.name)):
            gdal.Translate("/vsimem/temp_image.vrt", file_path, format="VRT")
            dataset = gdal.Open("/vsimem/temp_image.vrt")

            width = dataset.RasterXSize
            height = dataset.RasterYSize
            if (file_suffix == EnumTypeImportImage.tif.name):
                self._current_image_type = EnumTypeImportImage.tif
            else:
                self._current_image_type = EnumTypeImportImage.tiff
            self._current_image_file_path = file_path

        elif(file_suffix == EnumTypeImportImage.svs.name):
            dataset = openslide.OpenSlide(file_path)
            width = dataset.dimensions[0]
            height = dataset.dimensions[1]
            self._current_image_type = EnumTypeImportImage.svs
            self._current_image_file_path = file_path
        else:
            dataset = None
            width = 0
            height = 0
            self._current_image_type = None
            self._current_image_file_path = None
        if dataset is not None:
            self._ui.viewer.scene().setSceneRect(
                QRectF(0, 0, width, height)
            )
            self._factor = self._min_factor = max(
                self._ui.viewer.viewport().width() / width,
                self._ui.viewer.viewport().height() / height,
                )
            self._ui.viewer.resetTransform()
            self._ui.viewer.scale(self._factor, self._factor)
            self._render()
        del dataset
        gc.collect()

    def load_mask(self, file_path: str):
        if file_path == "":
            self._remove_mask()
            return
        gdal.Translate("/vsimem/temp_mask.vrt", file_path, format="VRT")
        self._render()

    def _remove_image(self):
        if self._image_item is not None:
            gdal.Unlink("/vsimem/temp_image.vrt")
            self._ui.viewer.scene().removeItem(self._image_item)
            self._image_item = None

    def _remove_mask(self):
        if self._mask_item is not None:
            gdal.Unlink("/vsimem/temp_mask.vrt")
            self._ui.viewer.scene().removeItem(self._mask_item)
            self._mask_item = None

    def _render_task(self):
        try:
            if ((self._current_image_type == EnumTypeImportImage.tiff) or (self._current_image_type == EnumTypeImportImage.tif)):
                if gdal.VSIStatL("/vsimem/temp_image.vrt") is not None:
                    image_dataset = gdal.Open("/vsimem/temp_image.vrt")
                    offset = self._ui.viewer.mapToScene(
                        self._ui.viewer.viewport().rect().topLeft()
                    )
                    # 窗口中显示的图像范围
                    view_width = self._ui.viewer.viewport().width() / self._factor
                    view_height = self._ui.viewer.viewport().height() / self._factor

                    x_start = max(int(offset.x() - view_width), 0)
                    y_start = max(int(offset.y() - view_height), 0)
                    x_end = min(int(offset.x() + view_width * 2), image_dataset.RasterXSize)
                    y_end = min(int(offset.y() + view_height * 2), image_dataset.RasterYSize)
                    # 确定需要读取的图像大小
                    origin_width = x_end - x_start
                    origin_height = y_end - y_start

                    target_width = int(origin_width * self._factor)
                    target_height = int(origin_height * self._factor)

                    arr = image_dataset.ReadAsArray(
                        xoff=x_start,
                        yoff=y_start,
                        xsize=origin_width,
                        ysize=origin_height,
                        buf_xsize=target_width,
                        buf_ysize=target_height,
                        resample_alg=gdal.GRA_NearestNeighbour,
                    )
                    if np.issubdtype(arr.dtype, np.uint16):
                        arr = (arr / 65535.0 * 255).clip(0, 255)
                    elif np.issubdtype(arr.dtype, np.int16):
                        arr = ((arr + 32768) / 65535.0 * 255).clip(0, 255)
                    elif np.issubdtype(arr.dtype, np.uint32):
                        arr = (arr / 4294967295.0 * 255).clip(0, 255)
                    elif np.issubdtype(arr.dtype, np.int32):
                        arr = ((arr + 2147483648) / 4294967295.0 * 255).clip(0, 255)
                    image = QGraphicsPixmapItem(
                        QPixmap.fromImage(
                            QImage(
                                np.moveaxis(
                                    arr.astype(np.uint8),
                                    0,
                                    -1,
                                ).tobytes(),
                                target_width,
                                target_height,
                                target_width * 3,
                                QImage.Format_RGB888,
                                )
                        )
                    )
                    image.setScale(1 / self._factor)
                    image.setPos(x_start, y_start)
                    image.setZValue(1000)
                    del image_dataset

            elif self._current_image_type == EnumTypeImportImage.svs:

                slide = openslide.OpenSlide(self._current_image_file_path)
                width = slide.dimensions[0]
                # 4. 将0层级的坐标和尺寸转换到目标读取层级
                # 公式：目标层级坐标 = 0层级坐标 / 目标层级的下采样因子
                height = slide.dimensions[1]
                # 当前窗口显示的图片范围，对应需要截取的原始图像的大小，窗口内需要容纳多少像素
                view_width = self._ui.viewer.viewport().width() / self._factor
                view_height = self._ui.viewer.viewport().height() / self._factor
                print("view_width %d"%view_width)
                print("view_height %d"%view_height)
                offset = self._ui.viewer.mapToScene(
                    self._ui.viewer.viewport().rect().topLeft()
                )

                # 边界处理，确保加载的图像是超过视图范围的，这个坐标也对应原始图像中的加载图像所在的坐标点
                x_start = max(int(offset.x()), 0)
                y_start = max(int(offset.y()), 0)

                x_end = min(int(offset.x() + view_width), width)
                y_end = min(int(offset.y() + view_height), height)
                # 需要将原始图像多少范围内的图像都加载到界面上显示
                origin_width = x_end - x_start
                origin_height = y_end - y_start
                # 按照当前的放缩比列，加载的图像应该被抽样到多大用于导入到视图中进行显示
                target_width = int(origin_width * self._factor)
                target_height = int(origin_height * self._factor)
                # 找分辨率大于当前图像的层级
                best_level = self._get_best_level_for_zoom(slide, origin_width, origin_height, target_width, target_height)
                print("best level %d"%(best_level))
                level_downsample = slide.level_downsamples[best_level] # 获取层级1的下采样因子

                # 4. 将0层级的坐标和尺寸转换到目标读取层级
                # 公式：目标层级坐标 = 0层级坐标 / 目标层级的下采样因子
                # read_x = int(x_start / level_downsample)
                # read_y = int(y_start / level_downsample)
                read_width = int(origin_width / level_downsample)
                read_height = int(origin_height / level_downsample)
                # 5. 从选择的低层级读取区域（这是内存高效的关键步骤）
                # 读取的是一个PIL.Image对象
                region_pil = read_region_parallel_pil_correct(slide, x_start, y_start, read_width, read_height, best_level, tile_size=4096)
                # 确保是RGB（去除Alpha通道如果存在的话）
                region_rgb = region_pil.convert('RGB')
                # 6. 对读取到的图像进行上采样，达到最终要求的尺寸
                # 使用PIL的高质量插值方法LANCZOS
                upsampled_pil = region_rgb.resize(
                    (target_width, target_height),
                    resample=Image.Resampling.NEAREST
                )
                # 7. 转换为NumPy数组并返回

                arr = np.array(upsampled_pil)

                if np.issubdtype(arr.dtype, np.uint16):
                    arr = (arr / 65535.0 * 255).clip(0, 255)
                elif np.issubdtype(arr.dtype, np.int16):
                    arr = ((arr + 32768) / 65535.0 * 255).clip(0, 255)
                elif np.issubdtype(arr.dtype, np.uint32):
                    arr = (arr / 4294967295.0 * 255).clip(0, 255)
                elif np.issubdtype(arr.dtype, np.int32):
                    arr = ((arr + 2147483648) / 4294967295.0 * 255).clip(0, 255)
                image = QGraphicsPixmapItem(
                    QPixmap.fromImage(
                        QImage(
                            arr.astype(np.uint8).tobytes(),
                            target_width,
                            target_height,
                            target_width * 3,
                            QImage.Format_RGB888,
                            )
                    )
                )
                print("factor: " + str(self._factor))

                image.setScale(1 / self._factor)
                image.setPos(x_start, y_start)
                image.setZValue(1000)

            else:
                return


            mask = None
            if gdal.VSIStatL("/vsimem/temp_mask.vrt") is not None:
                mask_dataset = gdal.Open("/vsimem/temp_mask.vrt")
                mask = QGraphicsPixmapItem(
                    QPixmap.fromImage(
                        QImage(
                            (
                                255
                                - mask_dataset.ReadAsArray(
                                    xoff=x_start,
                                    yoff=y_start,
                                    xsize=origin_width,
                                    ysize=origin_height,
                                    buf_xsize=target_width,
                                    buf_ysize=target_height,
                                    resample_alg=gdal.GRA_NearestNeighbour,
                                )
                            ).data,
                            target_width,
                            target_height,
                            target_width,
                            QImage.Format_Alpha8,
                        )
                    )
                )
                mask.setScale(1 / self._factor)
                mask.setPos(x_start, y_start)
                mask.setZValue(2000)
                del mask_dataset
            self._item_added.emit(image, mask)
        except Exception as e:
            print(f"Task Failed: {e}")
        finally:
            gc.collect()
            self._rendering_cnt -= 1

    def _render(self):
        # 界面更新这块是实时的，所以有可能当前线程还没处理完，又发了信号过来，ThreadPoolExecutor是会将信号任务加入到队列中（因为只启动一个线程）
        # 因此当前线程未处理完再有信号发送过来，只存储一个信号，表示最后要处理窗口的更新，中间的信号不再处理
        if self._rendering_cnt < 2:
            self._rendering_cnt += 1
            self._render_executor.submit(self._render_task)

    def _wheel_event(self, event: QWheelEvent):
        factor = 1.2 if event.angleDelta().y() > 0 else 0.8
        if factor < 1 and self._factor == self._min_factor:
            return
        if self._factor * factor < self._min_factor:
            factor = self._min_factor / self._factor
            self._factor = self._min_factor
        else:
            self._factor *= factor
        self._ui.viewer.scale(factor, factor)
        self._render()
        event.accept()

    def _add_to_scene(self, image: QGraphicsPixmapItem, mask: QGraphicsPixmapItem):
        self._ui.viewer.scene().addItem(image)
        if mask is not None:
            self._ui.viewer.scene().addItem(mask)
        self._ui.viewer.scene().update()
        self._ui.viewer.viewport().update()
        if self._image_item:
            self._ui.viewer.scene().removeItem(self._image_item)
        self._image_item = image
        if mask is not None:
            if self._mask_item:
                self._ui.viewer.scene().removeItem(self._mask_item)
            self._mask_item = mask

    def _get_best_level_for_zoom(self, slide, origin_width, origin_height, target_width, target_height):
        """
        获取最适合指定缩放比例的层级
        """
        best_level = 0
        for level in range(slide.level_count - 1, -1, -1):
            slide_level_width = origin_width / slide.level_downsamples[level]
            slide_level_height = origin_height / slide.level_downsamples[level]

            if (target_width <= slide_level_width) and (target_height <= slide_level_height):
                best_level = level
                break
        return best_level

def ReadRegionIdxTile(slide, x, y, level, x_idx, y_idx, result_image, tile_size=1024):

    read_step = int(slide.level_downsamples[level] * tile_size)
    tile = slide.read_region((x + x_idx * read_step, y + y_idx * read_step), level, (tile_size, tile_size))
    paste_x = tile_size * x_idx
    paste_y = tile_size * y_idx
    result_image.paste(tile, (paste_x, paste_y))
    # result_image.paste(tile, (paste_x, paste_y))
    return paste_x, paste_y, tile


def read_region_parallel_pil_correct(slide, x, y, width, height, level, tile_size=1024):
    """
    正确的并行分块读取区域，返回PIL Image对象

    参数:
    slide: OpenSlide对象
    x, y: 读取区域的左上角坐标（指定层级的坐标）
    width, height: 读取区域的尺寸（指定层级）
    level: 金字塔层级
    tile_size: 分块大小

    返回:
    PIL Image对象
    """
    # 计算分块数量
    x_tiles = (width + tile_size - 1) // tile_size
    y_tiles = (height + tile_size - 1) // tile_size

    # 创建空白RGBA图像
    result_image = Image.new('RGBA', (width, height))

    time1 = time.time()
    # 并行读取
    with ThreadPoolExecutor(max_workers=min(8, x_tiles * y_tiles)) as executor:
        futures = []
        for tile_y in range(y_tiles):
            for tile_x in range(x_tiles):
                futures.append(executor.submit(ReadRegionIdxTile, slide, x, y, level, tile_x, tile_y, result_image, tile_size))
    time2 = time.time()
    print("read image time: %.4fs"%(time2 - time1))
        # # # 按顺序处理结果，确保正确拼接
        # for future in as_completed(futures):
        #     paste_x, paste_y, tile = future.result()
        #     result_image.paste(tile, (paste_x, paste_y))
        # time3 = time.time()
        # print("merge image time: %.4fs"%(time3 - time2))

    return result_image