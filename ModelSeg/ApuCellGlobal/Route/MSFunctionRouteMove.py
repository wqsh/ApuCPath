from ApuCellGlobal.GeneralSystem.MSLogging import CLogProgressPercent

from enum import Enum, unique

import numpy as np
import time

@unique
class EnumMoveHorizontalDirection(Enum):

    Left = 0
    Right = 1

@unique
class EnumMoveVerticalDirection(Enum):

    Up = 0
    Down = 1

@unique
class EnmMoveDirection(Enum):
    Left = 0
    Right = 1
    Down = 2
    Up = 3

class CFunctionRouteMove:


    @staticmethod
    def GetMaskBoundary(img_mask:np.array, start_point_x:int, start_point_y:int, mask_value:int=255):
        last_direction = EnmMoveDirection.Right.value
        x = start_point_x
        y = start_point_y
        # 搜索闭合区域最外部轮廓
        # 上一步往右，下一步先往上走，没有就右走，再没有往下走
        # 上一步往左，下一步先往下走，没有就左走，再没有往上走
        # 上一步往下，下一步先往右走，没有就下走，再没有往左走
        # 上一步往上，下一步先往左走，没有就上走，再没有往右走
        #
        y_min, x_min = 0, 0
        y_max, x_max = img_mask.shape
        record_move = [(x, y)]
        finished = False
        while True:
            # print("x %d y %d" %(x,y))
            if img_mask[y][x] == 0:
                break
            if last_direction == EnmMoveDirection.Right.value:
                if not finished:
                    if y > y_min:
                        if img_mask[y - 1][x] == mask_value:
                            y = y - 1
                            last_direction = EnmMoveDirection.Up.value
                            finished = True
                if not finished:
                    if x < x_max - 1:
                        if img_mask[y][x + 1] == mask_value:
                            x = x + 1
                            last_direction = EnmMoveDirection.Right.value
                            finished = True
                if not finished:
                    if y < y_max - 1:
                        if img_mask[y + 1][x] == mask_value:
                            y = y + 1
                            last_direction = EnmMoveDirection.Down.value
                            finished = True
                if not finished:
                    if x > x_min:
                        if img_mask[y][x - 1] == mask_value:
                            x = x - 1
                            last_direction = EnmMoveDirection.Left.value
                            finished = True
            elif last_direction == EnmMoveDirection.Left.value:
                if not finished:
                    if y < y_max - 1:
                        if img_mask[y + 1][x] == mask_value:
                            y = y + 1
                            last_direction = EnmMoveDirection.Down.value
                            finished = True
                if not finished:
                    if x > x_min:
                        if img_mask[y][x - 1] == mask_value:
                            x = x - 1
                            last_direction = EnmMoveDirection.Left.value
                            finished = True
                if not finished:
                    if y > y_min:
                        if img_mask[y - 1][x] == mask_value:
                            y = y - 1
                            last_direction = EnmMoveDirection.Up.value
                            finished = True
                if not finished:
                    if x < x_max - 1:
                        if img_mask[y][x + 1] == mask_value:
                            x = x + 1
                            last_direction = EnmMoveDirection.Right.value
                            finished = True
            elif last_direction == EnmMoveDirection.Down.value:
                if not finished:
                    if x < x_max - 1:
                        if img_mask[y][x + 1] == mask_value:
                            x = x + 1
                            last_direction = EnmMoveDirection.Right.value
                            finished = True
                if not finished:
                    if y < y_max - 1:
                        if img_mask[y + 1][x] == mask_value:
                            y = y + 1
                            last_direction = EnmMoveDirection.Down.value
                            finished = True
                if not finished:
                    if x > x_min:
                        if img_mask[y][x - 1] == mask_value:
                            x = x - 1
                            last_direction = EnmMoveDirection.Left.value
                            finished = True
                if not finished:
                    if y > y_min:
                        if img_mask[y - 1][x] == mask_value:
                            y = y - 1
                            last_direction = EnmMoveDirection.Up.value
                            finished = True
            elif last_direction == EnmMoveDirection.Up.value:
                if not finished:
                    if x > x_min:
                        if img_mask[y][x - 1] == mask_value:
                            x = x - 1
                            last_direction = EnmMoveDirection.Left.value
                            finished = True
                if not finished:
                    if y > y_min:
                        if img_mask[y - 1][x] == mask_value:
                            y = y - 1
                            last_direction = EnmMoveDirection.Up.value
                            finished = True
                if not finished:
                    if x < x_max - 1:
                        if img_mask[y][x + 1] == mask_value:
                            x = x + 1
                            last_direction = EnmMoveDirection.Right.value
                            finished = True
                if not finished:
                    if y < y_max - 1:
                        if img_mask[y + 1][x] == mask_value:
                            y = y + 1
                            last_direction = EnmMoveDirection.Down.value
                            finished = True
            else:
                break
            if not finished:
                break
            record_move.append((x, y))
            finished = False
            if x == start_point_x and y == start_point_y:
                break
        return record_move

    @staticmethod
    def GetLabelBoundary(img_label:np.array, label_num:int, np_seed_point:np.array):
        list_boundary = []
        if label_num > 0:
            progress_percent = CLogProgressPercent(1, label_num, 0.05, "Get Boundary ... ")
            for label_idx in range(1, label_num + 1):
                seed_point_x = np_seed_point[0][label_idx * 2 + 1]
                seed_point_y = np_seed_point[0][label_idx * 2]
                boundary_move = CFunctionRouteMove.GetMaskBoundary(img_label, seed_point_x, seed_point_y, label_idx)
                list_boundary.append(boundary_move)
                progress_percent.ShowProgressPercent(label_idx)
            del progress_percent
        return list_boundary

    @staticmethod
    def GetBoundaryPointForMMI(list_boundary:list):
        new_list_boundary = []
        boundary_point_progress = CLogProgressPercent(0, len(list_boundary) - 1, 0.05, "Search boundary ... ")
        for boundary_idx, one_boundary in enumerate(list_boundary):
            boundary_point_progress.ShowProgressPercent(boundary_idx)
            new_one_boundary = CFunctionRouteMove.GetOneBoundaryForMMI(one_boundary)
            new_list_boundary.append(new_one_boundary)
        del boundary_point_progress
        return new_list_boundary

    @staticmethod
    def GetOneBoundaryForMMI(one_boundary:list):
        new_one_boundary = []
        if len(one_boundary) > 0:
            last_point_x, last_point_y = int(one_boundary[0][0]), int(one_boundary[0][1])
            cur_direction = 0
            last_direction = -1
            for point_idx in range(1, len(one_boundary)):
                point = one_boundary[point_idx]
                point = [int(point[0]), int(point[1])]
                if point[0] - last_point_x > 0:
                    cur_direction = EnmMoveDirection.Right.value
                if last_point_x - point[0] > 0:
                    cur_direction = EnmMoveDirection.Left.value
                if point[1] - last_point_y > 0:
                    cur_direction = EnmMoveDirection.Down.value
                if last_point_y - point[1] > 0:
                    cur_direction = EnmMoveDirection.Up.value
                if cur_direction != last_direction:
                    last_direction = cur_direction
                    new_one_boundary.append([last_point_x, last_point_y])
                last_point_x = point[0]
                last_point_y = point[1]
        new_one_boundary.append(one_boundary[0])
        return new_one_boundary

    @staticmethod
    def GetBoundaryTransformationForMMI(list_mmi_boundary:list, horizontal_ratio:float, vertical_ratio:float, mask_horizontal_start:int, mask_vertical_start:int, horizontal_start_pos:float, vertical_start_pos:float, multi_max:float=1.0):
        new_boundary = []
        for one_boundary in list_mmi_boundary:
            new_boundary.append(CFunctionRouteMove.GetOneBoundaryTransformationForMMI(one_boundary, horizontal_ratio, vertical_ratio, mask_horizontal_start, mask_vertical_start, horizontal_start_pos, vertical_start_pos, multi_max))
        return new_boundary
    @staticmethod
    def GetOneBoundaryTransformationForMMI(one_boundary: list, horizontal_ratio: float, vertical_ratio: float, mask_horizontal_start:int=0, mask_vertical_start:int=0, horizontal_start_pos:float=0.0, vertical_start_pos:float=0.0, multi_max:float=1.0):
        new_one_boundary = []
        for point_idx, point in enumerate(one_boundary):
            new_point_x = (point[0] - int(mask_horizontal_start)) * horizontal_ratio / multi_max + horizontal_start_pos
            new_point_y = (point[1] - int(mask_vertical_start)) * vertical_ratio / multi_max + vertical_start_pos
            new_one_boundary.append((new_point_x, new_point_y))
        return new_one_boundary

    @staticmethod
    def GetBoundaryTransformationForMMINew(list_mmi_boundary:list, pixel_width: float, pixel_height: float, wight_start_pixel:int, height_start_pixel:int, wight_start_pos:float, height_start_pos:float):
        new_boundary = []
        for one_boundary in list_mmi_boundary:
            new_boundary.append(CFunctionRouteMove.GetOneBoundaryTransformationForMMINew(one_boundary, pixel_width, pixel_height, wight_start_pixel, height_start_pixel, wight_start_pos, height_start_pos))
        return new_boundary

    @staticmethod
    def GetOneBoundaryTransformationForMMINew(one_boundary: list, pixel_width: float, pixel_height: float, wight_start_pixel:int, height_start_pixel:int, wight_start_pos:float, height_start_pos:float):

        new_one_boundary = []
        for point_idx, point in enumerate(one_boundary):
            new_point_x = (point[0] - wight_start_pixel) * pixel_width + wight_start_pos
            new_point_y = (point[1] - height_start_pixel) * pixel_height + height_start_pos
            new_one_boundary.append((new_point_x, new_point_y))
        return new_one_boundary

    @staticmethod
    def GetBoundaryRectangle(one_boundary:list):

        list_horizontal_point = sorted(one_boundary, key=lambda x:x[0], reverse=False)
        list_vertical_point = sorted(one_boundary, key=lambda x:x[1], reverse=False)

        return ((list_horizontal_point[0][0], list_horizontal_point[-1][0], list_vertical_point[0][1], list_vertical_point[-1][1]))
