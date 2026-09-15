import cv2
import numpy as np
from shapely.constructive import boundary

from ApuCellGlobal.MSApuCellGlobalImport import *
from ApuCellGlobal.GeneralFunction.MSFunctionImageOperator import CFunctionImageColorOperator, CFunctionImageOperator
from ApuCellGlobal.GeneralFunction.MSFunctionToolC import CFunctionToolImageC
from ApuCellGlobal.GeneralFunction.MSFunctionIO import CFunctionFileIO
from ApuCellGlobal.GeneralSystem.MSLogging import CLogging

from ApuCellGlobal.Memory.MSFunctionMemory import CFunctionMemory

from ApuCellGlobal.Route.MSFunctionRouteMove import CFunctionRouteMove
from ApuCellGlobal.Route.MSDataJsonLabel import CDataImageLabelFile, Polygon, PolygonAttributes, Label, LabelAttributes

import os
import json

class CFunctionDataJsonImageLabel:

    @staticmethod
    def ReadJsonInfo(image_path: str = "", annotation_path: str = "", mask_path: str = "")->CDataImageLabelFile:

        return CDataImageLabelFile(image_path, annotation_path, mask_path)

    @staticmethod
    def WriteJsonInfo(folder:str, name:str, data:CDataImageLabelFile):
        if '.' not in name:
            name += name + '.json'
        with open(os.path.join(folder, name), 'w', encoding="utf-8") as f:
            json.dump(data.json, f, indent=4)

    @staticmethod
    def MakeLabel(name:str, color:str=None, polygons:list[Polygon]=[], **kwargs):
        if color is None:
            color = CFunctionImageColorOperator.GetRandomRGBHex()
        return Label(LabelAttributes(name=name, color=color, **kwargs), polygons=polygons)

    @staticmethod
    def MakeMask2JsonDataImageLabelFile(image_path: str = "", annotation_path: str = "", mask_path: str = ""):

        return CDataImageLabelFile(image_path, annotation_path, mask_path)

    @staticmethod
    def PointsList2Polygons(name:str, color:str=None, points_list:list[list[list[int, int]]]=[]):
        if color is None:
            color = CFunctionImageColorOperator.GetRandomRGBHex()
        polygons = []
        for points_idx, points in enumerate(points_list):
            new_points  = []
            for point in points:
                new_points.append([int(point[0]), int(point[1])])
            polygons.append(Polygon(PolygonAttributes(name=name + "_%d"%points_idx, color=color), new_points))
        return polygons


    @staticmethod
    def Mask2JsonFile(image_mask:np.array, path_mask:str, polygon_fit:bool=False, h_offset:int=0, w_offset:int=0)->CDataImageLabelFile:
        image_label_file = CDataImageLabelFile(path_mask, "", mask_path="")
        image_label_file.labels = []
        if len(image_mask.shape) == 2:
            rgb = [0, 0, 0]
            rgb[0] = 255
            r, g, b = rgb
            CLogging.LogToUser('Search channel ')
            CLogging.LogToUser('Label the mask ... ')
            valid_polygon = []
            if not polygon_fit:
                image_mask_label, label_num = CFunctionImageOperator.GetImageMask2Label(image_mask[:, :])
                CLogging.LogToUser('Finish label the mask . ')
                CFunctionMemory.GetCurMemoryInfo()
                np_seed_point = CFunctionToolImageC.GetLabelBoundarySeedPoint(image_mask_label, label_num)
                list_boundary = CFunctionRouteMove.GetLabelBoundary(image_mask_label, label_num, np_seed_point)
                del image_mask_label
                CFunctionMemory.CleanMemory()
                for boundary in list_boundary:
                    if len(boundary) < 3:
                        continue
                    np_boundary = np.array(boundary)
                    x = np_boundary[:, 0] - w_offset
                    y = np_boundary[:, 1] - h_offset
                    offset_boundary = list(zip(x, y))
                    valid_polygon.append(offset_boundary)
            else:
                contours, hierarchy = cv2.findContours(image_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for one_contour in contours:
                    boundary = []
                    for one_contour_point_idx in range(one_contour.shape[0]):
                        boundary.append(one_contour[one_contour_point_idx, 0, :].tolist())
                    np_boundary = np.array(boundary)
                    x = np_boundary[:, 0] - w_offset
                    y = np_boundary[:, 1] - h_offset
                    offset_boundary = list(zip(x, y))
                    valid_polygon.append(offset_boundary)
            polygons = CFunctionDataJsonImageLabel.PointsList2Polygons("polygon", CFunctionImageColorOperator.GetRGBHex(r, g, b), valid_polygon)
            label = CFunctionDataJsonImageLabel.MakeLabel("Pos", CFunctionImageColorOperator.GetRGBHex(r, g, b), polygons)
            image_label_file.labels.append(label)
        elif len(image_mask.shape) == 3:
            c, h, w = image_mask.shape
            for c_idx in range(c):
                rgb = [0, 0, 0]
                rgb[c_idx] = 255
                r, g, b = rgb
                CLogging.LogToUser('Search channel %d ' % c_idx)
                CLogging.LogToUser('Label the mask ... ')
                valid_polygon = []
                if not polygon_fit:
                    image_mask_label, label_num = CFunctionImageOperator.GetImageMask2Label(image_mask[c_idx, :, :])
                    CLogging.LogToUser('Finish label the mask . ')
                    CFunctionMemory.GetCurMemoryInfo()
                    np_seed_point = CFunctionToolImageC.GetLabelBoundarySeedPoint(image_mask_label, label_num)
                    list_boundary = CFunctionRouteMove.GetLabelBoundary(image_mask_label, label_num, np_seed_point)
                    del image_mask_label
                    CFunctionMemory.CleanMemory()
                    valid_polygon = []
                    for boundary in list_boundary:
                        if len(boundary) < 3:
                            continue
                        valid_polygon.append(boundary)
                else:
                    contours, hierarchy = cv2.findContours(image_mask[c_idx, :, :], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    for one_contour in contours:
                        boundary = []
                        for one_contour_point_idx in range(one_contour.shape[0]):
                            boundary.append(one_contour[one_contour_point_idx, 0, :].tolist())
                        valid_polygon.append(boundary)
                polygons = CFunctionDataJsonImageLabel.PointsList2Polygons("polygon", CFunctionImageColorOperator.GetRGBHex(r, g, b), valid_polygon)
                label = CFunctionDataJsonImageLabel.MakeLabel("Pos", CFunctionImageColorOperator.GetRGBHex(r, g, b), polygons)
                image_label_file.labels.append(label)
        else:
            return
        return image_label_file

    @staticmethod
    def GetJson2Mask(path_image, path_json:str, label_name:str, set_value:int=255, x_length_large:int=0,y_length_large:int=0):
        # json from my boundary get the boundary use this function
        w, h = CFunctionFileIO.GetImageFileSize(path_image)
        img_mask = np.zeros((h, w), dtype=np.uint8)
        org_json_data = CFunctionDataJsonImageLabel.ReadJsonInfo(path_image, path_json)
        for label_info in org_json_data.labels:
            if label_info.attributes.name == label_name:
                for polygons in label_info.polygons:
                    points = polygons.json["points"]
                    np_points = np.array([points])
                    x_max = np.max(np_points[:, :, 0])
                    x_min = np.min(np_points[:, :, 0])
                    x_length = x_max - x_min

                    y_max = np.max(np_points[:, :, 1])
                    y_min = np.min(np_points[:, :, 1])
                    y_length = y_max - y_min

                    if ((x_length > x_length_large) or (y_length > y_length_large)):
                        cv2.fillPoly(img_mask, np_points, set_value)
                    else:
                        continue
        # boundary_num = 0
        # for label_info in org_json_data.labels:
        #     if label_info.attributes.name == label_name:
        #         for polygons in label_info.polygons:
        #             points = polygons.json["points"]
        #             for point_idx in range(1, len(points)):
        #                 cv2.line(img_mask, points[point_idx - 1], points[point_idx], 255, 1)
        #             cv2.line(img_mask, points[-1], points[0], 255, 1)
        #         boundary_num += 1
        # CFunctionImageOperator.FillPolygons(img_mask, 254, 255)
        # if set_value != 255:
        #     CFunctionToolImageC.SetMatrixValue(img_mask, set_value)
        return img_mask

    @staticmethod
    def GetJson2MaskForSimple(path_image, path_json:str, label_name:str, set_value:int=255, x_length_large:int=0,y_length_large:int=0):
        # json from cv2 get the boundary use this function

        w, h = CFunctionFileIO.GetImageFileSize(path_image)
        img_mask = np.zeros((h, w), dtype=np.uint8)
        org_json_data = CFunctionDataJsonImageLabel.ReadJsonInfo(path_image, path_json)
        boundary_num = 0
        for label_info in org_json_data.labels:
            if label_info.attributes.name == label_name:
                for polygons in label_info.polygons:
                    points = polygons.json["points"]
                    np_points = np.array([points])
                    x_max = np.max(np_points[:, :, 0])
                    x_min = np.min(np_points[:, :, 0])
                    x_length = x_max - x_min

                    y_max = np.max(np_points[:, :, 1])
                    y_min = np.min(np_points[:, :, 1])
                    y_length = y_max - y_min

                    if ((x_length > x_length_large) or (y_length > y_length_large)):
                        cv2.fillPoly(img_mask, np_points, set_value)
                    else:
                        continue
                boundary_num += 1
        # if boundary_num != 0:
        #     if set_value != 255:
        #         CFunctionToolImageC.SetMatrixValue(img_mask, set_value)
        return img_mask

if __name__ == "__main__":

    from ApuCellGlobal.GeneralFunction.MSFunctionIO import CFunctionFileIO
    from ApuCellGlobal.GeneralFunction.MSFunctionImageOperator import CFunctionImageOperator
    path_mask = r""
    mask = CFunctionFileIO.ReadImage(path_mask)