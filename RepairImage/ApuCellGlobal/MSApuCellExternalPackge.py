# ==============此文件导入包的顺序禁止随意修改！！！！！！！=================
# 如需调用ApuCellGlobal中的函数，请先导入该文件，再导入其他包，否则造成的错误自己查
import os
import json
import pickle
import imageio
import copy
import logging
import sys
import traceback
import time
import math

# from osgeo import gdal

# opencv 修改允许最大图片，需要先解除再导入
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2,60).__str__()
import cv2

from PIL import Image, ImageFile
# PIL 解除图片最大限制
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

from enum import Enum, unique
from scipy import ndimage
from skimage import io
from skimage import measure

from ctypes import *

import numpy as np
import pandas as pd
import scipy.io as sio
import tifffile as tf
import matplotlib.pyplot as plt

import gc
import psutil
from ApuCellGlobal.GeneralSystem.MSSystemDLL import FOLDER_DLL
if os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), FOLDER_DLL)):
    OPENSLIDE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), FOLDER_DLL)
elif os.path.exists(os.path.join(os.path.dirname(os.getcwd()), FOLDER_DLL)):
    OPENSLIDE_PATH = os.path.join(os.path.dirname(os.getcwd()), FOLDER_DLL)
else:
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
import imagesize
import pyvips
# 这个不能删,否则打包的exe会提示错误
# import imagecodecs._imcd
# import imagecodecs._shared_cython
# 这个不能删,否则打包的exe会提示错误
import platform