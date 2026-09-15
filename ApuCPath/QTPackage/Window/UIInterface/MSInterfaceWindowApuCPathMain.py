from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtCore import Qt, QRectF, QPoint, QPointF, pyqtSignal, QEvent
from PyQt5.QtGui import QImage, QPixmap, QPainterPath, QMouseEvent, QMoveEvent, QCursor

from QTPackage.Window.UICode.MSWindowApuCPathMain import Ui_WindowApuCPathMain
from QTPackage.Window.UIInterface.MSInterfaceWindowApuCPathSelectForDataset import CInterfaceWindowApuCPathSelectForDataset
from QTPackage.Window.MSSystemWindowGlobal import CSystemWindowSelectForDataset, EnumSelectForDatasetFileType, EnumModelType, EnumWindowState, EnumUIDatasetType
from QTPackage.Window.MSSystemWindowGlobal import EnumStackedWidget, LIST_STACKEDWIDGET_SUFFIX
from QTPackage.Window.MSSystemWindowGlobal import LIST_IMAGE_WINDOW_STATE, LIST_CHECK_IMAGE
from QTPackage.Window.MSSystemWindowGlobal import EXE_MODELSEG, EXE_APUCPATHCREATEDATASET, FOLDER_BIN, FOLDER_CONFIG

from QTPackage.Control.Interface.MSInterfaceControlGraphicsViewShowImage import CInterfaceInstanceGraphicsViewShowImage

from QTPackage.MSSystemGlobal import EnumCursorState, CInterfaceCursor

from ConfigExe.MSFunctionConfigWSISegModelExe import CConfigExeWSISegModel, CFunctionConfigWSISegModel, EnumEnumWSISegModelTypeIndex, EnumWSISegModelTypeModelInit, EnumWSISegModelTypeNegSample
from ConfigExe.MSFunctionConfigApuCPathCreateDataset import CFunctionConfigApuCPathCreateDataset, CConfigExeApuCPathCreateDataset
from ConfigExe.WSISegModel.MSSystemEnum import EnumFlow as EnumMaskRCNNEXEFlow
from ConfigExe.MSFunctionConfigApuCPathAnnotation import NAME_APUCPATHANNOTATION_EXE
from ConfigExe.MSFunctionConfigCheckGPU import NAME_CHECK_GPU, NAME_CHECK_GPU_LINUX

from ApuCellGlobal.GeneralSystem.MSLogging import CLogging, EnumInfoType
from ApuCellGlobal.GeneralSystem.MSSystemEnum import LIST_VISUAL_FILE_TYPE_SUFFIX, LIST_VISUAL_FILE_IMAGE_IDX, EnumVisualFileType
from ApuCellGlobal.WSIProcess.MSDataWSIDataset import EnumDatasetType

import os
import sys
import subprocess
import winreg
import ctypes

from enum import Enum, unique

@unique
class EnumImageViewZoomState(Enum):

    Close = 0
    ZoomIn = 1
    ZoomOut = 2

@unique
class EnumImageViewCrossState(Enum):

    Close = 0
    Open = 1

class CFunctionCMDEnv:
    @staticmethod
    def GetSystemEnv():
        """构建一个干净的系统环境字典（与未激活 conda 的 cmd 一致）"""
        # 1. 获取系统 PATH（合并 HKLM 和 HKCU）
        path_parts = []
        try:
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                 r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment")
            sys_path, _ = winreg.QueryValueEx(key, "Path")
            if sys_path:
                path_parts.append(sys_path)
        except WindowsError:
            pass
        try:
            key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Environment")
            user_path, _ = winreg.QueryValueEx(key, "Path")
            if user_path:
                path_parts.append(user_path)
        except WindowsError:
            pass
        clean_path = os.pathsep.join(path_parts)
        clean_path = os.path.expandvars(clean_path)

        # 2. 从当前环境复制一份，然后替换 PATH 并删除所有干扰变量
        env = os.environ.copy()
        # 删除所有 CONDA 开头的变量（包括 CONDA_PREFIX, CONDA_DEFAULT_ENV, CONDA_SHLVL, CONDA_BAT, CONDA_EXE 等）
        # 以及 PYTHONPATH, PYTHONHOME, PYCHARM_HOSTED, GDAL_DATA, PROJ_LIB
        keys_to_delete = [k for k in env.keys() if (
            k.startswith('CONDA') or
            k.startswith('PYTHON') or
            k in ('PYCHARM_HOSTED', 'GDAL_DATA', 'PROJ_LIB', 'OPENCV_IO_MAX_IMAGE_PIXELS')  # 可选
        )]
        for key in keys_to_delete:
            env.pop(key, None)
        # 替换 PATH 为干净的系统 PATH
        env["PATH"] = clean_path
        return env

    @staticmethod
    def RunSystemEnv(exe_path, args: list = []):
        """同步执行子进程，返回 (returncode, stdout, stderr)"""
        clean_env = CFunctionCMDEnv.GetSystemEnv()
        result = subprocess.run(
            [exe_path] + args,
            env=clean_env,
            cwd=os.path.dirname(exe_path),
            capture_output=True,
            text=True
        )
        return result.returncode, result.stdout, result.stderr

class SubprocessThread(QThread):
    # 信号：标准输出行、标准错误行、进程结束（退出码）、启动失败异常
    output = pyqtSignal(str)
    error = pyqtSignal(str)
    finished = pyqtSignal(int)
    failed = pyqtSignal(str)

    def __init__(self, exe_path, args=None, env=None, cwd=None):
        super().__init__()
        self.exe_path = exe_path
        self.args = args or []
        self.env = env          # 可以传入干净环境字典
        self.cwd = cwd or os.path.dirname(exe_path)

    def run(self):
        try:
            # 启动进程
            self._process = subprocess.Popen(
                [self.exe_path] + self.args,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=self.env,
                cwd=self.cwd,
                bufsize=1          # 行缓冲
            )
            # 创建两个线程分别读取 stdout 和 stderr（避免死锁）
            import threading
            def read_pipe(pipe, signal):
                for line in iter(pipe.readline, ''):
                    if line:
                        signal.emit(line.rstrip('\n'))
            t_out = threading.Thread(target=read_pipe, args=(self._process.stdout, self.output), daemon=True)
            t_err = threading.Thread(target=read_pipe, args=(self._process.stderr, self.error), daemon=True)
            t_out.start()
            t_err.start()
            # 等待进程结束
            exit_code = self._process.wait()
            self.finished.emit(exit_code)
        except Exception as e:
            self.failed.emit(str(e))

class CInterfaceWindowApuCPathMain(QMainWindow, Ui_WindowApuCPathMain):

    def __init__(self):
        # 面板切换状态
        self.__stacked_widget_state:EnumStackedWidget = EnumStackedWidget.CreateDataset

        # 左下角状态
        self.__run_state:EnumWindowState = EnumWindowState.Free

        # 参数文件
        self.config_create_dataset:CConfigExeApuCPathCreateDataset = CConfigExeApuCPathCreateDataset()
        self.config_model:CConfigExeWSISegModel = CConfigExeWSISegModel()

        # 参数文件保存路径
        self.config_create_dataset_path:str = None
        self.config_wsiseg_path:str = None
        self.cwd = os.getcwd()

        # ImageView界面的状态记录
        self.__ImageView_zoom_state:EnumImageViewZoomState = EnumImageViewZoomState.Close
        self.__ImageView_cross_state:EnumImageViewCrossState = EnumImageViewCrossState.Close

        super(CInterfaceWindowApuCPathMain, self).__init__()
        self.setupUi(self)

        self.interface_ImageView_GraphicsView_show_image = CInterfaceInstanceGraphicsViewShowImage(self.GraphicsView_ImageView_ShowImage)
        self.interface_Cursor = CInterfaceCursor()

        self.__captainInitConnectPyqtSlot()
        self.__captainInitControl()

    def __soldierInitCreateDatasetConfig(self):

        save_suffix = LIST_STACKEDWIDGET_SUFFIX[EnumStackedWidget.CreateDataset.value]
        default_name = self.__stacked_widget_state.name + save_suffix

        if not os.path.exists(os.path.join(os.getcwd(), FOLDER_CONFIG)):
            os.makedirs(os.path.join(os.getcwd(), FOLDER_CONFIG))

        path_cratedataset_default_config: str = os.path.join(os.getcwd(), FOLDER_CONFIG, default_name)
        if os.path.exists(path_cratedataset_default_config):
            CFunctionConfigApuCPathCreateDataset.File2Config(path_cratedataset_default_config, self.config_create_dataset)
        else:
            CFunctionConfigApuCPathCreateDataset.Config2File(self.config_create_dataset, os.path.dirname(path_cratedataset_default_config), os.path.basename(path_cratedataset_default_config))

    def __soldierInitTrainModelConfig(self):

        if not os.path.exists(os.path.join(os.getcwd(), FOLDER_CONFIG)):
            os.makedirs(os.path.join(os.getcwd(), FOLDER_CONFIG))

        save_suffix = LIST_STACKEDWIDGET_SUFFIX[EnumStackedWidget.TrainModel.value]
        default_name = self.__stacked_widget_state.name + save_suffix
        path_wsiseg_default_config:str = os.path.join(os.getcwd(), FOLDER_CONFIG, default_name)
        if os.path.exists(path_wsiseg_default_config):
            CFunctionConfigWSISegModel.File2Config(path_wsiseg_default_config, self.config_model)
        else:
            CFunctionConfigWSISegModel.Config2File(self.config_model, os.path.dirname(path_wsiseg_default_config), os.path.basename(path_wsiseg_default_config))

    def __soldierInitPredictConfig(self):

        save_suffix = LIST_STACKEDWIDGET_SUFFIX[EnumStackedWidget.Predict.value]
        default_name = self.__stacked_widget_state.name + save_suffix

        if not os.path.exists(os.path.join(os.getcwd(), FOLDER_CONFIG)):
            os.makedirs(os.path.join(os.getcwd(), FOLDER_CONFIG))

        path_maskrcnn_default_config:str = os.path.join(os.getcwd(), FOLDER_CONFIG, default_name)
        if os.path.exists(path_maskrcnn_default_config):
            CFunctionConfigWSISegModel.File2Config(path_maskrcnn_default_config, self.config_model)
        else:
            CFunctionConfigWSISegModel.Config2File(self.config_model, os.path.dirname(path_maskrcnn_default_config), os.path.basename(path_maskrcnn_default_config))

    def __captainInitConnectPyqtSlot(self):
        self.ComboBox_CreateDataset_DatasetType.currentTextChanged.connect(self.on_ComboBox_CreateDataset_DatasetType_currentTextChanged)
        self.ComboBox_TrainModel_InitType.currentIndexChanged.connect(self.on_ComboBox_TrainModel_InitType_currentIndexChanged)
        # 绑定图片画板中鼠标移动信号到更新实时坐标中
        self.GraphicsView_ImageView_ShowImage.moveMouseGraphicsView2WindowInterface.connect(self.__connnect_ImageView_ShowLocationControl)
        self.GraphicsView_ImageView_ShowImage.pressMouseGraphicsView2WindowInterface.connect(self.__connnect_ImageView_UpdateGraphicsView)

        self.CheckBox_TrainModel_ValidDatasetPath.toggled.connect(self.on_CheckBox_TrainModel_ValidDatasetPath_toggle)
        self.CheckBox_TrainModel_AddNeg.toggled.connect(self.on_CheckBox_TrainModel_AddNeg_toggle)
        self.ComboBox_TrainModel_NegSample_Scheme.currentIndexChanged.connect(self.on_ComboBox_TrainModel_NegSample_Scheme_currentIndexChanged)

    # 界面所有控件的初始化

    def __captainInitControl(self):

        self.__soldierInitControlRun()
        self.__soldierInitControlFlow()
        self.__soldierInitControlImageLabel()

        self.__soldierInitCreateDatasetConfig()
        self.__soldierInitControlCreateDataset()

        self.__soldierInitTrainModelConfig()
        self.__soldierInitControlTrainModel()

        self.__soldierInitPredictConfig()
        self.__soldierInitControlPredict()

        self.__soldierInitControlImageView()

    def __soldierInitControlRun(self):

        self.PushButton_Check.setCheckable(False)
        self.PushButton_Save.setCheckable(False)
        self.PushButton_SaveAs.setCheckable(False)
        self.PushButton_Run.setCheckable(False)
        self.LineEdit_PathConfig.setReadOnly(True)

        self.PushButton_Save.setEnabled(False)
        self.PushButton_SaveAs.setEnabled(False)
        self.PushButton_Run.setEnabled(False)
        self.ProgressBar_Run.setValue(0)
        self.LineEdit_PathConfig.setText('')

        if self.__stacked_widget_state == EnumStackedWidget.ImageView or self.__stacked_widget_state == EnumStackedWidget.ImageLabel:
            self.PushButton_Run.setVisible(False)
            self.PushButton_Save.setVisible(False)
            self.PushButton_SaveAs.setVisible(False)
            self.ProgressBar_Run.setVisible(False)
            self.PushButton_Check.setVisible(False)
            self.LineEdit_PathConfig.setVisible(False)
            self.Label_PathConfig.setVisible(False)
        else:
            self.PushButton_Run.setVisible(True)
            self.PushButton_Save.setVisible(True)
            self.PushButton_SaveAs.setVisible(True)
            self.ProgressBar_Run.setVisible(True)
            self.PushButton_Check.setVisible(True)
            self.LineEdit_PathConfig.setVisible(True)
            self.Label_PathConfig.setVisible(True)

    def __soldierInitControlFlow(self):

        # 面板切换状态
        self.SetStackedWidget(EnumStackedWidget.CreateDataset)

        self.PushButton_Flow_ImageLabel.setCheckable(True)
        self.PushButton_Flow_CrateDataset.setCheckable(True)
        self.PushButton_Flow_TrainModel.setCheckable(True)
        self.PushButton_Flow_Predict.setCheckable(True)
        self.PushButton_Flow_ImageView.setCheckable(True)

        self.PushButton_Flow_ImageView.setVisible(False)

        self.PushButton_Flow_ImageLabel.setText("Annotation")
        self.PushButton_Flow_ImageLabel.setCheckable(False)
        # 左下角状态
        self.SetFlowState(EnumWindowState.Free)

        self.Label_Flow_State.setStyleSheet(LIST_IMAGE_WINDOW_STATE[self.__run_state.value])
        self.LineEdit_Flow_Sate.setReadOnly(True)
        self.LineEdit_Flow_Sate.setText(self.__run_state.name)
        self.Label_Flow_GPU_State.setStyleSheet(LIST_CHECK_IMAGE[0])

        self.CheckBox_Flow_GPU_Use.setEnabled(False)
        self.CheckBox_Flow_GPU_Use.setChecked(False)

        self.on_PushButton_GPU_Check_clicked()

    def __soldierInitControlImageLabel(self):

        pass

    def __soldierInitControlCreateDataset(self):

        self.ComboBox_CreateDataset_DatasetType.addItems([item.name for item in EnumUIDatasetType] )
        self.ComboBox_CreateDataset_DatasetType.setCurrentIndex(EnumUIDatasetType.Predict.value)

        self.ListWidget_CreateDataset_Image.setSelectionMode(QAbstractItemView.SingleSelection)
        self.ListWidget_CreateDataset_Mask.setSelectionMode(QAbstractItemView.NoSelection)

        self.DoubleSpinBox_CreateDataset_MaskAreaRatio_Max.setMaximum(1)
        self.DoubleSpinBox_CreateDataset_MaskAreaRatio_Max.setMinimum(0)
        self.DoubleSpinBox_CreateDataset_MaskAreaRatio_Max.setSingleStep(0.0001)
        self.DoubleSpinBox_CreateDataset_MaskAreaRatio_Max.setDecimals(4)
        self.DoubleSpinBox_CreateDataset_MaskAreaRatio_Max.setValue(1)

        self.DoubleSpinBox_CreateDataset_MaskAreaRatio_Min.setMaximum(1)
        self.DoubleSpinBox_CreateDataset_MaskAreaRatio_Min.setMinimum(0)
        self.DoubleSpinBox_CreateDataset_MaskAreaRatio_Min.setSingleStep(0.0001)
        self.DoubleSpinBox_CreateDataset_MaskAreaRatio_Min.setDecimals(4)
        self.DoubleSpinBox_CreateDataset_MaskAreaRatio_Min.setValue(0.0001)

        self.CheckCreateDataset()

    def __soldierInitControlTrainModel(self):

        self.SpinBox_TrainModel_BatchSize.setMinimum(1)
        self.SpinBox_TrainModel_BatchSize.setMaximum(1024)
        self.SpinBox_TrainModel_BatchSize.setSingleStep(1)
        self.SpinBox_TrainModel_BatchSize.setValue(self.config_model.B4_BATCH_SIZE)

        self.SpinBox_TrainModel_Epoch.setMinimum(1)
        self.SpinBox_TrainModel_Epoch.setMaximum(1000)
        self.SpinBox_TrainModel_Epoch.setSingleStep(1)
        self.SpinBox_TrainModel_Epoch.setValue(self.config_model.B5_EPOCH_NUM)

        self.DoubleSpinBox_TrainModel_LearningRate.setDecimals(7)
        self.DoubleSpinBox_TrainModel_LearningRate.setMinimum(1e-7)
        self.DoubleSpinBox_TrainModel_LearningRate.setMaximum(10)
        self.DoubleSpinBox_TrainModel_LearningRate.setSingleStep(1e-7)
        self.DoubleSpinBox_TrainModel_LearningRate.setValue(self.config_model.B6_LEARNING_RATE)

        self.ComboBox_TrainModel_InitType.addItems([init_model_type.name for init_model_type in EnumWSISegModelTypeModelInit])
        self.ComboBox_TrainModel_InitType.setCurrentIndex(EnumWSISegModelTypeModelInit.Scratch.value)

        self.ComboBox_TrainModel_ModelType.addItems([model_type.name for model_type in EnumModelType])
        self.ComboBox_TrainModel_ModelType.setCurrentIndex(EnumModelType.UNIMask2Former.value)

        self.on_CheckBox_TrainModel_AddNeg_toggle(self.config_model.C4_MODEL_ADD_NEGATIVE)

        self.ComboBox_TrainModel_NegSample_Scheme.addItems([sample_type.name for sample_type in EnumWSISegModelTypeNegSample])
        self.ComboBox_TrainModel_NegSample_Scheme.setCurrentIndex(EnumWSISegModelTypeNegSample.MiniBatchKMeans.value)

        self.SpinBox_TrainModel_NegSample_Cluster.setMinimum(1)
        self.SpinBox_TrainModel_NegSample_Cluster.setMaximum(1000)
        self.SpinBox_TrainModel_NegSample_Cluster.setValue(self.config_model.C5_MODEL_NEGATIVE_SAMPLE_CLUSTER)

        self.LineEdit_TrainModel_PathUNIBackbone.setText(self.config_model.G4_PATH_UNI_CHECKPOINT)

        self.DoubleSpinBox_TrainModel_ValidDatasetPath.setMaximum(1.0)
        self.DoubleSpinBox_TrainModel_ValidDatasetPath.setMinimum(0.01)
        self.DoubleSpinBox_TrainModel_ValidDatasetPath.setSingleStep(0.01)
        self.DoubleSpinBox_TrainModel_ValidDatasetPath.setValue(self.config_model.B9_VALID_PERCENT)
        if self.CheckBox_TrainModel_ValidDatasetPath.isChecked():
            self.DoubleSpinBox_TrainModel_ValidDatasetPath.setEnabled(True)
        else:
            self.DoubleSpinBox_TrainModel_ValidDatasetPath.setEnabled(False)

        self.CheckTrainModel()

    def __soldierInitControlPredict(self):

        self.SpinBox_Predict_BatchSize.setMinimum(1)
        self.SpinBox_Predict_BatchSize.setMaximum(1024)
        self.SpinBox_Predict_BatchSize.setSingleStep(1)
        self.SpinBox_Predict_BatchSize.setValue(self.config_model.B4_BATCH_SIZE)

        self.ComboBox_Predict_ModelType.addItems([model_type.name for model_type in EnumModelType])
        self.ComboBox_Predict_ModelType.setCurrentIndex(EnumModelType.UNIMask2Former.value)

        self.SpinBox_Predict_MinArea.setMinimum(0)
        self.SpinBox_Predict_MinArea.setMaximum(100000000)
        self.SpinBox_Predict_MinArea.setSingleStep(1)
        self.SpinBox_Predict_MinArea.setValue(self.config_model.D3_MIN_AREA)

        self.DoubleSpinBox_Predict_FilterScore.setMinimum(0.0)
        self.DoubleSpinBox_Predict_FilterScore.setMaximum(1.0)
        self.DoubleSpinBox_Predict_FilterScore.setSingleStep(0.01)
        self.DoubleSpinBox_Predict_FilterScore.setValue(self.config_model.D1_1_FILTER_MASK_SCORE)

        self.DoubleSpinBox_Predict_MCAScore.setMinimum(0.0)
        self.DoubleSpinBox_Predict_MCAScore.setMaximum(1.0)
        self.DoubleSpinBox_Predict_MCAScore.setSingleStep(0.01)
        self.DoubleSpinBox_Predict_MCAScore.setValue(self.config_model.D4_MERGE_SCORE)

        self.CheckPredict()

    def __soldierInitControlImageView(self):

        self.interface_ImageView_GraphicsView_show_image.ClearImage()
        self.interface_ImageView_GraphicsView_show_image.GraphicsViewShowImage()
        self.GraphicsView_ImageView_ShowImage.setMouseTracking(True)

        self.PushButton_ImageView_ZoomIn.setEnabled(False)
        self.PushButton_ImageView_ZoomOut.setEnabled(False)
        self.PushButton_ImageView_Normal.setEnabled(False)
        self.PushButton_ImageView_Cross.setEnabled(False)

        self.LineEdit_ImageView_Location_X.setReadOnly(True)
        self.LineEdit_ImageView_Location_Y.setReadOnly(True)
        self.LineEdit_ImageView_Color_R.setReadOnly(True)
        self.LineEdit_ImageView_Color_G.setReadOnly(True)
        self.LineEdit_ImageView_Color_B.setReadOnly(True)

        self.TableWidget_ImageView_Info.setItem(1, 1, QTableWidgetItem(""))
        self.TableWidget_ImageView_Info.setItem(2, 1, QTableWidgetItem(""))
        self.TableWidget_ImageView_Info.setItem(3, 1, QTableWidgetItem(""))
        self.TableWidget_ImageView_Info.setItem(4, 1, QTableWidgetItem(""))

        # ImageView界面的状态记录
        self.__ImageView_zoom_state:EnumImageViewZoomState = EnumImageViewZoomState.Close
        self.__ImageView_cross_state:EnumImageViewCrossState = EnumImageViewCrossState.Close

        self.UpdataImageViewToolState()
        self.ImageView_ShowLocationControl(-1, -1)


    # 界面所有控件的初始化

    # 参数检查

    def CheckCreateDataset(self):
        ret:bool = True
        self.Label_CreateDataset_DatasetType.setStyleSheet(LIST_CHECK_IMAGE[1])

        if (self.DoubleSpinBox_CreateDataset_MaskAreaRatio_Min.value() > self.DoubleSpinBox_CreateDataset_MaskAreaRatio_Max.value()):
            self.Label_CreateDataset_MaskRatio.setStyleSheet(LIST_CHECK_IMAGE[0])
            ret = False
        else:
            self.Label_CreateDataset_MaskRatio.setStyleSheet(LIST_CHECK_IMAGE[1])

        if self.ListWidget_CreateDataset_Image.count() == 0:
            self.Label_CreateDataset_Input.setStyleSheet(LIST_CHECK_IMAGE[0])
            ret = False
        else:
            self.Label_CreateDataset_Input.setStyleSheet(LIST_CHECK_IMAGE[1])

        if len(self.LineEdit_CreateDataset_ExportPath.text()) == 0:
            self.Label_CreateDataset_ExportPath.setStyleSheet(LIST_CHECK_IMAGE[0])
            ret = False
        else:
            self.Label_CreateDataset_ExportPath.setStyleSheet(LIST_CHECK_IMAGE[1])

        return ret

    def CheckTrainModel(self):

        ret:bool = True

        if len(self.LineEdit_TrainModel_DatasetPath.text()) == 0:
            self.Label_TrainModel_DatasetPath.setStyleSheet(LIST_CHECK_IMAGE[0])
            ret = False
        else:
            if not os.path.exists(self.LineEdit_TrainModel_DatasetPath.text()):
                self.Label_TrainModel_DatasetPath.setStyleSheet([LIST_CHECK_IMAGE[0]])
                ret = False
            else:
                self.Label_TrainModel_DatasetPath.setStyleSheet(LIST_CHECK_IMAGE[1])

        if self.CheckBox_TrainModel_ValidDatasetPath.isChecked():
            self.Label_TrainModel_ValidDatasetPath.setStyleSheet(LIST_CHECK_IMAGE[1])
        else:
            if len(self.LineEdit_TrainModel_ValidDatasetPath.text()) == 0:
                self.Label_TrainModel_ValidDatasetPath.setStyleSheet(LIST_CHECK_IMAGE[0])
                ret = False
            else:
                if not os.path.exists(self.LineEdit_TrainModel_ValidDatasetPath.text()):
                    self.Label_TrainModel_ValidDatasetPath.setStyleSheet(LIST_CHECK_IMAGE[0])
                    ret = False
                else:
                    self.Label_TrainModel_ValidDatasetPath.setStyleSheet(LIST_CHECK_IMAGE[1])

        self.Label_TrainModel_BatchSize.setStyleSheet(LIST_CHECK_IMAGE[1])
        self.Label_TrainModel_Epoch.setStyleSheet(LIST_CHECK_IMAGE[1])
        self.Label_TrainModel_LearningRate.setStyleSheet(LIST_CHECK_IMAGE[1])
        self.Label_TrainModel_ModelType.setStyleSheet(LIST_CHECK_IMAGE[1])
        self.Label_TrainModel_InitType.setStyleSheet(LIST_CHECK_IMAGE[1])
        self.Label_TrainModel_NegSample.setStyleSheet(LIST_CHECK_IMAGE[1])
        if self.ComboBox_TrainModel_InitType.currentIndex() == EnumWSISegModelTypeModelInit.Scratch.value:
            self.Label_TrainModel_ModelPath.setStyleSheet(LIST_CHECK_IMAGE[1])
        else:
            if not os.path.exists(self.LineEdit_TrainModel_ModelPath.text()):
                self.Label_TrainModel_ModelPath.setStyleSheet(LIST_CHECK_IMAGE[0])
                ret = False
            else:
                self.Label_TrainModel_ModelPath.setStyleSheet(LIST_CHECK_IMAGE[1])

        if not os.path.exists(self.LineEdit_TrainModel_ExportPath.text()):
            self.Label_TrainModel_ExportPath.setStyleSheet(LIST_CHECK_IMAGE[0])
            ret = False
        else:
            self.Label_TrainModel_ExportPath.setStyleSheet(LIST_CHECK_IMAGE[1])

        if self.ComboBox_TrainModel_ModelType.currentIndex() == EnumModelType.UNIMask2Former.value:
            if not os.path.exists(self.LineEdit_TrainModel_PathUNIBackbone.text()):
                self.Label_TrainModel_PathUNIBackbone.setStyleSheet(LIST_CHECK_IMAGE[0])
                ret = False
            else:
                self.Label_TrainModel_PathUNIBackbone.setStyleSheet(LIST_CHECK_IMAGE[1])
        else:
            self.Label_TrainModel_PathUNIBackbone.setStyleSheet(LIST_CHECK_IMAGE[1])

        return ret

    def CheckPredict(self):

        ret:bool = True
        if len(self.LineEdit_Predict_DatasetPath.text()) == 0:
            self.Label_Predict_DatasetPath.setStyleSheet(LIST_CHECK_IMAGE[0])
            ret = False
        else:
            if not os.path.exists(self.LineEdit_Predict_DatasetPath.text()):
                self.Label_Predict_DatasetPath.setStyleSheet(LIST_CHECK_IMAGE[0])
                ret = False
            else:
                self.Label_Predict_DatasetPath.setStyleSheet(LIST_CHECK_IMAGE[1])

        self.Label_Predict_BatchSize.setStyleSheet(LIST_CHECK_IMAGE[1])
        self.Label_Predict_ModelType.setStyleSheet(LIST_CHECK_IMAGE[1])

        if len(self.LineEdit_Predict_ModelPath.text()) == 0:
            self.Label_Predict_ModelPath.setStyleSheet(LIST_CHECK_IMAGE[0])
            ret = False
        else:
            if not os.path.exists(self.LineEdit_Predict_ModelPath.text()):
                self.Label_Predict_ModelPath.setStyleSheet(LIST_CHECK_IMAGE[0])
                ret = False
            else:
                self.Label_Predict_ModelPath.setStyleSheet(LIST_CHECK_IMAGE[1])

        if len(self.LineEdit_Predict_ExportPath.text()) == 0:
            self.Label_Predict_ExportPath.setStyleSheet(LIST_CHECK_IMAGE[0])
            ret = False
        else:
            self.Label_Predict_ExportPath.setStyleSheet(LIST_CHECK_IMAGE[1])

        self.Label_Predict_FilterScore.setStyleSheet(LIST_CHECK_IMAGE[1])
        self.Label_Predict_MinArea.setStyleSheet(LIST_CHECK_IMAGE[1])
        self.Label_Predict_MCAScore.setStyleSheet(LIST_CHECK_IMAGE[1])

        return ret

    # 参数检查

    # 顶部按钮响应

    @pyqtSlot()
    def on_PushButton_Check_clicked(self):

        self.on_PushButton_Refresh_clicked()

        if self.__stacked_widget_state == EnumStackedWidget.CreateDataset:
            if self.CheckCreateDataset():
                self.PushButton_Save.setEnabled(True)
            else:
                QMessageBox.about(self, EnumInfoType.Warn.name,
                                  "Some parameters are incorrectly filled in, please check ! ")
        elif self.__stacked_widget_state == EnumStackedWidget.TrainModel:
            if self.CheckTrainModel():
                self.PushButton_Save.setEnabled(True)
            else:
                QMessageBox.about(self, EnumInfoType.Warn.name,
                                  "Some parameters are incorrectly filled in, please check ! ")
        elif self.__stacked_widget_state == EnumStackedWidget.Predict:
            if self.CheckPredict():
                self.PushButton_Save.setEnabled(True)
            else:
                QMessageBox.about(self, EnumInfoType.Warn.name,
                                  "Some parameters are incorrectly filled in, please check ! ")

    @pyqtSlot()
    def on_PushButton_Save_clicked(self):
        if self.__stacked_widget_state == EnumStackedWidget.CreateDataset:
            self.SaveCreateDatasetConfig()
            self.PushButton_SaveAs.setEnabled(True)
        elif self.__stacked_widget_state == EnumStackedWidget.TrainModel:
            self.SaveTrainModelConfig()
            self.PushButton_SaveAs.setEnabled(True)
        elif self.__stacked_widget_state == EnumStackedWidget.Predict:
            self.SavePredictConfig()
            self.PushButton_SaveAs.setEnabled(True)

    @pyqtSlot()
    def on_PushButton_SaveAs_clicked(self):

        save_suffix = LIST_STACKEDWIDGET_SUFFIX[self.__stacked_widget_state.value]
        default_name = self.__stacked_widget_state.name + save_suffix
        file_path, file_type = QFileDialog.getSaveFileName(self, 'File save', os.path.join(self.cwd, default_name), 'Text Files(*%s)'%save_suffix)
        if file_path == '':
            return
        else:
            folder = os.path.dirname(file_path)
            file_name = os.path.basename(file_path)
            if '.' not in file_name:
                file_name += save_suffix
            if not os.path.exists(os.path.join(folder)):
                os.makedirs(os.path.join(folder))
            if self.__stacked_widget_state == EnumStackedWidget.CreateDataset:
                self.config_create_dataset_path = file_path
                CFunctionConfigApuCPathCreateDataset.Config2File(self.config_create_dataset, folder, file_name)
                self.LineEdit_PathConfig.setText(self.config_create_dataset_path)
            elif self.__stacked_widget_state == EnumStackedWidget.TrainModel:
                self.config_wsiseg_path = file_path
                CFunctionConfigWSISegModel.Config2File(self.config_model, folder, file_name)
                self.LineEdit_PathConfig.setText(self.config_wsiseg_path)
            elif self.__stacked_widget_state == EnumStackedWidget.Predict:
                self.config_wsiseg_path = file_path
                CFunctionConfigWSISegModel.Config2File(self.config_model, folder, file_name)
                self.LineEdit_PathConfig.setText(self.config_wsiseg_path)
            else:
                return
            self.PushButton_Run.setEnabled(True)

    @pyqtSlot()
    def on_PushButton_Run_clicked(self):

        if self.__stacked_widget_state == EnumStackedWidget.CreateDataset:
            if os.path.exists(self.config_create_dataset_path):
                self.SetFlowState(EnumWindowState.Running)
                self.PushButton_Run.setDisabled(True)
                self.PushButton_Save.setDisabled(True)
                self.PushButton_SaveAs.setDisabled(True)
                self.ProgressBar_Run.setValue(20)
                cmd_str: str = os.path.join(os.getcwd(), FOLDER_BIN, EXE_APUCPATHCREATEDATASET) + " " + self.config_create_dataset_path
                CLogging.LogToUser(cmd_str)
                ret = self.__captainRunCoreExe(cmd_str)
                if not ret:
                    QMessageBox.about(self, EnumInfoType.Error.name, cmd_str)
                    self.SetFlowState(EnumWindowState.Error)
                else:
                    self.SetFlowState(EnumWindowState.Free)
                    self.ProgressBar_Run.setValue(100)
        elif self.__stacked_widget_state == EnumStackedWidget.TrainModel:
            if os.path.exists(self.config_wsiseg_path):
                self.SetFlowState(EnumWindowState.Running)
                self.PushButton_Run.setDisabled(True)
                self.PushButton_Save.setDisabled(True)
                self.PushButton_SaveAs.setDisabled(True)
                self.ProgressBar_Run.setValue(20)
                cmd_str: str = os.path.join(os.getcwd(), FOLDER_BIN, EXE_MODELSEG) + " " + self.config_wsiseg_path
                CLogging.LogToUser(cmd_str)
                ret = self.__captainRunCoreExe(cmd_str)
                if not ret:
                    QMessageBox.about(self, EnumInfoType.Error.name, cmd_str)
                    self.SetFlowState(EnumWindowState.Error)
                else:
                    self.SetFlowState(EnumWindowState.Free)
                    self.ProgressBar_Run.setValue(100)
        elif self.__stacked_widget_state == EnumStackedWidget.Predict:
            if os.path.exists(self.config_wsiseg_path):
                self.SetFlowState(EnumWindowState.Running)
                self.PushButton_Run.setDisabled(True)
                self.PushButton_Save.setDisabled(True)
                self.PushButton_SaveAs.setDisabled(True)
                cmd_str: str = os.path.join(os.getcwd(), FOLDER_BIN, EXE_MODELSEG) + " " + self.config_wsiseg_path
                CLogging.LogToUser(cmd_str)
                ret = self.__captainRunCoreExe(cmd_str)
                if not ret:
                    QMessageBox.about(self, EnumInfoType.Error.name, cmd_str)
                    self.SetFlowState(EnumWindowState.Error)
                else:
                    self.SetFlowState(EnumWindowState.Free)
                    self.ProgressBar_Run.setValue(100)

    # 顶部按钮响应

    # 左下角按钮响应
    @pyqtSlot()
    def on_PushButton_Refresh_clicked(self):
        self.SetFlowState(EnumWindowState.Free)
        self.ProgressBar_Run.setValue(0)
        self.LineEdit_PathConfig.setText('')
        self.PushButton_Run.setDisabled(True)
        self.PushButton_SaveAs.setDisabled(True)
        self.PushButton_Save.setDisabled(True)

    @pyqtSlot()
    def on_PushButton_GPU_Check_clicked(self):

        # 1. 可执行文件路径
        if os.name == 'nt':
            # name_check_gpu = NAME_CHECK_GPU
            name_check_gpu = EXE_MODELSEG
        elif os.name == 'posix':
            # name_check_gpu = NAME_CHECK_GPU_LINUX
            name_check_gpu = EXE_MODELSEG
        else:
            CLogging.LogWarning("Unknow system !")
        path_gpu_check = os.path.join(os.getcwd(), FOLDER_BIN, name_check_gpu)
        if not os.path.exists(path_gpu_check):
            QMessageBox.critical(self, "Warn", "The GPU check program does not exist！")
            self.Label_Flow_GPU_State.setStyleSheet(LIST_CHECK_IMAGE[0])
            self.CheckBox_Flow_GPU_Use.setEnabled(False)
            self.CheckBox_Flow_GPU_Use.setChecked(False)
            return

        dialog_title = r"Check GPU state, waiting..."

        self.__captainiCheckGPU(path_gpu_check, dialog_title)

    def __captainiCheckGPU(self, path_gpu_check, dialog_title:str):
        # 弹出等待对话框
        self.wait_dialog = QProgressDialog(dialog_title, None, 0, 0, self)
        self.wait_dialog.setWindowTitle("Waiting")
        self.wait_dialog.setModal(True)
        self.wait_dialog.setCancelButton(None)
        # 去掉问号按钮
        flags = self.wait_dialog.windowFlags()
        self.wait_dialog.setWindowFlags(flags & ~Qt.WindowContextHelpButtonHint)# 去掉问号按钮
        self.wait_dialog.show()

        clean_env = CFunctionCMDEnv.GetSystemEnv()
        # 创建线程
        self.gpu_thread = SubprocessThread(
            exe_path=path_gpu_check,
            args=["checkgpu"],
            env=clean_env
        )

        # 连接信号
        # 实时输出到控制台（也可转发到界面）
        self.gpu_thread.output.connect(lambda txt: print(txt))
        self.gpu_thread.error.connect(lambda txt: print(txt, file=sys.stderr))

        # 进程结束时关闭对话框并更新 UI
        def on_finished(exitCode):
            # 关闭等待对话框
            if hasattr(self, 'wait_dialog') and self.wait_dialog:
                self.wait_dialog.close()
                self.wait_dialog.deleteLater()
                delattr(self, 'wait_dialog')
            # 更新 UI 样式
            if exitCode == 0:
                self.Label_Flow_GPU_State.setStyleSheet(LIST_CHECK_IMAGE[1])
                self.CheckBox_Flow_GPU_Use.setChecked(True)
                self.CheckBox_Flow_GPU_Use.setEnabled(True)
            else:
                self.Label_Flow_GPU_State.setStyleSheet(LIST_CHECK_IMAGE[0])
                self.CheckBox_Flow_GPU_Use.setChecked(False)
                self.CheckBox_Flow_GPU_Use.setEnabled(False)
            # 清理线程
            self.gpu_thread.deleteLater()

        def on_failed(error_msg):
            print("启动失败:", error_msg)
            # 同样关闭对话框，UI 置失败状态
            if hasattr(self, 'wait_dialog') and self.wait_dialog:
                self.wait_dialog.close()
                self.wait_dialog.deleteLater()
                delattr(self, 'wait_dialog')
            self.Label_Flow_GPU_State.setStyleSheet(LIST_CHECK_IMAGE[0])
            self.CheckBox_Flow_GPU_Use.setChecked(False)
            self.CheckBox_Flow_GPU_Use.setEnabled(False)
            self.gpu_thread.deleteLater()

        self.gpu_thread.finished.connect(on_finished)
        self.gpu_thread.failed.connect(on_failed)

        # 启动线程（非阻塞）
        self.gpu_thread.start()

    # 左下角按钮响应


    # 左侧按钮响应

    @pyqtSlot()
    def on_PushButton_Flow_ImageLabel_clicked(self):

        # self.SetStackedWidget(EnumStackedWidget.ImageLabel)
        cmd_str: str = os.path.join(os.getcwd(), FOLDER_BIN, NAME_APUCPATHANNOTATION_EXE)
        os.chdir(FOLDER_BIN)
        CLogging.LogToUser(cmd_str)
        try:
            ret = self.__captainRunCoreExe(cmd_str)
        except:
            CLogging.LogWarning(NAME_APUCPATHANNOTATION_EXE + " meet some error ! ")
        os.chdir("..")

    @pyqtSlot()
    def on_PushButton_Flow_CrateDataset_clicked(self):

        self.SetStackedWidget(EnumStackedWidget.CreateDataset)

    @pyqtSlot()
    def on_PushButton_Flow_TrainModel_clicked(self):

        self.SetStackedWidget(EnumStackedWidget.TrainModel)

    @pyqtSlot()
    def on_PushButton_Flow_Predict_clicked(self):

        self.SetStackedWidget(EnumStackedWidget.Predict)

    @pyqtSlot()
    def on_PushButton_Flow_ImageView_clicked(self):

        self.SetStackedWidget(EnumStackedWidget.ImageView)

    # 左侧按钮响应

    # CreateDataset按钮响应

    @pyqtSlot()
    def on_PushButton_CreateDataset_Add_clicked(self):
        self.interface_window_select_dataset = CInterfaceWindowApuCPathSelectForDataset(self.ComboBox_CreateDataset_DatasetType.currentIndex())
        self.interface_window_select_dataset.show()
        self.interface_window_select_dataset.saveSelectDataset.connect(self.UpdateSelectFileForDataset)

    @pyqtSlot()
    def on_PushButton_CreateDataset_Clear_clicked(self):

        self.ListWidget_CreateDataset_Image.clear()
        self.ListWidget_CreateDataset_Mask.clear()

    @pyqtSlot()
    def on_PushButton_CreateDataset_Delete_clicked(self):
        if len(self.ListWidget_CreateDataset_Image.selectedIndexes()) == 0:
            pass
        else:
            select_index = self.ListWidget_CreateDataset_Image.selectedIndexes()[0].row()
            self.ListWidget_CreateDataset_Image.takeItem(select_index)
            self.ListWidget_CreateDataset_Mask.takeItem(select_index)

    @pyqtSlot()
    def on_PushButton_CreateDataset_ExportPath_clicked(self):

        dir_choose = QFileDialog.getExistingDirectory(self, 'Directory Select', os.getcwd())
        if len(dir_choose) == 0:
            pass
        else:
            self.LineEdit_CreateDataset_ExportPath.setText(dir_choose)

    @pyqtSlot()
    def on_ComboBox_CreateDataset_DatasetType_currentTextChanged(self):
        if self.ComboBox_CreateDataset_DatasetType.currentIndex() == EnumUIDatasetType.Train.value:
            self.Label_CreateDataset_Input_Mask.setHidden(False)
            self.ListWidget_CreateDataset_Mask.setHidden(False)
            self.Frame_CreateDataset_MaskAreaRatio.setHidden(False)
            self.DoubleSpinBox_CreateDataset_MaskAreaRatio_Min.setValue(0.0001)
            self.DoubleSpinBox_CreateDataset_MaskAreaRatio_Max.setValue(1.00)
            self.DoubleSpinBox_CreateDataset_MaskAreaRatio_Min.setMinimum(0.0001)
        elif self.ComboBox_CreateDataset_DatasetType.currentIndex() == EnumUIDatasetType.Assist.value:
            self.Label_CreateDataset_Input_Mask.setHidden(False)
            self.ListWidget_CreateDataset_Mask.setHidden(False)
            self.Frame_CreateDataset_MaskAreaRatio.setHidden(True)
            self.DoubleSpinBox_CreateDataset_MaskAreaRatio_Min.setValue(0.00)
            self.DoubleSpinBox_CreateDataset_MaskAreaRatio_Max.setValue(1.00)
            self.DoubleSpinBox_CreateDataset_MaskAreaRatio_Min.setMinimum(0)
        elif self.ComboBox_CreateDataset_DatasetType.currentIndex() == EnumUIDatasetType.Predict.value:
            self.Label_CreateDataset_Input_Mask.setHidden(True)
            self.ListWidget_CreateDataset_Mask.setHidden(True)
            self.Frame_CreateDataset_MaskAreaRatio.setHidden(True)
        elif self.ComboBox_CreateDataset_DatasetType.currentIndex() == EnumUIDatasetType.Valid.value:
            self.Label_CreateDataset_Input_Mask.setHidden(False)
            self.ListWidget_CreateDataset_Mask.setHidden(False)
            self.Frame_CreateDataset_MaskAreaRatio.setHidden(False)
            self.DoubleSpinBox_CreateDataset_MaskAreaRatio_Min.setValue(0.0001)
            self.DoubleSpinBox_CreateDataset_MaskAreaRatio_Max.setValue(1.00)
            self.DoubleSpinBox_CreateDataset_MaskAreaRatio_Min.setMinimum(0.0001)

    # CreateDataset按钮响应

    # TrainModel按钮响应

    @pyqtSlot()
    def on_PushButton_TrainModel_DatasetPath_clicked(self):
        model_select_filter = f"Dataset Files(*.{EnumDatasetType.Train.name})"
        path_cur_file, file_type = QFileDialog.getOpenFileName(self, 'Choose the file', self.cwd, model_select_filter)
        path_img = path_cur_file
        if len(path_img) == 0:
            pass
        else:
            self.LineEdit_TrainModel_DatasetPath.setText(path_img)
            
    @pyqtSlot()
    def on_PushButton_TrainModel_ValidDatasetPath_clicked(self):
        model_select_filter = f"Dataset Files(*.{EnumDatasetType.Valid.name})"
        path_cur_file, file_type = QFileDialog.getOpenFileName(self, 'Choose the file', self.cwd, model_select_filter)
        path_img = path_cur_file
        if len(path_img) == 0:
            pass
        else:
            self.LineEdit_TrainModel_ValidDatasetPath.setText(path_img)

    @pyqtSlot()
    def on_PushButton_TrainModel_ModelPath_clicked(self):
        model_select_filter = "Model Files(*.h5)"
        path_cur_file, file_type = QFileDialog.getOpenFileName(self, 'Choose the file', self.cwd, model_select_filter)
        path_model = path_cur_file
        if len(path_model) == 0:
            pass
        else:
            self.LineEdit_TrainModel_ModelPath.setText(path_model)

    @pyqtSlot()
    def on_PushButton_TrainModel_ExportPath_clicked(self):
        dir_choose = QFileDialog.getExistingDirectory(self, 'Directory Select', os.getcwd())
        if len(dir_choose) == 0:
            pass
        else:
            self.LineEdit_TrainModel_ExportPath.setText(dir_choose)

    @pyqtSlot()
    def on_PushButton_TrainModel_PathUNIBackbone_clicked(self):
        model_select_filter = "UNI Model Files(*.bin)"
        path_cur_file, file_type = QFileDialog.getOpenFileName(self, 'Choose the file', self.cwd, model_select_filter)
        path_model = path_cur_file
        if len(path_model) == 0:
            pass
        else:
            self.LineEdit_TrainModel_PathUNIBackbone.setText(path_model)
    def on_CheckBox_TrainModel_AddNeg_toggle(self, checked):
        if checked:
            self.ComboBox_TrainModel_NegSample_Scheme.setEnabled(True)
            self.on_ComboBox_TrainModel_NegSample_Scheme_currentIndexChanged()
        else:
            self.ComboBox_TrainModel_NegSample_Scheme.setEnabled(False)

    def on_CheckBox_TrainModel_ValidDatasetPath_toggle(self, checked):

        if checked:
            self.LineEdit_TrainModel_ValidDatasetPath.setText("")
            self.LineEdit_TrainModel_ValidDatasetPath.setEnabled(False)
            self.PushButton_TrainModel_ValidDatasetPath.setEnabled(False)
            self.DoubleSpinBox_TrainModel_ValidDatasetPath.setEnabled(True)
        else:
            self.LineEdit_TrainModel_ValidDatasetPath.setEnabled(True)
            self.PushButton_TrainModel_ValidDatasetPath.setEnabled(True)
            self.DoubleSpinBox_TrainModel_ValidDatasetPath.setEnabled(False)

    @pyqtSlot()
    def on_ComboBox_TrainModel_InitType_currentIndexChanged(self):
        if self.ComboBox_TrainModel_InitType.currentIndex() == EnumWSISegModelTypeModelInit.Scratch.value:
            self.LineEdit_TrainModel_ModelPath.setText("")
            self.Frame_TrainModel_ModelPath.setEnabled(False)
            self.Label_TrainModel_ModelPath.setStyleSheet(LIST_CHECK_IMAGE[1])
        elif self.ComboBox_TrainModel_InitType.currentIndex() == EnumWSISegModelTypeModelInit.Finetune.value:
            self.LineEdit_TrainModel_ModelPath.setText('')
            self.Frame_TrainModel_ModelPath.setEnabled(True)
            self.Label_TrainModel_ModelPath.setStyleSheet(LIST_CHECK_IMAGE[0])
        else:
            self.LineEdit_TrainModel_ModelPath.setText('')
            self.Label_TrainModel_ModelPath.setStyleSheet(LIST_CHECK_IMAGE[0])

    def on_ComboBox_TrainModel_NegSample_Scheme_currentIndexChanged(self):
        if self.CheckBox_TrainModel_AddNeg.isChecked() and (self.ComboBox_TrainModel_NegSample_Scheme.currentIndex() == EnumWSISegModelTypeNegSample.MiniBatchKMeans.value):
            self.SpinBox_TrainModel_NegSample_Cluster.setEnabled(True)
        else:
            self.SpinBox_TrainModel_NegSample_Cluster.setEnabled(False)

    def on_ComboBox_TrainModel_ModelType_currentIndexChanged(self):

        if self.ComboBox_TrainModel_ModelType.currentIndex() == EnumModelType.UNIMask2Former.value:
            self.LineEdit_TrainModel_PathUNIBackbone.setText(self.config_model.G4_PATH_UNI_CHECKPOINT)
            self.Frame_TrainModel_PathUNIBackbone.setEnabled(True)
        else:
            self.LineEdit_TrainModel_PathUNIBackbone.setText("")
            self.Frame_TrainModel_PathUNIBackbone.setEnabled(False)

    # TrainModel按钮响应

    # Predict按钮响应

    @pyqtSlot()
    def on_PushButton_Predict_DatasetPath_clicked(self):
        model_select_filter = f"Dataset Files(*{EnumDatasetType.Train.name} *.{EnumDatasetType.Predict.name} *.{EnumDatasetType.Assist.name})"
        path_cur_file, file_type = QFileDialog.getOpenFileName(self, 'Choose the file', self.cwd, model_select_filter)
        path_img = path_cur_file
        if len(path_img) == 0:
            pass
        else:
            self.LineEdit_Predict_DatasetPath.setText(path_img)

    @pyqtSlot()
    def on_PushButton_Predict_ModelPath_clicked(self):
        model_select_filter = "Model Files(*.pth)"
        path_cur_file, file_type = QFileDialog.getOpenFileName(self, 'Choose the file', self.cwd, model_select_filter)
        path_img = path_cur_file
        if len(path_img) == 0:
            pass
        else:
            self.LineEdit_Predict_ModelPath.setText(path_img)

    @pyqtSlot()
    def on_PushButton_Predict_ExportPath_clicked(self):
        dir_choose = QFileDialog.getExistingDirectory(self, 'Directory Select', os.getcwd())
        if len(dir_choose) == 0:
            pass
        else:
            self.LineEdit_Predict_ExportPath.setText(dir_choose)

    # Predict按钮响应

    # ImageView按钮响应

    @pyqtSlot()
    def on_PushButton_ImageView_ZoomIn_clicked(self):
        if self.__ImageView_cross_state == EnumImageViewCrossState.Open:
            self.__ImageView_cross_state = EnumImageViewCrossState.Close
            self.__ImageView_zoom_state = EnumImageViewZoomState.ZoomIn
        else:
            if self.__ImageView_zoom_state == EnumImageViewZoomState.ZoomIn:
                self.__ImageView_zoom_state = EnumImageViewZoomState.Close
            else:
                self.__ImageView_zoom_state = EnumImageViewZoomState.ZoomIn
        self.UpdataImageViewToolState()

    @pyqtSlot()
    def on_PushButton_ImageView_ZoomOut_clicked(self):
        if self.__ImageView_cross_state == EnumImageViewCrossState.Open:
            self.__ImageView_cross_state = EnumImageViewCrossState.Close
            self.__ImageView_zoom_state = EnumImageViewZoomState.ZoomOut
        else:
            if self.__ImageView_zoom_state == EnumImageViewZoomState.ZoomOut:
                self.__ImageView_zoom_state = EnumImageViewZoomState.Close
            else:
                self.__ImageView_zoom_state = EnumImageViewZoomState.ZoomOut
        self.UpdataImageViewToolState()

    @pyqtSlot()
    def on_PushButton_ImageView_Normal_clicked(self):
        self.interface_ImageView_GraphicsView_show_image.ImageRecover()

    @pyqtSlot()
    def on_PushButton_ImageView_Cross_clicked(self):
        if self.__ImageView_zoom_state != EnumImageViewZoomState.Close:
            self.__ImageView_zoom_state = EnumImageViewZoomState.Close
            self.__ImageView_cross_state = EnumImageViewCrossState.Open
        else:
            if self.__ImageView_cross_state == EnumImageViewCrossState.Open:
                self.__ImageView_cross_state = EnumImageViewCrossState.Close
            else:
                self.__ImageView_cross_state = EnumImageViewCrossState.Open
        self.UpdataImageViewToolState()

    @pyqtSlot()
    def on_PushButton_ImageView_Clear_clicked(self):

        self.__soldierInitControlImageView()

    @pyqtSlot()
    def on_PushButton_ImageView_Open_clicked(self):

        img_select_filter = "Image Files("
        for item in LIST_VISUAL_FILE_IMAGE_IDX:
            img_select_filter += f"*{LIST_VISUAL_FILE_TYPE_SUFFIX[item]} "
        img_select_filter += ")"
        path_cur_file, file_type = QFileDialog.getOpenFileName(self, 'Choose the file', self.cwd, img_select_filter)
        path_img = path_cur_file
        if len(path_img) == 0:
            pass
        else:
            self.interface_ImageView_GraphicsView_show_image.GraphicsViewShowImage(path_img)
            self.PushButton_ImageView_ZoomIn.setEnabled(True)
            self.PushButton_ImageView_ZoomOut.setEnabled(True)
            self.PushButton_ImageView_Normal.setEnabled(True)
            self.PushButton_ImageView_Cross.setEnabled(True)
            self.TableWidget_ImageView_Info.setItem(1, 1, QTableWidgetItem(os.path.basename(path_img)))
            self.TableWidget_ImageView_Info.setItem(2, 1, QTableWidgetItem(os.path.dirname(path_img)))
            h, w = self.interface_ImageView_GraphicsView_show_image.GetCurImageSize()
            self.TableWidget_ImageView_Info.setItem(3, 1, QTableWidgetItem(str(h)))
            self.TableWidget_ImageView_Info.setItem(4, 1, QTableWidgetItem(str(w)))

    @pyqtSlot()
    def on_PushButton_ImageView_Clear(self):

        self.__soldierInitControlImageView()

    # ImageView按钮响应

    # ===============相关函数=================

    # 左下状态相关的函数

    def SetFlowState(self, state:EnumWindowState):

        self.__run_state = state
        self.__captainUpdateFlowState()

    def __captainUpdateFlowState(self):

        self.LineEdit_Flow_Sate.setText(self.__run_state.name)
        self.Label_Flow_State.setStyleSheet(LIST_IMAGE_WINDOW_STATE[self.__run_state.value])

    # 左下状态相关的函数

    # 左侧按钮相关的函数
    def SetStackedWidget(self, stacked_widget_stat:EnumStackedWidget):

        self.__stacked_widget_state = stacked_widget_stat
        self.__captainUpadateStackedEWidget()

    def __captainUpadateStackedEWidget(self):
        # 左侧按钮切换界面管理的函数
        self.PushButton_Flow_ImageLabel.setChecked(False)
        self.PushButton_Flow_CrateDataset.setChecked(False)
        self.PushButton_Flow_TrainModel.setChecked(False)
        self.PushButton_Flow_Predict.setChecked(False)
        self.PushButton_Flow_ImageView.setChecked(False)
        if self.__stacked_widget_state == EnumStackedWidget.ImageLabel:
            self.PushButton_Flow_ImageLabel.setChecked(True)
        elif self.__stacked_widget_state == EnumStackedWidget.CreateDataset:
            self.PushButton_Flow_CrateDataset.setChecked(True)
            self.CheckCreateDataset()
        elif self.__stacked_widget_state == EnumStackedWidget.TrainModel:
            self.PushButton_Flow_TrainModel.setChecked(True)
            self.CheckTrainModel()
        elif self.__stacked_widget_state == EnumStackedWidget.Predict:
            self.PushButton_Flow_Predict.setChecked(True)
            self.CheckPredict()
        elif self.__stacked_widget_state == EnumStackedWidget.ImageView:
            self.PushButton_Flow_ImageView.setChecked(True)
        else:
            self.__stacked_widget_state = EnumStackedWidget.CreateDataset
            self.PushButton_Flow_CrateDataset.setChecked(True)
            self.CheckCreateDataset()

        self.stackedWidget.setCurrentIndex(self.__stacked_widget_state.value)
        self.__soldierInitControlRun()

    # 左侧按钮相关的函数

    # 顶部按钮相关的函数

    def SaveCreateDatasetConfig(self):
        ui_dataset_type_idx:EnumUIDatasetType = EnumUIDatasetType(self.ComboBox_CreateDataset_DatasetType.currentIndex())
        if ui_dataset_type_idx == EnumUIDatasetType.Train:
            self.config_create_dataset.A1_TYPE_DATASET = EnumDatasetType.Train
        elif ui_dataset_type_idx == EnumUIDatasetType.Assist:
            self.config_create_dataset.A1_TYPE_DATASET = EnumDatasetType.Assist
        elif ui_dataset_type_idx == EnumUIDatasetType.Predict:
            self.config_create_dataset.A1_TYPE_DATASET = EnumDatasetType.Predict
        elif ui_dataset_type_idx == EnumUIDatasetType.Valid:
            self.config_create_dataset.A1_TYPE_DATASET = EnumDatasetType.Valid
        else:
            return
        count = self.ListWidget_CreateDataset_Image.count()
        image_list_str = ''
        for i in range(count):
            image_list_str += self.ListWidget_CreateDataset_Image.item(i).text() + ';'
        self.config_create_dataset.A2_LIST_IMAGE_PATH = image_list_str

        if (self.ComboBox_CreateDataset_DatasetType.currentIndex() == EnumUIDatasetType.Train.value) or (self.ComboBox_CreateDataset_DatasetType.currentIndex() == EnumUIDatasetType.Assist.value) or (self.ComboBox_CreateDataset_DatasetType.currentIndex() == EnumUIDatasetType.Valid.value):
            mask_list_str = ""
            for i in range(count):
                mask_list_str += self.ListWidget_CreateDataset_Mask.item(i).text() + ";"
            self.config_create_dataset.A3_LIST_MASK_PATH = mask_list_str

        self.config_create_dataset.A4_FOLDER_EXPORT = self.LineEdit_CreateDataset_ExportPath.text()
        if (self.ComboBox_CreateDataset_DatasetType.currentIndex() == EnumUIDatasetType.Assist.value):
            self.config_create_dataset.B2_MASK_RATIO_MIN = 1e-6
            self.config_create_dataset.B3_MASK_RATIO_MAX = 1.0
        else:
            self.config_create_dataset.B2_MASK_RATIO_MIN = self.DoubleSpinBox_CreateDataset_MaskAreaRatio_Min.value()
            self.config_create_dataset.B3_MASK_RATIO_MAX = self.DoubleSpinBox_CreateDataset_MaskAreaRatio_Max.value()

    def SaveTrainModelConfig(self):
        if self.CheckBox_TrainModel_ValidDatasetPath.isChecked():
            self.config_model.A1_TYPE_FLOW = EnumMaskRCNNEXEFlow.NormalDatasetForTrain.value
        else:
            self.config_model.A1_TYPE_FLOW = EnumMaskRCNNEXEFlow.TrainDatasetValidDatasetForTrain.value
        self.config_model.A3_PATH_DATASET_TRAIN = self.LineEdit_TrainModel_DatasetPath.text()
        self.config_model.A4_PATH_DATASET_VALID = self.LineEdit_TrainModel_ValidDatasetPath.text()
        self.config_model.A6_FOLDER_OUTPUT = self.LineEdit_TrainModel_ExportPath.text()
        # self.config_maskrcnn.B1_IMAGE_DIM = self.TrainModel
        self.config_model.B4_BATCH_SIZE = self.SpinBox_TrainModel_BatchSize.value()
        self.config_model.B5_EPOCH_NUM = self.SpinBox_TrainModel_Epoch.value()
        self.config_model.B6_LEARNING_RATE = self.DoubleSpinBox_TrainModel_LearningRate.value()
        self.config_model.B9_VALID_PERCENT = self.DoubleSpinBox_TrainModel_ValidDatasetPath.value()

        self.config_model.C1_MODEL_INIT = EnumWSISegModelTypeModelInit(self.ComboBox_TrainModel_InitType.currentIndex())
        self.config_model.C2_MODEL_PATH = self.LineEdit_TrainModel_ModelPath.text()
        if self.ComboBox_TrainModel_ModelType.currentIndex() == EnumModelType.Unet.value:
            self.config_model.C3_MODEL_TYPE = EnumEnumWSISegModelTypeIndex.Unet
        elif self.ComboBox_TrainModel_ModelType.currentIndex() == EnumModelType.MaskRCNN.value:
            self.config_model.C3_MODEL_TYPE = EnumEnumWSISegModelTypeIndex.MaskRCNN
        elif self.ComboBox_TrainModel_ModelType.currentIndex() == EnumModelType.UNIMask2Former.value:
            self.config_model.C3_MODEL_TYPE = EnumEnumWSISegModelTypeIndex.UNIMask2Former
        else:
            self.config_model.C3_MODEL_TYPE = EnumEnumWSISegModelTypeIndex.UNIMask2Former
        self.config_model.C4_MODEL_ADD_NEGATIVE = self.CheckBox_TrainModel_AddNeg.isChecked()
        self.config_model.C5_MODEL_NEGATIVE_SAMPLE = EnumWSISegModelTypeNegSample(self.ComboBox_TrainModel_NegSample_Scheme.currentIndex())
        self.config_model.C5_MODEL_NEGATIVE_SAMPLE_CLUSTER = self.SpinBox_TrainModel_NegSample_Cluster.value()

        self.config_model.G4_PATH_UNI_CHECKPOINT = self.LineEdit_TrainModel_PathUNIBackbone.text()

        if self.CheckBox_Flow_GPU_Use.isChecked():
            self.config_model.Z1_USE_GPU = 1
        else:
            self.config_model.Z1_USE_GPU = 0

    def SavePredictConfig(self):
        self.config_model.A1_TYPE_FLOW = EnumMaskRCNNEXEFlow.PredictDatasetForPredict.value
        self.config_model.A5_PATH_DATASET_PREDICT = self.LineEdit_Predict_DatasetPath.text()
        self.config_model.A6_FOLDER_OUTPUT = self.LineEdit_Predict_ExportPath.text()

        self.config_model.B4_BATCH_SIZE = self.SpinBox_Predict_BatchSize.value()

        self.config_model.C2_MODEL_PATH = self.LineEdit_Predict_ModelPath.text()
        if self.ComboBox_Predict_ModelType.currentIndex() == EnumModelType.Unet.value:
            self.config_model.C3_MODEL_TYPE = EnumEnumWSISegModelTypeIndex.Unet
        elif self.ComboBox_Predict_ModelType.currentIndex() == EnumModelType.MaskRCNN.value:
            self.config_model.C3_MODEL_TYPE = EnumEnumWSISegModelTypeIndex.MaskRCNN
        elif self.ComboBox_Predict_ModelType.currentIndex() == EnumModelType.UNIMask2Former.value:
            self.config_model.C3_MODEL_TYPE = EnumEnumWSISegModelTypeIndex.UNIMask2Former
        else:
            self.config_model.C3_MODEL_TYPE = EnumEnumWSISegModelTypeIndex.UNIMask2Former

        self.config_model.D1_FILTER_SCORE = self.DoubleSpinBox_Predict_FilterScore.value()
        self.config_model.D3_MIN_AREA = self.SpinBox_Predict_MinArea.value()
        self.config_model.D4_MERGE_SCORE = self.DoubleSpinBox_Predict_MCAScore.value()

        if self.CheckBox_Flow_GPU_Use.isChecked():
            self.config_model.Z1_USE_GPU = 1
        else:
            self.config_model.Z1_USE_GPU = 0

    def __captainRunCoreExe(self, cmd_str:str)->bool:
        return self.__captainiStartCoreExe(cmd_str, "Running ... ")
        # if os.system(cmd_str) == 0:
        #     return True
        # else:
        #     return False

    def __captainiStartCoreExe(self, cmd_str, dialog_title):
        args = cmd_str.split(' ')
        # 显示模态等待对话框
        # self.wait_dialog = QProgressDialog(dialog_title, None, 0, 0, self)
        # self.wait_dialog.setWindowTitle("Waiting")
        # self.wait_dialog.setModal(True)
        # self.wait_dialog.setCancelButton(None)
        # flags = self.wait_dialog.windowFlags()
        # self.wait_dialog.setWindowFlags(flags & ~Qt.WindowContextHelpButtonHint)
        # self.wait_dialog.show()

        # 获取干净环境
        clean_env = CFunctionCMDEnv.GetSystemEnv()

        try:
            # 启动子进程，不捕获输出（直接继承控制台），同步等待
            proc = subprocess.Popen(
                args,
                env=clean_env,
                stdout=None,  # 继承父进程控制台
                stderr=None  # 继承父进程控制台
            )
            ret = proc.wait()  # 等待结束，返回退出码
        except Exception as e:
            print(f"启动失败: {e}")
            ret = -1

        # # 关闭对话框
        # if hasattr(self, 'wait_dialog') and self.wait_dialog:
        #     self.wait_dialog.close()
        #     self.wait_dialog.deleteLater()
        #     delattr(self, 'wait_dialog')

        # 更新 UI
        success = (ret == 0)
        return success

    # 顶部按钮相关的函数

    # CreateDataset相关的函数

    def UpdateSelectFileForDataset(self, save_type):
        # 选择图像数据
        del self.interface_window_select_dataset
        if save_type == EnumSelectForDatasetFileType.ImageFile.value:

            self.ListWidget_CreateDataset_Image.addItem(CSystemWindowSelectForDataset.PATH_IMAGE)
            self.ListWidget_CreateDataset_Mask.addItem(CSystemWindowSelectForDataset.PATH_MASK)

        elif save_type == EnumSelectForDatasetFileType.ImageFolder.value:
            list_image_path = []
            list_mask_path = []
            if self.ComboBox_CreateDataset_DatasetType.currentIndex() == EnumUIDatasetType.Predict.value:
                list_image_name_file = os.listdir(CSystemWindowSelectForDataset.PATH_IMAGE)
                for image_name in list_image_name_file:
                    list_image_path.append(os.path.join(CSystemWindowSelectForDataset.PATH_IMAGE, image_name))
                for idx, path_img in enumerate(list_image_path):
                    self.ListWidget_CreateDataset_Image.addItem(path_img)
            else:
                list_image_name_file = os.listdir(CSystemWindowSelectForDataset.PATH_IMAGE)
                list_mask_name_file = os.listdir(CSystemWindowSelectForDataset.PATH_MASK)
                list_not_find = []
                for img_idx, image_name in enumerate(list_image_name_file):
                    find:bool = False
                    for mask_idx, mask_name in enumerate(list_mask_name_file):
                        if (image_name.split('.')[0] == mask_name.split('.')[0] or image_name.split('.')[0] == mask_name.split('.')[0].replace("_MASK", '')) and (mask_name.endswith(LIST_VISUAL_FILE_TYPE_SUFFIX[EnumVisualFileType.JSON.value])):
                            find = True
                            break
                    if find:
                        list_image_path.append(os.path.join(CSystemWindowSelectForDataset.PATH_IMAGE, image_name))
                        list_mask_path.append(os.path.join(CSystemWindowSelectForDataset.PATH_MASK, mask_name))
                    else:
                        list_not_find.append(os.path.join(CSystemWindowSelectForDataset.PATH_IMAGE, image_name))
                if len(list_image_path) == 0:
                    QMessageBox.about(self, "Warn", "No files meeting the requirements were found.")
                for idx, path_img in enumerate(list_image_path):
                    path_mask = list_mask_path[idx]
                    self.ListWidget_CreateDataset_Image.addItem(path_img)
                    self.ListWidget_CreateDataset_Mask.addItem(path_mask)
        else:
            pass

    # CreateDataset相关的函数

    # ImageView相关的函数

    def UpdataImageViewToolState(self):

        # 顶部几个按钮(放大、缩小、指针)之间切换的联动关系

        if self.__ImageView_zoom_state == EnumImageViewZoomState.ZoomIn:
            self.PushButton_ImageView_ZoomIn.setChecked(True)
            self.PushButton_ImageView_ZoomOut.setChecked(False)
            self.__ImageView_cross_state = EnumImageViewCrossState.Close
        elif self.__ImageView_zoom_state == EnumImageViewZoomState.ZoomOut:
            self.PushButton_ImageView_ZoomIn.setChecked(False)
            self.PushButton_ImageView_ZoomOut.setChecked(True)
            self.__ImageView_cross_state = EnumImageViewCrossState.Close
        else:
            self.__ImageView_zoom_state = EnumImageViewZoomState.Close
            self.PushButton_ImageView_ZoomIn.setChecked(False)
            self.PushButton_ImageView_ZoomOut.setChecked(False)

        if self.__ImageView_cross_state == EnumImageViewCrossState.Open:
            self.PushButton_ImageView_Cross.setChecked(True)
        else:
            self.PushButton_ImageView_Cross.setChecked(False)

    def __connnect_ImageView_ShowLocationControl(self, y:int, x:int):

        self.ImageView_ShowLocationControl(y, x)

    def ImageView_ShowLocationControl(self, y:int, x:int):

        if self.__ImageView_cross_state == EnumImageViewCrossState.Open:
            if (y < 0) or (x < 0):
                self.LineEdit_ImageView_Location_X.setText("")
                self.LineEdit_ImageView_Location_Y.setText("")
                self.LineEdit_ImageView_Color_R.setText("")
                self.LineEdit_ImageView_Color_G.setText("")
                self.LineEdit_ImageView_Color_B.setText("")
                self.setCursor(self.interface_Cursor.GetCursor(EnumCursorState.Arrow.name))

            else:
                self.LineEdit_ImageView_Location_X.setText(str(x))
                self.LineEdit_ImageView_Location_Y.setText(str(y))
                r, g, b = self.interface_ImageView_GraphicsView_show_image.GetRGBValue(y, x)
                self.LineEdit_ImageView_Color_R.setText(str(r))
                self.LineEdit_ImageView_Color_G.setText(str(g))
                self.LineEdit_ImageView_Color_B.setText(str(b))

                self.setCursor(self.interface_Cursor.GetCursor(EnumCursorState.CROSS.name))
        else:
            self.LineEdit_ImageView_Location_X.setText("")
            self.LineEdit_ImageView_Location_Y.setText("")
            self.LineEdit_ImageView_Color_R.setText("")
            self.LineEdit_ImageView_Color_G.setText("")
            self.LineEdit_ImageView_Color_B.setText("")

            if (y < 0) or (x < 0):
                self.setCursor(self.interface_Cursor.GetCursor(EnumCursorState.Arrow.name))
            else:
                if self.__ImageView_zoom_state == EnumImageViewZoomState.ZoomIn:
                    self.setCursor(self.interface_Cursor.GetCursor(EnumCursorState.ZOOM_IN.name))
                elif self.__ImageView_zoom_state == EnumImageViewZoomState.ZoomOut:
                    self.setCursor(self.interface_Cursor.GetCursor(EnumCursorState.ZOOM_OUT.name))
                else:
                    self.setCursor(self.interface_Cursor.GetCursor(EnumCursorState.Arrow.name))

    def __connnect_ImageView_UpdateGraphicsView(self):

        if self.__ImageView_zoom_state == EnumImageViewZoomState.ZoomIn:

            self.interface_ImageView_GraphicsView_show_image.ZoomIn()

        elif self.__ImageView_zoom_state == EnumImageViewZoomState.ZoomOut:

            self.interface_ImageView_GraphicsView_show_image.ZoomOut()

        else:

            pass

    # ImageView相关的函数
