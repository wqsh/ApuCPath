import os
import shutil

work_folder = os.path.dirname(os.getcwd())

folder_out = os.path.join(work_folder, r"Release/ApuCPath")

if not os.path.exists(folder_out):
    os.makedirs(folder_out)

folder_bin = os.path.join(folder_out, "bin")
folder_dll = os.path.join(work_folder, r"ApuCPath/ApuCellGlobal/DLL")

path_exe_apucpath = os.path.join(work_folder, r"ApuCPath/dist/ApuCPath")
path_exe_apucpath_createdataset = os.path.join(work_folder, r"ApuCPathCreateDataset/dist/ApuCPathCreateDataset")
path_exe_apucpath_maskrcnn = os.path.join(work_folder, r"MaskRCNNExe/dist/MaskRCNNMain")
path_exe_apucpath_checkGPU = os.path.join(work_folder, r"MaskRCNNExe/dist/CheckGPU")
path_exe_apucpath_annotation = os.path.join(work_folder, r"ApuCPathAnnotation/dist/ApuCPathAnnotation")

path_exe_apucpath_platforms = os.path.join(work_folder, r"ApuCPath/dist/platforms")

path_exe_apucpath_annotation_ui = os.path.join(work_folder, r"ApuCPathAnnotation/ui")
path_exe_apucpath_annotation_resource = os.path.join(work_folder, r"ApuCPathAnnotation/resources")
path_exe_apucpath_annotation_platforms = os.path.join(work_folder, r"ApuCPathAnnotation/dist/platforms")

if not os.path.exists(folder_bin):
    os.makedirs(folder_bin)

if os.path.exists(path_exe_apucpath):
    shutil.copy(path_exe_apucpath, os.path.join(folder_out, os.path.basename(path_exe_apucpath)))
if os.path.exists(path_exe_apucpath_createdataset):
    shutil.copy(path_exe_apucpath_createdataset, os.path.join(folder_bin, os.path.basename(path_exe_apucpath_createdataset)))
if os.path.exists(path_exe_apucpath_maskrcnn):
    shutil.copy(path_exe_apucpath_maskrcnn, os.path.join(folder_bin, os.path.basename(path_exe_apucpath_maskrcnn)))
if os.path.exists(path_exe_apucpath_checkGPU):
    shutil.copy(path_exe_apucpath_checkGPU, os.path.join(folder_bin, os.path.basename(path_exe_apucpath_checkGPU)))
if os.path.exists(path_exe_apucpath_annotation):
    shutil.copy(path_exe_apucpath_annotation, os.path.join(folder_bin, os.path.basename(path_exe_apucpath_annotation)))

if os.path.exists(os.path.join(folder_out, os.path.basename(folder_dll))):
    shutil.rmtree(os.path.join(folder_out, os.path.basename(folder_dll)))

if os.path.exists(os.path.join(folder_bin, os.path.basename(folder_dll))):
    shutil.rmtree(os.path.join(folder_bin, os.path.basename(folder_dll)))

if os.path.exists(os.path.join(folder_bin, os.path.basename(path_exe_apucpath_annotation_ui))):
    shutil.rmtree(os.path.join(folder_bin, os.path.basename(path_exe_apucpath_annotation_ui)))

if os.path.exists(os.path.join(folder_bin, os.path.basename(path_exe_apucpath_annotation_resource))):
    shutil.rmtree(os.path.join(folder_bin, os.path.basename(path_exe_apucpath_annotation_resource)))

if os.path.exists(os.path.join(folder_out, os.path.basename(path_exe_apucpath_platforms))):
    shutil.rmtree(os.path.join(folder_out, os.path.basename(path_exe_apucpath_platforms)))

if os.path.exists(os.path.join(folder_bin, os.path.basename(path_exe_apucpath_annotation_platforms))):
    shutil.rmtree(os.path.join(folder_bin, os.path.basename(path_exe_apucpath_annotation_platforms)))

shutil.copytree(folder_dll, os.path.join(folder_out, os.path.basename(folder_dll)))
shutil.copytree(folder_dll, os.path.join(folder_bin, os.path.basename(folder_dll)))

shutil.copytree(path_exe_apucpath_platforms, os.path.join(folder_out, os.path.basename(path_exe_apucpath_platforms)))

shutil.copytree(path_exe_apucpath_annotation_ui, os.path.join(folder_bin, os.path.basename(path_exe_apucpath_annotation_ui)))
shutil.copytree(path_exe_apucpath_annotation_resource, os.path.join(folder_bin, os.path.basename(path_exe_apucpath_annotation_resource)))
shutil.copytree(path_exe_apucpath_annotation_platforms, os.path.join(folder_bin, os.path.basename(path_exe_apucpath_annotation_platforms)))
