import os
import shutil

work_folder = os.path.dirname(os.getcwd())

folder_out = os.path.join(work_folder, r"Release\RepairImage")

if not os.path.exists(folder_out):
    os.makedirs(folder_out)

folder_dll = os.path.join(work_folder, r"ApuCPath\ApuCellGlobal\DLL")

path_exe_repairimage = os.path.join(os.getcwd(), r"dist\RepairImage.exe")

shutil.copy(path_exe_repairimage, os.path.join(folder_out, os.path.basename(path_exe_repairimage)))

if os.path.exists(os.path.join(folder_out, os.path.basename(folder_dll))):
    shutil.rmtree(os.path.join(folder_out, os.path.basename(folder_dll)))

shutil.copytree(folder_dll, os.path.join(folder_out, os.path.basename(folder_dll)))