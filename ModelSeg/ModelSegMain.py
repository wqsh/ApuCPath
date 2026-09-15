# -*- coding: utf-8 -*-
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
if "CPU" in sys.argv:
    os.environ["CUDA_VISIBLE_DEVICES"]="-1"
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
from Core.MSStaff import CStaff

if __name__ == '__main__':
    staff = CStaff()
    staff.start(sys.argv)