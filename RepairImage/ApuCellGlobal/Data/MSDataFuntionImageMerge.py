import os


class CDataImageMergeInfo:

    def __init__(self, folder_path:str, y_min_idx:int, x_min_idx:int, y_max_idx:int, x_max_idx:int, y_size:int, x_size:int, shape:int):
        self.folder_path:str = folder_path
        self.y_min_idx:int = y_min_idx
        self.x_min_idx:int = x_min_idx
        self.y_max_idx:int = y_max_idx
        self.x_max_idx:int = x_max_idx
        self.y_idx_length:int = y_max_idx - y_min_idx + 1
        self.x_idx_length:int = x_max_idx - x_min_idx + 1
        self.y_size:int = y_size
        self.x_size:int = x_size
        self.shape:int = shape
        if y_size is None or x_size is None:
            self.y_full_size = None
            self.x_full_size: int = None
        else:
            self.y_full_size = (y_max_idx - y_min_idx + 1) * y_size
            self.x_full_size:int = (x_max_idx - x_min_idx + 1) * x_size

