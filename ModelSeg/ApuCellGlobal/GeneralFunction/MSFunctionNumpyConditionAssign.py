import numpy as np
from numba import jit

class CFunctionNumpyConditionAssign:
    """Numba 加速的条件赋值工具（原地修改，无临时布尔数组）"""
    
    @staticmethod
    @jit(nopython=True)
    def add_inplace(a, b):
        """
        原地矩阵加法：a += b
        直接修改第一个数组，无返回值（或返回 None）
        """
        for i in range(a.size):
            a.flat[i] += b.flat[i]

    @staticmethod
    @jit(nopython=True)
    def set_gt(arr, threshold, value, equal:bool=False):
        """arr[arr > threshold] = value"""
        for i in range(arr.size):
            if not equal:
                if arr.flat[i] > threshold:
                    arr.flat[i] = value
            else:
                if arr.flat[i] >= threshold:
                    arr.flat[i] = value

    @staticmethod
    @jit(nopython=True)
    def set_lt(arr, threshold, value, equal:bool=False):
        """arr[arr < threshold] = value"""
        for i in range(arr.size):
            if not equal:
                if arr.flat[i] < threshold:
                    arr.flat[i] = value
            else:
                if arr.flat[i] <= threshold:
                    arr.flat[i] = value

    @staticmethod
    @jit(nopython=True)
    def set_eq(arr, target, value):
        """arr[arr == target] = value"""
        for i in range(arr.size):
            if arr.flat[i] == target:
                arr.flat[i] = value

    @staticmethod
    @jit(nopython=True)
    def set_neq(arr, target, value):
        """arr[arr != target] = value"""
        for i in range(arr.size):
            if arr.flat[i] != target:
                arr.flat[i] = value

    @staticmethod
    @jit(nopython=True)
    def set_range(arr, low, high, value):
        """arr[low <= arr <= high] = value"""
        for i in range(arr.size):
            if low <= arr.flat[i] <= high:
                arr.flat[i] = value

    @staticmethod
    @jit(nopython=True)
    def set_outside_range(arr, low, high, value):
        """arr[arr < low or arr > high] = value"""
        for i in range(arr.size):
            v = arr.flat[i]
            if v < low or v > high:
                arr.flat[i] = value

    @staticmethod
    @jit(nopython=True)
    def set_where_mask(arr, mask, value):
        """arr[mask == True] = value，mask 必须与 arr 形状相同且为布尔类型（原地）"""
        # 注意：mask 本身需要预先存在，但不会额外创建临时布尔数组（用户可能已有 mask）
        for i in range(arr.size):
            if mask.flat[i]:
                arr.flat[i] = value

    @staticmethod
    @jit(nopython=True)
    def set_where_cond(arr, value, cond_func):
        """
        通用条件赋值，cond_func 是一个接受标量并返回 bool 的 Numba 函数
        用法示例见下方说明
        """
        for i in range(arr.size):
            if cond_func(arr.flat[i]):
                arr.flat[i] = value

    @staticmethod
    @jit(nopython=True)
    def where(target, condition_matrix, value):
        """
        实现 target[condition_matrix > 0] = value
        Parameters:
            target: 要被修改的矩阵（原地修改）
            condition_matrix: 条件矩阵（只读取，不修改）
            value: 赋值标量
        """
        # 假设都是二维矩阵，展平处理更快
        n = target.size
        target_flat = target.ravel()
        cond_flat = condition_matrix.ravel()
        for i in range(n):
            if cond_flat[i] > 0:
                target_flat[i] = value

    @staticmethod
    @jit(nopython=True)
    def where_gt(target, condition_matrix, threshold, value, equal:bool=False):
        """target[condition_matrix > threshold] = value"""
        n = target.size
        target_flat = target.ravel()
        cond_flat = condition_matrix.ravel()
        for i in range(n):
            if not equal:
                if cond_flat[i] > threshold:
                    target_flat[i] = value
            else:
                if cond_flat[i] >= threshold:
                    target_flat[i] = value

    @staticmethod
    @jit(nopython=True)
    def where_lt(target, condition_matrix, threshold, value, equal:bool=False):
        """target[condition_matrix < threshold] = value"""
        n = target.size
        target_flat = target.ravel()
        cond_flat = condition_matrix.ravel()
        for i in range(n):
            if not equal:
                if cond_flat[i] < threshold:
                    target_flat[i] = value
            else:
                if cond_flat[i] <= threshold:
                    target_flat[i] = value

    @staticmethod
    @jit(nopython=True)
    def where_eq(target, condition_matrix, target_value, value):
        """target[condition_matrix == target_value] = value"""
        n = target.size
        target_flat = target.ravel()
        cond_flat = condition_matrix.ravel()
        for i in range(n):
            if cond_flat[i] == target_value:
                target_flat[i] = value

    @staticmethod
    @jit(nopython=True)
    def where_neq(target, condition_matrix, target_value, value):
        """target[condition_matrix != target_value] = value"""
        n = target.size
        target_flat = target.ravel()
        cond_flat = condition_matrix.ravel()
        for i in range(n):
            if cond_flat[i] != target_value:
                target_flat[i] = value

    @staticmethod
    @jit(nopython=True)
    def where_range(target, condition_matrix, low, high, value):
        """target[low <= condition_matrix <= high] = value"""
        n = target.size
        target_flat = target.ravel()
        cond_flat = condition_matrix.ravel()
        for i in range(n):
            v = cond_flat[i]
            if low <= v <= high:
                target_flat[i] = value