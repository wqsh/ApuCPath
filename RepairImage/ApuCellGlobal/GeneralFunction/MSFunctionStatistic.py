import numpy as np
import matplotlib.pyplot as plt

class CFunctionStatistic:
    @staticmethod
    def Histogram(img: np.array, mask: np.array, show: bool, value_num: int = 256):
        point_in_circle = np.where(mask > 0)
        histogram_list = [0 for i in range(value_num)]
        histogram_data = []
        for idx in range(len(point_in_circle[0])):
            y = point_in_circle[0][idx]
            x = point_in_circle[1][idx]
            histogram_list[img[y, x]] += 1
            histogram_data.append(img[y, x])
        if show:
            plt.figure()
            plt.hist(histogram_data, range=(0, 255), bins=256)
            plt.show()

        return histogram_list