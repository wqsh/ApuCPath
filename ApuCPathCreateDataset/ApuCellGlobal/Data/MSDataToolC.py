from ApuCellGlobal.MSApuCellExternalPackge import *

class CDataShapeRectAngleInfo:

    def __init__(self, seed_x:int, seed_y:int, info:np.array):

        self.seed_x:int = int(seed_x)
        self.seed_y:int = int(seed_y)

        # mid_point_x = int((int(info[0][0]) + int(info[0][1])) / 2)
        # mid_point_y = int((int(info[0][2]) + int(info[0][3])) / 2)
        #
        # length = int((int(info[0][1]) - int(info[0][0]) + int(info[0][3]) - int(info[0][2])) / 2 / 2)

        self.x_min:int = int(info[0][0])
        self.x_max:int = int(info[0][1])
        self.y_min:int = int(info[0][2])
        self.y_max:int = int(info[0][3])

        # self.x_min:int = mid_point_x - length
        # self.x_max:int = mid_point_x + length
        # self.y_min:int = mid_point_y - length
        # self.y_max:int = mid_point_y + length

        self.perimeter:int = int(info[0][4])
        self.area:int = int(info[0][5])
        self.rectangle_area:int = (self.y_max - self.y_min) * (self.x_max - self.x_min)


class CDataShapeRectAngleApex:

    def __init__(self, info:list):


        self.seed_x:int = int(info[1])
        self.seed_y:int = int(info[0])

        self.x_min:int = int(info[3])
        self.x_max:int = int(info[5])
        self.y_min:int = int(info[2])
        self.y_max:int = int(info[4])

        self.x_mid:int = int((self.x_min + self.x_max) / 2)
        self.y_mid:int = int((self.y_min + self.y_max) / 2)