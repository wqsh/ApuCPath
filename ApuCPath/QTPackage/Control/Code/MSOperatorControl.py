from QTPackage.Control.Code.MSDataControl import CDataControlLocInfo, CDataControlSize

class COperatorDataControlSize:

    @staticmethod
    def Update(control_size:CDataControlSize, width, height):

        control_size.width = int(width)
        control_size.height = int(height)
        control_size.half_width = int((control_size.width - 1) / 2) - 1
        control_size.half_height = int((control_size.height - 1) / 2) - 1

class COperatorDataControlLoc:

    @staticmethod
    def IsInControl(control_loc_info:CDataControlLocInfo, *args)->bool:
        '''

        :param control_loc_info:
        :param args: [CDataLoc], [x, y]
        :return:
        '''
        control_x_min = control_loc_info.x
        control_x_max = control_loc_info.x + control_loc_info.width
        control_y_min = control_loc_info.y
        control_y_max = control_loc_info.y + control_loc_info.height
        if len(args) == 1:
            loc = args[0]
            if control_x_min < loc.x < control_x_max and control_y_min < loc.y < control_y_max:
                return True
            else:
                return False
        elif len(args) == 2:
            x = args[0]
            y = args[1]
            if control_loc_info.x < x < control_x_max and control_y_min < y < control_y_max:
                return True
            else:
                return False
        else:
            return False
