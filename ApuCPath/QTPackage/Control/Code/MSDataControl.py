class CDataControlSize:

    def __init__(self, width, height):

        self.width:int = int(width)
        self.height:int = int(height)
        self.half_height: int = int((self.height - 1) / 2) - 1
        self.half_width: int = int((self.width - 1) / 2) - 1
class CDataControlLocInfo:

    def __init__(self, x:int, y:int, width:int, height:int):

        self.x:int = x
        self.y:int = y
        self.width:int = width
        self.height:int = height