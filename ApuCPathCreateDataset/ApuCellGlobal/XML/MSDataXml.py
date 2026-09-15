from ApuCellGlobal.MSApuCellExternalPackge import *

@unique
class EnumNodeType(Enum):

    Null = 0
    Sibling = 1
    Child = 2
    Parent = 3

@unique
class EnumTypeLCMXML(Enum):

    Regular = 1
    MMI = 2

class CRegularRoot:

    start_tag:str = "?xml"
    end_tag:str = "?"

    attribute:dict = {
        "version":"1.0",
        "encoding":"utf-8",
    }

class CMMIRoot:

    start_tag:str = "GroupsAndShapes"
    end_tag:str = ""

    attribute:dict = {
        "xmlns":"http://molecular-machines.com"
    }

class CDataXmlElement:

    def __init__(self, tag:str, attribute:dict={}, text:str="", xml_type:int=EnumTypeLCMXML.Regular.value):

        self.tag:str = tag
        self.text:str = text
        self.attribute:dict = copy.deepcopy(attribute)

        self.xml_type:int = xml_type

class CDataXmlElementNode:

    def __init__(self, xml_element:CDataXmlElement):

        self.next_node:CDataXmlElementNode = None # 下一个兄弟节点

        self.head_node:CDataXmlElementNode = None # 当前节点的头节点

        self.child_node:CDataXmlElementNode = None # 子节点

        self.parent_node:CDataXmlElementNode = None # 父节点

        self.element:CDataXmlElement = xml_element

        self.state:bool = False  # False表示未被使用过，True表示已被加入到某个节点中

        self.grade:int=0 # 处于第几级（缩进数目）

class CDataXmlRoot:

    def __init__(self, tag_name:str="xml", attribute:dict={}, xml_type:int=EnumTypeLCMXML.Regular.value):

        self.xml_type:int = xml_type
        self.tag_name:str = tag_name
        self.xml_node_head:CDataXmlElementNode = None
        self.attribute = copy.deepcopy(attribute)


