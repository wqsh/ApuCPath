from ApuCellGlobal.XML.MSDataXml import CDataXmlRoot
from ApuCellGlobal.XML.MSOperatorDataXml import COperatorDataXmlElementNode
from ApuCellGlobal.XML.MMI.MSDataMMIXMLInfo import EnumApuCellTFMMITypeShape

class CFunctionExtractMMIXMLInfo:

    @staticmethod
    def GetRectAngleShape(root_xml:CDataXmlRoot, shape_idx:int):
        if shape_idx >= EnumApuCellTFMMITypeShape.Num.value:
            return
        list_info = []
        for i in range(COperatorDataXmlElementNode.GetNodeNum(root_xml.xml_node_head.child_node)):
            group_node = COperatorDataXmlElementNode.GetNode(root_xml.xml_node_head.child_node, i)
            if group_node.element.tag == "Group":
                if group_node.child_node is not None:
                    for j in range(COperatorDataXmlElementNode.GetNodeNum(group_node.child_node)):
                        shape_node = COperatorDataXmlElementNode.GetNode(group_node.child_node, j)
                        if int(shape_node.element.attribute["Type"]) == shape_idx:
                            list_info.append(shape_node)
