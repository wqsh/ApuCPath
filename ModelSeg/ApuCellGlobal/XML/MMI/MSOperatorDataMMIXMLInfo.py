from ApuCellGlobal.XML.MSDataXml import CDataXmlElementNode, CDataXmlRoot
from ApuCellGlobal.XML.MSOperatorDataXml import COperatorDataXmlElementNode
from ApuCellGlobal.XML.MMI.MSDataMMIXMLInfo import CDataMMIXMLBoundaryInfo, CDataMMIXMLShapeInfo, CDataMMIXMLInfo, CDataMMIXMLGroupInfo, CDataMMIXMLSlideInfo, CDataMMIXMLStageInsertsInfo

from ApuCellGlobal.GeneralSystem.MSLogging import CLogging
class COperatorGetRouteInfoFromXMLNode:

    @staticmethod
    def GetBoundaryInfo(input_node:CDataXmlElementNode):

        node_attribute = input_node.element.attribute
        left = float(node_attribute["BoundingBoxLeft"].replace('"', ""))
        right = float(node_attribute["BoundingBoxRight"].replace('"', ""))
        top = float(node_attribute["BoundingBoxTop"].replace('"', ""))
        bottom = float(node_attribute["BoundingBoxBottom"].replace('"', ""))
        return CDataMMIXMLBoundaryInfo(left, right, top, bottom)

    @staticmethod
    def GetCellsParent(input_node:CDataXmlElementNode):

        return input_node.element.attribute["CellsParent"]

    @staticmethod
    def GetName(input_node:CDataXmlElementNode):

        return input_node.element.attribute["Name"]

    @staticmethod
    def GetShapeType(input_node:CDataXmlElementNode):
        return int(input_node.element.attribute["Type"].replace('"', ''))

    @staticmethod
    def GetNumber(input_node:CDataXmlElementNode):

        return int(input_node.element.attribute["NumberOfSlides"].replace('"', ''))

    @staticmethod
    def GetShapeInfo(input_node:CDataXmlElementNode)->CDataMMIXMLShapeInfo:

        cell_parent = COperatorGetRouteInfoFromXMLNode.GetCellsParent(input_node)
        shape_rectangle_boundary = COperatorGetRouteInfoFromXMLNode.GetBoundaryInfo(input_node)
        shape_type = COperatorGetRouteInfoFromXMLNode.GetShapeType(input_node)
        shape_info = CDataMMIXMLShapeInfo(shape_type, cell_parent, shape_rectangle_boundary)

        return shape_info

    @staticmethod
    def GetMMIRouteInfo(input_xml_root:CDataXmlRoot)->CDataMMIXMLInfo:
        CLogging.LogToUser("Read xml file ... ")
        mmi_xml_info = CDataMMIXMLInfo()
        for i in range(COperatorDataXmlElementNode.GetNodeNum(input_xml_root.xml_node_head.child_node)):
            slide_node = COperatorDataXmlElementNode.GetNode(input_xml_root.xml_node_head.child_node, i)
            if slide_node.element.tag == "StageInserts":
                stage_inserts_boundary_info = COperatorGetRouteInfoFromXMLNode.GetBoundaryInfo(slide_node)
                stage_inserts_info = CDataMMIXMLStageInsertsInfo(stage_inserts_boundary_info, int(COperatorGetRouteInfoFromXMLNode.GetNumber(slide_node)))
                mmi_xml_info.stage_inserts = stage_inserts_info

            elif slide_node.element.tag == "Slide":
                slide_boundary_info = COperatorGetRouteInfoFromXMLNode.GetBoundaryInfo(slide_node)
                slide_name = COperatorGetRouteInfoFromXMLNode.GetName(slide_node)
                slide_info = CDataMMIXMLSlideInfo(slide_name, slide_boundary_info)
                mmi_xml_info.slide_info.append(slide_info)
                mmi_xml_info.slide_idx.append(slide_name)

            elif slide_node.element.tag == "Group":
                group_info = CDataMMIXMLGroupInfo(COperatorGetRouteInfoFromXMLNode.GetName(slide_node))
                group_name = COperatorGetRouteInfoFromXMLNode.GetName(slide_node)
                shape_node_head = slide_node.child_node
                mmi_xml_info.group_info.append(group_info)
                mmi_xml_info.group_idx.append(group_name)
                if shape_node_head is not None:
                    for j in range(COperatorDataXmlElementNode.GetNodeNum(shape_node_head)):
                        shape_node = COperatorDataXmlElementNode.GetNode(shape_node_head, j)
                        shape_info = COperatorGetRouteInfoFromXMLNode.GetShapeInfo(shape_node)
                        point_node_head = shape_node.child_node
                        shape_slide_idx = mmi_xml_info.slide_idx.index(shape_info.cell_parent)
                        shape_info.from_slide_idx = shape_slide_idx
                        shape_info.from_group_idx = len(mmi_xml_info.group_info) - 1
                        shape_info.slide_idx = len(mmi_xml_info.shape_info)
                        mmi_xml_info.shape_info.append(shape_info)
                        group_info.list_shape_idx.append(len(mmi_xml_info.shape_info))
                        mmi_xml_info.slide_info[shape_slide_idx].list_shape_idx.append(len(mmi_xml_info.shape_info))
                        for k in range(COperatorDataXmlElementNode.GetNodeNum(point_node_head)):
                            x = float(COperatorDataXmlElementNode.GetNode(point_node_head, k).element.attribute["X"].replace('"', ''))
                            y = float(COperatorDataXmlElementNode.GetNode(point_node_head, k).element.attribute["Y"].replace('"', ''))
                            shape_info.point.append((x, y))
        return mmi_xml_info

class COperatorGetInfoFromMMIXMLInfo:

    @staticmethod
    def GetSlideNames(input_MMIXML_info:CDataMMIXMLInfo)->list:
        return input_MMIXML_info.slide_idx

    @staticmethod
    def GetSlideInfo(input_MMIXML_info:CDataMMIXMLInfo, slide_name:str)->CDataMMIXMLSlideInfo:
        if slide_name in input_MMIXML_info.slide_idx:
            return input_MMIXML_info.slide_info[input_MMIXML_info.slide_idx.index(slide_name)]
        else:
            return None

    @staticmethod
    def SelectMMIXMLShapeByType(input_MMIXML_info:CDataMMIXMLInfo, select_type_list:list):
        select_list = []
        for i in range(len(input_MMIXML_info.shape_info)):
            if input_MMIXML_info.shape_info[i].type in select_type_list:
                select_list.append(input_MMIXML_info.shape_info[i])
        return select_list

    @staticmethod
    def SelectMMIXMLShapeBySlide(input_MMIXML_info:CDataMMIXMLInfo, select_slide:list):
        select_list = []
        select_idx_list = []
        for slide_idx, slide_name in enumerate(select_slide):
            for i in range(len(input_MMIXML_info.slide_idx)):
                if input_MMIXML_info.slide_idx[i] == slide_name:
                    select_idx_list.append(i)
        for i in range(len(input_MMIXML_info.shape_info)):
            if input_MMIXML_info.shape_info[i].from_slide_idx in select_idx_list:
                select_list.append(input_MMIXML_info.shape_info[i])
        return select_list
