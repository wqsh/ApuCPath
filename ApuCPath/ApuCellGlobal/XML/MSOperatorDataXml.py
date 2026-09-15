from ApuCellGlobal.XML.MSDataXml import CDataXmlRoot, CDataXmlElementNode, EnumTypeLCMXML

class COperatorDataXmlElementNode:

    @staticmethod
    def GetNodeNum(xml_node:CDataXmlElementNode)->int:
        if xml_node is None:
            return 0
        node_num:int = 0
        tmp_xml_node = xml_node.head_node
        while tmp_xml_node is not None:
            node_num += 1
            tmp_xml_node = tmp_xml_node.next_node
        return node_num

    @staticmethod
    def GetNode(xml_node:CDataXmlElementNode, idx: int = 0):
        # 默认返回当前节点所在的头节点
        # 索引为当前节点所属的头节点的第几个节点
        search_node = xml_node.head_node
        if idx >= 0:
            for i in range(idx):
                search_node = search_node.next_node
                if search_node is None:
                    return None
            return search_node
        else:
            return None

    @staticmethod
    def AddNode(xml_node:CDataXmlElementNode, add_node:CDataXmlElementNode)->bool:
        ret:bool = False
        if not add_node.state:
            if xml_node.next_node is None:
                xml_node.next_node = add_node

                if xml_node.head_node is None:
                    xml_node.head_node = xml_node
                    xml_node.next_node = add_node
                add_node.head_node = xml_node
                add_node.grade = xml_node.grade
                add_node.parent_node = xml_node.parent_node
                add_node.state = True
                ret = True
            else:
                tmp_node = xml_node
                while tmp_node.next_node is not None:
                    tmp_node = tmp_node.next_node
                tmp_node.next_node = add_node
                add_node.head_node = xml_node.head_node
        return ret

    @staticmethod
    def AddChildNode(xml_node:CDataXmlElementNode, child_node:CDataXmlElementNode)->bool:

        ret:bool = False
        if (not child_node.state):
            if xml_node.child_node is None:
                xml_node.child_node = child_node
                child_node.head_node = child_node
            else:
                COperatorDataXmlElementNode.AddNode(xml_node.child_node, child_node)

            child_node.grade = xml_node.grade + 1
            COperatorDataXmlElementNode.__captainUpdateNodeGrade(child_node)

            child_node.parent_node = xml_node
            child_node.state = True
            ret = True
        return ret

    @staticmethod
    def __captainGetNodeHead2EndStr(xml_node:CDataXmlElementNode, retract:bool=True, blank_num:int=1)->str:
        # 获取当前节点所在的头节点到尾节点的所有xml字符串（包含所有子节点）
        xml_str:str = ""
        search_node:CDataXmlElementNode = xml_node.head_node
        while search_node is not None:
            xml_str += COperatorDataXmlElementNode.GetCurNodeStr(search_node, retract, blank_num)
            search_node = search_node.next_node
        return xml_str

    @staticmethod
    def GetCurNodeStr(xml_node:CDataXmlElementNode, retract:bool=True, blank_num:int=1):
        # 返回当前节点囊括的字符串
        xml_str: str = ""
        xml_str += COperatorDataXmlElementNode.__captainGetStartTagStr(xml_node, retract, blank_num)
        if xml_node.child_node is None:
            xml_str += f"{xml_node.element.text}"
            xml_str += COperatorDataXmlElementNode.__captainGetEndTagStr(xml_node, False, blank_num)
        else:
            xml_str += COperatorDataXmlElementNode.__captainGetNodeHead2EndStr(xml_node.child_node)
            xml_str += COperatorDataXmlElementNode.__captainGetEndTagStr(xml_node, True, blank_num)

        return xml_str
    @staticmethod
    def GetNodeElementAttributeStr(xml_node:CDataXmlElementNode)->str:
        attribute_str:str = ""
        for attribute_key in xml_node.element.attribute.keys():
            attribute_str += f'{attribute_key}={xml_node.element.attribute[attribute_key]} '
        attribute_str = attribute_str.strip()
        return attribute_str

    @staticmethod
    def GetRetractStr(xml_node:CDataXmlElementNode, blank_num:int=1)->str:

        retract_str:str = ""
        blank_str:str = ""
        if EnumTypeLCMXML.Regular.value == xml_node.element.xml_type:
            character:str = "\t"
        elif EnumTypeLCMXML.MMI.value == xml_node.element.xml_type:
            character: str = "  "
        else:
            character: str = "\t"
        for i in range(blank_num):

            blank_str += character
        for i in range(xml_node.grade):
            retract_str += blank_str
        return retract_str

    @staticmethod
    def __captainGetStartTagStr(xml_node:CDataXmlElementNode, retract:bool=True, blank_num:int=1)->str:
        xml_node_str:str = ""
        if retract:
            xml_node_str += COperatorDataXmlElementNode.GetRetractStr(xml_node, blank_num)

        attribute_str: str = COperatorDataXmlElementNode.GetNodeElementAttributeStr(xml_node)

        if EnumTypeLCMXML.Regular.value == xml_node.element.xml_type:
            if len(attribute_str) <= 0:
                xml_node_str += f"<{xml_node.element.tag}>"
            else:
                xml_node_str += f"<{xml_node.element.tag} {attribute_str}>"
            if xml_node.child_node is not None:
                xml_node_str += "\n"
        elif EnumTypeLCMXML.MMI.value == xml_node.element.xml_type:
            xml_node_str += f"<{xml_node.element.tag} {attribute_str}"
            if xml_node.child_node is None:
                xml_node_str += "/>\n"
            else:
                xml_node_str += ">\n"
        else:
            pass
        return xml_node_str

    @staticmethod
    def __captainGetEndTagStr(xml_node:CDataXmlElementNode, retract:bool=True, blank_num:int=1)->str:
        xml_node_str: str = ""
        if retract:
            xml_node_str += COperatorDataXmlElementNode.GetRetractStr(xml_node, blank_num)
        if EnumTypeLCMXML.Regular.value == xml_node.element.xml_type:
            xml_node_str += f"</{xml_node.element.tag}>\n"
        elif EnumTypeLCMXML.MMI.value == xml_node.element.xml_type:
            if xml_node.child_node is not None:
                xml_node_str += f"</{xml_node.element.tag}>\n"
        else:
            pass
        return xml_node_str

    @staticmethod
    def __captainUpdateNodeGrade(xml_node:CDataXmlElementNode):

        if xml_node.child_node is None:
            return
        else:
            child_num = COperatorDataXmlElementNode.GetNodeNum(xml_node.child_node)
            xml_node_child_node = xml_node.child_node
            for i in range(child_num):
                xml_node_child_node.grade = xml_node.grade + 1
                if xml_node_child_node.child_node is None:
                    pass
                else:
                    COperatorDataXmlElementNode.__captainUpdateNodeGrade(xml_node_child_node)
                xml_node_child_node = xml_node_child_node.next_node
class COperatorDataXmlRoot:

    @staticmethod
    def CreateXmlRoot(root_name: str, attribute: dict = {}, xml_type: int = EnumTypeLCMXML.Regular.value) -> CDataXmlRoot:
        if EnumTypeLCMXML.Regular.value == xml_type:
            if "version" not in attribute.keys():
                attribute["version"] = "1.0"
            if "encoding" not in attribute.keys():
                attribute["encoding"] = "utf-8"
        elif EnumTypeLCMXML.MMI.value == xml_type:
            pass
        return CDataXmlRoot(root_name, attribute, xml_type)

    @staticmethod
    def AddNode(xml_root: CDataXmlRoot, xml_element_node: CDataXmlElementNode) -> bool:
        # 在最后面添加
        ret: bool = False
        if not xml_element_node.state:
            if xml_root.xml_node_head is None:
                xml_root.xml_node_head = xml_element_node
                xml_element_node.head_node = xml_element_node
            else:
                COperatorDataXmlElementNode.AddNode(xml_root.xml_node_head, xml_element_node)
            xml_element_node.state = True
            ret = True
        return ret

    @staticmethod
    def GetRootNodeStr(xml_root:CDataXmlRoot, retract:bool=True, blank_num:int=1)->str:
        xml_str:str = ""
        xml_str += COperatorDataXmlElementNode.GetCurNodeStr(xml_root.xml_node_head, retract, blank_num)
        return xml_str

    @staticmethod
    def ExportXML(xml_root: CDataXmlRoot) -> str:

        xml_str = ""
        if EnumTypeLCMXML.Regular.value == xml_root.xml_type:
            attribute_str:str = ""
            for attribute_key in xml_root.attribute.keys():
                attribute_str += f' {attribute_key}={xml_root.attribute[attribute_key]}'
            xml_str += f"<?{xml_root.tag_name}{attribute_str}?>\n"
            xml_str += COperatorDataXmlRoot.GetRootNodeStr(xml_root, True, 2)[:-1]
        elif EnumTypeLCMXML.MMI.value == xml_root.xml_type:
            xml_str += COperatorDataXmlRoot.GetRootNodeStr(xml_root, True, 2)
        return xml_str

def op_GetXmlClassVarAttribute(inputClass, add_cite:bool=True):

    AttributeDic = {}
    for attribute_name in dir(inputClass):
        if not attribute_name.startswith("_"):
            if add_cite:
                AttributeDic[attribute_name] = '"' + inputClass.__getattribute__(attribute_name) + '"'
            else:
                AttributeDic[attribute_name] = inputClass.__getattribute__(attribute_name)
    return AttributeDic