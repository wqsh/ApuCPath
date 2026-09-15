from ApuCellGlobal.MSApuCellExternalPackge import *

from ApuCellGlobal.GeneralSystem.MSLogging import CLogging
from ApuCellGlobal.XML.MSDataXml import CDataXmlRoot, CDataXmlElementNode, CDataXmlElement, EnumTypeLCMXML, EnumNodeType
from ApuCellGlobal.XML.MSOperatorDataXml import COperatorDataXmlElementNode, COperatorDataXmlRoot

from ApuCellGlobal.MSTool import toolGetDelimiterIndexList

class CFunctionParseXML:

    @staticmethod
    def ParseXML(path_xml:str, xml_type:int=EnumTypeLCMXML.Regular.value)->CDataXmlRoot:

        if not os.path.exists(path_xml):
            CLogging.LogForLackPath(path_xml)
        root:CDataXmlRoot = None
        if EnumTypeLCMXML.Regular.value == xml_type:

            root = CFunctionParseXML.ParseRegularXML(path_xml)

        elif EnumTypeLCMXML.MMI.value == xml_type:

            root = CFunctionParseXML.ParseMMIXML(path_xml)

        return root
    @staticmethod
    def ParseRegularXML(path_xml:str)->CDataXmlRoot:

        if not os.path.exists(path_xml):
            CLogging.LogForLackPath(path_xml)

        with open(path_xml, 'r') as f:
            buffer = f.readlines()

        line_idx = 0
        next_node_type:int = EnumNodeType.Null.value
        last_node: CDataXmlElementNode = None
        root:CDataXmlRoot = None
        for line in buffer:
            line = line.strip()
            if root is None:
                tag = CFunctionParseXML.GetRootTag(line)
                attribute_dict = CFunctionParseXML.GetAttributeDict(line, EnumTypeLCMXML.Regular.value)
                root = COperatorDataXmlRoot.CreateXmlRoot(tag, attribute_dict, xml_type=EnumTypeLCMXML.Regular.value)
            else:
                if line.replace(" ", "").startswith('</'):
                    next_node_type = EnumNodeType.Parent.value
                    last_node = last_node.parent_node
                else:
                    xml_element = CFunctionParseXML.GetNodeElement(line, EnumTypeLCMXML.Regular.value)
                    cur_node: CDataXmlElementNode = CDataXmlElementNode(xml_element)
                    if EnumNodeType.Null.value == next_node_type:
                        COperatorDataXmlRoot.AddNode(root, cur_node)
                    elif EnumNodeType.Sibling.value == next_node_type:
                        COperatorDataXmlElementNode.AddNode(last_node, cur_node)
                    elif EnumNodeType.Child.value == next_node_type:
                        COperatorDataXmlElementNode.AddChildNode(last_node, cur_node)
                    elif EnumNodeType.Parent.value == next_node_type:
                        COperatorDataXmlElementNode.AddNode(last_node, cur_node)
                    else:
                        CLogging.LogError("Unknown XML node ! ")
                    last_node = cur_node
                    flag_num: int = line.count('<')
                    if flag_num == 1:
                        if line.find('</') > 0 or line.endswith("/>"):
                            # 节点结束，有可能是，下一个节点可能是父节点也可能是兄弟节点，按兄弟节点假设
                            next_node_type = EnumNodeType.Sibling.value
                        else:
                            # 子节点
                            next_node_type = EnumNodeType.Child.value
                    elif flag_num == 2:
                        # 兄弟节点
                        next_node_type = EnumNodeType.Sibling.value
                    else:
                        CLogging.LogError("XML format error ! ")

            line_idx += 1
        return root

    @staticmethod
    def ParseMMIXML(path_xml:str)->CDataXmlRoot:

        if not os.path.exists(path_xml):
            CLogging.LogForLackPath(path_xml)

        with open(path_xml, 'r') as f:
            buffer = f.readlines()

        line_idx = 0
        next_node_type: int = EnumNodeType.Null.value
        last_node: CDataXmlElementNode = None
        root: CDataXmlRoot = COperatorDataXmlRoot.CreateXmlRoot("root", {}, xml_type=EnumTypeLCMXML.MMI.value)

        for line in buffer:
            line = line.strip()
            if line.replace(" ", "").startswith('</'):
                next_node_type = EnumNodeType.Parent.value
                last_node = last_node.parent_node
            else:
                xml_element = CFunctionParseXML.GetNodeElement(line, EnumTypeLCMXML.MMI.value)
                cur_node: CDataXmlElementNode = CDataXmlElementNode(xml_element)

                if EnumNodeType.Null.value == next_node_type:
                    COperatorDataXmlRoot.AddNode(root, cur_node)
                elif EnumNodeType.Sibling.value == next_node_type:
                    COperatorDataXmlElementNode.AddNode(last_node, cur_node)
                elif EnumNodeType.Child.value == next_node_type:
                    COperatorDataXmlElementNode.AddChildNode(last_node, cur_node)
                elif EnumNodeType.Parent.value == next_node_type:
                    COperatorDataXmlElementNode.AddNode(last_node, cur_node)
                else:
                    CLogging.LogError("Unknown XML node ! ")
                last_node = cur_node

                if line.replace("  ", "").startswith("</"):
                    # 子节点结束
                    next_node_type = EnumNodeType.Parent.value
                elif line.endswith("/>"):
                    # 兄弟节点结束
                    next_node_type = EnumNodeType.Sibling.value
                elif line.endswith(">"):
                    # 子节点
                    next_node_type = EnumNodeType.Child.value
                else:
                    CLogging.LogError("XML format error ! ")
            line_idx += 1
        return root

    @staticmethod
    def GetRootTag(line_str:str)->str:
        tag = line_str.replace("<?", "").replace("?>", "").split(' ')[0]
        return tag

    @staticmethod
    def GetNodeTag(line_str:str, xml_type:int)->str:
        tag = ""
        if EnumTypeLCMXML.Regular.value == xml_type:
            list_tag_start_idx = toolGetDelimiterIndexList(line_str, "<")
            list_tag_end_idx = toolGetDelimiterIndexList(line_str, ">")
            if len(list_tag_start_idx) != len(list_tag_end_idx):
                CLogging.LogError("len(list_tag_start_idx) != len(list_tag_end_idx)")
            if len(list_tag_start_idx) <= 0:
                CLogging.LogError("Tag format error !")
            elif len(list_tag_start_idx) == 1:
                if line_str[list_tag_start_idx[0] + 1: list_tag_end_idx[0]].find('=') == 0:
                    if line_str[list_tag_start_idx[0] + 1] == '\\':
                        tag += line_str[list_tag_start_idx[0] + 2: list_tag_end_idx[0]]
                    else:
                        tag += line_str[list_tag_start_idx[0] + 1: list_tag_end_idx[0]]
                else:
                    tag += line_str[list_tag_start_idx[0] + 1: list_tag_end_idx[0]].split(' ')[0]
            elif len(list_tag_start_idx) == 2:
                tag += line_str[list_tag_start_idx[0] + 1: list_tag_end_idx[0]]
            else:
                CLogging.LogError("Tag format error !")
        elif EnumTypeLCMXML.MMI.value == xml_type:
            tag += line_str.split(' ')[0].split("<")[1]
        return tag

    @staticmethod
    def GetNodeText(line_str:str)->str:
        text = line_str.split('>')[1].split('<')[0]
        return text

    @staticmethod
    def GetAttributeDict(line_str:str, xml_type:int)->dict:

        attribute_dict:dict = {}
        if EnumTypeLCMXML.Regular.value == xml_type:
            line_str = line_str.replace("<?", "").replace("?>", "")
            line_str = line_str.replace("</", "").replace("/>", "")
            line_str = line_str.replace("<", "").replace(">", "")
        elif EnumTypeLCMXML.MMI.value == xml_type:
            line_str = line_str.replace("</", "").replace("/>", "")
            line_str = line_str.replace("<", "").replace(">", "")
        list_idx = toolGetDelimiterIndexList(line_str, "=")
        list_line = []
        for idx in list_idx:
            key_idx = idx
            value_idx = idx
            while key_idx > 0:
                if line_str[key_idx] == " ":
                    break
                key_idx -= 1
            while (value_idx < len(line_str)):
                if line_str[value_idx] == " ":
                    break
                value_idx += 1
            list_line.append(line_str[key_idx + 1: value_idx])
        # list_line = line_str.split(' ')

        for i in range(len(list_line)):
            if len(list_line[i]) == 0:
                continue
            attribute_dict[list_line[i].split('=')[0]] = list_line[i].split('=')[1]

        return attribute_dict

    @staticmethod
    def GetNodeElement(line_str:str, xml_type:int)->CDataXmlElement:

        tag = CFunctionParseXML.GetNodeTag(line_str, xml_type)
        attribute_dict = CFunctionParseXML.GetAttributeDict(line_str, xml_type)
        text = CFunctionParseXML.GetNodeText(line_str)
        xml_element:CDataXmlElement = CDataXmlElement(tag, attribute_dict, text, xml_type)
        return xml_element
