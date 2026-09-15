from Core.MSDataPack import CDataPack, EnumDatasetType

from Core.MSTask import CTaskCreatePredictDataset, CTaskCreateTrainDataset, CTaskCreateAssistDataset, CTaskCreateValidDataset

from ApuCellGlobal.GeneralSystem.MSLogging import CLogging
class CFlow:

    def __init__(self, flow_enum:int=0):
        self.__FlowEnum:int = flow_enum

    def GetFlowEnum(self):

        return self.__FlowEnum

class CFlowCreateDataset(CFlow):

    def __init__(self, inpputDP:CDataPack, flow_enum:int=1):
        super(CFlowCreateDataset, self).__init__(flow_enum)
        self.dp = inpputDP

    def run(self):
        # 开始
        if self.dp.cfg.A1_TYPE_DATASET == EnumDatasetType.Normal:
            pass
        elif self.dp.cfg.A1_TYPE_DATASET == EnumDatasetType.Train:
            taskCreateTrainDataset = CTaskCreateTrainDataset(self.dp)
            taskCreateTrainDataset.do()
        elif self.dp.cfg.A1_TYPE_DATASET == EnumDatasetType.Valid:
            taskCreateValidDataset = CTaskCreateValidDataset(self.dp)
            taskCreateValidDataset.do()
        elif self.dp.cfg.A1_TYPE_DATASET == EnumDatasetType.Test:
            CLogging.LogWarning("Unknown flow : %d"%self.dp.cfg.A1_TYPE_DATASET.value)
        elif self.dp.cfg.A1_TYPE_DATASET == EnumDatasetType.Predict:
            taskCreatePredictDataset = CTaskCreatePredictDataset(self.dp)
            taskCreatePredictDataset.do()
        elif self.dp.cfg.A1_TYPE_DATASET == EnumDatasetType.Assist:
            taskCreateAssistDataset = CTaskCreateAssistDataset(self.dp)
            taskCreateAssistDataset.do()
        else:
            CLogging.LogWarning("Unknown flow : %d"%self.dp.cfg.A1_TYPE_DATASET.value)