from enum import Enum, unique

@unique
class EnumFlow(Enum):

    NormalDatasetForTrain = 0
    PredictDatasetForPredict = 1
    TrainDatasetValidDatasetForTrain = 2
    TestDatasetForPredict = 3