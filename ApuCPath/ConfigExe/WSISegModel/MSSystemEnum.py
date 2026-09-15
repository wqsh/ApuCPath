from enum import Enum, unique

@unique
class EnumFlow(Enum):

    NormalDatasetForTrain = 0
    PredictDatasetForPredict = 1
    TrainDatasetValidDatasetForTrain = 2
    TestDatasetForPredict = 3

    OnlyModelPredict = 4
    OnlyMergeWSIPredict = 5
    OnlyMultiChannelWSIMerge = 6
    PredictMergeWSIPredictMultiChannelWSIMerge = 7

    TrainBySetTrainValidDataset = 8