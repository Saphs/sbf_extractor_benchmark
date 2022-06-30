from abc import ABC, abstractmethod

from pandas import Series, DataFrame

from DataSets.DataSet import DataSet


class Classifier(ABC):

    @abstractmethod
    def name(self):
        pass

    @abstractmethod
    def detect_sample(self, row: Series) -> Series:
        """
        Run detection on a single element.
        :return: estimation result
        """
        pass

    @abstractmethod
    def detect(self, data: DataSet) -> DataFrame:
        """
        Runs detection over a full data set.

        This results in a DataFrame containing the ground truth provided by the dataset
        and the estimation result calculated by the detector. This can then be analyzed
        in a separate step by the SbfBenchmarkCore.

        :param data: incoming DataSet
        :return: estimation results from processing the whole dataset.
        """
        pass
