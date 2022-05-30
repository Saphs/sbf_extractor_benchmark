from abc import ABC, abstractmethod
from pandas import DataFrame, Series


class DataSet(ABC):

    @abstractmethod
    def name(self):
        pass

    @abstractmethod
    def sample(self) -> Series:
        """
        Should provide a single randomly selected entry of the dataset.
        """
        pass

    @abstractmethod
    def load(self, chunk: int) -> bool:
        """
        Should try to load the data into a pandas dataframe.
        """
        pass

    @abstractmethod
    def chunk_count(self) -> int:
        pass

    @abstractmethod
    def df(self) -> DataFrame:
        """
        Should return a dataframe representing the data.
        """
        pass
