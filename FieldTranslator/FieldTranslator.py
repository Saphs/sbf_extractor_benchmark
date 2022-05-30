from abc import ABC, abstractmethod

from pandas import DataFrame


class FieldTranslator(ABC):

    @abstractmethod
    def append_translated(self, df: DataFrame) -> DataFrame:
        """
        Appends n translated series to a known type of data frame
        """
        pass
