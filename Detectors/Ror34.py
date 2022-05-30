import re

import pandas as pd
from pandas import DataFrame, Series

from DataSets.DataSet import DataSet
from Detectors.Detector import Detector
import requests


class Ror34(Detector):

    def __init__(self):
        self.host = "http://127.0.0.1:5000/predict"
        self.age_pattern = re.compile(r'^.*?(\d+)-(\d+).*$')
        self.age_name = 'ror34_age'
        self.gender_name = 'ror34_gender'
        self.img_column = 'img_path'

    def name(self):
        return "Ror34"

    def detect_sample(self, row: Series) -> Series:
        age = self._estimate_age(row)
        gender = self._estimate_gender(row)
        extension = Series([age, gender], index=[self.age_name, self.gender_name])
        return pd.concat([row, extension])

    def _request_estimation(self, row: Series) -> str:
        response = requests.get(self.host + f"?path={row[self.img_column]}")
        if response.status_code == 200:
            return response.text
        else:
            print(f"unexpected response code: {response}")
            return "N/A"

    def _estimate_age(self, row: Series) -> str:
        txt = self._request_estimation(row)
        match = self.age_pattern.match(txt)
        return f"({match.group(1)} , {match.group(2)})"

    def _estimate_gender(self, row: Series) -> str:
        txt = self._request_estimation(row)
        return txt.split(",")[1].strip()[0]

    def detect(self, data: DataSet) -> DataFrame:
        data.df()[self.age_name] = data.df().apply(lambda row: self._estimate_age(row), axis=1)
        data.df()[self.gender_name] = data.df().apply(lambda row: self._estimate_gender(row), axis=1)
        return data.df()
