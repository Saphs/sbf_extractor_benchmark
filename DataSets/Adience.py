import numpy as np
import os.path
from typing import List, Optional

import pandas
import pandas as pd
from pandas import DataFrame, Series

from DataSets.DataSet import DataSet


class Adience(DataSet):

    def name(self):
        return "Adience"

    _part_paths: List[str] = [
        'fold_0_data.txt',
        'fold_1_data.txt',
        'fold_2_data.txt',
        'fold_3_data.txt',
        'fold_4_data.txt',
    ]

    _relevant_columns: List[str] = [
        'age', 'gender', 'img_path'
    ]

    def __init__(self, path):
        self._df = DataFrame()
        self._base_path = path

    def load(self, chunk: int) -> bool:
        base_p = self._base_path
        if os.path.exists(base_p) and os.path.isdir(base_p):
            print(f"Loading chunk {chunk}/{len(self._part_paths)-1} for Adience dataset...")
            self._df = pd.read_csv(base_p + "/" + self._part_paths[chunk], delimiter="\t")
            self._resolve_img_paths()
            self._prune()
            print(f"Loaded chunk of size: {self._df.shape}")
        else:
            print(f"Failed to load Adience Dataset.")
            return False

    def df(self):
        return self._df

    def chunk_count(self) -> int:
        return len(self._part_paths)

    def _resolve_img_paths(self):
        if self._df.shape[0] > 0:
            self._df['img_path'] = self._df.apply(lambda row: self._to_img_path(row), axis=1)

    def _to_img_path(self, row: Series):
        return f"{self._base_path}/faces/{row['user_id']}/coarse_tilt_aligned_face.{row['face_id']}.{row['original_image']}"

    def _prune(self):
        self._df = self._df.filter(self._relevant_columns)

    def sample(self):
        return self._df.iloc[0]
