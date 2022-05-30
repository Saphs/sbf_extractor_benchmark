import re

from pandas import DataFrame, Series

from FieldTranslator.FieldTranslator import FieldTranslator


class AdienceForLevi(FieldTranslator):
    digit_pattern = re.compile(r'^(\d+)$')

    def append_translated(self, df: DataFrame) -> DataFrame:
        df['LeviCNN_gt_age'] = df.apply(lambda row: self._validate(row), axis=1)
        return df

    def _validate(self, row: Series):
        digit_match = self.digit_pattern.match(row['age'])
        if bool(digit_match):
            return self._get_age_bucket(int(digit_match.group(1)))
        elif row['age'] == "(38, 48)":
            return "None"
        else:
            return row['age']

    def _get_age_bucket(self, x: int) -> str:
        if 0 <= x <= 2:
            return "(0 , 2)"
        elif 4 <= x <= 6:
            return "(4 , 6)"
        elif 8 <= x <= 12:
            return "(8 , 12)"
        elif 15 <= x <= 20:
            return "(15 , 20)"
        elif 25 <= x <= 32:
            return "(25 , 32)"
        elif 38 <= x <= 43:
            return "(38 , 43)"
        elif 48 <= x <= 53:
            return "(48 , 53)"
        elif 60 <= x <= 100:
            return "(60 , 100)"
        else:
            return "None"