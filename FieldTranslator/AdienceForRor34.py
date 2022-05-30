import re

from pandas import DataFrame, Series

from FieldTranslator.FieldTranslator import FieldTranslator


class AdienceForRor34(FieldTranslator):
    tuple_pattern = re.compile(r'^\((\d+),\s*(\d+)\)$')
    digit_pattern = re.compile(r'^(\d+)$')

    def append_translated(self, df: DataFrame) -> DataFrame:
        df['ror34_gt_age'] = df.apply(lambda row: self._age_translation(row), axis=1)
        df['ror34_gt_gender'] = df.apply(lambda row: self._gender_translation(row), axis=1)
        return df

    def _age_translation(self, row: Series) -> str:
        age_value = row['age']
        tuple_match = self.tuple_pattern.match(age_value)
        digit_match = self.digit_pattern.match(age_value)
        if bool(tuple_match):
            n0 = tuple_match.group(1)
            n1 = tuple_match.group(2)
            # This works 99% cleanly but will mix (48, 53) into the wrong bucket.
            return self._get_age_bucket(int(n0))
        elif bool(digit_match):
            return self._get_age_bucket(int(digit_match.group(1)))
        else:
            return "None"

    def _get_age_bucket(self, x: int) -> str:
        if 0 <= x <= 24:
            return "(0 , 24)"
        elif 25 <= x <= 49:
            return "(25 , 49)"
        elif 50 <= x <= 74:
            return "(50 , 74)"
        elif 75 <= x <= 99:
            return "(75 , 99)"
        elif 91 <= x <= 124:
            return "(91 , 124)"
        else:
            return "None"

    def _gender_translation(self, row: Series) -> str:
        return row['gender']
