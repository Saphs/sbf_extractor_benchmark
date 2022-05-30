from typing import List, Tuple

from DataSets.Adience import Adience
from DataSets.DataSet import DataSet
from Detectors.Detector import Detector
from Detectors.LeviHessner.CaffeLeviHessner import CaffeLeviHessner
from Detectors.Ror34 import Ror34
from FieldTranslator.AdienceForLevi import AdienceForLevi

ADIENCE_PATH = "/mnt/f/BSDatasets/Adience"
LEVI_MODEL_PATH = "/mnt/c/Users/Tiz/PycharmProjects/SBFBenchmark/Detectors/LeviHessner/model/"
DEFAULT_OUT = "/mnt/c/Users/Tiz/PycharmProjects/SBFBenchmark/out"

class SbfBenchmarkCore(object):

    def __init__(self):
        self._known_datasets: List[Tuple[str, DataSet]] = [
            ("Adience", Adience(ADIENCE_PATH))
        ]

        self._known_detectors: List[Tuple[str, Detector]] = [
            ("RoR-34", Ror34()),
            ("LeviCNN", CaffeLeviHessner(LEVI_MODEL_PATH))
        ]

    def list_datasets(self):
        print(list(map(lambda t: t[0], self._known_datasets)))

    def get_dataset(self, name: str):
        def _is_name_eq(dataset_entry: Tuple[str, DataSet]):
            return dataset_entry[0] == name

        return list(filter(_is_name_eq, self._known_datasets))[0][1]

    def run(self, detector: Detector, dataset: DataSet, out: str = DEFAULT_OUT):
        for i in range(dataset.chunk_count() - 1):
            dataset.load(i)
            chunk_result = detector.detect(dataset)
            p = f"{out}/{dataset.name()}_age_{detector.name()}/{dataset.name()}_chunk_{i}.csv"
            chunk_result.to_csv(p)


if __name__ == '__main__':
    core = SbfBenchmarkCore()
    core.list_datasets()
    ds = core.get_dataset('Adience')
    ds.load(0)
    print(f"{ds.sample()=}")