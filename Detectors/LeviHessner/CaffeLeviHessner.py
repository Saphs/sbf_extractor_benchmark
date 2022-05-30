import sys

import pandas as pd
from pandas import Series, DataFrame

from DataSets.DataSet import DataSet
from Detectors.Detector import Detector

sys.path.insert(1, '/mnt/f/BSClassifiers/caffe/python')
import caffe


class CaffeLeviHessner(Detector):

    def __init__(self, model_path: str):
        self.img_column = 'img_path'
        self.age_name = 'LeviCNN_age'
        self._age_list = ['(0, 2)', '(4, 6)', '(8, 12)', '(15, 20)', '(25, 32)', '(38, 43)', '(48, 53)', '(60, 100)']

        self._model_base_path = model_path
        self._caffe_model = self._model_base_path + "age_net.caffemodel"
        self._proto_file = self._model_base_path + "deploy.prototxt"
        self._mean_file = self._model_base_path + "mean.binaryproto"

        self._proto_data = open(self._mean_file, "rb").read()
        self._a = caffe.io.caffe_pb2.BlobProto.FromString(self._proto_data)
        self._mean = caffe.io.blobproto_to_array(self._a)[0]

        self._classifier = caffe.Classifier(self._proto_file, self._caffe_model,
                                   mean=self._mean,
                                   channel_swap=(2, 1, 0),
                                   raw_scale=255,
                                   image_dims=(256, 256))
        self._i = 0

    def name(self):
        return "LeviCNN"

    def detect_sample(self, row: Series) -> Series:
        age = self._classify(row[self.img_column])
        row_extension = Series([age], index=[self.age_name])
        return pd.concat([row, row_extension])

    def _classify(self, path: str) -> str:
        input_image = caffe.io.load_image(path)
        prediction = self._classifier.predict([input_image])
        age = self._age_list[prediction[0].argmax()]

        self._i += 1
        print(f"{self._i} -> {age}", end="\r")

        return age

    def detect(self, data: DataSet) -> DataFrame:
        data.df()[self.age_name] = data.df().apply(lambda row: self._classify(row[self.img_column]), axis=1)
        return data.df()
