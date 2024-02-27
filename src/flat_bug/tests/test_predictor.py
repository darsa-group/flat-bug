import unittest
from flat_bug.predictor import TensorPredictions
import numpy as np
import os

TEST_DIR = os.path.dirname(__file__)

SERIALISED_TENSOR_PREDS = os.path.join(TEST_DIR, "assets/metadata_ALUS_Non-miteArachnids_Unknown_2020_11_03_4545_UUID_XXXX.json")

class TestTensorPredictions(unittest.TestCase):
    def test_load(self):
        tp = TensorPredictions()
        tp.load(SERIALISED_TENSOR_PREDS)