import unittest
from flat_bug.predictor import TensorPredictions
import numpy as np
import os
from glob import glob

TEST_DIR = os.path.dirname(__file__)

SERIALISED_TENSOR_PREDS = os.path.join(TEST_DIR, "assets/metadata_ALUS_Non-miteArachnids_Unknown_2020_11_03_4545_UUID_XXXX.json")

class TestTensorPredictions(unittest.TestCase):
    def test_load(self):
        tp = TensorPredictions()
        tp.load(SERIALISED_TENSOR_PREDS)
        print("TensorPrediction.load test passed")

    def test_save(self):
        tp = TensorPredictions()
        tp = tp.load(SERIALISED_TENSOR_PREDS)
        save_dir = tp.save(TEST_DIR, mask_crops=True)
        for file in glob(os.path.join(save_dir, "**")):
            if os.path.isdir(file):
                [os.remove(f) for f in glob(os.path.join(file, "**"))]
                os.rmdir(file)
            else:
                os.remove(file)
        os.rmdir(save_dir)
        print("TensorPrediction.save test passed")

if __name__ == '__main__':
    unittest.main()