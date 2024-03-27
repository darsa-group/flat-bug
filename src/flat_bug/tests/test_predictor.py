import unittest

import os
import tempfile
from glob import glob

import numpy as np

from flat_bug.predictor import TensorPredictions

SERIALISED_TENSOR_PREDS = os.path.join(os.path.dirname(__file__), "assets/metadata_ALUS_Non-miteArachnids_Unknown_2020_11_03_4545_UUID_XXXX.json")
N_PREDICTIONS = 11

class TestTensorPredictions(unittest.TestCase):
    def test_load(self):
        tp = TensorPredictions()
        tp.load(SERIALISED_TENSOR_PREDS)
        self.assertEqual(len(tp), N_PREDICTIONS)

    def test_save(self):
        tp = TensorPredictions()
        tp = tp.load(SERIALISED_TENSOR_PREDS)
        with tempfile.TemporaryDirectory() as tmp_directory:
            save_dir = tp.save(tmp_directory, mask_crops=True)
            self.assertTrue(os.path.exists(os.path.join(save_dir, "crops")))
            self.assertEqual(len(glob(os.path.join(save_dir, "crops", "*"))), N_PREDICTIONS)
            centroid_initial = [i.float().mean(dim=0).numpy() for i in tp.contours]
            centroid_reloaded = [i.float().mean(dim=0).numpy() for i in TensorPredictions().load(glob(os.path.join(save_dir, "metadata*.json"))[0]).contours]
            centroid_initial = np.stack(centroid_initial)
            centroid_reloaded = np.stack(centroid_reloaded)
            abs_diff = np.abs(centroid_initial - centroid_reloaded).max()
            print(f"Reserialization centroid difference: {abs_diff}")
            self.assertTrue(abs_diff < 0.01)

if __name__ == '__main__':
    unittest.main()