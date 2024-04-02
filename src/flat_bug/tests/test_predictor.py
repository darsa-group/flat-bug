import unittest

import os, shutil
import tempfile
from glob import glob

import numpy as np
from torchvision.io import read_image

from flat_bug.predictor import TensorPredictions

ASSET_NAME = "ALUS_Non-miteArachnids_Unknown_2020_11_03_4545"
UUID = "XXXX"
SERIALISED_TENSOR_PREDS = os.path.join(os.path.dirname(__file__), f"assets/metadata_{ASSET_NAME}_UUID_{UUID}.json")
N_PREDICTIONS = {
    "XXXX" : 11,
    "ChangeThisTEMPORARY" : 10
}
N_PREDICTIONS = N_PREDICTIONS.get(UUID, None)
if N_PREDICTIONS is None:
    raise ValueError(f"Number of predictions for UUID {UUID} is not known")

class TestTensorPredictions(unittest.TestCase):
    def test_load(self):
        tp = TensorPredictions()
        tp.load(SERIALISED_TENSOR_PREDS)
        self.assertEqual(len(tp), N_PREDICTIONS, msg=f"Number of predictions ({len(tp)}) does not match the expected number of predictions ({N_PREDICTIONS})")

    def test_save(self):
        tp = TensorPredictions()
        tp = tp.load(SERIALISED_TENSOR_PREDS)
        tp.image = read_image(os.path.join(os.path.dirname(__file__), "assets", f"{ASSET_NAME}.jpg")) * 255
        with tempfile.TemporaryDirectory() as tmp_directory:
            save_dir = tp.save(tmp_directory, mask_crops=True)
            self.assertTrue(os.path.exists(os.path.join(save_dir, "crops")))
            crops = glob(os.path.join(save_dir, "crops", "*"))
            n_crops = len(crops)
            # ###### DEBUG ######
            # [shutil.move(c, os.path.join(os.path.dirname(__file__), "assets", os.path.basename(c))) for c in crops]
            # overview = glob(os.path.join(save_dir, "overview*"))[0] 
            # shutil.move(overview, os.path.join(os.path.dirname(__file__), "assets", os.path.basename(overview)))
            # ###################
            self.assertEqual(n_crops, N_PREDICTIONS, msg=f"Number of crops ({n_crops}) saved does not match the expected number of predictions ({N_PREDICTIONS})")
            centroid_initial = [i.float().mean(dim=0).numpy() for i in tp.contours]
            centroid_reloaded = [i.float().mean(dim=0).numpy() for i in TensorPredictions().load(glob(os.path.join(save_dir, "metadata*.json"))[0]).contours]
            centroid_initial = np.stack(centroid_initial)
            centroid_reloaded = np.stack(centroid_reloaded)
            abs_diff = np.abs(centroid_initial - centroid_reloaded).max()
            self.assertTrue(abs_diff < 0.01, msg=f"Centroid difference between initial and reloaded contours ({abs_diff}) is too large")

if __name__ == '__main__':
    unittest.main()