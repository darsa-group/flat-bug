import unittest

import os, shutil, re
import tempfile
from glob import glob

import torch
import numpy as np
from torchvision.io import read_image

from flat_bug.predictor import TensorPredictions, Predictor

ASSET_NAME = "ALUS_Non-miteArachnids_Unknown_2020_11_03_4545"
ASSET_DIR = os.path.join(os.path.dirname(__file__), "assets")
UUID = "XXXX"
SERIALISED_TENSOR_PREDS = os.path.join(ASSET_DIR, f"metadata_{ASSET_NAME}_UUID_{UUID}.json")
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


class DummyModel:
    def __init__(self, type : str, asset_dir : str):
        if type not in ["single_scale", "pyramid"]:
            raise ValueError(f"Invalid type {type}")
        self.type = type
        self.asset_dir = asset_dir
        self.index = 1

    def __call__(self, image):
        try:
            out = torch.load(os.path.join(self.asset_dir, f'{self.type}_tps_{self.index}.pt'), map_location=image.device)
        except Exception as e:
            print(f'Failed to load test file "{self.type}_tps_{self.index}.pt" - consider generating the test files with `python3 src/flat_bug/tests/generate_model_outputs.py --model model_snapshots/fb_2024-03-18_large_best.pt --image src/flat_bug/tests/assets/ALUS_Non-miteArachnids_Unknown_2020_11_03_4545.jpg --type both`')
            raise e
        self.index += 1
        return out

    def generate_single_scale_files(self, weights, image):
        dtype, device = image.dtype, image.device
        model = Predictor(model=weights, device=device, dtype=dtype)
        model.DEBUG = True
        model.TIME = True
        model.total_detection_time = 0
        model.total_forward_time = 0
        output = model._detect_instances(image, scale=1, max_scale=False)
        # Rename the files with the pattern "assets/tps_<NUMBER>.pt" to "assets/single_scale_tps_<NUMBER>.pt"
        [shutil.move(f, os.path.join(self.asset_dir, re.sub(r'tps_', "single_scale_tps_", f))) for f in glob("tps_*.pt")]
        # Create a file with the length of the output object as a reference - this is the number of detections in the final object
        with open(os.path.join(self.asset_dir, "single_scale_output_length.txt"), "w") as f:
            f.write(str(len(output)))

    def generate_pyramid_files(self, weights, image, image_path):
        dtype, device = image.dtype, image.device
        model = Predictor(model=weights, device=device, dtype=dtype)
        model.DEBUG = True
        model.TIME = True
        output = model.pyramid_predictions(image, image_path, scale_increment=1/2, scale_before=1, single_scale=False)
        # Rename the files with the pattern "assets/tps_<NUMBER>.pt" to "assets/pyramid_tps_<NUMBER>.pt"
        [shutil.move(f, os.path.join(self.asset_dir, re.sub(r'tps_', "pyramid_tps_", f))) for f in glob("tps_*.pt")]
        # Create a file with the length of the output object as a reference - this is the number of detections in the final object
        with open(os.path.join(self.asset_dir, "pyramid_output_length.txt"), "w") as f:
            f.write(str(len(output)))

class TestPredictor(unittest.TestCase):
    def test_single_scale(self):
        dtype = torch.float16
        predictor = Predictor(model=None, dtype=dtype)
        predictor._model = DummyModel("single_scale", ASSET_DIR)
        image_path = os.path.join(ASSET_DIR, ASSET_NAME + ".jpg")
        image = read_image(image_path).to(torch.device("cpu"), dtype=dtype)
        output = predictor._detect_instances(image, scale=1, max_scale=False)
        output_length = len(output)
        reference_length = int(open(os.path.join(ASSET_DIR, "single_scale_output_length.txt")).read())
        self.assertTrue(abs(1 - output_length/reference_length) < 0.1, msg=f"Output length ({output_length}) does not match the reference length ({reference_length})")
    
    def test_pyramid(self):
        dtype = torch.float16
        predictor = Predictor(model=None, dtype=dtype)
        predictor._model = DummyModel("pyramid", ASSET_DIR)
        image_path = os.path.join(ASSET_DIR, ASSET_NAME + ".jpg")
        image = read_image(image_path).to(torch.device("cpu"), dtype=dtype)
        output = predictor.pyramid_predictions(image, image_path, scale_increment=1/2, scale_before=1, single_scale=False)
        output_length = len(output)
        reference_length = int(open(os.path.join(ASSET_DIR, "pyramid_output_length.txt")).read())
        self.assertTrue(abs(1 - output_length/reference_length) < 0.1, msg=f"Output length ({output_length}) does not match the reference length ({reference_length})")

if __name__ == '__main__':
    unittest.main()