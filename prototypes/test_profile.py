import torch
from torchvision.io import read_image

import time

from flat_bug.predictor import Predictor
from pyremotedata.implicit_mount import *
from pyremotedata.dataloader import *

from flat_bug.predictor import *
from flat_bug.yolo_helpers import *


from glob import glob

datasets = {p.removeprefix("s3/") : glob(f'{p}/**.jpg') for p in glob("s3/**")}

# paths = glob("dev/input/**.jpg")[:3]
paths = sorted(glob("s3/AMI-traps/**.jpg"))[:1]

weights = "model_snapshots/fb_2024-02-19_best.pt"
device = torch.device("cuda:0")
dtype = torch.float16

pred = Predictor(weights, device=device, dtype=dtype)
pred.MIN_MAX_OBJ_SIZE = 16, 1024
pred.MINIMUM_TILE_OVERLAP = 256
pred.EDGE_CASE_MARGIN = 128
pred.SCORE_THRESHOLD = 0.5
pred.IOU_THRESHOLD = 0.25
pred.PREFER_POLYGONS = True # This wasn't a hyperparameter before, but it reproduces the old behavior
pred.EXPERIMENTAL_NMS_OPTIMIZATION = True

# torch.cuda.empty_cache()

start = time.time()
for i in range(len(paths)):
    test = pred.pyramid_predictions(paths[i], scale_increment=1/2, scale_before=0.5, single_scale=True)
    # test.save("dev/input_output", fast=True, crops = False)
    # del test
    # torch.cuda.empty_cache()

print(f'Average time: {(time.time() - start) / len(paths):.3f}s')