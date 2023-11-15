import copy

import torch
import os
import cv2
import math
import numpy as np
from pathlib import Path
import glob
import random
from ultralytics import YOLO
from ultralytics.models.yolo.segment import SegmentationTrainer
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils.torch_utils import de_parallel
# from ultralytics.yolo.data.dataloaders.v5loader import  LoadImagesAndLabels, InfiniteDataLoader
# from ultralytics.yolo.utils.instance import Instances
# from ultralytics.yolo.utils import colorstr
# from ultralytics.data.build import seed_worker, build_dataloader

# from torch.utils.data import DataLoader, distributed

# from ultralytics.yolo.utils import LOGGER
# from ultralytics.yolo.utils.torch_utils import torch_distributed_zero_first
# import albumentations as A
# from ultralytics.data import build_dataloader


from ultralytics.data import YOLODataset
from ultralytics.data.augment import RandomFlip, RandomHSV, Compose, Format, LetterBox, RandomPerspective
from ultralytics.utils.instance import Instances

# HELP_URL = 'See https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data'
# IMG_FORMATS = 'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp', 'pfm'  # include image suffixes
# VID_FORMATS = 'asf', 'avi', 'gif', 'm4v', 'mkv', 'mov', 'mp4', 'mpeg', 'mpg', 'ts', 'wmv'  # include video suffixes
# LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
# RANK = int(os.getenv('RANK', -1))
# PIN_MEMORY = str(os.getenv('PIN_MEMORY', True)).lower() == 'true'  # global pin_memory for dataloaders


# todo validation should be tiled... or at least, reproducible...


# model = YOLO()  # load a custom model
# Predict with the model


# results = model('test_positive.jpg')  # predict on an image
# cv2.imwrite("test.jpg", results[0].plot())

# results = model.train(data=DATASET, epochs=100, **overrides)
