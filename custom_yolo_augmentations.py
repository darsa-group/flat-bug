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

HELP_URL = 'See https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data'
IMG_FORMATS = 'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp', 'pfm'  # include image suffixes
VID_FORMATS = 'asf', 'avi', 'gif', 'm4v', 'mkv', 'mov', 'mp4', 'mpeg', 'mpg', 'ts', 'wmv'  # include video suffixes
LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', -1))
PIN_MEMORY = str(os.getenv('PIN_MEMORY', True)).lower() == 'true'  # global pin_memory for dataloaders


class MyRandomPerspective(RandomPerspective):
    fill_value = (0, 0, 0)

    def affine_transform(self, img, border):
        """Center."""
        C = np.eye(3, dtype=np.float32)

        C[0, 2] = -img.shape[1] / 2  # x translation (pixels)
        C[1, 2] = -img.shape[0] / 2  # y translation (pixels)

        # Perspective
        P = np.eye(3, dtype=np.float32)
        P[2, 0] = random.uniform(-self.perspective, self.perspective)  # x perspective (about y)
        P[2, 1] = random.uniform(-self.perspective, self.perspective)  # y perspective (about x)

        # Rotation and Scale
        R = np.eye(3, dtype=np.float32)
        a = random.uniform(-self.degrees, self.degrees)
        # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
        s = random.uniform(1 - self.scale, 1 + self.scale)
        # s = 2 ** random.uniform(-scale, scale)
        R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

        # Shear
        S = np.eye(3, dtype=np.float32)
        S[0, 1] = math.tan(random.uniform(-self.shear, self.shear) * math.pi / 180)  # x shear (deg)
        S[1, 0] = math.tan(random.uniform(-self.shear, self.shear) * math.pi / 180)  # y shear (deg)

        # Translation
        T = np.eye(3, dtype=np.float32)
        T[0, 2] = random.uniform(0.5 - self.translate, 0.5 + self.translate) * self.size[0]  # x translation (pixels)
        T[1, 2] = random.uniform(0.5 - self.translate, 0.5 + self.translate) * self.size[1]  # y translation (pixels)

        # Combined rotation matrix
        M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
        # Affine image
        if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
            if self.perspective:
                img = cv2.warpPerspective(img, M, dsize=self.size, borderValue=self.fill_value)
            else:  # affine
                img = cv2.warpAffine(img, M[:2], dsize=self.size, borderValue=self.fill_value)
        return img, M, s


class RandomCrop:
    bg_fill = (0,0,0)
    def __init__(self, imsize):
        self._imsize = imsize

    def crop_labels(self, labels, start_x, start_y, pad_before=None):

        or_img = labels["img"]
        debug = os.path.basename(labels["im_file"]) == "5595c586.2020-07-15_01-18-24.jpg"
        h, w = or_img.shape[:2]
        instances = labels.pop("instances")

        instances.convert_bbox(format="xywh")
        instances.denormalize(w, h)

        if pad_before:
            px, py = pad_before
            px0 = int(math.floor(px / 2))
            py0 = int(math.floor(py / 2))
            px1 = int(math.ceil(px / 2))
            py1 = int(math.ceil(py / 2))

            or_img = cv2.copyMakeBorder(or_img, py0, py1, px0, px1, cv2.BORDER_CONSTANT, value=self.bg_fill)
        else:
            px0, py0 = 0, 0
            # px1, py1 = 0, 0

        img = or_img[start_y: start_y + self._imsize, start_x: start_x + self._imsize, :]


        if img.shape != (self._imsize, self._imsize, 3):
            print("shape:", img.shape)
            print(labels["im_file"])

        assert img.shape == (self._imsize, self._imsize, 3), print(img.shape, self._imsize)

        labels["ori_shape"] = (self._imsize, self._imsize)
        labels["resized_shape"] = labels["ori_shape"]
        labels["img"] = np.copy(np.ascontiguousarray(img))

        labels['ratio_pad'] = ((1.0, 1.0), (0.0, 0.0))
        x_offset = -start_x + px0
        y_offset = -start_y + py0

        # positions in the cropped image
        instances._bboxes.add([x_offset, y_offset, 0, 0])

        for s in instances.segments:
            s[:, 0] += x_offset
            s[:, 1] += y_offset

        # print(start_x, start_y, self._imsize)
        b = instances._bboxes.bboxes

        if b.shape[0] == 0:
            labels["instances"] = Instances(np.empty([0, 4], dtype=np.float32), np.empty([0, 2], dtype=np.float32),
                                            normalized=False)
            labels["cls"] = []
            return labels

        valid = np.all([(b[:, 0] - b[:, 2] / 2) > 0,
                        (b[:, 1] - b[:, 3] / 2) > 0,
                        (b[:, 0] + b[:, 2] / 2) < self._imsize,
                        (b[:, 1] + b[:, 3] / 2) < self._imsize], axis=0)

        # here, we paint the edge cases (partially outside the image, as (0,0,0)),
        # this should help learning. Indeed it would be very confusing if an image if an insect that is
        # 10% outside is flagged as NOT insect!
        invalid = np.bitwise_not(valid)

        invalid_i = np.nonzero(invalid)[0]
        invalid_segments = instances.segments[invalid_i]
        invalid_segments = [np.array(s, dtype=np.int32) for s in invalid_segments]
        if len(invalid_segments):
            cv2.drawContours(or_img,
                             invalid_segments,
                             contourIdx=-1,
                             color=self.bg_fill,
                             thickness=-1,
                             lineType=cv2.LINE_4,
                             offset=(px0-x_offset, py0-y_offset)
                             )

            cv2.imwrite(f"/tmp/{os.path.basename(labels['im_file'])}", or_img)
        valid_i = np.nonzero(valid)[0]

        if len(valid_i) == 0:
            labels["instances"] = Instances(np.empty([0, 4], dtype=np.float32), np.empty([0, 2], dtype=np.float32),
                                            normalized=False)
            labels["cls"] = []  # np.empty_like(labels["cls"])
            return labels

        instances.segments = instances.segments[valid_i]
        # fixme, should be                                 b[valid_i, :] ?!
        instances._bboxes.bboxes = b[valid_i, :]
        labels["cls"] = labels["cls"][valid_i]
        # print(os.path.basename(labels["im_file"]), labels["cls"] = labels["cls"][valid_i])

        labels["instances"] = instances
        # print(labels)
        return labels

    def __call__(self, labels):
        h, w = labels["img"].shape[:2]
        start_x = np.random.randint(w - self._imsize, size=1)[0]

        start_y = np.random.randint(h - self._imsize, size=1)[0]
        return self.crop_labels(labels, start_x, start_y)


# class MyLetterBox(LetterBox):
#     def __call__(self, *args, **kwargs):
#         out = super().__call__(*args, **kwargs)
#         print(out)
#         return out
class MyCrop(RandomCrop):
    pass


class MyYOLODataset(YOLODataset):

    def _debug_write_loaded_images(self, out, index):
        m = np.ascontiguousarray(out["masks"].detach().numpy().transpose(1, 2, 0)) * 255
        m = cv2.cvtColor(cv2.resize(m, (self.imgsz, self.imgsz)), cv2.COLOR_GRAY2BGR)
        n = np.ascontiguousarray(out["img"].detach().numpy().transpose(1, 2, 0)) * 255
        bbs = out["bboxes"].detach().numpy() * self.imgsz
        bbs = bbs.astype(int)
        for k in range(bbs.shape[0]):
            x, y, w, h = bbs[k, :]
            # print(n.shape)
            # print((x, y))
            # print((x+w, y+h))
            n = cv2.rectangle(n, (x - w // 2, y - w // 2), (x + w // 2, y + h // 2), (255), 3)
        # n = n + m/3
        cv2.imwrite("/tmp/test/%i-img.jpg" % index, n + m / 3)
        # cv2.imwrite("/tmp/test/%i-mask.jpg" % index, m)

    def load_image(self, i):
        # Loads 1 image from dataset index 'i', returns (im, resized hw)

        im, f, fn = self.ims[i], self.im_files[i], self.npy_files[i]

        if im is None:  # not cached in RAM
            if fn.exists():  # load npy
                im = np.load(fn)

            else:  # read image
                im = cv2.imread(f)  # BGR
                if im is None:
                    raise FileNotFoundError(f"Image Not Found {f}")

            h0, w0 = im.shape[:2]  # orig hw

            # fixme
            if os.path.basename(self.im_files[i]).startswith("2023-"):
                im = cv2.resize(im, (round(w0 / 4.0), round(h0 / 4.0)), interpolation=cv2.INTER_AREA)
                # print("not cached", f, (h0, w0), im.shape[:2])
                h0, w0 = im.shape[:2]
            return im, (h0, w0), im.shape[:2]  # im, hw_original, hw_resized
        # print("cached", f, self.im_hw0[i], self.im_hw[i])
        return self.ims[i], self.im_hw0[i], self.im_hw[i]  # im, hw_original, hw_resized

    def build_transforms(self, hyp=None):
        import torchvision.transforms as T
        return Compose([
            RandomHSV(hgain=hyp.hsv_h, sgain=hyp.hsv_s, vgain=hyp.hsv_v),
            RandomFlip(direction="vertical", p=hyp.flipud),
            RandomFlip(direction="horizontal", p=hyp.fliplr),
            # T.RandomRotation(180),
            MyRandomPerspective(degrees=180, scale=0, translate=0),
            RandomCrop(self.imgsz),
            # MyAlbumentations(self.imgsz),
            # LetterBox(new_shape=(self.imgsz, self.imgsz), scaleup=False),
            Format(bbox_format="xywh",
                   normalize=True,
                   return_mask=self.use_segments,
                   return_keypoint=self.use_keypoints,
                   batch_idx=True,
                   mask_ratio=hyp.mask_ratio,
                   mask_overlap=hyp.overlap_mask)
        ])

    def __getitem__(self, index):
        out = self.transforms(self.get_image_and_label(index))
        dbg = os.path.basename(out["im_file"]) == "PureBarleySample_p1-20230801_104337.jpg"
        if dbg:
            self._debug_write_loaded_images(out, index)
        return out


class MyYOLOValidationDataset(MyYOLODataset):
    keep_every = 3

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
        # labs = self.get_labels()

        self._custom_crop = MyCrop(self.imgsz)
        new_im_files = []
        new_labels = []
        new_npy_files = []
        new_ims = []
        self.crops = []
        for i in range(len(self)):
            if i % self.keep_every == 0:
                h, w = self.load_image(i)[1]

                h_padding = self.imgsz - (h % self.imgsz)
                w_padding = self.imgsz - (w % self.imgsz)
                n_y = (h + h_padding) // self.imgsz
                n_x = (w + w_padding) // self.imgsz
                lab = self.labels[i]
                lab["shape"] = (h, w)
                for n in range(n_x):
                    for m in range(n_y):
                        new_labels.append(copy.copy(lab))
                        new_im_files.append(copy.copy(self.im_files[i]))
                        new_npy_files.append(copy.copy(self.npy_files[i]))
                        new_ims.append(copy.copy(self.ims[i]))
                        d = {"x0": n * self.imgsz,
                             "y0": m * self.imgsz,
                             "w_padding": w_padding,
                             "h_padding": h_padding,
                             }
                        self.crops.append(d)
        self.im_files, self.labels, self.npy_files, self.ims = new_im_files, new_labels, new_npy_files, new_ims

        self.ni = len(self.labels)  # number of images

        self.set_rectangle()

    def build_transforms(self, hyp=None):
        return Compose([
            Format(bbox_format="xywh",
                   normalize=True,
                   return_mask=self.use_segments,
                   return_keypoint=self.use_keypoints,
                   batch_idx=True,
                   mask_ratio=hyp.mask_ratio,
                   mask_overlap=hyp.overlap_mask)
        ])

    def get_image_and_label(self, index, print_me=False):
        """Get and return label information from the dataset."""
        label = copy.deepcopy(
            self.labels[index])  # requires deepcopy() https://github.com/ultralytics/ultralytics/pull/1948
        label.pop('shape', None)  # shape is for rect, remove it
        label['img'], label['ori_shape'], label['resized_shape'] = self.load_image(index)
        label['ratio_pad'] = (label['resized_shape'][0] / label['ori_shape'][0],
                              label['resized_shape'][1] / label['ori_shape'][1])  # for evaluation
        if self.rect:
            label['rect_shape'] = self.batch_shapes[self.batch[index]]
        out = self.update_labels_info(label)
        if print_me:
            print(label["instances"]._bboxes.bboxes)
            print("----------------------------")
            print(out["instances"]._bboxes.bboxes)
            print("=====================================================get_image_and_label")
        return out

    def __getitem__(self, index):
        out00 = self.get_image_and_label(index)
        cr = self.crops[index]
        out0 = self._custom_crop.crop_labels(copy.deepcopy(out00), cr["x0"], cr["y0"],
                                             pad_before=(cr["w_padding"], cr["h_padding"]))
        out = self.transforms(copy.deepcopy(out0))
        return out

    def get_img_files(self, img_path):
        """Read image files."""

        try:
            f = []  # image files
            for p in img_path if isinstance(img_path, list) else [img_path]:
                p = Path(p)  # os-agnostic
                if p.is_dir():  # dir
                    f += glob.glob(str(p / '**' / '*.*'), recursive=True)
                    # F = list(p.rglob('*.*'))  # pathlib
                elif p.is_file():  # file
                    with open(p) as t:
                        t = t.read().strip().splitlines()
                        parent = str(p.parent) + os.sep
                        f += [x.replace('./', parent) if x.startswith('./') else x for x in t]  # local to global path
                        # F += [p.parent / x.lstrip(os.sep) for x in t]  # local to global path (pathlib)
                else:
                    raise FileNotFoundError(f'{self.prefix}{p} does not exist')
            im_files = sorted(x.replace('/', os.sep) for x in f if x.split('.')[-1].lower() in IMG_FORMATS)
            # self.img_files = sorted([x for x in f if x.suffix[1:].lower() in IMG_FORMATS])  # pathlib
            assert im_files, f'{self.prefix}No images found'
        except Exception as e:
            raise FileNotFoundError(f'{self.prefix}Error loading data from {img_path}\n{HELP_URL}') from e
        if self.fraction < 1:
            im_files = im_files[:round(len(im_files) * self.fraction)]

        return im_files


class MySegmentationTrainer(SegmentationTrainer):

    def build_dataset(self, img_path, mode='train', batch=None):
        from ultralytics.utils import yaml_load
        if mode == "train":
            dataset = MyYOLODataset(
                data=yaml_load(self.args.data),
                img_path=img_path,
                imgsz=self.args.imgsz,
                cache=False,
                augment=mode == "train",
                hyp=self.args,
                rect=self.args.rect if mode == "train" else True,
                batch_size=batch,
                # stride=int(stride),
                pad=0.0 if mode == "train" else 0.5,
                single_cls=self.args.single_cls or False,
                use_segments=True
            )
        else:
            print("VALIDATION!!")
            dataset = MyYOLOValidationDataset(
                data=yaml_load(self.args.data),
                img_path=img_path,
                imgsz=self.args.imgsz,
                cache=False,
                augment=mode == "train",
                hyp=self.args,
                rect=self.args.rect if mode == "train" else True,
                batch_size=batch,
                # stride=int(stride),
                pad=0.0 if mode == "train" else 0.5,  # fixme... does not make sense...
                single_cls=self.args.single_cls or False,
                use_segments=True,

            )

        return dataset


overrides = {
    "data": "./data.yaml",
    "batch": 6,
    # "imgsz": 1216,
    "imgsz": 1024,
    # "model": "/home/quentin/repos/flat-bug-git/runs/segment/train12/weights/best.pt",
    "model": "yolov8m-seg.pt",
    "task": "detect",
    "epochs": 1000,
    "device": "cuda",
    "patience": 100,
    "workers": 4  # fixme

}

# todo validation should be tiled... or at least, reproducible...

# t = MyDetectionTrainer(overrides=overrides)
t = MySegmentationTrainer(overrides=overrides)
t.train()

# model = YOLO()  # load a custom model
# Predict with the model


# results = model('test_positive.jpg')  # predict on an image
# cv2.imwrite("test.jpg", results[0].plot())

# results = model.train(data=DATASET, epochs=100, **overrides)
