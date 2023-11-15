import os
import cv2
import math
import numpy as np
import random

from ultralytics.data.augment import RandomPerspective
from ultralytics.utils.instance import Instances


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
        s = random.uniform(self.scale[0], self.scale[1])

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
    bg_fill = (0, 0, 0)
    min_size = 20  # px
    max_targets = 50  # randomely inpaint when too many targets?! # fixme implement

    def __init__(self, imsize):
        self._imsize = imsize

    def crop_labels(self, labels, start_x, start_y, pad_before=None, apply_max_targets=False):

        or_img = labels["img"]
        debug = os.path.basename(labels["im_file"]) == "a_63-20190826004609-00.jpg"

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
            print("or-shape:", or_img.shape)
            print("x,y:", start_x, start_y)
            print(labels["im_file"])  # fixme, this is also done during validation?!

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
                        (b[:, 1] + b[:, 3] / 2) < self._imsize,
                        b[:, 2] > self.min_size,
                        b[:, 3] > self.min_size],
                       axis=0)

        # here, we paint the edge cases (partially outside the image, as (0,0,0)),
        # this should help learning. Indeed it would be very confusing if an image if an insect that is
        # 10% outside is flagged as NOT insect!

        if apply_max_targets:
            if np.sum(valid) > self.max_targets:
                w = np.where(valid)[0]
                kept = np.random.choice(w, size=self.max_targets, replace=False)
                valid.fill(False)
                valid[kept] = True

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
                             offset=(px0 - x_offset, py0 - y_offset)
                             )

            # cv2.imwrite(f"/tmp/{os.path.basename(labels['im_file'])}", or_img)
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
        labels["img"] = np.copy(np.ascontiguousarray(img))
        return labels

    def __call__(self, labels):
        h, w = labels["img"].shape[:2]
        start_x = np.random.randint(w - self._imsize, size=1)[0]
        start_y = np.random.randint(h - self._imsize, size=1)[0]
        return self.crop_labels(labels, start_x, start_y, apply_max_targets=True)


class MyCrop(RandomCrop):
    pass


class RandomColorInv(object):

    def __call__(self, labels):
        img = labels['img']
        r = random.uniform(0, 1)
        if r > 0.5:
            assert img.dtype == np.uint8
            labels['img'] = 255 - img
        return labels
