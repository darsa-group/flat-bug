import glob
import logging
import os
from ultralytics.data.converter import convert_coco
import hashlib
import shutil
import tempfile


tmp_dir = tempfile.mkdtemp()
try:
    os.chdir(tmp_dir)

    # DATASET_DIR = "/home/quentin/Desktop/task_sticky-pi-2023_09_27_06_38_28-coco 1.0/annotations"
    # DATASET_DIR = "/tmp/task_pitfall-2023_10_11_13_38_31-coco 1.0/annotations"
    DATASET_DIR = "/tmp/ami-pretrain/annotations"

    # RAW_IMAGE_DIR = "/home/quentin/Desktop/flat-bug_data"
    RAW_IMAGE_DIR = "/tmp/ami"

    OUT_TARGET = "/tmp/ami/dataset"

    #-------------------------------------------------------------------#
    OUT_COCO_CONVERTER_BASE = "./yolo_labels"
    OUT_COCO_CONVERTER = "./yolo_labels/labels/default/"
    OUT_COCO_CONVERTER_IMAGES = "./yolo_labels/images/default/"


    o = convert_coco(labels_dir=DATASET_DIR, use_segments=True)

    os.makedirs(os.path.join(OUT_COCO_CONVERTER, "train"), exist_ok=True)
    os.makedirs(os.path.join(OUT_COCO_CONVERTER, "val"), exist_ok=True)

    os.makedirs(os.path.join(OUT_COCO_CONVERTER_IMAGES, "train"), exist_ok=True)
    os.makedirs(os.path.join(OUT_COCO_CONVERTER_IMAGES, "val"), exist_ok=True)


    # labels = {os.path.basename(l) for l in glob.glob(os.path.join(OUT_COCO_CONVERTER, "*.txt"))}
    # with open(os.path.join(DATASET_DIR, "im_list.txt"), "r") as f:
    #     files = set([os.path.splitext(l.rstrip())[0] + ".txt" for l in f.readlines()])

    # empty_files = files - labels
    # print("images with no annotations:", len(empty_files))
    # for f in empty_files:
    #     ann = os.path.join(OUT_COCO_CONVERTER, f)
    #     print(ann)
    #     open(ann, "w")
    #     assert os.path.isfile(ann)

    images = {os.path.basename(f) for f in glob.glob(os.path.join(RAW_IMAGE_DIR, "*.jpg"))}


    def collapse_in_parent_dir(child):
        assert os.path.isdir(child)
        target = os.path.abspath(os.path.join(child, os.pardir))
        print(child, target)
        for dd in os.listdir(child, ):
            shutil.move(os.path.join(child, dd), os.path.join(target, dd))
        shutil.rmtree(child)

    for f in glob.glob(os.path.join(OUT_COCO_CONVERTER, "*.txt")):

        expected_image_basename = os.path.splitext(os.path.basename(f))[0] + ".jpg"
        if expected_image_basename not in images:
            logging.warning("Missing image: " + expected_image_basename)
            os.remove(f)
            continue
        im_path = os.path.join(RAW_IMAGE_DIR,expected_image_basename)
        assert os.path.isfile(im_path)
        s = bytes(os.path.basename(os.path.splitext(f)[0]), 'ascii')
        d = hashlib.md5(s).hexdigest()
        if d < "5":
            subset="val/"
        else:
            subset = "train/"
        print(f"{expected_image_basename} -> {subset}")
        shutil.move(f, os.path.join(OUT_COCO_CONVERTER, subset))
        shutil.copy(im_path, os.path.join(OUT_COCO_CONVERTER_IMAGES, subset))

    collapse_in_parent_dir(OUT_COCO_CONVERTER)
    collapse_in_parent_dir(OUT_COCO_CONVERTER_IMAGES)



    shutil.move(OUT_COCO_CONVERTER_BASE + "/", OUT_TARGET)
    print()
finally:
    # if os.path.isdir(tmp_dir):
    #     shutil.rmtree(tmp_dir)
    pass
