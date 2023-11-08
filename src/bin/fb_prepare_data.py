import glob
import os
from ultralytics.data.converter import convert_coco
import hashlib
import shutil
import tempfile
import yaml
import argparse
import logging


def collapse_in_parent_dir(child):
    assert os.path.isdir(child)
    target = os.path.abspath(os.path.join(child, os.pardir))

    for dd in os.listdir(child, ):
        shutil.move(os.path.join(child, dd), os.path.join(target, dd))
    shutil.rmtree(child)


OUT_COCO_CONVERTER_BASE = "./yolo_labels"
OUT_COCO_CONVERTER = "./yolo_labels/labels/default/"
OUT_COCO_CONVERTER_IMAGES = "./yolo_labels/images/default/"
JSON_FILE_BASENAME = "instances_default.json"
DATASET_NAME = "insects"


out_structure = """
├── data.yaml
└── insects
    ├── images
    │   ├── train
    │   └── val
    └── labels
        ├── train
        ├── val
"""

if __name__ == '__main__':
    args_parse = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

    args_parse.add_argument("-i", "--input-data", dest="coco_data_root",
                            help="A directory that contains subdirectories for each COCO sub-datasets."
                                 "Each sub-dataset contains a single json file named 'instances_default.json' "
                                 "and the associated images"
                            )
    args_parse.add_argument("-o", "--output-dir", dest="prepared_data_target",
                            help="The output compiled YOLO dataset joining together all sub-datasets in a single dataset with the structure:"
                                 f"{out_structure}")

    args = args_parse.parse_args()
    option_dict = vars(args)

    datasets = {"sticky-pi", "pitfall"}

    data_yaml = {"path": DATASET_NAME,
                 "train": "images/train",
                 "val": "images/val",
                 "nc": 1,
                 "names": ['insect']
                 }

    COCO_DATA_ROOT = option_dict["coco_data_root"]
    PREPARED_DATA_TARGET = option_dict["prepared_data_target"]
    PREPARED_DATA_TARGET_SUBDIR = os.path.join(PREPARED_DATA_TARGET, DATASET_NAME)
    os.makedirs(PREPARED_DATA_TARGET_SUBDIR, exist_ok=True)
    with open(os.path.join(PREPARED_DATA_TARGET, "data.yaml"), "w") as f:
        yaml.dump(data_yaml, f)

    for d in datasets:
        source_dir = os.path.join(COCO_DATA_ROOT, d)
        assert os.path.isdir(source_dir)
        assert os.path.isfile(os.path.join(source_dir, JSON_FILE_BASENAME))

    for d in datasets:
        source_dir = os.path.join(COCO_DATA_ROOT, d)
        tmp_dir = tempfile.mkdtemp()
        try:
            os.chdir(tmp_dir)
            o = convert_coco(labels_dir=source_dir, use_segments=True)

            os.makedirs(os.path.join(OUT_COCO_CONVERTER, "train"), exist_ok=True)
            os.makedirs(os.path.join(OUT_COCO_CONVERTER, "val"), exist_ok=True)

            os.makedirs(os.path.join(OUT_COCO_CONVERTER_IMAGES, "train"), exist_ok=True)
            os.makedirs(os.path.join(OUT_COCO_CONVERTER_IMAGES, "val"), exist_ok=True)
            images = {os.path.basename(f) for f in glob.glob(os.path.join(source_dir, "*.jpg"))}

            assert len(images) > 0

            for f in glob.glob(os.path.join(OUT_COCO_CONVERTER, "*.txt")):

                expected_image_basename = os.path.splitext(os.path.basename(f))[0] + ".jpg"
                if expected_image_basename not in images:
                    logging.warning("Missing image: " + expected_image_basename)
                    os.remove(f)
                    continue
                im_path = os.path.join(source_dir, expected_image_basename)
                assert os.path.isfile(im_path)
                s = bytes(os.path.basename(os.path.splitext(f)[0]), 'ascii')
                d = hashlib.md5(s).hexdigest()
                if d < "5":
                    subset = "val/"
                else:
                    subset = "train/"
                logging.info(f"{expected_image_basename} -> {subset}")
                shutil.move(f, os.path.join(OUT_COCO_CONVERTER, subset))
                shutil.copy(im_path, os.path.join(OUT_COCO_CONVERTER_IMAGES, subset))

            collapse_in_parent_dir(OUT_COCO_CONVERTER)
            collapse_in_parent_dir(OUT_COCO_CONVERTER_IMAGES)
            #fixme here should add a subdir like "insects/" same as the name in data.yaml
            shutil.copytree(OUT_COCO_CONVERTER_BASE , PREPARED_DATA_TARGET_SUBDIR, dirs_exist_ok=True)
        finally:
            if os.path.isdir(tmp_dir):
                shutil.rmtree(tmp_dir)

