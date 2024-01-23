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



# A help sting
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
# fixme, now, we ignores BG images!

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

    args_parse.add_argument("-p", "--validation-proportion", dest="validation_proportion",
                            help="the proportion of data allocated to the validation set, based on md5 (pseudorandom)",
                            default=0.15)
    
    args_parse.add_argument("-f", "--force", dest="delete_target_before",
                            help="Delete output directory before, this avoids duplicating data etc",
                            action="store_true")
    args = args_parse.parse_args()
    option_dict = vars(args)



    data_yaml = {"path": DATASET_NAME,
                 "train": "images/train",
                 "val": "images/val",
                 "nc": 1,
                 "names": ['insect']
                 }

    COCO_DATA_ROOT = option_dict["coco_data_root"]
    PREPARED_DATA_TARGET = option_dict["prepared_data_target"]

    PREPARED_DATA_TARGET_SUBDIR = os.path.join(PREPARED_DATA_TARGET, DATASET_NAME)

    if option_dict["delete_target_before"] and os.path.isdir(PREPARED_DATA_TARGET):
        logging.warning("Removing old output data directory")
        shutil.rmtree(PREPARED_DATA_TARGET)

    os.makedirs(PREPARED_DATA_TARGET_SUBDIR, exist_ok=True)

    with open(os.path.join(PREPARED_DATA_TARGET, "data.yaml"), "w") as f:
        yaml.dump(data_yaml, f)
    datasets = []
    for d in os.listdir(COCO_DATA_ROOT):
        source_dir = os.path.join(COCO_DATA_ROOT, d)
        if os.path.isdir(source_dir):
            if os.path.isfile(os.path.join(source_dir, JSON_FILE_BASENAME)):
                logging.info(f"Registering dataset: {d}")
                datasets.append(d)
    assert len(datasets) > 0, f"Did not find any datasets in {COCO_DATA_ROOT}"
    for d in datasets:
        source_dir = os.path.join(COCO_DATA_ROOT, d)
        tmp_dir = tempfile.mkdtemp()
        try:
            os.chdir(tmp_dir)
            o = convert_coco(labels_dir=source_dir, save_dir=OUT_COCO_CONVERTER_BASE, use_segments=True)

            os.makedirs(os.path.join(OUT_COCO_CONVERTER, "train"), exist_ok=True)
            os.makedirs(os.path.join(OUT_COCO_CONVERTER, "val"), exist_ok=True)

            os.makedirs(os.path.join(OUT_COCO_CONVERTER_IMAGES, "train"), exist_ok=True)
            os.makedirs(os.path.join(OUT_COCO_CONVERTER_IMAGES, "val"), exist_ok=True)
            images = {os.path.basename(f) for f in glob.glob(os.path.join(source_dir, "*.jpg"))}

            assert len(images) > 0
            for f in glob.glob(os.path.join(OUT_COCO_CONVERTER, "*.txt")):

                basename_sans_ext = os.path.splitext(os.path.basename(f))[0]
                expected_image_basename = basename_sans_ext + ".jpg"
                if expected_image_basename not in images:
                    logging.warning("Missing image: " + expected_image_basename)
                    os.remove(f)
                    continue
                im_path = os.path.join(source_dir, expected_image_basename)
                assert os.path.isfile(im_path)

                with open(im_path, 'rb') as file_obj:
                    file_hash = hashlib.md5(file_obj.read()).hexdigest()

                # s = bytes(basename_sans_ext, 'ascii')
                p = int(file_hash[0:4], 16) / int("ffff", 16)
                if p < option_dict["validation_proportion"]:
                    subset = "val/"
                else:
                    subset = "train/"
                logging.info(f"{expected_image_basename} -> {subset}")
                new_bn_se = f"{d}_{basename_sans_ext}"
                shutil.move(f, os.path.join(OUT_COCO_CONVERTER, os.path.join(subset, new_bn_se + ".txt")))
                shutil.copy(im_path, os.path.join(OUT_COCO_CONVERTER_IMAGES, subset, new_bn_se + ".jpg"))

            collapse_in_parent_dir(OUT_COCO_CONVERTER)
            collapse_in_parent_dir(OUT_COCO_CONVERTER_IMAGES)
            #fixme here should add a subdir like "insects/" same as the name in data.yaml
            shutil.copytree(OUT_COCO_CONVERTER_BASE , PREPARED_DATA_TARGET_SUBDIR, dirs_exist_ok=True)
        finally:
            if os.path.isdir(tmp_dir):
                shutil.rmtree(tmp_dir)


