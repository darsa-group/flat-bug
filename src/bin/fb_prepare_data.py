#!/usr/bin/env python

import glob
import json
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



OUT_COCO_CONVERTER = "labels/default/"
OUT_COCO_CONVERTER_IMAGES = "images/default/"
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


def merge_cocos(files, out_file, delete=False):
    im_id = 1
    an_id = 1
    out = None
    for c in files:
        with open(c) as f:
            coco = json.load(f)
        id_map = {} ## old: new
        new_images = []
        for i in coco["images"]:
            id_map[i["id"]] = im_id
            i["id"] = im_id
            new_images.append(i)
            im_id += 1
        coco["images"] = new_images

        for a in coco["annotations"]:
            a["image_id"] = id_map[a["image_id"]]
            a["id"] = an_id
            an_id += 1
        if out is None:
            out = coco
        else:
            out["images"].extend(coco["images"])
            out["annotations"].extend(coco["annotations"])
    with open(out_file, "w") as f:
        json.dump(out, f)

    if delete:
        for c in files:
            os.remove(c)

def prepare_coco_file(source_file, image_list, out):
    with open(source_file) as f:
        coco = json.load(f)
    images_to_keep = []
    image_ids_to_keep = []

    new_image = []
    for i in coco["images"]:
        if i["file_name"] in image_list:
            images_to_keep.append(i)
            i["file_name"] = image_list[i["file_name"]]
            image_ids_to_keep.append(i["id"])
            new_image.append(i)
    assert len(images_to_keep) > 0


    new_annots = []
    for a in coco["annotations"]:
        if a["image_id"] in image_ids_to_keep:
            new_annots.append(a)

    coco["annotations"] = new_annots
    coco["images"] = images_to_keep
    with open(out, "w") as f:
        json.dump(coco, f)

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
        tmp_dir = tempfile.mkdtemp(prefix="tmp-fb-")
        shutil.rmtree(tmp_dir)

        try:

            coco_files = [f for f in sorted(glob.glob(os.path.join( source_dir, "*.json")))]
            assert len(coco_files) == 1, os.path.join(source_dir, "*.json") #,"Multiple label files, only supporting one"


            convert_coco(labels_dir=source_dir, save_dir=tmp_dir, use_segments=True)
            os.makedirs(os.path.join(tmp_dir, OUT_COCO_CONVERTER, "train"), exist_ok=True)
            os.makedirs(os.path.join(tmp_dir, OUT_COCO_CONVERTER, "val"), exist_ok=True)

            os.makedirs(os.path.join(tmp_dir, OUT_COCO_CONVERTER_IMAGES, "train"), exist_ok=True)
            os.makedirs(os.path.join(tmp_dir, OUT_COCO_CONVERTER_IMAGES, "val"), exist_ok=True)

            images = {os.path.basename(f) for f in sorted(glob.glob(os.path.join(source_dir, "*.jpg")))}

            assert len(images) > 0

            validation_files = {}
            training_files = {}
            for f in sorted(glob.glob(os.path.join(tmp_dir,OUT_COCO_CONVERTER, "*.txt"))):
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


                new_bn_se = f"{d}_{basename_sans_ext}"
                p = int(file_hash[0:4], 16) / int("ffff", 16)
                if p < option_dict["validation_proportion"]:
                    subset = "val/"
                    validation_files[expected_image_basename] = new_bn_se + ".jpg"
                else:
                    subset = "train/"
                    training_files[expected_image_basename] = new_bn_se + ".jpg"
                logging.info(f"{expected_image_basename} -> {subset}")

                shutil.move(f, os.path.join(tmp_dir, OUT_COCO_CONVERTER, os.path.join(subset, new_bn_se + ".txt")))
                shutil.copy(im_path, os.path.join(tmp_dir, OUT_COCO_CONVERTER_IMAGES, subset, new_bn_se + ".jpg"))

            if len(validation_files) == 0:
                logging.warning(f"No validation files for {d}")
            else:
                prepare_coco_file(coco_files[0], validation_files, os.path.join(tmp_dir, OUT_COCO_CONVERTER, "val", f"{d}"+JSON_FILE_BASENAME))

            if len(validation_files) == 0:
                logging.warning(f"No train files for {d}")
            else:
                prepare_coco_file(coco_files[0], training_files, os.path.join(tmp_dir, OUT_COCO_CONVERTER, "train", f"{d}"+JSON_FILE_BASENAME))

            collapse_in_parent_dir(os.path.join(tmp_dir, OUT_COCO_CONVERTER))
            collapse_in_parent_dir(os.path.join(tmp_dir,  OUT_COCO_CONVERTER_IMAGES))
            #fixme here should add a subdir like "insects/" same as the name in data.yaml
            shutil.copytree(tmp_dir, PREPARED_DATA_TARGET_SUBDIR, dirs_exist_ok=True)
        finally:
            if os.path.isdir(tmp_dir):
                shutil.rmtree(tmp_dir)


    for subset in {"val", "train"}:
        all_json = [f for f in sorted(glob.glob(os.path.join(PREPARED_DATA_TARGET_SUBDIR, "labels", subset, "*.json")))]
        merge_cocos(all_json, os.path.join(PREPARED_DATA_TARGET_SUBDIR, "labels", subset,JSON_FILE_BASENAME), delete=True)
