import argparse
import logging
import os
import glob
from flat_bug.trainers import MySegmentationTrainer
from ultralytics import YOLO, settings

settings.update({'datasets_dir': "/home/quentin/Desktop/flat-bug-sorted-data/yolo"})
data= "/home/quentin/Desktop/flat-bug-sorted-data/yolo/data.yaml"
if __name__ == '__main__':
    args_parse = argparse.ArgumentParser()

    overrides = {
        "data": data,
        "batch": 6,
        # "imgsz": 1216,
        "imgsz": 1024,
        "model": "/home/quentin/repos/flat-bug-git/runs/segment/train78/weights/last.pt",
        # "model": "yolov8m-seg.pt",
        "task": "detect",
        "epochs": 5000,
        "device": "cuda",
        "patience": 200,
        "workers": 4  # fixme
    }

    t = MySegmentationTrainer(overrides=overrides)
    t.train()
    # args_parse.add_argument("action", help=str(valid_actions))
    # args_parse.add_argument("-b", "--bundle-dir", dest="bundle_dir")
    #
    # args_parse.add_argument("-v", "--verbose", dest="verbose", default=False,
    #                   help="verbose",
    #                   action="store_true")
    #
    # args_parse.add_argument("-D", "--debug", dest="debug", default=False,
    #                   help="debug",
    #                   action="store_true")
    # args_parse.add_argument("-g", "--gpu", dest="gpu", default=False, help="Whether to use GPU/Cuda", action="store_true")
    #
    # # predict specific
    # args_parse.add_argument("-t", "--target", dest="target")
    # args_parse.add_argument("-n", "--de-novo", dest="de_novo", default=False, help="Whether to just make empty SVGs "
    #                                                                                "for manual annotation.",
    #                         action="store_true")
    # args_parse.add_argument("-f", "--force", dest="force", default=False, help="force", action="store_true")
    #
    # # training specific
    # args_parse.add_argument("-r", "--restart-training", dest="restart_training", default=False, action="store_true")
    #
    # args = args_parse.parse_args()
    # option_dict = vars(args)
    #
    # if option_dict['verbose']:
    #     for handler in logging.root.handlers[:]:
    #         logging.root.removeHandler(handler)
    #
    #     logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    #                         datefmt='%Y-%m-%d %H:%M:%S',
    #                         level=logging.INFO)
    #
    # if not option_dict["bundle_dir"] and not option_dict["de_novo"]:
    #     raise ValueError("--bundle-dir (-b) not defined")
    #
    # if not option_dict["de_novo"] and not os.path.isdir(option_dict["bundle_dir"]):
    #     raise ValueError(f"--bundle-dir refers to a directory that does NOT exist: {option_dict['bundle_dir']}")
    # if option_dict["gpu"]:
    #     device = "cuda"
    # else:
    #     device = "cpu"
    #
    # if option_dict['action'] not in valid_actions:
    #     raise ValueError(f"Unexpected action{option_dict['action']}. Valid actions are:{str(valid_actions)}")
    #
    # if not option_dict["de_novo"]:
    #     bundle = MLBundle(option_dict["bundle_dir"], device=device)
    #
    # if option_dict['action'] == 'predict_dir':
    #     if not option_dict["target"]:
    #         raise ValueError("--target (-t) not defined")
    #
    #     if not os.path.isdir(option_dict['target']):
    #         raise ValueError(f"Target directory does not exist: {option_dict['target']}")
    #     # fixme (could be a other formats/patterns)
    #     if not option_dict["de_novo"]:
    #         pred = Predictor(bundle)
    #
    #     valid_imgs = sorted(glob.glob(os.path.join(option_dict['target'], "**", "*.jpg"), recursive=True))
    #     assert len(valid_imgs) > 0, f"No image found in {option_dict['target']}"
    #     logging.info(f"Found {len(valid_imgs)} images")
    #     for img in valid_imgs:
    #         # foreign image may have arbitrary filenames
    #         new_name = os.path.join(os.path.dirname(img), os.path.splitext(os.path.basename(img))[0] + ".svg")
    #         new_name_manual_annotation = os.path.join(os.path.dirname(img), MANUAL_ANNOTATION_PREFIX +
    #                                                   os.path.splitext(os.path.basename(img))[0] + ".svg")
    #
    #         img = Image(img, foreign=True)
    #
    #         if (os.path.exists(new_name_manual_annotation) or os.path.exists(new_name)) and not option_dict["force"]:
    #             logging.info(f"SVG output file exist: {os.path.relpath(new_name, option_dict['target'])}. Skipping. "
    #                          f"Use --force to overwrite")
    #             continue
    #         if option_dict["de_novo"]:
    #             annotated = img
    #         else:
    #             logging.info(f"Detecting in {os.path.relpath(img.path, option_dict['target'])}")
    #             annotated = pred.detect(img)
    #             logging.info(f"Saving results in {os.path.relpath(new_name, option_dict['target'])}")
    #
    #         annotated.to_svg(target=new_name)
    #         assert os.path.exists(new_name)
    #
    # if option_dict['action'] == 'train':
    #
    #     t = Trainer(bundle)
    #     t.resume_or_load(resume=not option_dict['restart_training'])
    #     t.train()
    #
    # if option_dict['action'] == 'check_data':
    #
    #     t = Trainer(bundle) # prepare data, implicitly
    #     loader = t._detectron_trainer.build_test_loader(bundle.config, bundle.name + "_val")
    #
    #     # for i in loader:
    #     #     pass
    #     # loader = t._detectron_trainer.build_test_loader(bundle.config,  bundle.name + "_train")
    #     # for i in loader:
    #     #     pass
    #
    # elif option_dict['action'] == 'validate':
    #     t = Trainer(bundle)
    #     pred = Predictor(bundle)
    #     os.makedirs(option_dict['target'], exist_ok=True)
    #     t.validate(pred, out_dir=option_dict['target'])
    #
    # elif option_dict['action'] == 'visualise':
    #     # bundle.dataset.visualise(subset="val")
    #     bundle.dataset.visualise()