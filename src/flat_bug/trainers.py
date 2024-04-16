import json, glob, os, random

import numpy as np
from ultralytics.models.yolo.segment import SegmentationTrainer
from flat_bug.datasets import MyYOLODataset, MyYOLOValidationDataset
from ultralytics.utils import yaml_load, DEFAULT_CFG, RANK, LOGGER

import torch
from ultralytics.nn.tasks import attempt_load_one_weight, attempt_load_weights
from ultralytics.utils import (DEFAULT_CFG, LOGGER, RANK, SETTINGS, TQDM_BAR_FORMAT, __version__, callbacks, clean_url,
                               colorstr, emojis, yaml_save)

from ultralytics.utils.torch_utils import (EarlyStopping, ModelEMA, de_parallel, init_seeds, one_cycle, select_device,
                                           strip_optimizer, smart_inference_mode)

def data2labels(data):
    if hasattr(data, "__iter__") and not isinstance(data, str):
        return [data2labels(d) for d in data]
    data = data.replace("images", "labels")
    # Remove possible trailing directory separator
    if data[-1] == os.sep:
        data = data[:-1]
    return data + f'{os.sep}instances_default.json'

def get_latest_weight(weight_dir):
    weights = glob.glob(f"{weight_dir}{os.sep}*.pt")
    if not weights:
        LOGGER.warning(f"No weights found in {weight_dir}")
        return None
    return max(weights, key=os.path.getctime)

def _custom_end_to_end_validation(self):
        if not self._do_custom_eval:
            return
        self._do_custom_eval = False
        # Get image and label paths
        train_data, val_data = self.data["train"], self.data["val"]
        train_labels, val_labels = data2labels(train_data), data2labels(val_data)
        train_paths, val_paths = self.training_image_paths, self.val_image_paths
        if self._custom_num_images > -1 and self._custom_num_images < len(val_paths):
            # Sample n images
            # train_paths = random.sample(train_paths, self._custom_num_images)
            val_paths = random.sample(val_paths, min(len(val_paths), self._custom_num_images))
        escape_dots = lambda f: os.path.basename(f).replace(".", r"\.")
        val_pattern = f'({"|".join([escape_dots(f) for f in val_paths])})'
        # Get latest model path
        weight_dir = self.wdir
        latest_weights = get_latest_weight(weight_dir)
        # Construct end-to-end evaluation command
        command = f'bash "scripts{os.sep}eval{os.sep}end_to_end_eval.sh" -w "{latest_weights}" -d "{val_data}" -l "{val_labels}" -o "{self.save_dir}{os.sep}e2e_val{os.sep}{self.epoch}" -g "{self.args.device}" -p "{val_pattern}"'
        LOGGER.info(f"Running custom end-to-end validation command: `{command}`")
        # Run command
        os.system(command)

class MySegmentationTrainer(SegmentationTrainer):
    def __init__(self, max_instances, max_images, custom_eval, custom_eval_num_images, exclude_datasets, cfg=DEFAULT_CFG, overrides=None, _callbacks=None, *args, **kwargs):
        """Initialize a SegmentationTrainer object with given arguments."""
        self._max_instances = max_instances
        self._max_images = max_images
        self._exclude_datasets = exclude_datasets
        self.custom_eval = custom_eval
        self._do_custom_eval = False
        self._custom_num_images = custom_eval_num_images
        assert self._custom_num_images != 0, 'fb_custom_eval_num_images/custom_eval_num_images cannot be 0. If you mean to disable custom eval set fb_custom_eval/custom_eval=False.'
        assert self._max_instances != 0, 'fb_max_instances/max_instances cannot be 0.'
        assert self._max_images != 0, "fb_max_images/max_images cannot be 0."
        super().__init__(cfg, overrides, _callbacks, *args, **kwargs)
        if overrides["resume"]:
            self.args.resume = True
        self.add_callback("on_train_epoch_start", MySegmentationTrainer.log_lr)
        if self.custom_eval:
            self.add_callback("on_model_save", _custom_end_to_end_validation)
            self.add_callback("on_train_end", _custom_end_to_end_validation)

        self._val_metrics = None
        self._val_fitness = None
        self.cfg = cfg

        # Reproducibility
        self._reproducibility_setup()

    @staticmethod
    def log_lr(self):
        LOGGER.info(f"LR: {self.scheduler.get_last_lr()}")

    def setup_model(self):
        if isinstance(self.model, torch.nn.Module):  # if model is loaded beforehand. No setup needed
            return

        model, weights = self.model, None
        ckpt = None
        if str(model).endswith('.pt'):
            weights, ckpt = attempt_load_one_weight(model)
            cfg = ckpt['model'].yaml
        else:
            cfg = model
        self.model = self.get_model(cfg=cfg, weights=weights, verbose=RANK == -1)  # calls Model(cfg, weights)
        if not self.args.resume:
            return None
        return ckpt
    
    @property
    def exclude_pattern(self):
        return f'^(?!({"|".join([d + "_" for d in self._exclude_datasets])}))'

    def build_dataset(self, img_path, mode='train', batch=None):
        print(f"Building dataset with max instances {self._max_instances}, max images {self._max_images} and exclude pattern {self.exclude_pattern}")
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
                max_instances=self._max_instances,
                task="segment"
            ).subset(n=self._max_images, pattern=self.exclude_pattern)
        else:
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
                max_instances=np.Inf,
                task="segment"
            ).subset(n=self._max_images, pattern=self.exclude_pattern)

        return dataset

    @smart_inference_mode()
    def validate(self):
        """
        Runs validation on test set using self.validator. The returned dict is expected to contain "fitness" key.
        """
        if self.epoch % self.save_period == 0 or self._val_metrics is None:
            metrics, fitness = super().validate()
            self._val_metrics, self._val_fitness = metrics, fitness

            # Custom end-to-end validation
            if self.custom_eval:
                self._do_custom_eval = True
        else:
            metrics, fitness = self._val_metrics, self._val_fitness
            LOGGER.info(f"Skipped validation at epoch {self.epoch}, using old values")
        return metrics, fitness

    @property
    def training_image_paths(self):
        try:
            return self.train_loader.dataset.im_files
        except Exception as e:
            print("Perhaps the trainer has not been activated yet. Accessing the training image paths is not possible, while training has not started.")
            raise e
        
    @property
    def val_image_paths(self):
        try:
            return self.test_loader.dataset.im_files
        except Exception as e:
            print("Perhaps the trainer has not been activated yet. Accessing the validation image paths is not possible, while training has not started.")
            raise e

    def _reproducibility_setup(self):
        def log_data(self):
            with open(self.save_dir / "data_log.json", "w") as f:
                json.dump({**{k : str(v) for k, v in self.data.items()}, **{"train_images" : self.training_image_paths, "val_images" : self.val_image_paths}}, f)
        self.add_callback("on_train_start", log_data)
