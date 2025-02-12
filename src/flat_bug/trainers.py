import glob
import json
import os
import random
from copy import copy
from tempfile import NamedTemporaryFile
from typing import Any, Dict, List, Optional, Self, Tuple, Union

import numpy as np
import torch
from ultralytics.data import build_dataloader
from ultralytics.data.build import InfiniteDataLoader
from ultralytics.models import yolo
from ultralytics.models.yolo.segment import SegmentationTrainer
from ultralytics.nn.tasks import attempt_load_one_weight
from ultralytics.utils import (DEFAULT_CFG, LOGGER, RANK,
                               IterableSimpleNamespace, __version__, yaml_load)
from ultralytics.utils.files import increment_path
from ultralytics.utils.torch_utils import (smart_inference_mode,
                                           torch_distributed_zero_first)

from flat_bug import logger
from flat_bug.datasets import FlatBugYOLODataset, FlatBugYOLOValidationDataset


def remove_custom_fb_args(args : Union[Dict, IterableSimpleNamespace, Any]) -> Union[Dict, IterableSimpleNamespace, Any]:
    if isinstance(args, dict):
        for k in list(args.keys()):
            if k.startswith("fb_"):
                del args[k]
    elif isinstance(args, IterableSimpleNamespace):
        for k in list(args.__dict__.keys()):
            if k.startswith("fb_"):
                del args.__dict__[k]

    return args

def extract_custom_fb_args(args : Dict) -> Dict:
    custom_fb_args = {}
    for k, v in args.items():
        if k.startswith("fb_"):
            custom_fb_args[k] = v

    return custom_fb_args

def data2labels(data : Union[str, List[str]]) -> Union[str, List[str]]:
    if hasattr(data, "__iter__") and not isinstance(data, str):
        return [data2labels(d) for d in data]
    data = data.replace("images", "labels")
    # Remove possible trailing directory separator
    if data[-1] == os.sep:
        data = data[:-1]
    return data + f'{os.sep}instances_default.json'

def get_latest_weight(weight_dir : str) -> Union[str, None]:
    weights = glob.glob(f"{weight_dir}{os.sep}*.pt")
    if not weights:
        logger.warning(f"No weights found in {weight_dir}")
        return None
    return max(weights, key=os.path.getctime)

def _custom_end_to_end_validation(self : "FlatBugSegmentationTrainer"):
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
        custom_eval_path = os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "eval", "end_to_end_eval.sh")
        command = f'bash "{custom_eval_path}" -w "{latest_weights}" -d "{val_data}" -l "{val_labels}" -o "{self.save_dir}{os.sep}e2e_val{os.sep}{self.epoch}" -g "{self.args.device}" -p "{val_pattern}"'
        logger.debug(f"Running custom end-to-end validation command: `{command}`")
        # Run command
        os.system(command)

def findattr(o, name : str, filters : list=[lambda _ : True], exclude_prefix="_", label : str="object"):
    isd = isinstance(o, dict)
    if isd:
        attrs = list(o.keys())
    else:
        try:
            attrs = list(o.__dict__.keys())
        except:
            return {}
    values = {}
    for attr in attrs:
        if (isinstance(attr, str) and attr.startswith(exclude_prefix)):
            continue
        new_label = f'{label}["{attr}"]' if isd else f"{label}.{attr}"
        val = o.get(attr) if isd else getattr(o, attr)
        if name == attr and all(map(lambda f : f(val), filters)):
            values[new_label] = val
        else:
            values.update(findattr(val, name, filters=filters, exclude_prefix=exclude_prefix, label=new_label))
    return values

def replaceattr(o, name : str, value, filters : list=[lambda _ : True], exclude_prefix="_", label : str="object"):
    isd = isinstance(o, dict)
    if isd:
        attrs = list(o.keys())
    else:
        try:
            attrs = list(o.__dict__.keys())
        except:
            return False
    for attr in attrs:
        if (isinstance(attr, str) and attr.startswith(exclude_prefix)):
            continue
        new_label = f'{label}["{attr}"]' if isd else f"{label}.{attr}"
        val = o.get(attr) if isd else getattr(o, attr)
        if name == attr and all(map(lambda f : f(val), filters)):
            logger.debug(f'{new_label} : {val} ==> {value}')
            if isd:
                o.update({attr : value})
            else:
                setattr(o, attr, value)
        else:
            replaceattr(o.get(attr) if isd else getattr(o, attr), name, value, filters=filters, exclude_prefix=exclude_prefix, label=new_label)

def apply_overrides_to_checkpoint(overrides):
    if not overrides.get("resume", False):
        return
    resume_model = overrides["resume"]
    _, ckpt_ext = os.path.splitext(resume_model)
    if not isinstance(resume_model, str):
        raise NotImplementedError(f"`flat-bug` currently onyl supports resuming training from a file. Please specify resume=<checkpoint>.pt instead of resume={resume_model}")
    if not os.path.exists(resume_model):
        raise FileNotFoundError(f"Resume checkpoint {resume_model} not found.")
    # Load original checkpoint
    logger.debug(f'Loading checkpoint for resuming {resume_model} to `resume_ckpt`')
    resume_ckpt = torch.load(resume_model)
    logger.debug("Replacing values in `resume_ckpt`...")
    # Enforce overrides
    for k, v in overrides.items():
        if not k.startswith("fb_") and v is not None:
            replaceattr(resume_ckpt, k, v, [lambda x : isinstance(x, (str, int, float)) or x is None], label="resume_ckpt")
    # Change save dir
    if "name" not in overrides:
        overrides["name"] = (list(findattr(resume_ckpt, "name", [lambda x : isinstance(x, str)]).values()) or ["train"])[0]
    if "project" not in overrides:
        overrides["project"] = (list(findattr(resume_ckpt, "project", [lambda x : isinstance(x, str)]).values()) or ["runs/segment"])[0]
    new_save_dir = increment_path(os.path.join(overrides["project"], overrides["name"]), False)
    replaceattr(resume_ckpt, "save_dir", new_save_dir, [lambda x : isinstance(x, (str, int, float)) or x is None], label="resume_ckpt")
    # Set epoch appropriately
    prior_epochs = resume_ckpt["train_results"]["epoch"]
    if len(prior_epochs) == 0 or max(prior_epochs) < 1:
        raise ValueError("Checkpoint doesn't contain enough information to restart training.")
    resume_ckpt["epoch"] = max(prior_epochs)
    # Save the new checkpoint to a temporary file
    tmp_resume_weight_dir = os.path.join(overrides["project"], "resume_weights")
    os.makedirs(tmp_resume_weight_dir, exist_ok=True)
    with NamedTemporaryFile(
        delete=False, 
        suffix=ckpt_ext, 
        prefix="resume--" + "__".join(os.path.splitext(resume_model)[0].split(os.sep)) + "--", 
        dir=tmp_resume_weight_dir
    ) as tmp_model:
        torch.save(resume_ckpt, tmp_model)
    logger.debug(f"Saved altered checkpoint for resuming `resume_ckpt` as {tmp_model.name}")
    # Replace the resume checkpoint with the updated checkpoint in the temporary file
    overrides["resume"] = tmp_model.name
    logger.debug(f'Set {overrides["resume"]=}')
    if "model" in overrides:
        overrides["model"] = tmp_model.name
        logger.debug(f'Set {overrides["model"]=}')
    # Return overrides for convenience, in fact this function mutates the original overrides object
    return overrides

class FlatBugSegmentationTrainer(SegmentationTrainer):
    def __init__(
            self, 
            cfg : IterableSimpleNamespace=DEFAULT_CFG, 
            overrides : Dict=None, 
            _callbacks : Any=None, 
            *args, 
            **kwargs
        ):
        """Initialize a SegmentationTrainer object with given arguments."""
        cfg = DEFAULT_CFG # In DDP mode, a CFG is created for each rank, but we always want the default one
        custom_fb_args = extract_custom_fb_args(overrides)
        self._max_instances = custom_fb_args["fb_max_instances"]
        self._max_images = custom_fb_args["fb_max_images"]
        self._exclude_datasets = custom_fb_args["fb_exclude_datasets"]
        self.custom_eval = custom_fb_args["fb_custom_eval"]
        self._do_custom_eval = False # This is a dynamic signalling flag, not a hyperparameter
        self._custom_num_images = custom_fb_args["fb_custom_eval_num_images"]
        assert self._custom_num_images != 0, 'fb_custom_eval_num_images/custom_eval_num_images cannot be 0. If you mean to disable custom eval set fb_custom_eval/custom_eval=False.'
        assert self._max_instances != 0, 'fb_max_instances/max_instances cannot be 0.'
        assert self._max_images != 0, "fb_max_images/max_images cannot be 0."
        overrides = remove_custom_fb_args(overrides) # The custom arguments must be removed before calling super.__init___
        # To use overrides we must apply these to the checkpoint file itself (only applies if we resume a training run)
        # otherwise the overrides are overwritten by the old training arguments stored within the checkpoint file
        apply_overrides_to_checkpoint(overrides)
        super().__init__(cfg, overrides, _callbacks, *args, **kwargs)
        if overrides.get("resume", False):
            self.args.__dict__.update(overrides)
        
        self.args.__dict__.update(custom_fb_args) # But we need to add them back, otherwise they will be missing in DDP mode
        if overrides.get("resume", False):
            self.args.resume = True
        self.add_callback("on_train_epoch_start", FlatBugSegmentationTrainer.log_lr)
        # self.use_ewa_sampler()

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

    def setup_model(self : Self) -> Optional[Dict]:
        if isinstance(self.model, torch.nn.Module):  # if model is loaded beforehand. No setup needed
            return

        model, weights = self.model, None
        ckpt = None
        if str(model).endswith('.pt'):
            weights, ckpt = attempt_load_one_weight(model)
            if hasattr(ckpt['model'], 'yaml'):
                cfg = ckpt['model'].yaml
            else:
                cfg = weights.yaml
        else:
            cfg = model
        self.model = self.get_model(cfg=cfg, weights=weights, verbose=RANK == -1)  # calls Model(cfg, weights)
        if not self.args.resume:
            return None
        return ckpt
    
    @property
    def exclude_pattern(self : Self) -> str:
        return f'^(?!({"|".join(self._exclude_datasets)}))' if self._exclude_datasets else ""

    def build_dataset(
            self : Self, 
            img_path : str, 
            mode : str='train', 
            batch : Optional[int]=None
        ) -> Union[FlatBugYOLODataset, FlatBugYOLOValidationDataset]:
        LOGGER.info(f"Building dataset with max instances ({self._max_instances}), max images ({self._max_images}) and exclude pattern ({self.exclude_pattern}).")
        if mode == "train":
            dataset = FlatBugYOLODataset(
                data=yaml_load(self.args.data),
                img_path=img_path,
                imgsz=self.args.imgsz,
                cache=self.args.cache,
                augment=mode == "train",
                hyp=self.args,
                rect=self.args.rect if mode == "train" else True,
                batch_size=batch,
                # stride=int(stride),
                pad=0.0 if mode == "train" else 0.5,
                single_cls=self.args.single_cls or False,
                max_instances=self._max_instances,
                task="segment",
                subset_args={"n" : self._max_images, "pattern" : self.exclude_pattern}
            )
        else:
            dataset = FlatBugYOLOValidationDataset(
                data=yaml_load(self.args.data),
                img_path=img_path,
                imgsz=self.args.imgsz,
                cache=self.args.cache,
                augment=mode == "train",
                hyp=self.args,
                rect=self.args.rect if mode == "train" else True,
                batch_size=batch,
                # stride=int(stride),
                pad=0.0 if mode == "train" else 0.5,  # fixme... does not make sense...
                single_cls=self.args.single_cls or False,
                max_instances=np.inf,
                task="segment",
                subset_args={"n" : self._max_images, "pattern" : self.exclude_pattern}
            )

        return dataset
    
    def get_dataloader(
            self : Self, 
            dataset_path : str, 
            batch_size : Optional[int]=16, 
            rank : int=0, 
            mode : str="train"
        ) -> InfiniteDataLoader:
        """Construct and return dataloader."""
        assert mode in {"train", "val"}, f"Mode must be 'train' or 'val', not {mode}."
        with torch_distributed_zero_first(rank):  # init dataset *.cache only once if DDP
            dataset = self.build_dataset(dataset_path, mode, batch_size)
        shuffle = mode == "train"
        if getattr(dataset, "rect", False) and shuffle:
            LOGGER.warning("WARNING ⚠️ 'rect=True' is incompatible with DataLoader shuffle, setting shuffle=False")
            shuffle = False
        workers = self.args.workers
        return build_dataloader(dataset, batch_size, workers, shuffle, rank)  # return dataloader

    @smart_inference_mode()
    def validate(self : Self) -> Tuple[Dict, float]:
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
    def training_image_paths(self : Self) -> List[str]:
        try:
            return self.train_loader.dataset.im_files
        except Exception as e:
            logger.error("Perhaps the trainer has not been activated yet. Accessing the training image paths is not possible, while training has not started.")
            raise e
        
    @property
    def val_image_paths(self : Self) -> List[str]:
        try:
            return self.test_loader.dataset.im_files
        except Exception as e:
            logger.error("Perhaps the trainer has not been activated yet. Accessing the validation image paths is not possible, while training has not started.")
            raise e

    def _reproducibility_setup(self : Self):
        if not RANK in {-1, 0}:
            logger.warning("Reproducibility setup skipped for non-master rank.")
            return
        def log_data(self):
            with open(self.save_dir / "data_log.json", "w") as f:
                json.dump({**{k : str(v) for k, v in self.data.items()}, **{"train_images" : self.training_image_paths, "val_images" : self.val_image_paths}}, f)
        self.add_callback("on_train_start", log_data)

    def get_validator(self : Self) -> yolo.segment.SegmentationValidator:
        """Return an instance of SegmentationValidator for validation of YOLO model."""
        self.loss_names = "box_loss", "seg_loss", "cls_loss", "dfl_loss"
        return yolo.segment.SegmentationValidator(
            self.test_loader, save_dir=self.save_dir, args=remove_custom_fb_args(copy(self.args)), _callbacks=self.callbacks
        )