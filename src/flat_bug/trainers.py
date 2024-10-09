import os, glob, random, json

from copy import copy

from typing import Self, Union, List, Tuple, Dict, Any, Optional

import numpy as np
import torch

from ultralytics.models import yolo
from ultralytics.models.yolo.segment import SegmentationTrainer
from ultralytics.nn.tasks import attempt_load_one_weight
from ultralytics.utils import DEFAULT_CFG, LOGGER, RANK, __version__, yaml_load, IterableSimpleNamespace
from ultralytics.utils.torch_utils import smart_inference_mode, torch_distributed_zero_first
from ultralytics.data import build_dataloader
from ultralytics.data.utils import PIN_MEMORY
from ultralytics.data.build import InfiniteDataLoader, seed_worker
from ultralytics.utils.checks import print_args
from ultralytics.cfg import get_save_dir

from flat_bug import logger
from flat_bug.datasets import MyYOLODataset, MyYOLOValidationDataset

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
        LOGGER.warning(f"No weights found in {weight_dir}")
        return None
    return max(weights, key=os.path.getctime)

def _custom_end_to_end_validation(self : "MySegmentationTrainer"):
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
        LOGGER.info(f"Running custom end-to-end validation command: `{command}`")
        # Run command
        os.system(command)

# ### WORK IN PROGRESS - NOT YET FUNCTIONAL ###
# class TrackingWeightedRandomSampler(torch.utils.data.WeightedRandomSampler):
#     """Sampler that tracks the indices it samples."""
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.sampled_indices = []

#     def __iter__(self):
#         self.sampled_indices = []
#         for idx in super().__iter__():
#             self.sampled_indices.append(idx)
#             yield idx

# ### WORK IN PROGRESS - NOT YET FUNCTIONAL ###
# class DynamicWeightedDataLoader(InfiniteDataLoader):
#     def __init__(self, dataset, batch_size, num_workers, pin_memory, generator, alpha=0.9, **kwargs):

#         self.dataset = dataset
#         self.alpha = alpha  # Smoothing factor for EWA
#         self.ewa_losses = None  # Initialize EWA loss tracker
#         weights = torch.ones(len(dataset))  # Start with uniform weights
#         self.sampler = TrackingWeightedRandomSampler(weights, num_samples=len(dataset), generator=generator)
#         if "sampler" in kwargs:
#             del kwargs["sampler"]
#         super().__init__(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, generator=generator, sampler=self.sampler, **kwargs)

#     def update_sampler(self, current_epoch_losses):
#         # Apply the EWA to update the losses based on the indices sampled in the last epoch
#         current_losses = np.array(current_epoch_losses)[self.sampler.sampled_indices]

#         if self.ewa_losses is None:
#             self.ewa_losses = current_losses
#         else:
#             self.ewa_losses = self.alpha * self.ewa_losses + (1 - self.alpha) * current_losses

#         weights = np.exp(-self.ewa_losses)
#         weights /= np.sum(weights)
#         self.sampler.weights = torch.as_tensor(weights, dtype=torch.double)
    
#     def reset(self):
#         self.ewa_losses = None
#         self.current_epoch_losses = None
#         super().reset()

# def build_dataloader(dataset, batch, workers, shuffle=True, rank=-1, alpha=0.9):
#     """Return an InfiniteDataLoader or DataLoader for training or validation set."""
#     batch = min(batch, len(dataset))
#     nd = torch.cuda.device_count()  # number of CUDA devices
#     nw = min([os.cpu_count() // max(nd, 1), workers])  # number of workers
#     generator = torch.Generator()
#     generator.manual_seed(6148914691236517205 + rank)
    
#     if rank != -1:
#         # Distributed case: use the old implementation
#         sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=shuffle)
#         return InfiniteDataLoader(
#             dataset=dataset,
#             batch_size=batch,
#             shuffle=shuffle and sampler is None,
#             num_workers=nw,
#             sampler=sampler,
#             pin_memory=PIN_MEMORY,
#             collate_fn=getattr(dataset, "collate_fn", None),
#             worker_init_fn=seed_worker,
#             generator=generator,
#         )
#     else:
#         # Non-distributed case: use DynamicWeightedDataLoader with updated sampler based on EWA of losses
#         return DynamicWeightedDataLoader(
#             dataset=dataset,
#             batch_size=batch,
#             shuffle=False,
#             num_workers=nw,
#             sampler=None,
#             pin_memory=PIN_MEMORY,
#             collate_fn=getattr(dataset, "collate_fn", None),
#             worker_init_fn=seed_worker,
#             generator=generator,
#             alpha=alpha
#         )
    
# def save_loss_items(self):
#     logger.info(self.loss_items)
#     raise NotImplementedError
#     self.current_epoch_losses += self.loss_items.tolist()

# def update_sampler(self):
#     self.train_loader.update_sampler(self.current_epoch_losses)
#     self.current_epoch_losses = []

# def override_get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
#         """Construct and return dataloader."""
#         assert mode in {"train", "val"}, f"Mode must be 'train' or 'val', not {mode}."
#         with torch_distributed_zero_first(rank):  # init dataset *.cache only once if DDP
#             dataset = self.build_dataset(dataset_path, mode, batch_size)
#         shuffle = mode == "train"
#         if getattr(dataset, "rect", False) and shuffle:
#             LOGGER.warning("WARNING ⚠️ 'rect=True' is incompatible with DataLoader shuffle, setting shuffle=False")
#             shuffle = False
#         workers = self.args.workers if mode == "train" else self.args.workers * 2
#         return build_dataloader(dataset, batch_size, workers, shuffle, rank)  # return dataloader

# def use_ewa_sampler(self):
#     raise NotImplementedError("EWA (Exponentially Weighted Average of losses) sampler is not yet implemented.")
#     self.current_epoch_losses = []
#     self.add_callback("on_train_batch_end", save_loss_items)
#     self.add_callback("on_train_epoch_end", update_sampler)
#     self.get_dataloader = override_get_dataloader


class MySegmentationTrainer(SegmentationTrainer):
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
        super().__init__(cfg, overrides, _callbacks, *args, **kwargs)
        if overrides.get("resume", False):
            self.args.__dict__.update(overrides)
            delattr(self.args, "save_dir")
            self.args.save_dir = get_save_dir(self.args)
            self.args.name = self.args.save_dir.name  # update name for loggers
            print("Resuming with:")
            print(print_args(vars(self.args)))
        self.args.__dict__.update(custom_fb_args) # But we need to add them back, otherwise they will be missing in DDP mode
        if overrides["resume"]:
            self.args.resume = True
        self.add_callback("on_train_epoch_start", MySegmentationTrainer.log_lr)
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
        ) -> Union[MyYOLODataset, MyYOLOValidationDataset]:
        logger.info(f"Building dataset with max instances ({self._max_instances}), max images ({self._max_images}) and exclude pattern ({self.exclude_pattern}).")
        if mode == "train":
            dataset = MyYOLODataset(
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
            dataset = MyYOLOValidationDataset(
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
                max_instances=np.Inf,
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