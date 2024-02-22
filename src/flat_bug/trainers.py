from ultralytics.models.yolo.segment import SegmentationTrainer
from flat_bug.datasets import MyYOLODataset, MyYOLOValidationDataset
from ultralytics.utils import yaml_load, DEFAULT_CFG, RANK, LOGGER

import torch
from ultralytics.nn.tasks import attempt_load_one_weight, attempt_load_weights
from ultralytics.utils import (DEFAULT_CFG, LOGGER, RANK, SETTINGS, TQDM_BAR_FORMAT, __version__, callbacks, clean_url,
                               colorstr, emojis, yaml_save)

from ultralytics.utils.torch_utils import (EarlyStopping, ModelEMA, de_parallel, init_seeds, one_cycle, select_device,
                                           strip_optimizer, smart_inference_mode)


class MySegmentationTrainer(SegmentationTrainer):
    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None, val_every=10, *args, **kwargs):
        """Initialize a SegmentationTrainer object with given arguments."""

        super().__init__(cfg, overrides, _callbacks, *args, **kwargs)
        if overrides["resume"]:
            self.args.resume = True
        self.add_callback("on_train_epoch_start", MySegmentationTrainer.log_lr)

        self._val_metrics = None
        self._val_fitness = None
        self._val_every = val_every
        self.cfg = cfg

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

    def build_dataset(self, img_path, mode='train', batch=None):

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
                use_segments=True,
                max_instances=self.args.max_instances
            )
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
                use_segments=True,
                max_instances=self.args.max_instances
            )

        return dataset

    @smart_inference_mode()
    def validate(self):
        """
        Runs validation on test set using self.validator. The returned dict is expected to contain "fitness" key.
        """
        if self.epoch % self._val_every == 0 or self._val_metrics is None:
            metrics, fitness = super().validate()
            self._val_metrics, self._val_fitness = metrics, fitness

        else:
            metrics, fitness = self._val_metrics, self._val_fitness
            LOGGER.info(f"Skipped validation at epoch {self.epoch}, using old values")
        return metrics, fitness
