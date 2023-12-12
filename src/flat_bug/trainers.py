from ultralytics.models.yolo.segment import SegmentationTrainer
from flat_bug.datasets import MyYOLODataset, MyYOLOValidationDataset
from ultralytics.utils import yaml_load, DEFAULT_CFG, RANK, LOGGER

import torch
from ultralytics.nn.tasks import attempt_load_one_weight, attempt_load_weights
from ultralytics.utils import (DEFAULT_CFG, LOGGER, RANK, SETTINGS, TQDM_BAR_FORMAT, __version__, callbacks, clean_url,
                               colorstr, emojis, yaml_save)

from ultralytics.utils.torch_utils import (EarlyStopping, ModelEMA, de_parallel, init_seeds, one_cycle, select_device,
                                           strip_optimizer, smart_inference_mode)


# class CustomValidator(SegmentationValidator):
#     @smart_inference_mode()
# def __call__(self, trainer=None, model=None):
#     """Supports validation of a pre-trained model if passed or a model being trained if trainer is passed (trainer
#     gets priority).
#     """
#     self.training = trainer is not None
#
#     if self.training:
#         self.device = trainer.device
#         self.data = trainer.data
#         self.args.half = self.device.type != 'cpu'  # force FP16 val during training
#         model = trainer.ema.ema or trainer.model
#         model = model.half() if self.args.half else model.float()
#         # self.model = model
#         self.loss = torch.zeros_like(trainer.loss_items, device=trainer.device)
#         self.args.plots &= trainer.stopper.possible_stop or (trainer.epoch == trainer.epochs - 1)
#         model.eval()
#     else:
#         callbacks.add_integration_callbacks(self)
#         model = AutoBackend(model or self.args.model,
#                             device=select_device(self.args.device, self.args.batch),
#                             dnn=self.args.dnn,
#                             data=self.args.data,
#                             fp16=self.args.half)
#         # self.model = model
#         self.device = model.device  # update device
#         self.args.half = model.fp16  # update half
#         stride, pt, jit, engine = model.stride, model.pt, model.jit, model.engine
#         imgsz = check_imgsz(self.args.imgsz, stride=stride)
#         if engine:
#             self.args.batch = model.batch_size
#         elif not pt and not jit:
#             self.args.batch = 1  # export.py models default to batch-size 1
#             LOGGER.info(f'Forcing batch=1 square inference (1,3,{imgsz},{imgsz}) for non-PyTorch models')
#
#         if isinstance(self.args.data, str) and self.args.data.split('.')[-1] in ('yaml', 'yml'):
#             self.data = check_det_dataset(self.args.data)
#         elif self.args.task == 'classify':
#             self.data = check_cls_dataset(self.args.data, split=self.args.split)
#         else:
#             raise FileNotFoundError(emojis(f"Dataset '{self.args.data}' for task={self.args.task} not found ❌"))
#
#         if self.device.type in ('cpu', 'mps'):
#             self.args.workers = 0  # faster CPU val as time dominated by inference, not dataloading
#         if not pt:
#             self.args.rect = False
#         self.stride = model.stride  # used in get_dataloader() for padding
#         self.dataloader = self.dataloader or self.get_dataloader(self.data.get(self.args.split), self.args.batch)
#
#     model.eval()
#     predictor = Predictor(model, cfg=self.args)
#     all_comps = []
#     all_recall, all_precisions = [], []
#     for l in self.dataloader.dataset.labels:
#         print(l["im_file"])
#
#         pred = predictor.pyramid_predictions(l["im_file"])
#         gt = LabelPredictions(l, model.names)
#         comp = pred.compare(gt, 0.5)
#
#         tp, fn, fp = 0, 0, 0
#         for i in comp:
#             if i["in_gt"] and i["in_im"]:
#                 tp += 1
#             elif i["in_gt"] and not i["in_im"]:
#                 fn += 1
#             elif not i["in_gt"] and i["in_im"]:
#                 fp += 1
#
#         if len(pred) == 0 and len(gt) > 0:
#             recall, precision = 0, 0
#         else:
#             recall = tp / (tp + fn)
#             precision = tp / (tp + fp)
#         all_comps.append(comp)
#         print(recall, precision)
#         all_recall.append(recall)
#         all_precisions.append(precision)
#     m_precision = np.mean(all_precisions)
#     m_recall = np.mean(all_recall)
#
#     fitness = (m_recall + m_precision) / 2.0
#
#     metrics = {"metrics/precision": m_precision,
#                "metrics/recall": m_recall}
#     return metrics, fitness


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
                use_segments=True
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
