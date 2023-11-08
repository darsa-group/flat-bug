from ultralytics.models.yolo.segment import SegmentationTrainer
from flat_bug.datasets import MyYOLODataset, MyYOLOValidationDataset
from ultralytics.utils import yaml_load, DEFAULT_CFG

class MySegmentationTrainer(SegmentationTrainer):
    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """Initialize a SegmentationTrainer object with given arguments."""
        if overrides is None:
            overrides = {}
        overrides['task'] = 'segment'

        print(cfg)
        super().__init__(cfg, overrides, _callbacks)
        print(self.args)
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
