
## Downloading data
```sh
# change as you see fit
DATA_DIR=~/Desktop/flat-bug-sorted-data
```

## Preparing data
```sh

fb_prepare_data.py -i ${DATA_DIR}/pre-pro/ -o ${DATA_DIR}/yolo
```

## Configuration file:

Example `fb_config.yaml` file:
```yaml
batch: 6
model: yolov8m-seg.pt # or path to other model, including pretrained
epochs: 4000
device: "cuda"
patience: 500
lr0: 0.01
lrf: 0.0005
workers: 4
```

## Training 

```sh
fb_train.py -d ${DATA_DIR}/yolo -c fb_config.yaml
```


## Inference 

```sh
fb_predict.py -i ${DATA_DIR}/pre-pro/AMI-traps -w runs/segment/train59/weights/last.pt -o /tmp/AMI-traps-preds
```
