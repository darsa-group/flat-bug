
mkdir /tmp/val-results
mkdir /tmp/val-preds
fb_predict.py -i /tmp/yolo/insects/images/val/  -w ~/Desktop/fb_2024-02-19_best.pt -o /tmp/val-preds
fb_eval.py -g /tmp/yolo/insects/labels/val/instances_default.json -p '/tmp/val-preds/**/*.json' -I /tmp/yolo/insects/images/val -o /tmp/val-results -P
