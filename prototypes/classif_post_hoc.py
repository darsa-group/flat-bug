import shutil
import glob
import os
import hashlib
from ultralytics import YOLO
ROOT_DIR = "/home/quentin/Insync/qgeissmann@gmail.com/Google Drive - Shared with me/PIEE Lab - Shared/Projects/Sticky Card Scans/Michelle Franklin Weevil Sticky Cards/crops"
DATASET_DIR = "/tmp/classif_weevil"
MAX_PER_CLASS = 200



dir_to_classes = {"non-targets": "00_non-target",
				  "Weevils": "01_weevil",
				  "Medium to High Confidence Drosophila Parasitoids": "02_parasitoid",
  				  "Low Confidence Drosophila Parasitoids": "02_parasitoid"
}


counts = {c:0 for c in set(dir_to_classes.values())}

def make_dataset():
	if os.path.isdir(DATASET_DIR):
		shutil.rmtree(DATASET_DIR)
	for k, v in dir_to_classes.items():

		for p in glob.glob(os.path.join(ROOT_DIR, k, "*.png")):
			if counts[v] >= MAX_PER_CLASS:
				continue
			counts[v] = counts[v] + 1
			print(counts)

			with open(p, 'rb') as f:
				file_hash = hashlib.md5(f.read()).hexdigest()

			prob = int(file_hash[0:4], 16) / int("ffff", 16)

			if prob >= 0.20 :
				subset = "train"
			else:
				subset = "val"

			dst = os.path.join(DATASET_DIR, subset, v)

			os.makedirs(dst, exist_ok=True)
			shutil.copy(p, dst)


# make_dataset()
# model = YOLO('yolov8m-cls.pt')  # load a pretrained model (recommended for training)
# results = model.train(data=DATASET_DIR, batch=32, epochs=400, imgsz=256)
model = YOLO('runs/classify/train10/weights/best.pt')  # load a pretrained model (recommended for training)
out = "/tmp/prosp_class_weev"

crop_path = "/home/quentin/Insync/qgeissmann@gmail.com/Google Drive - Shared with me/PIEE Lab - Shared/Projects/Sticky Card Scans/Michelle Franklin Weevil Sticky Cards/crops"
for f in glob.glob(os.path.join(crop_path, "*.png")):
	o = model.predict(f)[0]
	conf = o.probs.top1conf
	cls = o.names[o.probs.top1]
	print(cls, float(conf))
	if conf < 0.95:
		cls = "99_undefined"
	dst = os.path.join(out, cls)
	print ("---------------------")
	os.makedirs(dst, exist_ok=True)
	shutil.copy(f, dst)

