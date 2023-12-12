from flat_bug.predictor import Predictor


IMG = "images/AMI-traps_LV1_20230530025959-10-snapshot.jpg"
MOD = "../runs/segment/train75/weights/last.pt"
pred = Predictor(MOD)

out = pred.pyramid_predictions(IMG, scale_before=0.5)
out2 = pred.pyramid_predictions(IMG, scale_before=.7)

comp = out.compare(out2, 0.5)



tp,  fn, fp = 0,0,0
for i in comp:
    if i["in_gt"] and i["in_im"]:
        tp += 1
    elif i["in_gt"] and not i["in_im"]:
        fn += 1
    elif not i["in_gt"] and i["in_im"]:
        fp += 1

recall =  tp/ (tp + fn)
precision =  tp/ (tp + fp)
print(tp, fn, fp, recall, precision)


