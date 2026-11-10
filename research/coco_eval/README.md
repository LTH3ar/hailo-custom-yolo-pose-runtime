# COCO keypoint accuracy (official protocol)

The training log scores the 13-keypoint models with Ultralytics' validator. Without `kpt_oks_sigmas`
in the dataset file, it gives every keypoint the same tolerance (1/13). Those numbers can't be
compared with published COCO results. This folder re-scores the models with pycocotools and COCO's
own per-keypoint sigmas, on COCO val2017 (5,000 images), the same way for every model:

| Rows | What runs | Where |
| --- | --- | --- |
| stock (17 → 13) | Ultralytics `yolov8{n,s,m}-pose.pt`, cut to the 13 upper-body points | GPU machine |
| trained .pt | the 13-keypoint models from `hailo-custom-yolo-pose`, FP32 | GPU machine |
| HEF INT8 | the deployed `upper_body_{n,s,m}.hef` on the Hailo-8 | Pi 5 |

All three use hcpose's square 640×640 letterbox, so trained .pt vs HEF isolates the cost of INT8
quantisation. Settings: score threshold 0.001, NMS IoU 0.7, at most 100 detections per image
(COCO counts the top 20).

```bash
# GPU machine (Ultralytics + pycocotools in the venv)
P=.venv/bin/python; E=research/coco_eval/coco_kpt_eval.py; C=~/coco
$P $E gt13 $C/annotations/person_keypoints_val2017.json $C/gt13.json
$P $E pt yolov8n-pose.pt $C/val2017 $C/preds/stock_n.json --keep 17
$P $E pt TRAINED_N.pt   $C/val2017 $C/preds/pt_n.json
# Pi (hailo_platform + pycocotools in the venv)
$P $E hef ~/models/upper_body_n.hef ~/coco/val2017 ~/coco/preds/hef_n.json
# scoring (anywhere)
$P $E score $C/gt13.json stock_n=$C/preds/stock_n.json pt_n=$C/preds/pt_n.json hef_n=$C/preds/hef_n.json --md out.md
$P $E score $C/gt13.json ... --sigmas uniform        # close to the training log's protocol
$P $E score $C/annotations/person_keypoints_val2017.json stock_n=$C/preds/stock_n.json   # 17 points: sanity check against the published numbers
```

`--limit 50` on `pt`/`hef` and `--only-predicted` on `score` make a quick smoke test.
Results: [results.md](results.md).
