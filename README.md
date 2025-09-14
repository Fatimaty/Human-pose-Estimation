
# Human Pose Estimation (Torchvision Keypoint R-CNN)

**Goal:** Detect 17 COCO keypoints (nose, shoulders, elbows, wrists, hips, knees, ankles, etc.) and draw a skeleton on images or webcam video.

- Framework: **PyTorch + torchvision**
- Model: **Keypoint R-CNN ResNet50 FPN** (pretrained on COCO keypoints)
- Scripts: image inference and webcam/video inference
- No training required — ready to run

## Setup
```bash
pip install torch torchvision opencv-python
```

## Run on an image
```bash
python infer_image.py --image path/to/person.jpg --out pose_out.jpg --score 0.6
```

## Run on webcam / video
```bash
# Webcam 0
python infer_video.py --source 0
# Or a video file
python infer_video.py --source path/to/video.mp4
```

## Files
- `model.py` — loads the pretrained Keypoint R-CNN model
- `utils_vis.py` — drawing utilities for keypoints and skeleton
- `infer_image.py` — run pose estimation on a single image
- `infer_video.py` — realtime pose on webcam/video
- `README.md` — instructions

## Notes
- COCO **17 keypoints** are used. See `utils_vis.py` for the names and the skeleton connections.
- The `--score` flag filters out low-confidence detections.

