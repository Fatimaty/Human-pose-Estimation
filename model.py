
# model.py
# Human pose estimation using torchvision's Keypoint R-CNN (COCO 17-keypoints).
import torch
from torchvision.models.detection import keypointrcnn_resnet50_fpn

def load_model(device=None, score_thresh: float = 0.6):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = keypointrcnn_resnet50_fpn(weights='DEFAULT')
    model.to(device).eval()
    model.score_thresh = score_thresh  # used by torchvision's postprocess
    return model, device
