
# infer_image.py
import argparse, cv2, torch
from torchvision.transforms.functional import to_tensor
from model import load_model
from utils_vis import draw_keypoints_and_skeleton

def main():
    parser = argparse.ArgumentParser(description="Human Pose Estimation on an image")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--out", default="pose_out.jpg", help="Output image path")
    parser.add_argument("--score", type=float, default=0.6, help="Min person score")
    args = parser.parse_args()

    img_bgr = cv2.imread(args.image)
    if img_bgr is None:
        raise FileNotFoundError(args.image)

    model, device = load_model(score_thresh=args.score)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    x = to_tensor(img_rgb).to(device)
    with torch.inference_mode():
        pred = model([x])[0]

    keep = pred["scores"] >= args.score
    keypoints = pred["keypoints"][keep].cpu().numpy()  # (N,17,3)

    vis = draw_keypoints_and_skeleton(img_bgr, keypoints, kp_thresh=0.2)
    cv2.imwrite(args.out, vis)
    print(f"Saved: {args.out} (people detected: {len(keypoints)})")

if __name__ == "__main__":
    main()
