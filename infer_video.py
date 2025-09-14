
# infer_video.py
import argparse, cv2, torch
from torchvision.transforms.functional import to_tensor
from model import load_model
from utils_vis import draw_keypoints_and_skeleton

def main():
    parser = argparse.ArgumentParser(description="Human Pose Estimation on webcam/video")
    parser.add_argument("--source", default="0", help="Webcam index like '0' or video path")
    parser.add_argument("--score", type=float, default=0.6, help="Min person score")
    args = parser.parse_args()

    src = 0 if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError("Cannot open source")

    model, device = load_model(score_thresh=args.score)

    while True:
        ok, frame = cap.read()
        if not ok: break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        x = to_tensor(rgb).to(device)

        with torch.inference_mode():
            pred = model([x])[0]

        keep = pred["scores"] >= args.score
        kps = pred["keypoints"][keep].cpu().numpy()
        vis = draw_keypoints_and_skeleton(frame, kps, kp_thresh=0.2)

        cv2.imshow("Pose", vis)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
