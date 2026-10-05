import argparse
import cv2
from pathlib import Path


def variance_of_laplacian(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()


def extract_frames(input_dir, output_dir, target_fps, blur_thresh):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    video_exts = {".mp4", ".mov", ".avi", ".mkv"}
    videos = [p for p in input_dir.rglob("*") if p.suffix.lower() in video_exts]

    if not videos:
        print(f"No videos found in {input_dir}")
        return

    total_saved, total_skipped_blur = 0, 0

    for video_path in videos:
        cap = cv2.VideoCapture(str(video_path))
        native_fps = cap.get(cv2.CAP_PROP_FPS) or 30
        frame_interval = max(1, round(native_fps / target_fps))

        stem = video_path.stem
        frame_idx, saved_idx = 0, 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_interval == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                blur_score = variance_of_laplacian(gray)

                if blur_score < blur_thresh:
                    total_skipped_blur += 1
                else:
                    out_name = f"{stem}_f{saved_idx:05d}.jpg"
                    cv2.imwrite(
                        str(output_dir / out_name),
                        frame,
                        [cv2.IMWRITE_JPEG_QUALITY, 95],
                    )
                    saved_idx += 1
                    total_saved += 1

            frame_idx += 1

        cap.release()
        print(f"{video_path.name}: saved {saved_idx} frames")

    print(
        f"\nTotal: {total_saved} frames saved, {total_skipped_blur} skipped (too blurry)"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="data/raw_videos")
    parser.add_argument("--output_dir", default="data/frames")
    parser.add_argument("--fps", type=float, default=5.0, help="Target sampling rate")
    parser.add_argument(
        "--blur_thresh",
        type=float,
        default=50.0,
        help="Lower = more permissive of blur. Start here, adjust after inspecting output.",
    )
    args = parser.parse_args()

    extract_frames(args.input_dir, args.output_dir, args.fps, args.blur_thresh)