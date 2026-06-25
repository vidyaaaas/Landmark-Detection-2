from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from landmark_detection.detector import DetectorOptions, LandmarkDetector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect face, hand, and pose landmarks in images or webcam video."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", type=Path, help="Path to an input image.")
    source.add_argument("--webcam", action="store_true", help="Use the default webcam.")
    parser.add_argument("--output", type=Path, help="Where to save an annotated image.")
    parser.add_argument("--faces", action="store_true", help="Enable face landmarks.")
    parser.add_argument("--hands", action="store_true", help="Enable hand landmarks.")
    parser.add_argument("--pose", action="store_true", help="Enable pose landmarks.")
    parser.add_argument("--camera-index", type=int, default=0, help="Webcam index.")
    return parser.parse_args()


def build_options(args: argparse.Namespace) -> DetectorOptions:
    selected = args.faces or args.hands or args.pose
    return DetectorOptions(
        detect_faces=args.faces if selected else True,
        detect_hands=args.hands if selected else True,
        detect_pose=args.pose if selected else True,
    )


def run_image(args: argparse.Namespace, detector: LandmarkDetector) -> None:
    image = cv2.imread(str(args.image))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {args.image}")

    output = detector.process_frame(image)
    if args.output:
        cv2.imwrite(str(args.output), output)
        print(f"Saved annotated image to {args.output}")
        return

    cv2.imshow("Landmark Detection", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def run_webcam(args: argparse.Namespace, detector: LandmarkDetector) -> None:
    capture = cv2.VideoCapture(args.camera_index)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open webcam index {args.camera_index}")

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            output = detector.process_frame(frame)
            cv2.imshow("Landmark Detection", output)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()


def main() -> None:
    args = parse_args()
    detector = LandmarkDetector(build_options(args))
    try:
        if args.webcam:
            run_webcam(args, detector)
        else:
            run_image(args, detector)
    finally:
        detector.close()


if __name__ == "__main__":
    main()
