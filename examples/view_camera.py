"""Preview a single OpenCV camera in a window.

Usage:
    python examples/view_camera.py --index 0
    python examples/view_camera.py --index 1

Press 'q' or ESC to quit. The window title shows the camera index and measured FPS.
"""

import argparse
import time

import cv2


def main():
    parser = argparse.ArgumentParser(description="Preview a single OpenCV camera.")
    parser.add_argument("--index", type=int, required=True, help="Camera index, e.g. 0 or 1")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.index, cv2.CAP_MSMF)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {args.index}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)

    window = f"camera {args.index} (press q to quit)"
    frame_count = 0
    t_start = time.perf_counter()
    fps_text = "..."

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Frame read failed, retrying...")
            time.sleep(0.1)
            continue

        frame_count += 1
        elapsed = time.perf_counter() - t_start
        if elapsed >= 1.0:
            fps_text = f"{frame_count / elapsed:.1f}"
            frame_count = 0
            t_start = time.perf_counter()

        cv2.putText(frame, f"cam {args.index}  fps {fps_text}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow(window, frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):  # q or ESC
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
