import cv2
import argparse
import os
from video_stabilization import stabilization

def parse_args():
    parser = argparse.ArgumentParser(description="Video Stabilization using ORB + Optical Flow")

    parser.add_argument('--input', type=str, required=True,
                        help='Path to input video file (or "webcam" to use webcam)')

    parser.add_argument('--output', type=str, default=None,
                        help='Path to save stabilized output video')

    parser.add_argument('--show', action='store_true',
                        help='Flag to display live output')

    parser.add_argument('--resize', type=int, nargs=2, metavar=('width', 'height'), default=None,
                        help='Resize video to this resolution (width height)')

    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    # Open input video or webcam
    if args.input.lower() == 'webcam':
        cap = cv2.VideoCapture(0)
    else:
        if not os.path.exists(args.input):
            print(f"[ERROR] Input file not found: {args.input}")
            return
        cap = cv2.VideoCapture(args.input)

    # Prepare writer if output is set
    out_writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if args.resize:
            width, height = args.resize
        out_writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    # Read first frame
    ret, prev_frame = cap.read()
    if not ret:
        print("[ERROR] Failed to read first frame.")
        return

    while True:
        ret, curr_frame = cap.read()
        if not ret:
            break

        if args.resize:
            prev_frame = cv2.resize(prev_frame, tuple(args.resize))
            curr_frame = cv2.resize(curr_frame, tuple(args.resize))

        stabilized_frame = stabilization(prev_frame, curr_frame)

        if args.show:
            cv2.imshow("Stabilized Output", stabilized_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        if out_writer:
            out_writer.write(stabilized_frame)

        prev_frame = curr_frame.copy()

    cap.release()
    if out_writer:
        out_writer.release()
    cv2.destroyAllWindows()
    print("[INFO] Video stabilization completed.")

if __name__ == '__main__':
    main()
