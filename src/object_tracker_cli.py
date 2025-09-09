import argparse
from object_tracker_main import Object_Tracker

def parse_args():
    parser = argparse.ArgumentParser(
        description="Label a single object across a video"
    )
    parser.add_argument(
        "video_path",
        type=str,
        help="Path to the input video file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save annotations(default: <video_name>.annotations)"
    )
    parser.add_argument("--prompt", type=str, help="Text prompt for object detection", default=None)
    return parser.parse_args()

def main():
    args = parse_args()
    OT = Object_Tracker(args.video_path, args.output, args.prompt)
    OT.run()

if __name__ == "__main__":
    main()      