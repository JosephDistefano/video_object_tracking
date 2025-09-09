import argparse
from validation_main  import visualize_video
import sys
import os

def parse_args():
    parser = argparse.ArgumentParser(
        description="View video annotations with navigation"
    )
    parser.add_argument(
        "video_path",
        type=str,
        help="Path to the input video file"
    )
    parser.add_argument(
        "--annotations",
        type=str,
        default=None,
        help="Path to the annotation file (default: <video_name>.annotations)"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    annotation_file = args.annotations or os.path.splitext(args.video_path)[0] + ".annotations"
    visualize_video(args.video_path, annotation_file)

if __name__ == "__main__":
    main()
