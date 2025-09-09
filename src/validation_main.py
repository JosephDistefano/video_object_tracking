import cv2
import os

def load_annotations(annotation_path):
    """
    Load annotations from a file into a dictionary.

    Args:
        annotation_path (str): Path to annotation file.

    Returns:
        dict: Dictionary with frame index as key and annotation list as value.
              Example:
              {
                  0: ['V', '320', '240', '50', '60'],
                  1: ['S', '-1', '-1', '-1', '-1']
              }

    Raises:
        FileNotFoundError: If annotation file does not exist.
    """
    annotations = {}
    with open(annotation_path, "r") as f:
        for i, line in enumerate(f):
            parts = line.strip().split()
            if not parts:
                continue
            key = int(i)  # frame index
            annotations[key] = parts
    return annotations

def visualize_video(video_path, annotation_path):
    """
    Visualize video frames with annotations overlayed.

    This function opens a video, reads annotations, and displays each frame
    with bounding boxes or status labels. Navigation keys allow moving through frames.

    Args:
        video_path (str): Path to the video file.
        annotation_path (str): Path to the annotation file.

    Raises:
        FileNotFoundError: If video or annotation file is missing.

    Key Controls:
        'n'  - Next frame
        'N'  - Next 10 frames
        'p'  - Previous frame
        'P'  - Previous 10 frames
        'q'  - Quit visualization
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    
    if not os.path.exists(annotation_path):
        raise FileNotFoundError(f"Annotation file not found: {annotation_path}")

    # Load annotations into a dictionary
    annotations = load_annotations(annotation_path)

    # Open video capture
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idx = 0

    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()

        if not ret:
            print("End of video reached.")
            break

        # Display annotations
        if frame_idx in annotations:
            parts = annotations[frame_idx]
            status = parts[0]
            if status == 'V':
                _, cx, cy, w, h = parts
                cx, cy, w, h = map(int, [cx, cy, w, h])
                x = cx - w // 2
                y = cy - h // 2
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            else:
                cv2.putText(frame, f"Frame {frame_idx}: {status}", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Display Frame Index
        cv2.putText(frame, f"Frame {frame_idx}/{total_frames}", (10, frame.shape[0]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.imshow("Annotation Viewer", frame)

        # Handle key inputs
        key = cv2.waitKey(0) & 0xFF
        if key == ord('n'):
            frame_idx = min(frame_idx + 1, total_frames - 1)
        elif key == ord('N'):
            frame_idx = min(frame_idx + 10, total_frames - 1)
        elif key == ord('p'):
            frame_idx = max(frame_idx - 1, 0)
        elif key == ord('P'):
            frame_idx = max(frame_idx - 10, 0)
        elif key == ord('q'):
            break
        
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
