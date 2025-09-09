# Video Object Tracker

A command-line toolkit for annotating and tracking a single object across videos.  
Supports **manual labeling**, **tracker-assisted tracking**, **prompt-based detection via GroundingDINO**, and interactive **video validation**.

Perfect for dataset creation, research in object tracking, and AI-assisted video labeling pipelines.

---

## Features

- Track and annotate a single object throughout a video.
- Manual annotation and tracker-assisted labeling.
- Optional prompt-based detection using GroundingDINO.
- Save annotations in a simple, human-readable text format.
- Validate annotations interactively with frame-by-frame navigation.
- Handles skipped and invisible frames for robust dataset creation.

---

## Installation

### System Requirements

- Ubuntu / Debian-based Linux recommended.
- Python >= 3.10
- GPU recommended for GroundingDINO prompt detection (optional).

### Setup Instructions

```bash
# Clone the repository with submodules
git clone --recurse-submodules https://github.com/JosephDistefano/video_object_tracking.git
cd video_object_tracker

# Install Python virtual environment
sudo apt install python3.10-venv

# Run setup script (downloads dependencies and pretrained models)
sudo bash setup.sh

# Activate the virtual environment
source ./object_tracker/bin/activate
```

###  Project Structure

```bash
video_object_tracker/
├── src/
│   ├── object_tracker_main.py    # Core annotation class (Object_Tracker)
│   ├── validation_main.py        # Functions to visualize and validate annotations
│   ├── cli_annotate.py           # Command-line interface for video annotation
│   └── cli_validate.py           # Command-line interface for annotation validation
├── GroundingDINO/                # Submodule for prompt-based object detection
│   ├── config/                   # Model configuration files
│   └── weights/                  # Pretrained model weights
├── data/                         # Directory for sample videos or datasets
├── requirements.txt              # Python dependencies
├── setup.sh                      # Setup script to install dependencies and models
└── README.md                     # Project documentation
```

### CLI Usage
```bash
python cli_annotate.py <video_path> [--output <output_file>] [--prompt <text>]

```

| Argument     | Description                                                               |
| ------------ | ------------------------------------------------------------------------- |
| `video_path` | Path to the input video file                                              |
| `--output`   | Optional path to save annotations. Defaults to `<video_name>.annotations` |
| `--prompt`   | Optional text prompt for object detection via GroundingDINO               |


Example:
```bash
python3 src/object_tracker_cli.py data/sample_1.mp4
```
Example with GroundingDINO text tracking:
```bash
python3 src/object_tracker_cli.py data/sample_1.mp4 --prompt "car"
```

During Annotation:

| Key | Action                                     |
| --- | ------------------------------------------ |
| L   | Label manually (or via prompt if provided) |
| A   | Accept tracker bounding box                |
| F   | Fix tracker manually                       |
| S   | Skip frame                                 |
| I   | Mark frame as invisible                    |
| Q   | Quit annotation                            |


### Validate Annotations
```bash
python3 src/cli_validate.py <video_path> [--annotations <file>]
```

Arguments:

| Argument        | Description                                                              |
| --------------- | ------------------------------------------------------------------------ |
| `video_path`    | Path to the input video file                                             |
| `--annotations` | Optional path to annotation file. Defaults to `<video_name>.annotations` |


Example:
```bash
python3 src/validation_cli.py data/sample_1.mp4 --annotations data/sample_1.annotations
```

Navigation Keys:

| Key | Action             |
| --- | ------------------ |
| n   | Next frame         |
| N   | Next 10 frames     |
| p   | Previous frame     |
| P   | Previous 10 frames |
| q   | Quit viewer        |


## Annotation Format

| Action | x_center | y_center | width | height | Description                     |
|--------|----------|----------|-------|--------|---------------------------------|
| V      | int      | int      | int   | int    | Visible object with bounding box|
| S      | -1       | -1       | -1    | -1     | Frame skipped                   |
| I      | -1       | -1       | -1    | -1     | Object invisible                |

- Coordinates are **center-based** (`x_center`, `y_center`) relative to the frame.
- Width and height define the size of the bounding box.

## Functions Reference

### `object_tracker_main.py`

#### Class: `Object_Tracker`

- **`__init__(video_path, output_path=None, prompt=None)`**  
  Initialize the video tracker, annotation system, and optional GroundingDINO model.  
  **Args:**  
  - `video_path` (str): Path to the video file.  
  - `output_path` (str, optional): Path to save annotations. Defaults to `<video_name>.annotations`.  
  - `prompt` (str, optional): Text prompt for GroundingDINO object detection.  
  **Raises:** `FileNotFoundError` if the video does not exist.

- **`draw_bbox(frame)`**  
  Manually draw a bounding box on a frame, initialize the tracker, and log the annotation.

- **`accept_tracked_bbox()`**  
  Accepts the current tracker bounding box and logs it as a tracked object.

- **`add_annotation(action, bbox=None)`**  
  Adds an annotation for a frame.  
  **Args:**  
  - `action` (str): 'V', 'S', or 'I'  
  - `bbox` (tuple, optional): `(x, y, w, h)` bounding box for visible frames  
  **Raises:** `ValueError` if action is invalid or `bbox` is missing for visible frames.

- **`detect_with_prompt(frame)`**  
  Runs GroundingDINO prompt-based detection on a frame.  
  **Args:** `frame` (numpy array) – the frame to detect objects on.  
  **Returns:** Bounding box `(x, y, w, h)` or `None` if no object detected.

- **`cleanup()`**  
  Releases video resources, closes annotation file, and performs cleanup.

- **`run()`**  
  Main loop for annotation and tracking. Handles tracker updates, key commands, prompt detection, and interactive labeling.

---

### `validation_main.py`

- **`load_annotations(annotation_path)`**  
  Loads annotations into a dictionary keyed by frame index.  
  **Args:** `annotation_path` (str) – path to the annotation file.  
  **Returns:** Dictionary `{frame_index: [action, x, y, w, h]}`.

- **`visualize_video(video_path, annotation_path)`**  
  Visualizes video frames with annotations overlayed.  
  **Args:**  
  - `video_path` (str): Path to the video file.  
  - `annotation_path` (str): Path to the annotation file.  
  **Behavior:** Draws bounding boxes for visible frames, shows skipped or invisible frames, allows frame navigation with interactive keys.

---

### `cli_annotate.py`

- **`parse_args()`**  
  Parses command-line arguments for video annotation.  
  **Returns:** Namespace containing `video_path`, `output`, and `prompt`.

- **`main()`**  
  Entry point for the annotation CLI. Initializes `Object_Tracker` and runs the annotation loop.

---

### `cli_validate.py`

- **`parse_args()`**  
  Parses command-line arguments for annotation validation.  
  **Returns:** Namespace containing `video_path` and `annotations`.

- **`main()`**  
  Entry point for the validation CLI. Calls `visualize_video` with the specified video and annotation file.

## Contact

- **Author:** Joseph Distefano  
- **GitHub:** [https://github.com/JosephDistefano/video_object_tracker](https://github.com/JosephDistefano/video_object_tracker)  
- **Email:** joedistefano789@gmail.com
