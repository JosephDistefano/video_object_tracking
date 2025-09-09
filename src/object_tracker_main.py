import cv2
import os
import sys
import torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../GroundingDINO")))
from groundingdino.util.inference import load_model,predict,annotate,load_image

class Object_Tracker:
    """
    Object_Tracker class for video annotation and object tracking.

    Supports:
    - Manual annotation via GUI.
    - Automatic tracking using OpenCV trackers.
    - Prompt-based detection using GroundingDINO.
    - Frame-wise annotation logging.
    """
    def __init__(self, video_path, output_path=None,prompt=None):
        """
        Initialize tracker and annotation system.

        Args:
            video_path (str): Path to the input video file.
            output_path (str, optional): Path to save annotation file.
            prompt (str, optional): Text prompt for GroundingDINO detection.

        Raises:
            FileNotFoundError: If video_path does not exist.
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

        # Initializing all variables, inputs, and paths
        self.video = cv2.VideoCapture(video_path)
        self.frame_num = 0
        self.annotations = []
        self.tracker_initialized = False
        self.last_bbox = None
        self.output_path = output_path or os.path.splitext(video_path)[0] + ".annotations"
        self.prompt = prompt

        # Loading GroundDINO MOdel
        if self.prompt is not None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.dino_model = load_model("GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py","GroundingDINO/weights/groundingdino_swint_ogc.pth",)
            self.dino_model.eval()

        # Initialize tracker
        self.tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
        self.tracker_type = self.tracker_types[0]
        if self.tracker_type == 'BOOSTING':
            self.tracker = cv2.legacy.TrackerBoosting_create()
        if self.tracker_type == 'MIL':
            self.tracker = cv2.TrackerMIL_create() 
        if self.tracker_type == 'KCF':
            self.tracker = cv2.TrackerKCF_create() 
        if self.tracker_type == 'TLD':
            self.tracker = cv2.legacy.TrackerTLD_create() 
        if self.tracker_type == 'MEDIANFLOW':
            self.tracker = cv2.legacy.TrackerMedianFlow_create() 
        if self.tracker_type == 'GOTURN':
            tracker = cv2.TrackerGOTURN_create()
        if self.tracker_type == 'MOSSE':
            self.tracker = cv2.legacy.TrackerMOSSE_create()
        if self.tracker_type == "CSRT":
            self.tracker = cv2.TrackerCSRT_create()
        
        #Open annotation file
        self.annotation_file = open(self.output_path, "w", newline='')

    def draw_bbox(self, frame):
        """
        Manually draw bounding box on frame.

        Args:
            frame: Video frame for annotation.
        """
        x, y, w, h = cv2.selectROI("Video Labeling", frame, showCrosshair=True, fromCenter=True)
        x_center = x + w // 2
        y_center = y + h // 2
        self.annotations.append((self.frame_num, "label", x_center, y_center, w, h))
        self.tracker.init(frame, (x, y, w, h))
        self.last_bbox = (x, y, w, h)
        self.tracker_initialized = True

    def accept_tracked_bbox(self):
        """
        Accept current tracked bounding box and add annotation.
        """
        x, y, w, h = map(int, self.last_bbox)
        x_center = x + w // 2
        y_center = y + h // 2
        self.annotations.append((self.frame_num, "tracked", x_center, y_center, w, h))

    def add_annotation(self, action, bbox=None):
        """
        Write annotation to memory and file.

        Args:
            action (str): 'V' (visible), 'S' (skip), 'I' (invisible)
            bbox (tuple, optional): Bounding box (x, y, w, h) for visible frames.

        Raises:
            ValueError: Invalid action or missing bbox.
        """
        if action == 'V' and bbox is not None:
            x, y, w, h = map(int, bbox)
            x_center = x + w // 2
            y_center = y + h // 2
            line = f"V {x_center} {y_center} {w} {h}"
        elif action in ['S', 'I']:
            line = f"{action} -1 -1 -1 -1"
        else:
            raise ValueError("Invalid action or missing bbox for visible frame")

        self.annotations.append(line)
        self.annotation_file.write(line + "\n")
        self.annotation_file.flush() 

    def detect_with_prompt(self, frame):
        """
        Run GroundingDINO detection on frame using a text prompt.

        Args:
            frame (ndarray): Frame to run detection on.

        Returns:
            tuple: Bounding box (x, y, w, h) or None if detection fails.
        """
        cv2.imwrite('temp.jpg', frame)
        image_source, image = load_image('temp.jpg')
        boxes, scores, phrases = predict(
            model=self.dino_model,
            image=image,
            caption=self.prompt,
            box_threshold=0.25,
            text_threshold=0.20,
            device='cpu'
        )

        h_img, w_img, _ = frame.shape
        cx, cy, w_rel, h_rel = boxes[0]  # take first detection
        cx, cy, w_rel, h_rel = float(cx), float(cy), float(w_rel), float(h_rel)

        # scale back to pixels
        x_center = int(cx * w_img)
        y_center = int(cy * h_img)
        w = int(w_rel * w_img)
        h = int(h_rel * h_img)

        # convert center format to top-left corner
        x = x_center - w // 2
        y = y_center - h // 2

        bbox = (x, y, w, h)
        return bbox
    
    def cleanup(self):
        """
        Release resources and close annotation file.
        """
        self.video.release()
        cv2.destroyAllWindows()
        self.annotation_file.close()
        print(f"Annotations saved to {self.output_path}")


    def run(self):
        """
        Main loop to annotate and track video frames.
        """
        while True:
            ret, frame = self.video.read()
            if not ret:
                break

            display_frame = frame.copy()
            # display_frame = cv2.resize(display_frame, (self.DISPLAY_WIDTH,self.DISPLAY_HEIGHT))

             # Initialize tracker with prompt detection if provided
            if self.prompt and not self.tracker_initialized:
                bbox = self.detect_with_prompt(display_frame)
                if bbox is not None:
                    x, y, w, h = bbox
                    self.tracker.init(frame, (x, y, w, h))
                    self.last_bbox = (x, y, w, h)
                    self.tracker_initialized = True
                    self.add_annotation('V', self.last_bbox)

            # Update Tracker if initialized
            if self.tracker_initialized:
                success, bbox = self.tracker.update(frame)
                if success:
                    self.last_bbox = bbox
                    x, y, w, h = map(int, bbox)
                    cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    x_center = x + w // 2
                    y_center = y + h // 2
                    cv2.circle(display_frame, (x_center, y_center), 3, (255, 0, 0), -1)
                else:
                    print("Tracker lost object, fallback to manual labeling")
                    self.tracker_initialized = False

                if self.prompt:
                    text = f"Frame: {self.frame_num} | [A]ccept [L] Re-run prompt [F]ix manual [S]kip [I]nvisible [Q]uit"
                else:
                    text = f"Frame: {self.frame_num} | [A]ccept [F]ix manual [S]kip [I]nvisible [Q]uit"
                valid_keys = [ord(c) for c in "aAfFsSiIqQ"]
                if self.prompt:
                    valid_keys += [ord('l'), ord('L')]
            else:
                if self.prompt:
                    text = f"Frame: {self.frame_num} | [L]abel (prompt/manual) [S]kip [I]nvisible [Q]uit"
                else:
                    text = f"Frame: {self.frame_num} | [L]abel manual [S]kip [I]nvisible [Q]uit"
                valid_keys = [ord(c) for c in "lLsSiIqQ"]

            cv2.putText(display_frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Video Labeling", display_frame)

            key = cv2.waitKey(0) & 0xFF
            if key not in valid_keys:
                continue

            # Handle Key Actions
            if key in [ord('q'), ord('Q')]:
                print("Quitting")
                break

            elif key in [ord('s'), ord('S')]:
                self.add_annotation('S')
                self.last_bbox = None
                self.tracker_initialized = False

            elif key in [ord('i'), ord('I')]:
                self.add_annotation('I')
                self.last_bbox = None
                self.tracker_initialized = False

            elif key in [ord('a'), ord('A')] and self.tracker_initialized:
                self.accept_tracked_bbox()
                self.add_annotation('V', self.last_bbox)


            elif key in [ord('l'), ord('L')]:
                if self.prompt:
                    self.tracker = cv2.legacy.TrackerBoosting_create()
                    bbox = self.detect_with_prompt(display_frame)
                    x, y, w, h = bbox
                    self.tracker.init(display_frame, (x, y, w, h))
                    self.last_bbox = (x, y, w, h)
                    self.tracker_initialized = True
                    self.add_annotation('V', self.last_bbox)
                else:
                    self.draw_bbox(display_frame)
                    self.add_annotation('V', self.last_bbox)

            elif key in [ord('f'), ord('F')]:
                self.tracker = cv2.legacy.TrackerBoosting_create()
                self.draw_bbox(display_frame)

        self.frame_num += 1
        self.cleanup()


