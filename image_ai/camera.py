import cv2
from ultralytics import YOLO

# Load pre-trained YOLOv8 model
model = YOLO("yolov8n.pt")  # use yolov8x.pt for higher accuracy

class VideoCamera:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)  # open default webcam

    def __del__(self):
        self.cap.release()

    def get_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None

        # Run YOLO detection
        results = model(frame)

        # Annotate frame
        annotated_frame = results[0].plot()

        # Encode frame as JPEG
        ret, jpeg = cv2.imencode('.jpg', annotated_frame)
        return jpeg.tobytes()
