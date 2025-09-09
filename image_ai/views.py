from django.shortcuts import render
from django.http import StreamingHttpResponse
from django.conf import settings
from ultralytics import YOLO
import os
import cv2
import numpy as np
from google.cloud import vision

# =========================
# Load YOLO model once
# =========================
model = YOLO("yolov8n.pt")

# =========================
# Google Vision Client
# =========================
client = vision.ImageAnnotatorClient()

# --- Home Page ---
def home(request):
    return render(request, "image_ai/home.html")


# =========================
# YOLO: Live Detection
# =========================
def live_page(request):
    return render(request, "image_ai/live.html")


def gen_frames():
    cap = cv2.VideoCapture(0)  # 0 = default webcam
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            # Run YOLO detection
            results = model(frame)
            annotated_frame = results[0].plot()

            # Encode frame as JPEG
            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            frame = buffer.tobytes()

            # Yield frame in HTTP response
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


def live_feed(request):
    return StreamingHttpResponse(
        gen_frames(),
        content_type='multipart/x-mixed-replace; boundary=frame'
    )


# =========================
# YOLO: Image Detection
# =========================
def image_detect_page(request):
    result_url = None
    objects = []

    if request.method == "POST" and request.FILES.get("image"):
        image_file = request.FILES["image"]

        upload_path = os.path.join("uploads", "input.jpg")
        full_upload_path = os.path.join(settings.MEDIA_ROOT, upload_path)
        os.makedirs(os.path.dirname(full_upload_path), exist_ok=True)

        with open(full_upload_path, "wb+") as f:
            for chunk in image_file.chunks():
                f.write(chunk)

        results = model(full_upload_path)
        annotated_img = results[0].plot()

        result_path = os.path.join("results", "detected.jpg")
        full_result_path = os.path.join(settings.MEDIA_ROOT, result_path)
        os.makedirs(os.path.dirname(full_result_path), exist_ok=True)
        cv2.imwrite(full_result_path, annotated_img)

        result_url = settings.MEDIA_URL + result_path

        cls_ids = results[0].boxes.cls
        objects = [model.names[int(c)] for c in cls_ids]

    return render(request, "image_ai/image_detect.html", {
        "result_url": result_url,
        "objects": objects
    })


# =========================
# Google Vision: Image Detection
# =========================
def vision_image_page(request):
    result_url = None
    objects = []

    if request.method == "POST" and request.FILES.get("image"):
        image_file = request.FILES["image"]

        # Save uploaded image
        upload_path = os.path.join("uploads", image_file.name)
        full_upload_path = os.path.join(settings.MEDIA_ROOT, upload_path)
        os.makedirs(os.path.dirname(full_upload_path), exist_ok=True)

        with open(full_upload_path, "wb+") as f:
            for chunk in image_file.chunks():
                f.write(chunk)

        # Load image using OpenCV
        img = cv2.imread(full_upload_path)
        if img is None:
            return render(request, "image_ai/vision_image.html", {
                "result_url": None,
                "objects": [],
                "error": "Failed to load image."
            })

        height, width, _ = img.shape

        # Google Vision API
        with open(full_upload_path, "rb") as img_file:
            content = img_file.read()
        vision_image = vision.Image(content=content)
        response = client.object_localization(image=vision_image)

        # Draw bounding boxes
        for obj in response.localized_object_annotations:
            objects.append(obj.name)
            box = obj.bounding_poly.normalized_vertices
            pts = [(int(v.x * width), int(v.y * height)) for v in box]
            pts = np.array(pts, dtype=np.int32)
            cv2.polylines(img, [pts], isClosed=True, color=(0, 255, 0), thickness=3)

        # Save annotated image
        result_path = os.path.join("results", "vision_detected.jpg")
        full_result_path = os.path.join(settings.MEDIA_ROOT, result_path)
        os.makedirs(os.path.dirname(full_result_path), exist_ok=True)
        cv2.imwrite(full_result_path, img)
        result_url = settings.MEDIA_URL + result_path

    return render(request, "image_ai/vision_image.html", {
        "result_url": result_url,
        "objects": objects
    })


# =========================
# Google Vision: Live Detection
# =========================
def vision_live_page(request):
    return render(request, "image_ai/vision_live.html")


def vision_gen_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break

        height, width, _ = frame.shape
        ret, buffer = cv2.imencode('.jpg', frame)
        content = buffer.tobytes()
        vision_image = vision.Image(content=content)

        response = client.object_localization(image=vision_image)

        for obj in response.localized_object_annotations:
            box = obj.bounding_poly.normalized_vertices
            pts = [(int(v.x * width), int(v.y * height)) for v in box]
            pts = np.array(pts, dtype=np.int32)
            cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

            # Add label (object name) at the top-left of the bounding box
            if pts.shape[0] > 0:
                text_pos = (pts[0][0], pts[0][1] - 10)  # slightly above the top-left corner
                cv2.putText(frame, obj.name, text_pos, 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')



def vision_live_feed(request):
    return StreamingHttpResponse(
        vision_gen_frames(),
        content_type='multipart/x-mixed-replace; boundary=frame'
    )
