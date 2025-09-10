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

        upload_path = os.path.join(settings.MEDIA_ROOT, "uploads", image_file.name)
        os.makedirs(os.path.dirname(upload_path), exist_ok=True)

        with open(upload_path, "wb+") as f:
            for chunk in image_file.chunks():
                f.write(chunk)

        results = model(upload_path)
        annotated_img = results[0].plot()

        result_filename = f"detected_{image_file.name}"
        result_path = os.path.join(settings.MEDIA_ROOT, "results", result_filename)
        os.makedirs(os.path.dirname(result_path), exist_ok=True)
        cv2.imwrite(result_path, annotated_img)

        result_url = f"{settings.MEDIA_URL}results/{result_filename}"

        cls_ids = results[0].boxes.cls
        objects = [model.names[int(c)] for c in cls_ids]

    return render(request, "image_ai/image_detect.html", {
        "result_url": result_url,
        "objects": objects
    })


# =========================
# Google Vision: Image Detection (EDITED SECTION)
# =========================
def vision_image_page(request):
    result_url = None
    objects = []

    if request.method == "POST" and request.FILES.get("image"):
        image_file = request.FILES["image"]

        upload_path = os.path.join(settings.MEDIA_ROOT, "uploads", image_file.name)
        os.makedirs(os.path.dirname(upload_path), exist_ok=True)

        with open(upload_path, "wb+") as f:
            for chunk in image_file.chunks():
                f.write(chunk)

        img = cv2.imread(upload_path)
        if img is None:
            return render(request, "image_ai/vision_image.html", {"error": "Failed to load image."})

        height, width, _ = img.shape

        with open(upload_path, "rb") as img_content_file:
            content = img_content_file.read()
        vision_image = vision.Image(content=content)
        response = client.object_localization(image=vision_image)

        # Draw bounding boxes and labels on the image
        for obj in response.localized_object_annotations:
            objects.append(f"{obj.name} ({obj.score:.2f})")

            box = obj.bounding_poly.normalized_vertices
            x_min = int(box[0].x * width)
            y_min = int(box[0].y * height)
            x_max = int(box[2].x * width)
            y_max = int(box[2].y * height)

            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            label = f"{obj.name} {obj.score:.2f}"
            (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(img, (x_min, y_min - text_height - baseline), (x_min + text_width, y_min), (0, 255, 0), -1)
            cv2.putText(img, label, (x_min, y_min - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        result_filename = f"vision_detected_{image_file.name}"
        result_path = os.path.join(settings.MEDIA_ROOT, "results", result_filename)
        os.makedirs(os.path.dirname(result_path), exist_ok=True)
        cv2.imwrite(result_path, img)

        result_url = f"{settings.MEDIA_URL}results/{result_filename}"

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

            if pts.shape[0] > 0:
                text_pos = (pts[0][0], pts[0][1] - 10)
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