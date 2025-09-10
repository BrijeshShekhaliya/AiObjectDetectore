from django.shortcuts import render
from django.http import StreamingHttpResponse
from django.conf import settings
from ultralytics import YOLO
import os
import cv2
import numpy as np
from google.cloud import vision
from rest_framework.views import APIView
from rest_framework.response import Response
import base64
from django.core.files.base import ContentFile

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
    context = {
        'on_render': settings.IS_RENDER
    }
    return render(request, "image_ai/home.html", context)

# =========================
# WebRTC Detection Pages
# =========================
def webrtc_page(request):
    return render(request, "image_ai/webrtc_detect.html")

def vision_webrtc_page(request):
    return render(request, "image_ai/vision_webrtc_detect.html")

# =========================
# YOLO: Live Detection (Local Only)
# =========================
def live_page(request):
    return render(request, "image_ai/live.html")

def gen_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            results = model(frame)
            annotated_frame = results[0].plot()
            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def live_feed(request):
    return StreamingHttpResponse(gen_frames(), content_type='multipart/x-mixed-replace; boundary=frame')

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
    return render(request, "image_ai/image_detect.html", {"result_url": result_url, "objects": objects})

# =========================
# Google Vision: Image Detection
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
        for obj in response.localized_object_annotations:
            objects.append(f"{obj.name} ({obj.score:.2f})")
            box = obj.bounding_poly.normalized_vertices
            x_min, y_min = int(box[0].x * width), int(box[0].y * height)
            x_max, y_max = int(box[2].x * width), int(box[2].y * height)
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
    return render(request, "image_ai/vision_image.html", {"result_url": result_url, "objects": objects})

# =========================
# Google Vision: Live Detection (Local Only)
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
                cv2.putText(frame, obj.name, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def vision_live_feed(request):
    return StreamingHttpResponse(vision_gen_frames(), content_type='multipart/x-mixed-replace; boundary=frame')

# =========================
# API Views for WebRTC
# =========================
class YoloDetectAPIView(APIView):
    def post(self, request, *args, **kwargs):
        image_data = request.data.get('image')
        if not image_data: return Response({"error": "No image data"}, status=400)
        try:
            format, imgstr = image_data.split(';base64,')
            data = ContentFile(base64.b64decode(imgstr), name='temp.jpg')
        except: return Response({"error": "Invalid image data"}, status=400)
        
        image_array = np.frombuffer(data.read(), np.uint8)
        img_opencv = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if img_opencv is None: return Response({"error": "Failed to decode image"}, status=400)
        
        results = model(img_opencv)
        detections = []
        for box in results[0].boxes:
            class_id = int(box.cls)
            detections.append({
                'class_name': model.names[class_id],
                'confidence': round(float(box.conf), 2),
                'box': [int(coord) for coord in box.xyxy[0]]
            })
        return Response(detections)

class GoogleVisionDetectAPIView(APIView):
    def post(self, request, *args, **kwargs):
        image_data = request.data.get('image')
        if not image_data: return Response({"error": "No image data"}, status=400)
        try:
            format, imgstr = image_data.split(';base64,')
            content = base64.b64decode(imgstr)
        except: return Response({"error": "Invalid image data"}, status=400)

        vision_image = vision.Image(content=content)
        response = client.object_localization(image=vision_image)
        
        detections = []
        for obj in response.localized_object_annotations:
            # We need image dimensions to convert normalized vertices
            # Since we don't have them, we send back normalized values
            box = obj.bounding_poly.normalized_vertices
            detections.append({
                'class_name': obj.name,
                'confidence': round(obj.score, 2),
                'box': [{'x': v.x, 'y': v.y} for v in box]
            })
        return Response(detections)