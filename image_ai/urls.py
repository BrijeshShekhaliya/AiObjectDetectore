from django.urls import path
from . import views

urlpatterns = [
    path("", views.home, name="home"),

    # Image Upload Pages
    path("image_detect/", views.image_detect_page, name="image_detect_page"),
    path("vision_image/", views.vision_image_page, name="vision_image_page"),

    # WebRTC Pages (Work on Live Site)
    path("webrtc/", views.webrtc_page, name="webrtc_page"),
    path("vision_webrtc/", views.vision_webrtc_page, name="vision_webrtc_page"),
    
    # Live Camera Pages (Local Development Only)
    path("live_page/", views.live_page, name="live_page"),
    path("vision_live/", views.vision_live_page, name="vision_live_page"),
    
    # Live Feeds (Local Development Only)
    path("live_feed/", views.live_feed, name="live_feed"),
    path("vision_live_feed/", views.vision_live_feed, name="vision_live_feed"),

    # API Endpoints
    path("api/yolo_detect/", views.YoloDetectAPIView.as_view(), name="yolo_detect_api"),
    path("api/vision_detect/", views.GoogleVisionDetectAPIView.as_view(), name="vision_detect_api"),
]