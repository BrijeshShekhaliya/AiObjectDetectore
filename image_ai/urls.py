from django.urls import path
from . import views

urlpatterns = [
    path("", views.home, name="home"),

    # YOLO Live
    path("live_page/", views.live_page, name="live_page"),
    path("live_feed/", views.live_feed, name="live_feed"),

    # YOLO Image Detection
    path("image_detect/", views.image_detect_page, name="image_detect_page"),

    # Google Vision Image Detection
    path("vision_image/", views.vision_image_page, name="vision_image_page"),

    # Google Vision Live Detection
    path("vision_live/", views.vision_live_page, name="vision_live_page"),
    path("vision_live_feed/", views.vision_live_feed, name="vision_live_feed"),  # <-- added
]
