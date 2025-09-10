"""
Django settings for AiObjectDetectore project.
"""

from pathlib import Path
import os
from dotenv import load_dotenv
import dj_database_url  # Add this import

# --------------------------------------------------------
# Base directory
# --------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

# --------------------------------------------------------
# Load environment variables from .env
# --------------------------------------------------------
load_dotenv(dotenv_path=BASE_DIR / ".env")

# --------------------------------------------------------
# Google Vision API
# --------------------------------------------------------
google_key_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
if google_key_path:
    # Convert relative path to absolute
    google_key_path = str(Path(BASE_DIR / google_key_path).resolve())
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = google_key_path
else:
    # This will only raise an error if the variable is missing locally.
    # On Render, the Secret File will set the variable.
    print("Warning: GOOGLE_APPLICATION_CREDENTIALS not set in .env")

# --------------------------------------------------------
# Security
# --------------------------------------------------------
SECRET_KEY = os.getenv("SECRET_KEY")

# Check if we are running on Render
IS_RENDER = 'RENDER' in os.environ

if IS_RENDER:
    DEBUG = False
else:
    DEBUG = True  # Keep True for local development

# Automatically configure ALLOWED_HOSTS for Render
ALLOWED_HOSTS = []
if IS_RENDER:
    RENDER_EXTERNAL_HOSTNAME = os.environ.get('RENDER_EXTERNAL_HOSTNAME')
    if RENDER_EXTERNAL_HOSTNAME:
        ALLOWED_HOSTS.append(RENDER_EXTERNAL_HOSTNAME)
else:
    # For local development
    ALLOWED_HOSTS.extend(['127.0.0.1', 'localhost'])

# --------------------------------------------------------
# Media (uploads/results)
# --------------------------------------------------------
MEDIA_URL = '/media/'
MEDIA_ROOT = BASE_DIR / 'media'

# --------------------------------------------------------
# Static files (CSS, JavaScript, Images)
# --------------------------------------------------------
STATIC_URL = '/static/'
STATICFILES_DIRS = [BASE_DIR / "static"]
STATIC_ROOT = BASE_DIR / "staticfiles"

# --------------------------------------------------------
# Installed apps
# --------------------------------------------------------
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'rest_framework',
    'image_ai',  # custom app
]

# --------------------------------------------------------
# Middleware
# --------------------------------------------------------
MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'whitenoise.middleware.WhiteNoiseMiddleware',  # for static files in production
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

# --------------------------------------------------------
# URL configuration
# --------------------------------------------------------
ROOT_URLCONF = 'AiObjectDetectore.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'AiObjectDetectore.wsgi.application'

# --------------------------------------------------------
# Database
# --------------------------------------------------------
if IS_RENDER:
    # Use Render's PostgreSQL database
    DATABASES = {
        'default': dj_database_url.config(
            conn_max_age=600,
            ssl_require=True
        )
    }
else:
    # Use local SQLite database for development
    DATABASES = {
        'default': {
            'ENGINE': 'django.db.backends.sqlite3',
            'NAME': BASE_DIR / 'db.sqlite3',
        }
    }

# --------------------------------------------------------
# Password validation
# --------------------------------------------------------
AUTH_PASSWORD_VALIDATORS = [
    {'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator'},
    {'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator'},
    {'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator'},
    {'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator'},
]

# --------------------------------------------------------
# Internationalization
# --------------------------------------------------------
LANGUAGE_CODE = 'en-us'
TIME_ZONE = 'UTC'
USE_I18N = True
USE_TZ = True

# --------------------------------------------------------
# Default primary key field type
# --------------------------------------------------------
DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'