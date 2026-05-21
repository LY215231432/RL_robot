import os


SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPTS_DIR)
VIDEOS_DIR = os.path.join(BASE_DIR, "videos")
RUNTIME_LOGS_DIR = os.path.join(BASE_DIR, "logs", "runtime")
ARCHIVES_DIR = os.path.join(BASE_DIR, "archives")


def ensure_artifact_dirs():
    for path in (VIDEOS_DIR, RUNTIME_LOGS_DIR, ARCHIVES_DIR):
        os.makedirs(path, exist_ok=True)
