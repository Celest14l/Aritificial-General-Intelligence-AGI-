# Handles loading environment variables, defining paths, and storing configuration constants.
# config.py
import os
from dotenv import load_dotenv
# No longer need pathlib here for ontology loading
# import pathlib

# Load environment variables
load_dotenv(dotenv_path="pass.env")

# --- Core Paths ---
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

STATIC_DIR = os.path.join(BASE_DIR, "static")
AUDIO_DIR = os.path.join(STATIC_DIR, "responses_output")
DOWNLOADS_DIR = os.path.join(BASE_DIR, "Downloads")
MEMORY_DIR = os.path.join(BASE_DIR, "memory")

# --- File Paths ---
USER_PREFS_FILE = os.path.join(MEMORY_DIR, "user_prefs.json")
ERROR_LOG_FILE = os.path.join(MEMORY_DIR, "error_log.txt")
CHAT_HISTORY_FILE = os.path.join(MEMORY_DIR, "chat_history.json")

# --- Ontology Configuration ---
# Define the standard Windows path to your file
LTM_ONTOLOGY_FILE = os.path.join(BASE_DIR, "knowledge_base.owl")

# Define the Identity IRI (declared *inside* the knowledge_base.owl file)
LTM_ONTOLOGY_IDENTITY_IRI = "http://test.org/knowledge_base.owl"

# **REMOVED:** LTM_ONTOLOGY_LOAD_URI is not needed for the onto_path method.


# --- API Keys & Service URLs ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY") # Replace placeholder
NEWS_SERVICE_URL = os.getenv("NEWS_SERVICE_URL", "http://localhost:5000/news")
EMAIL_USER = os.getenv("EMAIL_USER")
EMAIL_PASS = os.getenv("EMAIL_PASS")
SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", 465))
FFMPEG_PATH = os.getenv("FFMPEG_PATH", None)

# --- Model & App Settings ---
TTS_MODEL_ID = "facebook/mms-tts-eng"
LLM_MODEL_NAME = "llama3-8b-8192"
CONVERSATIONAL_MEMORY_LENGTH = 15
STM_MAX_SIZE = 200
STM_TTL_SECONDS = 24 * 60 * 60 # 24 hours

# --- Function to ensure directories exist ---
def ensure_directories():
    """Creates necessary directories if they don't exist."""
    print("Ensuring core directories...")
    os.makedirs(STATIC_DIR, exist_ok=True)
    os.makedirs(AUDIO_DIR, exist_ok=True)
    os.makedirs(DOWNLOADS_DIR, exist_ok=True)
    os.makedirs(MEMORY_DIR, exist_ok=True)
    print(f"  Static: {STATIC_DIR}")
    print(f"  Audio: {AUDIO_DIR}")
    print(f"  Downloads: {DOWNLOADS_DIR}")
    print(f"  Memory: {MEMORY_DIR}")
    print("✅ Core directories ensured.")

# --- Initial Check for Critical Config ---
def check_critical_config():
    """Checks if essential configurations like API keys are set."""
    print("Checking critical configuration...")
    critical_ok = True
    if not GROQ_API_KEY or "YOUR_GROQ_API_KEY" in GROQ_API_KEY or len(GROQ_API_KEY) < 10:
        print("❌ Critical Error: GROQ_API_KEY is not set correctly in environment variables or config.py.")
        critical_ok = False
    else:
        print("  GROQ API Key: Set")

    # Check if ontology file exists
    if os.path.exists(LTM_ONTOLOGY_FILE):
         print(f"  Ontology file found at: {LTM_ONTOLOGY_FILE}")
    else:
         print(f"⚠️ Warning: Ontology file not found at expected location: {LTM_ONTOLOGY_FILE}")
         print(f"  Loading will fail unless an empty file is created or path is corrected.")
         # If loading relies on onto_path, existence is crucial before loading
         critical_ok = False # Consider making this critical if LTM is essential


    if critical_ok:
        print("✅ Critical configuration check passed.")
    else:
        print("❌ Critical configuration check failed. Please review settings.")
    return critical_ok
