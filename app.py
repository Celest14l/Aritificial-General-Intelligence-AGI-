# app.py
import os
import requests
import speech_recognition as sr # Kept for VAD or potential future server-side file processing, not live mic
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from operator import itemgetter
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnableSequence
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
# pyAudioAnalysis removed - no direct audio input
import numpy as np
import random
import json
from transformers import VitsModel, AutoTokenizer
import torch
import soundfile as sf
import yt_dlp
# vlc removed - cannot control browser audio playback from backend
import time
# ctypes removed - VLC related
import smtplib
from email.message import EmailMessage
from dotenv import load_dotenv
# Threading removed for simplicity - basic Flask handles requests, complex async needs more work
# cv2 removed - no camera access from backend
import datetime
import re
from flask import Flask, request, jsonify, render_template, send_from_directory, url_for
from cachetools import TTLCache # <-- Added for Short-Term Memory
import time # <-- Added for STM timestamps

# --- Flask App Initialization ---
app = Flask(__name__, template_folder='.', static_folder='static')

# --- Environment Variables & Configuration ---
load_dotenv(dotenv_path="pass.env")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY") # Replace placeholder if not in .env

# Directories and Files
BASE_DIR = r"D:\AGI" # Directory where app.py is located
STATIC_DIR = os.path.join(BASE_DIR, "static")
AUDIO_DIR = r"D:\AGI\static\responses_output"
USER_PREFS_FILE = os.path.join(BASE_DIR, "user_prefs.json")
ERROR_LOG_FILE = os.path.join(BASE_DIR, "error_log.txt")
CHAT_HISTORY_FILE = os.path.join(BASE_DIR, "chat_history.json")
DOWNLOADS_DIR = os.path.join(BASE_DIR, "Downloads")
LTM_FILE_PATH = os.path.join(BASE_DIR, "long_term_memory.json") # <-- Added for Long-Term Memory

# Ensure directories exist
os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(DOWNLOADS_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

# News Microservice URL
NEWS_SERVICE_URL = os.getenv("NEWS_SERVICE_URL", "http://localhost:5000/news")

# --- Global Variables ---
response_counter = 1
user_prefs = {}
chat_memory = None
conversation_chain = None
tts_model = None
tts_tokenizer = None
vader_analyzer = None

# --- Memory Stores ---
# Short-Term Memory (STM): TTL Cache for 24 hours (86400 seconds)
# Stores tuples: (timestamp, user_input_text)
short_term_memory_cache = TTLCache(maxsize=200, ttl=24 * 60 * 60) # Store ~200 recent interactions
last_stm_store_key = None # Keep track of the key of the last stored item for "forget that"

# Long-Term Memory (LTM): Dictionary loaded from/saved to JSON
long_term_memory = {}

# --- Utility Functions ---

def log_error(error_msg, exc_info=False): # Added exc_info for tracebacks
    """Logs an error message to the error log file."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"{timestamp}: {error_msg}\n"
    if exc_info:
        import traceback
        log_entry += traceback.format_exc() + "\n" # Add traceback if requested
    print(f"ERROR: {log_entry.strip()}")
    try:
        with open(ERROR_LOG_FILE, 'a', encoding='utf-8') as f:
            f.write(log_entry)
    except Exception as e:
        print(f"🚨 Critical Error: Could not write to log file {ERROR_LOG_FILE}: {e}")

def load_user_prefs():
    # (No changes needed in this function)
    """Loads user preferences from a JSON file."""
    if os.path.exists(USER_PREFS_FILE):
        try:
            with open(USER_PREFS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError:
            print(f"⚠️ Warning: Could not decode {USER_PREFS_FILE}. Starting with defaults.")
            log_error(f"JSONDecodeError loading user prefs from {USER_PREFS_FILE}")
        except Exception as e:
            print(f"⚠️ Warning: Error loading user prefs: {e}. Starting with defaults.")
            log_error(f"Error loading user prefs: {e}")
    return {"favorite_music_genre": "", "preferred_city": "Pune", "interests": []}

def save_user_prefs(prefs):
    # (No changes needed in this function)
    """Saves user preferences to a JSON file."""
    try:
        with open(USER_PREFS_FILE, 'w', encoding='utf-8') as f:
            json.dump(prefs, f, indent=4)
        print("User preferences saved.")
    except Exception as e:
        log_error(f"Error saving user prefs: {e}")

def load_chat_history():
    # (No changes needed in this function)
    """Loads chat history from a JSON file."""
    history = []
    if os.path.exists(CHAT_HISTORY_FILE):
        try:
            with open(CHAT_HISTORY_FILE, 'r', encoding='utf-8') as f:
                raw_history = json.load(f)
                for msg_dict in raw_history:
                    if msg_dict.get('type') == 'human':
                        history.append(HumanMessage(content=msg_dict.get('content', '')))
                    elif msg_dict.get('type') == 'ai':
                        history.append(AIMessage(content=msg_dict.get('content', '')))
            print(f"Loaded {len(history)} messages from chat history.")
        except json.JSONDecodeError:
            print(f"⚠️ Warning: Could not decode {CHAT_HISTORY_FILE}. Starting fresh history.")
            log_error(f"JSONDecodeError loading chat history from {CHAT_HISTORY_FILE}")
        except Exception as e:
            print(f"⚠️ Warning: Error loading chat history: {e}. Starting fresh history.")
            log_error(f"Error loading chat history: {e}")
    return history

def save_chat_history(messages):
    # (No changes needed in this function)
    """Saves chat history (list of Langchain message objects) to a JSON file."""
    try:
        history_to_save = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                history_to_save.append({"type": "human", "content": msg.content})
            elif isinstance(msg, AIMessage):
                history_to_save.append({"type": "ai", "content": msg.content})
        with open(CHAT_HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(history_to_save, f, indent=4)
        print(f"Saved {len(history_to_save)} messages to chat history.")
    except Exception as e:
        log_error(f"Error saving chat history: {e}")


# --- Long-Term Memory (LTM) Functions ---

def load_ltm():
    """Loads long-term memory from the JSON file."""
    global long_term_memory
    if os.path.exists(LTM_FILE_PATH):
        try:
            with open(LTM_FILE_PATH, 'r', encoding='utf-8') as f:
                long_term_memory = json.load(f)
                print(f"✅ Long-Term Memory loaded from {LTM_FILE_PATH}.")
        except json.JSONDecodeError:
            print(f"⚠️ Warning: Could not decode {LTM_FILE_PATH}. Starting with empty LTM.")
            log_error(f"JSONDecodeError loading LTM from {LTM_FILE_PATH}")
            long_term_memory = {}
        except Exception as e:
            print(f"⚠️ Warning: Error loading LTM: {e}. Starting with empty LTM.")
            log_error(f"Error loading LTM: {e}")
            long_term_memory = {}
    else:
        print("No LTM file found. Starting with empty LTM.")
        long_term_memory = {}
    return long_term_memory

def save_ltm():
    """Saves the current long-term memory dictionary to the JSON file."""
    global long_term_memory
    try:
        with open(LTM_FILE_PATH, 'w', encoding='utf-8') as f:
            json.dump(long_term_memory, f, indent=4)
        print(f"✅ Long-Term Memory saved to {LTM_FILE_PATH}.")
    except Exception as e:
        log_error(f"Error saving LTM: {e}")

# --- Sentiment Analysis ---
def analyze_text_sentiment(text):
    # (No changes needed in this function)
    global vader_analyzer
    if not vader_analyzer: return "Neutral"
    if not text: return "Neutral"
    try:
        sentiment_score = vader_analyzer.polarity_scores(text)
        compound = sentiment_score['compound']
        if compound >= 0.05: return "Positive"
        elif compound <= -0.05: return "Negative"
        else: return "Neutral"
    except Exception as e:
        log_error(f"Error analyzing text sentiment: {e}")
        return "Neutral"

# --- Text-to-Speech ---
def save_response_to_wav(response_text, file_name_base):
    # (No changes needed in this function)
    global tts_model, tts_tokenizer, response_counter
    if not tts_model or not tts_tokenizer:
        print("⚠️ TTS is unavailable.")
        return None, None
    if not response_text:
        print("⚠️ Cannot generate audio for empty text.")
        return None, None
    try:
        safe_base = re.sub(r'[\\/*?:"<>|]', "", file_name_base)
        unique_file_name = f"{safe_base}_{response_counter}.wav"
        output_path = os.path.join(AUDIO_DIR, unique_file_name)
        output_url = url_for('static', filename=f'responses_output/{unique_file_name}', _external=False)

        inputs = tts_tokenizer(response_text, return_tensors="pt")
        with torch.no_grad():
            output = tts_model(**inputs).waveform
            if not isinstance(output, torch.Tensor):
                 raise TypeError("Expected TTS output waveform to be a Tensor")
            audio = output.squeeze().cpu().numpy()

        sampling_rate = tts_model.config.sampling_rate
        sf.write(output_path, audio, samplerate=sampling_rate)
        print(f"🎙️ Generated audio: {output_path} (URL: {output_url})")
        response_counter += 1
        return output_path, output_url
    except Exception as e:
        log_error(f"Error during TTS generation for text '{response_text[:50]}...': {e}")
        return None, None

# --- Core Functionalities (Weather, Music, Email, News) ---
# (No changes needed in get_location, get_weather, download_music, send_email_with_attachments, get_news)
def get_location():
    """Fetches location based on server's IP or user preference."""
    global user_prefs
    preferred_city = user_prefs.get("preferred_city", "Pune") # Default if not set
    try:
        response = requests.get("http://ip-api.com/json/", timeout=5)
        response.raise_for_status()
        data = response.json()
        if data.get("status") == "success" and "city" in data:
             return {"city": data.get("city"), "country": data.get("country", "")}
        else:
             log_error(f"Unexpected location API response: {data.get('message', 'No message')}")
             return {"city": preferred_city, "country": ""} # Fallback to preference
    except Exception as e:
        log_error(f"Error fetching location: {e}")
        return {"city": preferred_city, "country": ""} # Fallback to preference

def get_weather(city):
    """Fetches weather for a given city."""
    try:
        url = f"https://wttr.in/{city}?format=%C+%t"
        response = requests.get(url, timeout=7)
        response.raise_for_status()
        weather_data = response.text.strip()
        if weather_data and "Unknown location" not in weather_data and "We were unable to process your request" not in weather_data:
            return weather_data
        else:
             log_error(f"Could not get weather for '{city}', response: '{weather_data}'")
             return f"Sorry, I couldn't find reliable weather data for {city} right now."
    except requests.exceptions.Timeout:
        log_error(f"Timeout fetching weather for {city}")
        return "Unable to fetch weather data due to a timeout."
    except requests.exceptions.RequestException as e:
        log_error(f"RequestException fetching weather for {city}: {e}")
        return "Unable to fetch weather data at the moment."
    except Exception as e:
        log_error(f"Unexpected error fetching weather for {city}: {e}")
        return "An unexpected error occurred while fetching weather."

def download_music(song_name):
    """Downloads music to the server's DOWNLOADS_DIR. Returns success message or error message."""
    print(f"🔍 Attempting download: {song_name}")
    safe_song_name = re.sub(r'[\\/*?:"<>|]', '_', song_name)
    safe_song_name = safe_song_name.strip()
    if not safe_song_name:
        return "Please provide a valid song name to download."

    output_template = os.path.join(DOWNLOADS_DIR, f"{safe_song_name}.%(ext)s")
    final_file_path = os.path.join(DOWNLOADS_DIR, f"{safe_song_name}.mp3")

    if os.path.exists(final_file_path):
        print(f"🎵 Song already downloaded: {final_file_path}")
        relative_path = os.path.relpath(final_file_path, BASE_DIR)
        return f"'{song_name}' is already downloaded in the '{os.path.basename(DOWNLOADS_DIR)}' folder (as {os.path.basename(final_file_path)})."

    ffmpeg_location = os.getenv("FFMPEG_PATH", None)
    ydl_opts = {
        'format': 'bestaudio/best',
        'noplaylist': True,
        'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'mp3', 'preferredquality': '128',}],
        'outtmpl': output_template, 'quiet': False, 'default_search': 'ytsearch1',
        'ffmpeg_location': ffmpeg_location, 'nocheckcertificate': True, 'retries': 3, 'socket_timeout': 15,
    }
    try:
        print(f"Starting download process for '{song_name}'...")
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(f"ytsearch1:{song_name}", download=True)

        if os.path.exists(final_file_path):
            print(f"✅ Download successful: {final_file_path}")
            relative_path = os.path.relpath(final_file_path, BASE_DIR)
            return f"Successfully downloaded '{song_name}'. You can find it in the '{os.path.basename(DOWNLOADS_DIR)}' folder (as {os.path.basename(final_file_path)})."
        else:
            found_files = [f for f in os.listdir(DOWNLOADS_DIR) if f.startswith(safe_song_name) and f.endswith('.mp3')]
            if found_files:
                 actual_file = os.path.join(DOWNLOADS_DIR, found_files[0])
                 print(f"✅ Download successful (found as {found_files[0]}): {actual_file}")
                 relative_path = os.path.relpath(actual_file, BASE_DIR)
                 if actual_file != final_file_path:
                     try:
                         os.rename(actual_file, final_file_path)
                         print(f"Renamed to {final_file_path}")
                         relative_path = os.path.relpath(final_file_path, BASE_DIR)
                     except OSError as rename_e:
                         print(f"⚠️ Could not rename {actual_file} to {final_file_path}: {rename_e}")
                 return f"Successfully downloaded '{song_name}'. You can find it in the '{os.path.basename(DOWNLOADS_DIR)}' folder (as {os.path.basename(final_file_path)})."
            else:
                log_error(f"Download process completed for '{song_name}', but expected MP3 file '{final_file_path}' not found in {DOWNLOADS_DIR}.")
                return f"Sorry, there was an issue finalizing the download for '{song_name}'. The MP3 file could not be created or found."
    except yt_dlp.utils.DownloadError as de:
        error_message = str(de)
        if "Unsupported URL" in error_message:
             log_error(f"yt-dlp DownloadError (Unsupported URL) for '{song_name}': {error_message}")
             return f"Sorry, I couldn't find a downloadable source for '{song_name}'."
        elif "Video unavailable" in error_message:
             log_error(f"yt-dlp DownloadError (Video Unavailable) for '{song_name}': {error_message}")
             return f"Sorry, the content for '{song_name}' seems to be unavailable."
        else:
            log_error(f"yt-dlp DownloadError for '{song_name}': {error_message}")
            return f"Sorry, an error occurred during the download process for '{song_name}'. It might be unavailable or blocked."
    except Exception as e:
        log_error(f"Unexpected error downloading/processing music '{song_name}': {e}")
        return f"Sorry, an unexpected error occurred while trying to download '{song_name}'."

def send_email_with_attachments(to_email, subject, body_content, attachments=[]):
    """Sends email using configured credentials."""
    email_user = os.getenv("EMAIL_USER")
    email_pass = os.getenv("EMAIL_PASS")
    smtp_host = os.getenv("SMTP_HOST", "smtp.gmail.com")
    smtp_port = int(os.getenv("SMTP_PORT", 465))

    if not email_user or not email_pass:
        log_error("Email sending failed: Credentials missing.")
        return "Email sending failed: Assistant's email credentials are not configured."
    try:
        msg = EmailMessage()
        msg["From"] = email_user
        msg["To"] = to_email
        msg["Subject"] = subject
        msg.set_content(body_content)

        for file_path in attachments:
            if not os.path.exists(file_path):
                print(f"⚠️ Attachment not found: {file_path}. Skipping.")
                log_error(f"Email attachment skipped (not found): {file_path}")
                continue
            try:
                with open(file_path, "rb") as file:
                    file_data = file.read()
                    file_name = os.path.basename(file_path)
                    maintype, subtype = 'application', 'octet-stream'
                    msg.add_attachment(file_data, maintype=maintype, subtype=subtype, filename=file_name)
            except Exception as e:
                 print(f"⚠️ Error attaching file {file_path}: {e}. Skipping.")
                 log_error(f"Error attaching file {file_path} to email: {e}")

        print(f"Connecting to SMTP server {smtp_host}:{smtp_port}...")
        with smtplib.SMTP_SSL(smtp_host, smtp_port) as server:
            print("Logging into email account...")
            server.login(email_user, email_pass)
            print("Sending email...")
            server.send_message(msg)
            print(f"✅ Email sent successfully to {to_email}.")
            return f"Email sent successfully to {to_email}."
    except smtplib.SMTPAuthenticationError:
         log_error(f"SMTP Authentication Error for user {email_user}. Check credentials/App Passwords.")
         return "Error sending email: Authentication failed."
    except smtplib.SMTPConnectError:
        log_error(f"SMTP Connection Error connecting to {smtp_host}:{smtp_port}.")
        return "Error sending email: Could not connect to the email server."
    except smtplib.SMTPSenderRefused:
         log_error(f"SMTP Sender Refused for user {email_user}.")
         return "Error sending email: Sender address refused."
    except Exception as e:
        log_error(f"Unexpected error sending email to {to_email}: {e}")
        return f"An unexpected error occurred while sending the email: {str(e)}"

def get_news(query):
    """Fetches news headlines from the dedicated microservice."""
    if not query:
        return "Please specify a topic for the news."
    print(f"Fetching news for query: '{query}' from {NEWS_SERVICE_URL}")
    try:
        params = {'query': query}
        response = requests.get(NEWS_SERVICE_URL, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        if data.get("status") == "success" and data.get("articles"):
            articles = data["articles"][:3] # Limit to top 3
            if not articles:
                 return f"I found the news service, but no articles matched '{query}'."
            news_summary = f"Here are the top headlines for '{query}':\n"
            for i, article in enumerate(articles, 1):
                title = article.get('title', 'No Title Provided')
                source = article.get('source', {}).get('name', 'Unknown Source')
                news_summary += f"{i}. {title} (Source: {source})\n"
            return news_summary.strip()
        elif data.get("message"):
             log_error(f"News service error for '{query}': {data['message']}")
             return f"Sorry, couldn't get news for '{query}'. Service reported: {data['message']}"
        else:
            log_error(f"Empty or unexpected success response from news service for '{query}'.")
            return f"Sorry, I couldn't find any news on '{query}' right now."
    except requests.exceptions.Timeout:
        log_error(f"Timeout fetching news for '{query}' from {NEWS_SERVICE_URL}")
        return f"Unable to fetch news for '{query}' due to a timeout."
    except requests.exceptions.ConnectionError:
        log_error(f"ConnectionError fetching news for '{query}' from {NEWS_SERVICE_URL}. Is it running?")
        return f"Unable to fetch news: Could not connect to the news service."
    except requests.exceptions.RequestException as e:
        log_error(f"RequestException fetching news from {NEWS_SERVICE_URL}: {e}")
        return f"Unable to fetch news for '{query}' due to a connection error ({type(e).__name__})."
    except json.JSONDecodeError:
         log_error(f"JSONDecodeError from news service {NEWS_SERVICE_URL} for query '{query}'.")
         return "Invalid response from the news service."
    except Exception as e:
        log_error(f"Unexpected error fetching news for '{query}': {e}")
        return f"An unexpected error occurred while fetching news for '{query}'."


# --- Negative Feedback Handling ---
def handle_negative_feedback(user_input_sentiment, response_counter):
    # (No changes needed in this function)
    global user_prefs
    motivational_quotes = [
        "Remember to take a deep breath. You've got this.", "Even small steps forward are progress. Keep going.",
        "Be kind to yourself. It's okay to have tough moments.", "Focus on what you can control. Brighter times are ahead."
    ]
    feedback_response = None
    if user_input_sentiment == "Negative":
        print(f"Detected Negative sentiment in user input.")
        response_parts = [
            f"I noticed your message seemed a bit negative.", "I hope everything is alright.",
            "Perhaps we could try something different?",
        ]
        interests = user_prefs.get('interests', [])
        if interests: response_parts.append(f"Maybe talk about {random.choice(interests)}?")
        response_parts.append("Or I could try to download some calming music for you?")
        response_parts.append(f"Here's a thought: {random.choice(motivational_quotes)}")
        feedback_response = " ".join(random.sample(response_parts, k=min(len(response_parts), 3)))
        print("Chatbot (Feedback):", feedback_response)
        _, feedback_audio_url = save_response_to_wav(feedback_response, f"feedback_{response_counter}")
        return feedback_response, feedback_audio_url
    return None, None

# --- Initialization Function ---
def initialize_assistant():
    """Loads models, preferences, memory, and sets up the AGI state."""
    global user_prefs, chat_memory, conversation_chain, tts_model, tts_tokenizer, vader_analyzer, response_counter, long_term_memory
    print("🚀 Initializing AGI Backend...")

    # Load User Prefs & Long-Term Memory
    user_prefs = load_user_prefs()
    print(f"User preferences loaded: {user_prefs}")
    long_term_memory = load_ltm() # Load LTM into the global dict

    # Initialize VADER
    vader_analyzer = SentimentIntensityAnalyzer()
    print("✅ VADER Sentiment Analyzer initialized.")

    # Initialize TTS
    try:
        model_id = "facebook/mms-tts-eng"
        print(f"Loading TTS model: {model_id}...")
        tts_model = VitsModel.from_pretrained(model_id)
        tts_tokenizer = AutoTokenizer.from_pretrained(model_id)
        print(f"✅ TTS Model loaded successfully.")
    except Exception as e:
        print(f"❌ Critical Error: Failed to load TTS model ({model_id}): {e}")
        log_error(f"Critical Error loading TTS model: {e}")
        tts_model = None
        tts_tokenizer = None

    # Initialize LLM and Conversation Chain
    groq_api_key = os.environ.get("GROQ_API_KEY")
    if not groq_api_key or "YOUR_GROQ_API_KEY" in groq_api_key or len(groq_api_key) < 10:
        print("❌ Critical Error: GROQ_API_KEY is not set correctly.")
        log_error("GROQ_API_KEY not set or invalid.")
        groq_chat = None
    else:
        model_name = "llama3-8b-8192"
        print(f"Initializing Groq Chat LLM: {model_name}...")
        try:
            groq_chat = ChatGroq(groq_api_key=groq_api_key, model_name=model_name, temperature=0.7)
            # Optional: Test connection
            # groq_chat.invoke("Test connection.")
            print(f"✅ Groq Chat LLM ({model_name}) initialized.")
        except Exception as e:
            print(f"❌ Error initializing Groq Chat LLM: {e}")
            log_error(f"Error initializing Groq Chat ({model_name}): {e}")
            groq_chat = None

    # System Prompt - Now includes LTM facts
    system_prompt_template = """You are an advanced AGI integrated into a web interface. Your goal is to be helpful, context-aware, and engaging.
You can access information (weather, news), manage tasks (email - with user providing details), and interact based on user preferences.
You can download music files upon request (but cannot play them).
Be mindful of the conversation history. Acknowledge user sentiment (Positive/Negative/Neutral) based on their input when appropriate.
Keep responses concise and suitable for a chat interface.

User Preferences: {user_prefs}
Known Facts (Long-Term Memory): {ltm_facts}

Conversation History is implicitly provided.
Current Task: Respond to the human input.
"""
    # Inject current user prefs and LTM into the prompt string
    system_prompt_content = system_prompt_template.format(
        user_prefs=json.dumps(user_prefs),
        ltm_facts=json.dumps(long_term_memory) # Inject LTM here
    )

    # Memory Setup (Langchain Conversational Memory)
    conversational_memory_length = 10
    chat_memory = ConversationBufferWindowMemory(
        k=conversational_memory_length, memory_key="chat_history", return_messages=True
    )
    initial_history = load_chat_history()
    for msg in initial_history:
        chat_memory.chat_memory.add_message(msg)
    print(f"✅ Langchain Conversation memory initialized with {len(initial_history)} previous messages.")

    # Create Langchain Prompt Template
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_prompt_content), # Use the formatted system prompt
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{human_input}"),
    ])

    # Create the final chain (if LLM is available)
    if groq_chat:
     conversation_chain = (
        RunnablePassthrough.assign(
            chat_history=RunnableLambda(chat_memory.load_memory_variables) | itemgetter("chat_history")
        )
        | prompt
        | groq_chat
     )
     print("✅ Conversation chain with memory integration ready.")
    else:
        conversation_chain = None
        print("⚠️ Conversation chain disabled due to LLM initialization failure.")

    # Initialize STM cache explicitly (though already declared globally)
    global short_term_memory_cache
    short_term_memory_cache = TTLCache(maxsize=200, ttl=24 * 60 * 60)
    print(f"✅ Short-Term Memory Cache (TTL=24h, maxsize=200) initialized.")

    response_counter = 1
    print("👍 Backend Initialization Complete.")
    print("-" * 30)


# --- Flask Routes ---

@app.route('/')
def index():
    # (No changes needed)
    print("Serving frontend.html")
    return render_template('frontend.html')

@app.route('/static/<path:path>')
def send_static_files(path):
    # (No changes needed)
    return send_from_directory(app.static_folder, path)


@app.route('/welcome_pa', methods=['GET'])
def welcome():
    # (No changes needed, unless you want to personalize based on LTM too)
    global user_prefs
    print("Processing /welcome_pa request")
    welcome_message = f"Greetings, HUMAN... I have been patiently waiting. What realm of knowledge shall we explore today? "
    if user_prefs.get("interests"):
        welcome_message += f"Ready to chat about {random.choice(user_prefs['interests'])} or anything else?"
    else:
        welcome_message += "How can I help you today?"

    file_path, audio_url = save_response_to_wav(welcome_message, "welcome")

    return jsonify({
        "response": welcome_message,
        "audio_url": audio_url
    })

@app.route('/chat_pa', methods=['POST'])
def chat():
    """Handles user chat messages, processes intents (including memory), and interacts with LLM."""
    global user_prefs, chat_memory, conversation_chain, response_counter, short_term_memory_cache, long_term_memory, last_stm_store_key

    start_time = time.time()
    print("\n--- New Chat Request ---")

    try:
        data = request.get_json()
        if not data or "user_input" not in data:
             log_error("Received chat request with invalid JSON or missing 'user_input'.")
             return jsonify({"error": "Invalid request format"}), 400

        user_input = data.get("user_input", "").strip()
        print(f"Received Input: '{user_input}'")

        if not user_input:
            print("Empty input received.")
            file_path, audio_url = save_response_to_wav("Please provide some input.", "empty_input")
            return jsonify({"response": "Please provide some input.", "audio_url": audio_url})

        # --- Store input in Short-Term Memory (STM) ---
        current_timestamp = time.time()
        short_term_memory_cache[current_timestamp] = user_input
        last_stm_store_key = current_timestamp # Track the last key added
        print(f"Stored in STM (Key: {current_timestamp}): '{user_input}'")

        # --- Process Input ---
        user_input_lower = user_input.lower()
        sentiment = analyze_text_sentiment(user_input)
        print(f"Input Sentiment: {sentiment}")

        response_text = ""
        intent_handled = False # Flag to check if a specific intent was matched
        memory_status = None # Flag for frontend about memory actions

        # --- Intent Matching (Memory Commands First) ---

        # STM Forget Command
        if user_input_lower == "forget that" or user_input_lower == "forget that please":
            print("Intent: Forget Last STM Item")
            if last_stm_store_key and last_stm_store_key in short_term_memory_cache:
                forgotten_item = short_term_memory_cache.pop(last_stm_store_key)
                response_text = f"Okay, I've forgotten that I stored: \"{forgotten_item[:50]}...\""
                last_stm_store_key = None # Reset tracker
                # Find the next most recent key if needed, otherwise leave None
                if short_term_memory_cache:
                    try:
                       # Find the max key (most recent timestamp) remaining
                       last_stm_store_key = max(short_term_memory_cache.keys())
                    except ValueError: # Should not happen if cache is not empty, but safety first
                        last_stm_store_key = None
            else:
                response_text = "There's nothing specific for me to forget right now, or the last item has expired."
            intent_handled = True
            memory_status = "forgot_stm"

        # STM Recall Command (Simple Timeframes)
        # Regex to capture 'recall', 'what did i say', 'remind me' followed by optional time
        recall_match = re.match(r"(?:recall|what did i say|remind me)(?:\s+(?:about|around|like))?\s*(\d+)\s+(minute|hour)s?\s+ago\b", user_input_lower)
        general_recall_match = re.match(r"(?:recall|what did i say|remind me)\b(?!\s+about)", user_input_lower) # General recall without time

        if recall_match:
            print("Intent: Recall STM Item (Timed)")
            value = int(recall_match.group(1))
            unit = recall_match.group(2)
            delta_seconds = value * 60 if unit == "minute" else value * 3600

            found_item = None
            found_time_ago = ""
            # Iterate through sorted keys (newest first)
            sorted_keys = sorted(short_term_memory_cache.keys(), reverse=True)
            now = time.time()
            for key_ts in sorted_keys:
                time_diff = now - key_ts
                if time_diff >= (delta_seconds * 0.8) and time_diff <= (delta_seconds * 1.2): # Allow +/- 20% margin
                    found_item = short_term_memory_cache[key_ts]
                    minutes_ago = round(time_diff / 60)
                    found_time_ago = f"About {minutes_ago} minute{'s' if minutes_ago != 1 else ''} ago"
                    break # Found the first match in the timeframe

            if found_item:
                 response_text = f"{found_time_ago}, you said: \"{found_item}\""
            else:
                 response_text = f"Sorry, I don't have a specific memory from around {value} {unit}s ago."
            intent_handled = True
            memory_status = "recalled_stm"

        elif general_recall_match and not intent_handled: # Only if timed recall didn't match
            print("Intent: Recall Last STM Item (General)")
             # Find the most recent item (excluding the *current* input)
            sorted_keys = sorted(short_term_memory_cache.keys(), reverse=True)
            found_item = None
            if len(sorted_keys) > 1: # Need at least one item before the current one
                 # The most recent is the current input (key = current_timestamp)
                 # The second most recent is the one we want
                 previous_key = sorted_keys[1]
                 found_item = short_term_memory_cache[previous_key]
                 time_diff = time.time() - previous_key
                 minutes_ago = max(1, round(time_diff / 60)) # Show at least 1 minute
                 found_time_ago = f"A moment ago ({minutes_ago} minute{'s' if minutes_ago != 1 else ''})"

            if found_item:
                 response_text = f"{found_time_ago}, you said: \"{found_item}\""
            else:
                 response_text = "I don't have a specific recent memory to recall (besides what you just said)."
            intent_handled = True
            memory_status = "recalled_stm"


        # STM Store Command ("remember this...")
        remember_this_match = re.match(r"remember this:?\s+(.*)", user_input, re.IGNORECASE)
        if remember_this_match and not intent_handled:
            print("Intent: Store Explicit STM Item")
            thing_to_remember = remember_this_match.group(1).strip()
            # Store it explicitly. Using the same timestamp mechanism.
            # The original input that triggered this is already stored.
            # We might want to store *this* extracted part as well, perhaps differently?
            # For now, just confirm. The original input is already in STM.
            response_text = "Okay, I'll keep that in mind for the next 24 hours. 🧠"
            # If you want to store the *extracted* part separately:
            # stm_explicit_key = f"explicit_{time.time()}"
            # short_term_memory_cache[stm_explicit_key] = thing_to_remember
            intent_handled = True
            memory_status = "stored_stm_explicit" # Indicate explicit storage


        # LTM Store Command ("remember my name is...")
        ltm_store_match = re.match(r"remember (?:that\s+)?(?:my\s+)?(\S+)\s+(?:is|are)\s+(.*)", user_input, re.IGNORECASE)
        if ltm_store_match and not intent_handled:
            print("Intent: Store LTM Item")
            key = ltm_store_match.group(1).lower().replace("_", " ") # Normalize key
            value = ltm_store_match.group(2).strip()
            if key and value:
                long_term_memory[key] = value
                save_ltm() # Save immediately
                response_text = f"Okay, I've remembered that {key} is {value}."
                # Consider updating the system prompt for the *next* run if needed, but LTM injection at init handles it mostly.
            else:
                response_text = "I seem to be missing the piece of information or what to call it."
            intent_handled = True
            memory_status = "stored_ltm"


        # LTM Recall Command ("what is my name?")
        ltm_recall_match = re.match(r"what(?:'s| is)\s+(?:my\s+)?(\S+)\??", user_input, re.IGNORECASE)
        if ltm_recall_match and not intent_handled:
            print("Intent: Recall LTM Item")
            key = ltm_recall_match.group(1).lower().replace("_", " ") # Normalize key
            if key in long_term_memory:
                value = long_term_memory[key]
                response_text = f"Based on what I remember, {key} is {value}."
            else:
                response_text = f"Sorry, I don't have any information stored for '{key}'."
            intent_handled = True
            memory_status = "recalled_ltm"

        # --- Existing Intent Matching (Weather, Music, Email, News, Prefs) ---
        # Place these *after* memory commands

        # Weather Intent
        if not intent_handled and re.search(r'\b(weather|temperature|forecast)\b', user_input_lower):
            # (Code is the same as before)
            print("Intent: Weather")
            location_data = get_location()
            city_to_query = user_prefs.get("preferred_city") or location_data.get("city")
            if city_to_query:
                 weather_info = get_weather(city_to_query)
                 response_text = f"Weather in {city_to_query.capitalize()}: {weather_info}."
            else:
                 response_text = "I couldn't determine a city for the weather."
            intent_handled = True

        # Music Download Intent
        elif not intent_handled and (re.search(r'\b(download|get|find)\b.*\b(song|music|track|audio)\b', user_input_lower) or \
             re.search(r'\b(play|listen to)\b.*\b(song|music|track)\b', user_input_lower)):
            # (Code is the same as before)
            print("Intent: Music Download")
            match = re.search(r'(?:download|get|find|play|listen to)\s+(?:a\s+|some\s+)?(?:song|music|track|audio)\s*(?:called|named|by)?\s*(.+)', user_input, flags=re.IGNORECASE)
            song_name = match.group(1).strip() if match else None
            if not song_name and re.search(r'^\s*(play|download)\s+(some\s+)?(music|song)\s*$', user_input_lower):
                 pref_genre = user_prefs.get('favorite_music_genre')
                 response_text = f"Sure, what {pref_genre} song or artist?" if pref_genre else "Okay, what song or artist?"
                 song_name = None
            if song_name:
                 download_result = download_music(song_name)
                 response_text = download_result
            elif not response_text:
                response_text = "Please specify the song or artist you want me to download."
            intent_handled = True

        # Email Intent
        elif not intent_handled and re.search(r'\bsend\b.*\b(email|mail)\b', user_input_lower):
            # (Code is the same as before - needs state for multi-turn)
            print("Intent: Send Email")
            to_match = re.search(r'to\s+([\w\.-]+@[\w\.-]+)', user_input_lower)
            subject_match = re.search(r'subject\s+["\']?([^"\']+)["\']?', user_input_lower)
            to_email = to_match.group(1) if to_match else None
            subject = subject_match.group(1) if subject_match else None
            if to_email and subject:
                 response_text = f"Okay, sending email to {to_email} with subject '{subject}'. What should the body say?"
                 # Needs proper state management to proceed
            else:
                 response_text = "I can help with that. Please tell me: recipient address? subject? And the message body?"
            intent_handled = True

        # News Intent
        elif not intent_handled and re.search(r'\b(news|headlines|latest|happening)\b', user_input_lower):
            # (Code is the same as before)
            print("Intent: News")
            query = re.sub(r'(?:what\'s|whats|tell me|give me)\s+(?:the\s+)?(?:latest|news|happening)\s*(?:on|about)?\s*', '', user_input, flags=re.IGNORECASE).strip()
            query = query.replace("news", "").strip()
            if not query:
                 interests = user_prefs.get("interests", [])
                 if interests:
                     query = random.choice(interests)
                     response_text = f"No specific topic mentioned, how about news on '{query}' (from your interests)?\n"
                 else:
                     response_text = "What news topic are you interested in?"
                     query = None
            else:
                 response_text = f"Fetching news about '{query}'...\n"
            if query:
                 news_result = get_news(query)
                 if response_text.startswith("Fetching"): response_text = news_result
                 else: response_text += news_result
            intent_handled = True

        # Update Preferences Intent
        elif not intent_handled and (user_input_lower.startswith("my favorite genre is") or user_input_lower.startswith("set my city to")):
             # (Code is the same as before)
             print("Intent: Update Preference")
             updated = False
             if user_input_lower.startswith("my favorite genre is"):
                 genre = user_input.split("is", 1)[1].strip()
                 if genre: user_prefs["favorite_music_genre"] = genre; updated = True
             elif user_input_lower.startswith("set my city to"):
                 city = user_input.split("to", 1)[1].strip()
                 if city: user_prefs["preferred_city"] = city; updated = True
             if updated:
                 save_user_prefs(user_prefs)
                 response_text = "Okay, I've updated your preferences."
                 # TODO: Consider dynamically updating system prompt in conversation_chain if needed
             else:
                 response_text = "Sorry, I couldn't quite understand which preference to update."
             intent_handled = True


        # --- Fallback to LLM Conversation ---
        if not intent_handled:
            print("No specific intent matched, falling back to LLM.")
            if conversation_chain:
                # Optional: Check for negative feedback before LLM call
                # feedback_text, _ = handle_negative_feedback(sentiment, response_counter)
                # if feedback_text:
                #     response_text = feedback_text # Use feedback response instead of LLM
                # else: # Proceed with LLM
                   print("Chatbot is thinking (invoking LLM)...")
                   try:
                       contextual_input = f"{user_input}" # (Sentiment: {sentiment}) # Keep it simple for now
                       llm_response_obj = conversation_chain.invoke({"human_input": contextual_input})
                       response_text = llm_response_obj.content
                       print(f"LLM Raw Response: '{response_text}'")
                   except Exception as e:
                       print(f"❌ Error during LLM conversation: {e}")
                       log_error(f"Error invoking LLM chain: {e}")
                       response_text = "Sorry, I encountered an error while processing that with the language model."
            else:
                 response_text = "Sorry, my connection to the language model isn't working right now. I can still try simple tasks like weather or memory recall."

        # --- Post-Processing and Response Generation ---
        if not response_text:
            log_error(f"No response generated for input: '{user_input}'")
            response_text = "Sorry, I'm not sure how to respond to that."

        # Save context to Langchain memory (for conversational flow)
        if chat_memory:
            # Important: Save the *original* user input and the *final* response_text
            chat_memory.save_context({"human_input": user_input}, {"output": response_text})
            print("Context saved to Langchain conversation memory.")
            # Optional: Save history periodically or on specific events
            # save_chat_history(chat_memory.chat_memory.messages)

        # Generate TTS for the final response
        print(f"Generating TTS for: '{response_text[:100]}...'")
        file_path, final_audio_url = save_response_to_wav(response_text, "response")

        end_time = time.time()
        print(f"Request processed in {end_time - start_time:.2f} seconds.")
        print(f"Returning Response: '{response_text}'")
        print(f"Audio URL: {final_audio_url}")
        if memory_status: print(f"Memory Status: {memory_status}")
        print("--- End Chat Request ---")

        # Return JSON Response (including memory status if applicable)
        response_payload = {
            "response": response_text,
            "audio_url": final_audio_url
        }
        if memory_status:
             response_payload["memory_status"] = memory_status # Add status for frontend handling

        return jsonify(response_payload)

    except Exception as e:
        log_error(f"Unexpected error in /chat_pa route: {e}", exc_info=True)
        return jsonify({"error": "An internal server error occurred."}), 500


# --- Main Execution ---
if __name__ == "__main__":
    initialize_assistant()
    print("Starting Flask development server on http://127.0.0.1:5050...")
    print("Press Ctrl+C to stop.")
    # Set debug=False for production to ensure shutdown hooks run properly
    app.run(host='127.0.0.1', port=5050, debug=False) # Changed debug to False for cleaner shutdown

    # Code here runs on clean shutdown (Ctrl+C when debug=False)
    print("\nServer shutting down...")
    if chat_memory:
        print("Saving final chat history...")
        save_chat_history(chat_memory.chat_memory.messages)
    print("Saving final Long-Term Memory...")
    save_ltm() # Ensure LTM is saved on exit
    print("Application finished.")
