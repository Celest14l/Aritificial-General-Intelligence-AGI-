#Contains functions that interact with external APIs or perform distinct tasks like TTS, weather, news, music download, email, location.
# services.py
import os
import requests
import smtplib
import re
import json
import time
from email.message import EmailMessage
import yt_dlp
import torch
import soundfile as sf

from utils import log_error
from config import (
    NEWS_SERVICE_URL, EMAIL_USER, EMAIL_PASS, SMTP_HOST, SMTP_PORT,
    DOWNLOADS_DIR, BASE_DIR, AUDIO_DIR, FFMPEG_PATH
)

# --- Location Service ---
def get_location(user_prefs):
    """Fetches location based on server's IP or user preference."""
    preferred_city = user_prefs.get("preferred_city", "Pune")
    try:
        response = requests.get("http://ip-api.com/json/", timeout=5)
        response.raise_for_status()
        data = response.json()
        if data.get("status") == "success" and "city" in data:
            return {"city": data.get("city"), "country": data.get("country", "")}
        else:
            log_error(f"Unexpected location API response: {data.get('message', 'No message')}")
            return {"city": preferred_city, "country": ""}
    except Exception as e:
        log_error(f"Error fetching location: {e}")
        return {"city": preferred_city, "country": ""}

# --- Weather Service ---
def get_weather(city):
    """Fetches weather for a given city from wttr.in."""
    if not city:
        return "Please specify a city for the weather."
    try:
        # Ensure city name is URL-encoded if it contains spaces or special chars
        safe_city = requests.utils.quote(city)
        url = f"https://wttr.in/{safe_city}?format=%C+%t"
        response = requests.get(url, timeout=7)
        response.raise_for_status()
        weather_data = response.text.strip()
        if weather_data and "Unknown location" not in weather_data and "We were unable to process your request" not in weather_data:
            return f"Weather in {city.capitalize()}: {weather_data}."
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


# --- Music Download Service ---
def download_music(song_name):
    """
    Downloads music using yt-dlp.
    Returns a tuple: (message_for_user, filename_or_None)
    """
    print(f"🔍 Attempting download: {song_name}")
    safe_song_name = re.sub(r'[\\/*?:"<>|]', '_', song_name).strip()
    if not safe_song_name:
        return "Please provide a valid song name to download.", None

    # Define expected final path *before* ydl_opts modifies it
    final_filename = f"{safe_song_name}.mp3"
    final_file_path = os.path.join(DOWNLOADS_DIR, final_filename)
    output_template = os.path.join(DOWNLOADS_DIR, f"{safe_song_name}.%(ext)s") # Template for yt-dlp

    if os.path.exists(final_file_path):
        print(f"🎵 Song already downloaded: {final_file_path}")
        # Return message and the existing filename
        return f"'{song_name}' is already available. Playing it now!", final_filename

    ydl_opts = {
        'format': 'bestaudio/best',
        'noplaylist': True,
        'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'mp3', 'preferredquality': '128'}],
        'outtmpl': output_template, # Use the template yt-dlp needs
        'quiet': False, 'default_search': 'ytsearch1',
        'ffmpeg_location': FFMPEG_PATH,
        'nocheckcertificate': True,
        'retries': 3,
        'socket_timeout': 15,
    }
    try:
        print(f"Starting download process for '{song_name}'...")
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(f"ytsearch1:{song_name}", download=True) # Search and download
            # Try to get the exact filename yt-dlp *intended* to use after processing
            # This is complex as hooks might be needed for certainty.
            # We will rely on checking if our *expected* final_file_path exists.

        # Check if the expected MP3 file exists now
        if os.path.exists(final_file_path):
            print(f"✅ Download successful: {final_file_path}")
            # Return success message and the filename
            return f"Successfully downloaded '{song_name}'. Playing it now!", final_filename
        else:
            # Fallback: Check if *any* mp3 file matching the safe name exists (less reliable)
            found_files = [f for f in os.listdir(DOWNLOADS_DIR) if f.startswith(safe_song_name) and f.endswith('.mp3')]
            if found_files:
                actual_filename = found_files[0]
                actual_file_path = os.path.join(DOWNLOADS_DIR, actual_filename)
                # Optional: Rename to the cleaner final_file_path if different
                if actual_file_path != final_file_path:
                    try:
                        os.rename(actual_file_path, final_file_path)
                        print(f"Renamed downloaded file to {final_filename}")
                        actual_filename = final_filename # Use the new name
                    except OSError as rename_err:
                        log_error(f"Could not rename {actual_filename} to {final_filename}: {rename_err}")
                        # Proceed with the original found filename if rename failed
                print(f"✅ Download successful (as {actual_filename}): {actual_file_path}")
                return f"Successfully downloaded '{song_name}'. Playing it now!", actual_filename
            else:
                log_error(f"Download for '{song_name}' completed, but expected MP3 '{final_file_path}' not found.")
                return f"Sorry, there was an issue finalizing the download for '{song_name}'.", None

    except yt_dlp.utils.DownloadError as de:
        log_error(f"yt-dlp DownloadError for '{song_name}': {de}")
        # Check if the error message indicates it's already downloaded (less common now with pre-check)
        if 'already been downloaded' in str(de):
             if os.path.exists(final_file_path):
                 print(f"🎵 Song already downloaded (detected by yt-dlp error): {final_file_path}")
                 return f"'{song_name}' was already downloaded. Playing it now!", final_filename
             else:
                 # This case is unlikely but possible if the file was moved/deleted after yt-dlp's check
                 return f"It seems '{song_name}' was downloaded previously but I can't find the file now.", None
        return f"Sorry, an error occurred during download for '{song_name}'. It might be unavailable or blocked.", None
    except Exception as e:
        log_error(f"Unexpected error downloading music '{song_name}': {e}", exc_info=True) # Add traceback logging
        return f"Sorry, an unexpected error occurred while trying to download '{song_name}'.", None

# --- Email Service ---
def send_email(to_email, subject, body_content, attachments=[]):
    """Sends email using configured credentials."""
    if not EMAIL_USER or not EMAIL_PASS:
        log_error("Email sending failed: Credentials missing.")
        return "Email sending failed: Assistant's email credentials are not configured."
    try:
        msg = EmailMessage()
        msg["From"] = EMAIL_USER
        msg["To"] = to_email
        msg["Subject"] = subject
        msg.set_content(body_content)
        # Attachment logic (simplified - ensure paths are valid server paths)
        for file_path in attachments:
             if os.path.exists(file_path):
                 # ... (add attachment logic from original code if needed) ...
                 pass
             else:
                log_error(f"Email attachment skipped (not found): {file_path}")

        print(f"Connecting to SMTP server {SMTP_HOST}:{SMTP_PORT}...")
        with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT) as server:
            server.login(EMAIL_USER, EMAIL_PASS)
            server.send_message(msg)
            print(f"✅ Email sent successfully to {to_email}.")
            return f"Email sent successfully to {to_email}."
    except smtplib.SMTPAuthenticationError:
         log_error(f"SMTP Auth Error for user {EMAIL_USER}.")
         return "Error sending email: Authentication failed."
    except Exception as e:
        log_error(f"Unexpected error sending email to {to_email}: {e}")
        return f"An unexpected error occurred while sending the email."

# --- News Service ---
def get_news(query):
    """Fetches news headlines from the dedicated microservice."""
    if not query: return "Please specify a topic for the news."
    print(f"Fetching news for query: '{query}' from {NEWS_SERVICE_URL}")
    try:
        params = {'query': query}
        response = requests.get(NEWS_SERVICE_URL, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        if data.get("status") == "success" and data.get("articles"):
            articles = data["articles"][:3] # Limit to top 3
            if not articles: return f"No articles found for '{query}'."
            news_summary = f"Headlines for '{query}':\n"
            for i, article in enumerate(articles, 1):
                title = article.get('title', 'N/A')
                source = article.get('source', {}).get('name', 'N/A')
                news_summary += f"{i}. {title} ({source})\n"
            return news_summary.strip()
        elif data.get("message"):
             log_error(f"News service error for '{query}': {data['message']}")
             return f"News service error: {data['message']}"
        else:
             return f"Couldn't get news for '{query}'. Unexpected response."
    except Exception as e:
        log_error(f"Unexpected error fetching news for '{query}': {e}")
        return f"An error occurred while fetching news for '{query}'."


# --- Text-to-Speech (TTS) Service ---
# This function needs the loaded model/tokenizer and a counter
def save_response_to_wav(response_text, file_name_base, tts_model, tts_tokenizer, response_counter_ref):
    """Generates audio, saves it, returns URL and file path."""
    if not tts_model or not tts_tokenizer:
        print("⚠️ TTS service unavailable (models not loaded).")
        return None, None
    if not response_text:
        print("⚠️ Cannot generate TTS for empty text.")
        return None, None

    output_path = None
    output_url = None
    try:
        # Increment counter using mutable reference (e.g., list or dict)
        current_count = response_counter_ref['count']
        response_counter_ref['count'] += 1

        safe_base = re.sub(r'[\\/*?:"<>|]', "", file_name_base)
        unique_file_name = f"{safe_base}_{current_count}.wav"
        output_path = os.path.join(AUDIO_DIR, unique_file_name)
        # Generate URL path relative to the static folder base
        # Need Flask app context for url_for
        # We pass the function from app.py or generate relative path manually
        # Manual relative path (less robust than url_for):
        relative_audio_path_for_url = f'responses_output/{unique_file_name}'
        # output_url = url_for('static', filename=f'responses_output/{unique_file_name}', _external=False)

        inputs = tts_tokenizer(response_text, return_tensors="pt")
        with torch.no_grad():
            output = tts_model(**inputs).waveform
            if not isinstance(output, torch.Tensor):
                 raise TypeError("Expected TTS output waveform to be a Tensor")
            audio = output.squeeze().cpu().numpy()

        sampling_rate = tts_model.config.sampling_rate
        sf.write(output_path, audio, samplerate=sampling_rate)
        print(f"🎙️ Generated audio: {output_path}")
        # Return absolute path and the relative path for URL generation
        return output_path, relative_audio_path_for_url # <<< CHANGE HERE
    except Exception as e:
        log_error(f"Error during TTS generation for text '{response_text[:50]}...': {e}")
        return None, None