# app.py
import os
import time
import json
import random # Needed for welcome message interest choice
from flask import Flask, request, jsonify, render_template, send_from_directory, url_for
from cachetools import TTLCache
from operator import itemgetter
import services


# Langchain & Model Imports
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_groq import ChatGroq
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import VitsModel, AutoTokenizer

# Local Module Imports
import config
from utils import log_error # Only log_error needed directly here
import storage
import services
import memory_manager
import core_logic

# --- Flask App Initialization ---
# Make sure DOWNLOADS_DIR is configured correctly
if not os.path.isdir(config.DOWNLOADS_DIR):
     print(f"🚨 WARNING: Downloads directory does not exist or is not configured: {config.DOWNLOADS_DIR}")
     # Consider creating it: os.makedirs(config.DOWNLOADS_DIR, exist_ok=True)
     # Or exiting if music playing is critical

app = Flask(__name__, template_folder='.', static_folder='static')

# --- Global State Dictionary ---
# Holds initialized models, memory stores, preferences, etc.
app_state = {
    "user_prefs": {},
    "ontology": None, # Ontology object placeholder
    "ontology_world": None, # Owlready2 World object placeholder
    "stm_cache": None,
    "last_stm_key_tracker": {'key': None}, # Use dict for mutability
    "response_counter": {'count': 1}, # Use dict for mutability
    "vader_analyzer": None,
    "tts_model": None,
    "tts_tokenizer": None,
    "conversation_chain": None,
    "chat_memory": None, # Langchain conversation memory
    "save_response_to_wav_func": None # Placeholder for the TTS function reference
}

# --- Initialization ---
def initialize_assistant():
    """Loads models, preferences, memory, ontology and sets up the AGI state."""
    print("🚀 Initializing AGI Backend...")
    start_time = time.time()

    # Ensure directories exist
    config.ensure_directories()

    # Check critical config
    if not config.check_critical_config():
        print("🚨 Initialization halted due to critical configuration errors.")
        # Consider exiting or running with limited features
        # exit(1)

    # Load User Prefs
    app_state["user_prefs"] = storage.load_user_prefs()
    print(f"User preferences loaded: {app_state['user_prefs']}")

    # Load Long-Term Memory Ontology using storage module
    app_state["ontology"], app_state["ontology_world"] = storage.load_ltm_ontology()

    # Initialize VADER
    app_state["vader_analyzer"] = SentimentIntensityAnalyzer()
    print("✅ VADER Sentiment Analyzer initialized.")

    # Initialize TTS
    try:
        print(f"Loading TTS model: {config.TTS_MODEL_ID}...")
        app_state["tts_model"] = VitsModel.from_pretrained(config.TTS_MODEL_ID)
        app_state["tts_tokenizer"] = AutoTokenizer.from_pretrained(config.TTS_MODEL_ID)
        # Store a reference to the service function within app_state for reuse
        app_state["save_response_to_wav_func"] = lambda text, base: services.save_response_to_wav(
            text, base, app_state["tts_model"], app_state["tts_tokenizer"], app_state["response_counter"]
        )
        print(f"✅ TTS Model loaded successfully.")
    except Exception as e:
        print(f"❌ Critical Error: Failed to load TTS model ({config.TTS_MODEL_ID}): {e}")
        log_error(f"Critical Error loading TTS model: {e}")
        app_state["tts_model"] = None
        app_state["tts_tokenizer"] = None
        app_state["save_response_to_wav_func"] = None


    # Initialize LLM
    groq_chat = None
    if config.GROQ_API_KEY and config.GROQ_API_KEY != "YOUR_GROQ_API_KEY_HERE":
        print(f"Initializing Groq Chat LLM: {config.LLM_MODEL_NAME}...")
        try:
            groq_chat = ChatGroq(groq_api_key=config.GROQ_API_KEY, model_name=config.LLM_MODEL_NAME, temperature=0.7)
            # Optional: Test connection
            # groq_chat.invoke("Test connection.")
            print(f"✅ Groq Chat LLM ({config.LLM_MODEL_NAME}) initialized.")
        except Exception as e:
            print(f"❌ Error initializing Groq Chat LLM: {e}")
            log_error(f"Error initializing Groq Chat ({config.LLM_MODEL_NAME}): {e}")
            groq_chat = None
    else:
        print("⚠️ Groq API Key not found or invalid. LLM functionality disabled.")


    # --- Modify System Prompt ---
    print("Generating LTM summary for system prompt...")
    ltm_summary_str = memory_manager.get_ltm_summary(app_state["ontology"])
    print(f"LTM Summary: {ltm_summary_str}")

    # Define the full system prompt template - UPDATED
    system_prompt_template = """You are an advanced AGI integrated into a web interface. Your goal is to be helpful, context-aware, and engaging.
You can access information (weather, news), manage tasks (email - with user providing details), and interact based on user preferences.
You can download music files upon request and trigger playback in the user's browser. Let the user know when you are playing the music.
You have access to a short-term memory (last 24 hours, recallable via commands like 'what did I say X minutes ago?') and a persistent knowledge base (recallable via 'what is my [fact]?').
Be mindful of the conversation history provided. Acknowledge user sentiment (Positive/Negative/Neutral) based on their input when appropriate.
Keep responses concise and suitable for a chat interface.

User Preferences: {user_prefs}
Known Facts (Knowledge Base Summary): {ltm_facts}

Conversation History is implicitly provided.
Current Task: Respond to the human input.
"""
    system_prompt_content = system_prompt_template.format(
        user_prefs=json.dumps(app_state["user_prefs"]),
        ltm_facts=ltm_summary_str
    )


    # Langchain Memory Setup
    app_state["chat_memory"] = ConversationBufferWindowMemory(
        k=config.CONVERSATIONAL_MEMORY_LENGTH, memory_key="chat_history", return_messages=True
    )
    initial_history = storage.load_chat_history()
    for msg in initial_history:
        app_state["chat_memory"].chat_memory.add_message(msg)
    print(f"✅ Langchain Conversation memory initialized with {len(initial_history)} previous messages.")


    # --- Create Langchain Prompt Template (Update with new system prompt) ---
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_prompt_content), # Use updated content
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{human_input}"),
    ])
    

    # Create the final LLM chain
    if groq_chat:
     app_state["conversation_chain"] = (
        RunnablePassthrough.assign(
            chat_history=RunnableLambda(app_state["chat_memory"].load_memory_variables) | itemgetter("chat_history")
        )
        | prompt
        | groq_chat
     )
     print("✅ Conversation chain ready.")
    else:
        app_state["conversation_chain"] = None
        print("⚠️ Conversation chain disabled (LLM init failed or API key missing).")

    # Initialize STM cache
    app_state["stm_cache"] = TTLCache(maxsize=config.STM_MAX_SIZE, ttl=config.STM_TTL_SECONDS)
    print(f"✅ Short-Term Memory Cache (TTL={config.STM_TTL_SECONDS}s, maxsize={config.STM_MAX_SIZE}) initialized.")

    end_time = time.time()
    print(f"👍 Backend Initialization Complete ({end_time - start_time:.2f} seconds).")
    print("-" * 30)


# --- Flask Routes ---
@app.route('/')
def index():
    """Serves the main HTML frontend page."""
    print("Serving frontend.html")
    return render_template('frontend.html')

@app.route('/static/<path:path>')
def send_static_files(path):
    """Serves static files from the static folder, including generated audio."""
    # print(f"Serving static file: {path}") # Debugging
    return send_from_directory(app.static_folder, path)
# --- NEW ROUTE for serving downloaded music ---
@app.route('/downloads/<path:filename>')
def serve_downloaded_music(filename):
    """Serves downloaded music files from the DOWNLOADS_DIR."""
    print(f"Serving downloaded file: {filename}")
    # IMPORTANT: Add security checks if needed, e.g., ensure filename is safe
    # For basic use, send_from_directory handles basic path safety.
    try:
        return send_from_directory(config.DOWNLOADS_DIR, filename, as_attachment=False) # as_attachment=False for streaming/playing
    except FileNotFoundError:
        log_error(f"Attempted to serve non-existent download file: {filename}")
        return jsonify({"error": "File not found"}), 404
    except Exception as e:
        log_error(f"Error serving download file {filename}: {e}", exc_info=True)
        return jsonify({"error": "Server error serving file"}), 500


@app.route('/welcome_pa', methods=['GET'])
def welcome():
    """Provides the initial welcome message and TTS."""
    welcome_message = f"Greetings, HUMAN... I have been patiently waiting. What realm of knowledge shall we explore today? "
    # ... (existing welcome logic) ...
    interests = app_state["user_prefs"].get("interests", [])
    if interests:
        welcome_message += f"Ready to chat about {random.choice(interests)} or anything else?"
    else:
        welcome_message += "How can I help you today?"
    # Optionally, personalize further using LTM facts if relevant facts are stored

    # Ensure url_for uses the correct relative path from save_response_to_wav
    if app_state["save_response_to_wav_func"]:
        _, relative_audio_path_for_url = app_state["save_response_to_wav_func"](welcome_message, "welcome")
        if relative_audio_path_for_url:
             try:
                 # Use the relative path returned by the service function
                 audio_url = url_for('static', filename=relative_audio_path_for_url, _external=False)
                 print(f"Welcome audio URL: {audio_url}")
             except RuntimeError as e:
                 log_error(f"Error generating URL for welcome audio (Flask context issue?): {e}")
                 # Fallback if url_for fails (should not happen in request context)
                 audio_url = f"/static/{relative_audio_path_for_url}"

    # ... (rest of welcome logic) ...
    return jsonify({
        "response": welcome_message,
        "audio_url": audio_url # URL for browser TTS playback
    })

@app.route('/chat_pa', methods=['POST'])
def chat():
    """Handles user chat messages, delegates processing, returns response."""
    start_time = time.time()
    print("\n--- New Chat Request ---")

    try:
        data = request.get_json()
        if not data or "user_input" not in data:
             log_error("Received chat request with invalid JSON or missing 'user_input'.")
             return jsonify({"error": "Invalid request format"}), 400

        user_input = data.get("user_input", "").strip()
        response_text = ""
        memory_status = None
        final_audio_url = None
        final_music_url = None # <<< Add variable for music URL

        if not user_input:
            response_text = "Please provide some input."
            if app_state["save_response_to_wav_func"]:
                 _, relative_audio_path = app_state["save_response_to_wav_func"](response_text, "empty_input")
                 if relative_audio_path: final_audio_url = url_for('static', filename=relative_audio_path, _external=False)

        elif user_input.lower() == "exit":
            response_text = "Goodbye! Feel free to refresh the page."
            memory_status = "exit"
            # ... (save history logic) ...
            if app_state["save_response_to_wav_func"]:
                 _, relative_audio_path = app_state["save_response_to_wav_func"](response_text, "farewell")
                 if relative_audio_path: final_audio_url = url_for('static', filename=relative_audio_path, _external=False)

        else:
            # --- Core Logic ---
            current_timestamp = time.time()
            memory_manager.add_to_stm(
                app_state['stm_cache'],
                app_state['last_stm_key_tracker'],
                user_input
            )

            # Process request using core_logic module
            # Expect core_logic to potentially return music filename now
            response_data = core_logic.process_chat_request(
                app_state,
                user_input,
                current_timestamp
            )

            # Unpack response data (adjust based on core_logic's return type)
            # Assuming it returns a dictionary now:
            response_text = response_data.get("response_text", "Sorry, I couldn't process that.")
            memory_status = response_data.get("memory_status")
            music_filename = response_data.get("music_filename") # <<< Get filename if available

            # --- Generate Music URL if filename provided ---
            if music_filename:
                try:
                    # Use the new '/downloads/' route
                    final_music_url = url_for('serve_downloaded_music', filename=music_filename, _external=False)
                    print(f"Music URL generated: {final_music_url}")
                except Exception as e:
                    log_error(f"Error generating URL for music file {music_filename}: {e}")
                    # Don't send a broken URL, maybe adjust response_text?
                    response_text += " (But there was an issue preparing it for playback)."


            # Save context AFTER potentially modifying response_text or getting music URL
            if app_state.get("chat_memory"):
                if not isinstance(response_text, str):
                    log_error(f"Non-string response generated, converting: {response_text}")
                    response_text = str(response_text)
                app_state["chat_memory"].save_context({"human_input": user_input}, {"output": response_text})
                print("Context saved to Langchain conversation memory.")

            # Generate TTS for the textual response
            if response_text and app_state["save_response_to_wav_func"]:
                print(f"Generating TTS for: '{response_text[:100]}...'")
                _, relative_audio_path = app_state["save_response_to_wav_func"](response_text, "response")
                if relative_audio_path:
                     try:
                        final_audio_url = url_for('static', filename=relative_audio_path, _external=False)
                        print(f"TTS Audio URL: {final_audio_url}")
                     except RuntimeError as e:
                         log_error(f"Error generating TTS URL (Flask context issue?): {e}")
                         final_audio_url = f"/static/{relative_audio_path}" # Fallback


        # --- Prepare and Return Response ---
        end_time = time.time()
        print(f"Request processed in {end_time - start_time:.2f} seconds.")
        print(f"Returning Response Text: '{response_text}'")
        if final_music_url: print(f"Returning Music URL: {final_music_url}")
        if final_audio_url: print(f"Returning TTS URL: {final_audio_url}")
        if memory_status: print(f"Memory Status: {memory_status}")
        print("--- End Chat Request ---")

        # Build the JSON payload, including the music URL if available
        response_payload = {
            "response": response_text,
            "audio_url": final_audio_url, # TTS audio
            "music_url": final_music_url # Music audio <<< ADDED
        }
        if memory_status:
             response_payload["memory_status"] = memory_status

        return jsonify(response_payload)

    except Exception as e:
        log_error(f"Unexpected error in /chat_pa route: {e}", exc_info=True)
        return jsonify({"error": "An internal server error occurred."}), 500


# --- Main Execution ---
if __name__ == "__main__":
    initialize_assistant() # Load models, prefs, ontology etc.
    print("Starting Flask development server on http://127.0.0.1:5050...")
    print("Press Ctrl+C to stop.")
    # Set debug=False for production to ensure shutdown hooks run properly
    # Use debug=True during development for auto-reload & better tracebacks
    app.run(host='127.0.0.1', port=5050, debug=False)

    # --- Shutdown Hook (runs when debug=False and Ctrl+C is pressed) ---
    print("\nServer shutting down...")
    # Save Langchain history
    if app_state.get("chat_memory"):
        print("Saving final chat history...")
        storage.save_chat_history(app_state["chat_memory"].chat_memory.messages)
    # Save LTM Ontology
    if app_state.get("ontology_world"): # Check if world object exists
        print("Saving final Long-Term Memory Ontology...")
        storage.save_ltm_ontology(app_state["ontology_world"])
    print("Application finished.")