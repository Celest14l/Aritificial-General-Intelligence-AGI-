# core_logic.py
import re
import random
import time
import json # Needed for get_ltm_summary fallback if used directly here

# Import specific functions from modules
from utils import analyze_text_sentiment, handle_negative_feedback, log_error
from services import get_weather, download_music, send_email, get_news, get_location
from memory_manager import (
    forget_last_stm, recall_stm_time, recall_stm_general,
    store_ltm_fact, recall_ltm_fact, get_ltm_summary # Import LTM functions
)
from storage import save_user_prefs

def process_chat_request(app_state, user_input, current_input_timestamp):
    """
    Processes the user's chat input, determines intent, interacts with
    services/memory, and returns a dictionary containing the response details.

    Args:
        app_state (dict): Dictionary holding initialized components.
        user_input (str): The raw input from the user.
        current_input_timestamp (float): Timestamp when the input was received/cached.

    Returns:
        dict: {
            "response_text": str,
            "memory_status": str or None,
            "music_filename": str or None # <<< Added
        }
    """
    print(f"Processing Input: '{user_input}'")
    user_input_lower = user_input.lower()

    # ... (extract necessary components from app_state) ...
    ontology = app_state.get('ontology', None) # Handle case where ontology might be None


    # Extract necessary components from app_state
    user_prefs = app_state['user_prefs']
    short_term_memory_cache = app_state['stm_cache']
    last_stm_key_tracker = app_state['last_stm_key_tracker']
    ontology = app_state.get('ontology') # Get the ontology object
    conversation_chain = app_state['conversation_chain']
    vader_analyzer = app_state['vader_analyzer']
    # Get the TTS function reference for feedback handler
    save_response_to_wav_func = app_state.get('save_response_to_wav_func')
    response_counter_ref = app_state['response_counter'] # Mutable counter ref

    sentiment = analyze_text_sentiment(user_input, vader_analyzer)
    print(f"Input Sentiment: {sentiment}")

    response_text = ""
    intent_handled = False
    memory_status = None
    music_filename = None # <<< Initialize filename

    # --- Intent Matching (Memory Commands First) ---
    # ... (STM/LTM commands remain the same, they only set response_text and memory_status) ...
    # Example:
    if user_input_lower == "forget that" or user_input_lower == "forget that please":
        print("Intent: Forget Last STM Item")
        response_text = forget_last_stm(short_term_memory_cache, last_stm_key_tracker)
        intent_handled = True
        memory_status = "forgot_stm"

    # STM Recall Command (Timed)
    recall_match = re.match(r"(?:recall|what did i say|remind me)(?:\s+(?:about|around|like))?\s*(\d+)\s+(minute|hour)s?\s+ago\b", user_input_lower)
    if not intent_handled and recall_match:
        print("Intent: Recall STM Item (Timed)")
        value = int(recall_match.group(1))
        unit = recall_match.group(2)
        response_text = recall_stm_time(short_term_memory_cache, value, unit)
        intent_handled = True
        memory_status = "recalled_stm"

    # STM Recall Command (General)
    general_recall_match = re.match(r"(?:recall|what did i say|remind me)\b(?!\s+about)", user_input_lower)
    if not intent_handled and general_recall_match:
        print("Intent: Recall Last STM Item (General)")
        response_text = recall_stm_general(short_term_memory_cache, current_input_timestamp)
        intent_handled = True
        memory_status = "recalled_stm"

    # STM Store Command ("remember this...")
    remember_this_match = re.match(r"remember this:?\s+(.*)", user_input, re.IGNORECASE)
    if not intent_handled and remember_this_match:
        print("Intent: Store Explicit STM Item")
        # The input is already stored implicitly by memory_manager.add_to_stm
        # We just provide confirmation.
        response_text = "Okay, I'll keep that in mind for the next 24 hours. 🧠"
        intent_handled = True
        memory_status = "stored_stm_explicit"

    # LTM Store Command ("remember my name is...")
    ltm_store_match = re.match(r"remember (?:that\s+)?(?:my\s+)?(\S+)\s+(?:is|are)\s+(.*)", user_input, re.IGNORECASE)
    if not intent_handled and ltm_store_match:
        print("Intent: Store LTM Item")
        key = ltm_store_match.group(1)
        value = ltm_store_match.group(2)
        # Pass the actual ontology object from app_state
        response_text = store_ltm_fact(ontology, key, value) # Pass onto
        intent_handled = True
        memory_status = "stored_ltm"

    # LTM Recall Command ("what is my name?")
    ltm_recall_match = re.match(r"what(?:'s| is)\s+(?:my\s+)?(\S+)\??", user_input, re.IGNORECASE)
    if not intent_handled and ltm_recall_match:
        print("Intent: Recall LTM Item")
        key = ltm_recall_match.group(1)
        # Pass the actual ontology object from app_state
        response_text = recall_ltm_fact(ontology, key) # Pass onto
        intent_handled = True
        memory_status = "recalled_ltm"

    # --- Standard Intents ---
    # Weather Intent
    elif not intent_handled and re.search(r'\b(weather|temperature|forecast)\b', user_input_lower):
        print("Intent: Weather")
        location_data = get_location(user_prefs) # Pass prefs
        city_to_query = user_prefs.get("preferred_city") or location_data.get("city")
        response_text = get_weather(city_to_query) # Get full response from service
        intent_handled = True

    # Music Download Intent
    elif not intent_handled and (re.search(r'\b(download|get|find|play|listen to)\b.*\b(song|music|track|audio)\b', user_input_lower)):
        print("Intent: Music Download/Play")
        # Keep the improved regex
        match = re.search(r'(?:download|get|find|play|listen to)\s+(?:a\s+|the\s+|some\s+)?(?:song|music|track|audio)\s*(?:called|named|by)?\s*(.+)', user_input, flags=re.IGNORECASE)
        song_name = match.group(1).strip() if match else None

        if not song_name and re.search(r'^\s*(play|download)\s+(some\s+)?(music|song)\s*$', user_input_lower):
             pref_genre = user_prefs.get('favorite_music_genre')
             response_text = f"Sure, what {pref_genre} song or artist?" if pref_genre else "Okay, what song or artist should I find?"
             song_name = None

        if song_name:
             # Call the modified service function
             download_message, downloaded_filename = download_music(song_name)
             response_text = download_message # Use the message from the service
             if downloaded_filename:
                 music_filename = downloaded_filename # Store the filename to return <<<
                 print(f"Music filename to return: {music_filename}")
             # No need to set response_text again here unless download failed specifically

        elif not response_text: # Only set if not already asking for clarification
             response_text = "Please specify the song or artist you want me to play or download."

        intent_handled = True



    # Email Intent (Requires multi-turn state not implemented here)
    elif not intent_handled and re.search(r'\bsend\b.*\b(email|mail)\b', user_input_lower):
        print("Intent: Send Email")
        # Extract details if possible (simple regex, likely insufficient)
        to_match = re.search(r'to\s+([\w\.-]+@[\w\.-]+)', user_input_lower)
        subject_match = re.search(r'subject\s+["\']?([^"\']+)["\']?', user_input_lower)
        to_email = to_match.group(1) if to_match else None
        subject = subject_match.group(1) if subject_match else None

        if to_email and subject:
             # Still need the body - ask for it
             response_text = f"Okay, sending email to {to_email} with subject '{subject}'. What should the body say? (Email sending not fully implemented yet)"
             # Need state here to remember 'to' and 'subject' for the next turn
        else:
             # Ask for all details
             response_text = "I can help with that. Please tell me: Who is the email for (recipient address)? What is the subject? And what message should be in the body? (Email sending not fully implemented yet)"
        # Actual sending would require state management across requests
        # response_text = send_email(to_email, subject, body_content)
        intent_handled = True


    # News Intent
    elif not intent_handled and re.search(r'\b(news|headlines|latest|happening)\b', user_input_lower):
        print("Intent: News")
        # Extract topic, default to general interests if specific topic absent
        query = re.sub(r'(?:what\'s|whats|tell me|give me)\s+(?:the\s+)?(?:latest|news|happening)\s*(?:on|about)?\s*', '', user_input, flags=re.IGNORECASE).strip()
        query = query.replace("news", "").strip() # Remove the word "news" itself if it's the query

        if not query:
             interests = user_prefs.get("interests", [])
             if interests:
                 query = random.choice(interests)
                 response_text = f"No specific topic mentioned, how about news on '{query}' (from your interests)?\n"
             else:
                 response_text = "What news topic are you interested in (e.g., technology, world, sports)?"
                 query = None # Prevent search
        else:
             response_text = f"Fetching news about '{query}'...\n" # Initial confirmation

        if query:
             news_result = get_news(query)
             # Append result to confirmation or replace if it was just asking
             if response_text.startswith("Fetching"):
                  response_text = news_result # Replace placeholder
             elif response_text.endswith("?\n"):
                 response_text += news_result # Append results to the question asking for topic
             else:
                  response_text = news_result # Default to just showing result if query was specific

        intent_handled = True


    # Update Preferences Intent
    elif not intent_handled and (user_input_lower.startswith("my favorite genre is") or user_input_lower.startswith("set my city to")):
        print("Intent: Update Preference")
        updated = False
        if user_input_lower.startswith("my favorite genre is"):
             genre = user_input.split("is", 1)[1].strip()
             if genre: user_prefs["favorite_music_genre"] = genre; updated = True
        elif user_input_lower.startswith("set my city to"):
             city = user_input.split("to", 1)[1].strip()
             if city: user_prefs["preferred_city"] = city; updated = True
             # Add more preference updates (e.g., interests)

        if updated:
             save_user_prefs(user_prefs) # Save immediately
             response_text = "Okay, I've updated your preferences."
             # TODO: Update system prompt in conversation chain if necessary (complex)
             # This usually requires rebuilding the chain or modifying the SystemMessage
             # in the memory object if the LLM needs the updated pref *immediately*.
             # For now, it will be reflected on next restart or if prompt regenerated.
        else:
             response_text = "Sorry, I couldn't quite understand which preference to update."
        intent_handled = True


    # --- Fallback to LLM ---
    if not intent_handled:
        print("No specific intent matched, falling back to LLM.")
        if conversation_chain:
            
            print("Chatbot is thinking (invoking LLM)...")
            try:
                # Pass history via chain's memory integration
                llm_response_obj = conversation_chain.invoke({"human_input": user_input})
                response_text = llm_response_obj.content
                print(f"LLM Raw Response: '{response_text}'")
            except Exception as e:
                log_error(f"Error invoking LLM chain: {e}", exc_info=True)
                response_text = "Sorry, I encountered an error processing that with the language model."
        else:
             response_text = "My language model connection isn't working. I can still try simple tasks like memory recall or weather."

     # --- Final Response Preparation ---
    if not response_text:
        log_error(f"No response generated for input: '{user_input}'")
        response_text = "Sorry, I'm not sure how to respond to that."

    # Return a dictionary
    return {
        "response_text": response_text,
        "memory_status": memory_status,
        "music_filename": music_filename
    }