#Contains general utility functions like logging, sentiment analysis, and feedback handling.
# utils.py
import datetime
import random
import json
# No vader import here yet, will be passed in or initialized within a class

def log_error(error_msg, exc_info=False):
    """Logs an error message to the error log file."""
    # Requires ERROR_LOG_FILE path from config
    from config import ERROR_LOG_FILE # Import here to avoid circular dependency issues at load time
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"{timestamp}: {error_msg}\n"
    if exc_info:
        import traceback
        log_entry += traceback.format_exc() + "\n"
    print(f"ERROR: {log_entry.strip()}")
    try:
        with open(ERROR_LOG_FILE, 'a', encoding='utf-8') as f:
            f.write(log_entry)
    except Exception as e:
        print(f"🚨 Critical Error: Could not write to log file {ERROR_LOG_FILE}: {e}")


# Sentiment Analysis needs the analyzer object
def analyze_text_sentiment(text, vader_analyzer):
    """Analyzes the sentiment of a given text using VADER."""
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


# Feedback handling needs user_prefs and the TTS function (save_response_to_wav)
def handle_negative_feedback(user_input_sentiment, user_prefs, save_response_to_wav_func, response_counter):
    """Checks for negative sentiment and constructs a response."""
    motivational_quotes = [
        "Remember to take a deep breath. You've got this.", "Even small steps forward are progress. Keep going.",
        "Be kind to yourself. It's okay to have tough moments.", "Focus on what you can control. Brighter times are ahead."
    ]
    feedback_response = None
    feedback_audio_url = None

    if user_input_sentiment == "Negative":
        print("Detected Negative sentiment in user input.")
        response_parts = [
            "I noticed your message seemed a bit negative.", "I hope everything is alright.",
            "Perhaps we could try something different?",
        ]
        interests = user_prefs.get('interests', [])
        if interests: response_parts.append(f"Maybe talk about {random.choice(interests)}?")
        response_parts.append("Or I could try to download some calming music for you?")
        response_parts.append(f"Here's a thought: {random.choice(motivational_quotes)}")
        feedback_response = " ".join(random.sample(response_parts, k=min(len(response_parts), 3)))

        print("Chatbot (Feedback):", feedback_response)
        # Generate TTS using the provided function
        if save_response_to_wav_func:
            _, feedback_audio_url = save_response_to_wav_func(feedback_response, f"feedback_{response_counter}")

    return feedback_response, feedback_audio_url