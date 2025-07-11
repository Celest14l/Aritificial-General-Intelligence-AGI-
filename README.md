# 🤖 AGI Project: Hybrid AI Conversational Assistant

**Last Updated:** July 11, 2025

Welcome to **AGI Project** – an ambitious initiative to create a conversational AI assistant that fuses **Subsymbolic AI (Llama 3)** with **Symbolic AI (Owlready2 Ontology)** for a powerful, context-aware, and personalized user experience.

---

## 🌟 Overview

### Why This Project?

AGI Project builds a **Hybrid AI** system by integrating:

- **Subsymbolic AI (Llama 3 via Groq)**
  - Natural language understanding and fluent generation.
  - Contextual reasoning and conversation continuity.
- **Symbolic AI (Owlready2 Ontology)**
  - Structured knowledge representation.
  - Reliable fact storage, retrieval, and logical inference.

This synergy ensures your assistant is both **human-like in conversation** and **precise in knowledge handling** – making it ideal for intelligent dialogue, personal task automation, and memory management.

---

## 🚀 Key Features

✅ **Conversational Interaction:** Natural, context-aware dialogue powered by Llama 3  
✅ **Information Retrieval:** Real-time weather, news, and API data integrated with user preferences  
✅ **Task Execution:** Perform actions like music downloads, email sending, and more  
✅ **Memory System:**  
   - **Short-Term Memory (STM):** Recent interaction cache  
   - **Long-Term Memory (LTM):** Persistent OWL ontology-based storage  
   - **Conversational Memory:** Maintains context seamlessly  
✅ **Preference Management:** Stores and retrieves user preferences intelligently  
✅ **Voice Input/Output:** Web Speech API + VITS TTS for natural voice interaction  
✅ **Sentiment Awareness:** Tailors responses based on user sentiment analysis

---

## 🛠️ Technical Stack

### **Backend**

- **Languages/Frameworks:** Python, Flask, Langchain, Langchain-Groq
- **Subsymbolic AI:** Llama 3 (Groq API)
- **Symbolic AI:** Owlready2
- **Key Libraries:** Transformers, PyTorch, VADER, cachetools, yt-dlp, Requests, smtplib
- **Data Storage:** JSON (preferences/history), OWL (knowledge_base.owl)
- **TTS:** VITS (Hugging Face)

### **Frontend**

- **Technologies:** HTML, CSS, JavaScript
- **APIs:** Web Speech API (voice input), Web Audio API (TTS playback)
- **Features:** Chat interface, memory indicators, async communication, graceful error handling

---

## 🔄 Hybrid Architecture

The AGI Project’s **hybrid pipeline** seamlessly combines subsymbolic and symbolic processing:

1. **Subsymbolic Core (Llama 3)**
   - Processes input, generates responses.
   - Identifies intents/facts for symbolic handling.
   - Uses ConversationBufferWindowMemory for continuity.
2. **Symbolic Component (Owlready2)**
   - Stores structured facts in `knowledge_base.owl`.
   - Supports queries like “What is my name?”
   - Future-ready for ontology-based logical reasoning.

### **Interaction Flow**

- **LLM ➡️ Symbolic:** Commands to store/recall facts routed via `memory_manager.py`.
- **Symbolic ➡️ LLM:** Injects key facts back into prompts for personalized responses.
- **External Services:** Enrich interactions with real-time data retrieval.

---

## 🧠 Memory System

| **Type**               | **Technology**                                              | **Role**                                | **Features**                      |
|-------------------------|--------------------------------------------------------------|------------------------------------------|-------------------------------------|
| **Short-Term Memory**  | `cachetools.TTLCache`                                       | Temporary cache for quick recall         | Time-based expiry, limited capacity |
| **Long-Term Memory**   | `Owlready2` (`knowledge_base.owl`)                          | Persistent structured storage            | Reliable retrieval, supports inference |
| **Conversational**     | `ConversationBufferWindowMemory` (Langchain)                | Maintains dialogue context               | Tracks recent conversation turns |

---

## 🖥️ Backend Modules

- **`app.py`** – Main Flask app, routes processing to core logic.  
- **`core_logic.py`** – Intent recognition and central routing.  
- **`config.py`** – API keys and paths.  
- **`storage.py`** – JSON and OWL file I/O management.  
- **`memory_manager.py`** – STM/LTM operations and summaries.  
- **`services.py`** – Integrates weather, news, TTS, and other services.  
- **`utils.py`** – Logging, sentiment analysis, and utilities.  
- **LLM Integration:** Llama 3 via Langchain  
- **Symbolic AI:** Owlready2 ontology management

---

## 🌐 Frontend Components

- **Chat Interface:** Displays conversation history and input.  
- **Voice Input:** Mic button with Web Speech API visualizer.  
- **Audio Output:** TTS playback using Web Audio API.  
- **Memory Indicators:** STM/LTM status visuals.  
- **Async Communication:** Fetch API for seamless backend integration.  
- **Error Handling:** Graceful user/system error management.

---

📦 Setup & Installation
Follow these steps to set up the AGI Project on your local machine:

🔧 1. Prerequisites
✅ Ensure you have Python 3.8+ installed.
✅ Install Git for cloning the repository.
✅ (Optional) Create a virtual environment to manage dependencies:

bash
Copy
Edit
python -m venv venv
source venv/bin/activate    # Linux/macOS
venv\Scripts\activate       # Windows
💻 2. Clone the Repository
bash
Copy
Edit
git clone https://github.com/your-username/agi-project.git
cd agi-project
📚 3. Install Dependencies
Install all required packages listed in requirements.txt:

bash
Copy
Edit
pip install -r requirements.txt
⚙️ 4. Configure Environment
Update config.py:

Add your Groq API key.

Insert keys for weather, news, or other external services as needed.

Initialize Ontology:

Ensure knowledge_base.owl exists in the root directory.

If missing, create it using your ontology schema.

🚀 5. Run the Application
Start the Flask server:

bash
Copy
Edit
python app.py
🌐 6. Access the Web Interface
Open your browser and navigate to:

http://localhost:5000

✅ Setup Complete
You’re now ready to experience your Hybrid AGI Conversational Assistant in action. 🎯
