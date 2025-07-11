AGI Project: Hybrid AI Conversational Assistant
Last Updated: July 11, 2025
Welcome to the AGI Project, an innovative conversational AI assistant that combines the power of Subsymbolic AI (Large Language Model - Llama 3) and Symbolic AI (Ontology-based Knowledge Base with Owlready2) to deliver a robust, context-aware, and personalized user experience. This project aims to create a versatile web-based AI assistant capable of natural language understanding, structured knowledge management, and seamless interaction with external services.

🌟 Project Overview
The AGI Project integrates two complementary AI paradigms to create a Hybrid AI architecture:

Subsymbolic AI (Llama 3 via Groq): Excels in natural language understanding, fluent response generation, pattern recognition, and contextual reasoning. It drives conversational flow and leverages short-term conversational memory for coherence.
Symbolic AI (Owlready2 Ontology): Manages structured, persistent knowledge in a Long-Term Memory (LTM) system, enabling reliable fact storage, retrieval, and logical reasoning.

This hybrid approach ensures the assistant is both conversationally fluent and capable of precise, structured data handling, making it suitable for a wide range of applications, from casual dialogue to task execution and personalized memory management.

🚀 Features

Conversational Interaction: Engage in natural, context-aware dialogue powered by Llama 3.
Information Retrieval: Access real-time data (e.g., weather, news, location) using external APIs, informed by user preferences stored in the LTM.
Task Execution: Perform tasks like music downloads and email sending, guided by procedural logic and LLM insights.
Memory System:
Short-Term Memory (STM): Volatile cache for recent interactions.
Long-Term Memory (LTM): Persistent OWL ontology for structured fact storage and retrieval.
Conversational Memory: Maintains dialogue context for seamless interactions.


Preference Management: Store and apply user preferences, leveraging the symbolic LTM for structure.
Voice Input/Output: Supports voice-based interaction using Web Speech API and Text-to-Speech (TTS) with VITS.
Sentiment Awareness: Analyzes user sentiment to tailor response strategies.


🛠️ Technical Stack
Backend

Languages/Frameworks: Python, Flask, Langchain, Langchain-Groq
AI Components:
Subsymbolic AI: Llama 3 (via Groq API)
Symbolic AI: Owlready2 for OWL ontology management


Libraries:
Machine Learning: Transformers (Hugging Face), PyTorch
Sentiment Analysis: VADER
Caching: cachetools
External Services: yt-dlp, Requests, smtplib
Data Storage: JSON (preferences, history), OWL (knowledge_base.owl)


TTS: VITS (Hugging Face)

Frontend

Technologies: HTML, CSS, JavaScript
APIs: Web Speech API (voice input), Web Audio API (TTS playback)
Features: Chat history, input area, memory indicators, asynchronous communication (fetch API), error handling


🔄 Hybrid Architecture
The AGI Project's hybrid architecture seamlessly integrates subsymbolic and symbolic AI:

Subsymbolic Core (Llama 3):

Processes natural language input and generates fluent responses.
Identifies intents and facts for symbolic processing.
Uses ConversationBufferWindowMemory for dialogue continuity.


Symbolic Component (Owlready2):

Stores facts in a persistent OWL ontology (knowledge_base.owl).
Supports structured queries (e.g., "What is my name?").
Enables future logical reasoning with ontology reasoners.


Interaction Flow:

LLM → Symbolic: Commands to store/recall facts are routed to the ontology via memory_manager.py.
Symbolic → LLM: Key facts from the LTM are injected into the LLM's system prompt for personalized context.
External Services: Supplement core functionality with real-time data (e.g., weather, news).




🧠 Memory System
The AGI employs a multi-layered memory system to support its hybrid architecture:

Short-Term Memory (STM):

Technology: cachetools.TTLCache
Role: Temporary storage for recent inputs, supporting quick recall.
Features: Time-based expiration, limited capacity.


Long-Term Memory (LTM):

Technology: Owlready2 (knowledge_base.owl)
Role: Persistent, structured storage for facts and preferences.
Features: Reliable retrieval, supports logical inference.


Conversational Memory:

Technology: langchain.chains.conversation.memory.ConversationBufferWindowMemory
Role: Maintains dialogue context for the LLM.
Features: Tracks recent conversation for coherent responses.




🖥️ Backend Components
The backend is modular, reflecting the hybrid AI approach:

app.py: Main Flask app, orchestrates web requests, initializes app state, and routes processing to core_logic.py.
core_logic.py: Central hub for intent recognition, routing requests to memory, services, or LLM.
config.py: Centralizes paths, API keys, and settings.
storage.py: Manages file I/O for JSON (preferences, history) and OWL (LTM).
memory_manager.py: Handles STM and LTM operations, including store_ltm_fact, recall_ltm_fact, and get_ltm_summary.
services.py: Integrates external services (weather, news, TTS, etc.).
utils.py: Provides utilities like logging and sentiment analysis.
LLM Integration: Manages Llama 3 interactions via Langchain.
Symbolic AI: Uses Owlready2 for ontology management.


🌐 Frontend Components
The frontend provides a user-friendly interface for interacting with the AGI:

Chat Interface: Displays conversation history and input area.
Voice Input: Mic button with visualizer, powered by Web Speech API.
Audio Output: TTS playback via Web Audio API.
Memory Indicators: Visual cues for STM/LTM status.
Asynchronous Communication: Uses fetch API for seamless backend interaction.
Error Handling: Graceful management of user and system errors.


📦 Setup and Installation

Clone the Repository:
git clone https://github.com/your-username/agi-project.git
cd agi-project


Install Dependencies:
pip install -r requirements.txt


Configure Environment:

Update config.py with API keys (e.g., Groq, weather, news).
Ensure knowledge_base.owl is initialized or created.


Run the Application:
python app.py


Access the Web Interface:

Open http://localhost:5000 in your browser.




🤝 Contributing
We welcome contributions to enhance the AGI Project! To contribute:

Fork the repository.
Create a feature branch (git checkout -b feature/your-feature).
Commit your changes (git commit -m "Add your feature").
Push to the branch (git push origin feature/your-feature).
Open a Pull Request.

Please ensure your code adheres to the project's coding standards and includes relevant tests.

📜 License
This project is licensed under the MIT License. See the LICENSE file for details.

📬 Contact
For questions or feedback, reach out via GitHub Issues or contact the project maintainers.

Built with ❤️ by the AGI Project Team
