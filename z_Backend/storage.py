# Manages loading and saving data to files (user preferences, chat history, LTM).
# storage.py
import os
import json
import traceback

# Import specific config variables needed
from config import (
    USER_PREFS_FILE,
    CHAT_HISTORY_FILE,
    LTM_ONTOLOGY_FILE,         # The actual file path
    LTM_ONTOLOGY_IDENTITY_IRI  # The IRI declared inside the OWL file
    # LTM_ONTOLOGY_LOAD_URI is no longer needed
)
from utils import log_error

try:
    from langchain_core.messages import HumanMessage, AIMessage
except ImportError:
    log_error("Langchain core components not found. Chat history may not load/save correctly.")
    HumanMessage = None
    AIMessage = None

try:
    import owlready2
    from owlready2 import World, Thing, DataProperty, ObjectProperty, onto_path # <-- Import onto_path
    print("Owlready2 imported successfully.")
except ImportError:
    log_error("Owlready2 library not found. LTM functionality will be disabled. Run 'pip install Owlready2'")
    owlready2 = None

# --- User Preferences ---
# (load_user_prefs and save_user_prefs remain the same)
def load_user_prefs():
    """Loads user preferences from the JSON file defined in config."""
    if os.path.exists(USER_PREFS_FILE):
        try:
            with open(USER_PREFS_FILE, 'r', encoding='utf-8') as f:
                prefs = json.load(f)
                return prefs
        except json.JSONDecodeError:
            print(f"⚠️ Warning: Could not decode {USER_PREFS_FILE}. Starting with defaults.")
            log_error(f"JSONDecodeError loading user prefs from {USER_PREFS_FILE}")
        except Exception as e:
            print(f"⚠️ Warning: Error loading user prefs: {e}. Starting with defaults.")
            log_error(f"Error loading user prefs: {e}", exc_info=True)
    else:
        print(f"User preferences file not found at {USER_PREFS_FILE}. Using defaults.")
    return {"favorite_music_genre": "", "preferred_city": "Pune", "interests": []}

def save_user_prefs(prefs):
    """Saves user preferences to the JSON file defined in config."""
    try:
        os.makedirs(os.path.dirname(USER_PREFS_FILE), exist_ok=True)
        with open(USER_PREFS_FILE, 'w', encoding='utf-8') as f:
            json.dump(prefs, f, indent=4)
        print(f"User preferences saved to {USER_PREFS_FILE}.")
    except Exception as e:
        log_error(f"Error saving user prefs: {e}", exc_info=True)

# --- Chat History ---
# (load_chat_history and save_chat_history remain the same)
def load_chat_history():
    """Loads chat history from the JSON file defined in config."""
    history = []
    if not HumanMessage or not AIMessage:
        print("⚠️ Langchain components missing, cannot load chat history correctly.")
        return history
    if os.path.exists(CHAT_HISTORY_FILE):
        try:
            with open(CHAT_HISTORY_FILE, 'r', encoding='utf-8') as f:
                raw_history = json.load(f)
                for msg_dict in raw_history:
                    if msg_dict.get('type') == 'human':
                        history.append(HumanMessage(content=msg_dict.get('content', '')))
                    elif msg_dict.get('type') == 'ai':
                        history.append(AIMessage(content=msg_dict.get('content', '')))
        except json.JSONDecodeError:
            print(f"⚠️ Warning: Could not decode {CHAT_HISTORY_FILE}. Starting fresh history.")
            log_error(f"JSONDecodeError loading chat history from {CHAT_HISTORY_FILE}")
        except Exception as e:
            print(f"⚠️ Warning: Error loading chat history: {e}. Starting fresh history.")
            log_error(f"Error loading chat history: {e}", exc_info=True)
    else:
         print(f"Chat history file not found at {CHAT_HISTORY_FILE}. Starting fresh history.")
    return history

def save_chat_history(messages):
    """Saves chat history (list of Langchain message objects) to the JSON file."""
    if not HumanMessage or not AIMessage:
        print("⚠️ Langchain components missing, cannot save chat history correctly.")
        return
    try:
        history_to_save = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                history_to_save.append({"type": "human", "content": msg.content})
            elif isinstance(msg, AIMessage):
                history_to_save.append({"type": "ai", "content": msg.content})
        os.makedirs(os.path.dirname(CHAT_HISTORY_FILE), exist_ok=True)
        with open(CHAT_HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(history_to_save, f, indent=4)
    except Exception as e:
        log_error(f"Error saving chat history: {e}", exc_info=True)


# --- Long-Term Memory (LTM) Ontology ---
def load_ltm_ontology():
    """
    Loads the LTM ontology by adding its directory to owlready2's search path
    and then loading via its identity IRI.
    Returns the loaded ontology object and the owlready2 world object.
    Returns (None, None) if owlready2 is not available or loading fails.
    """
    if owlready2 is None:
        print("⚠️ Owlready2 not available. Cannot load LTM Ontology.")
        return None, None

    # Check if the ontology file actually exists before trying to load
    if not os.path.exists(LTM_ONTOLOGY_FILE):
        print(f"❌ ERROR: Ontology file not found at '{LTM_ONTOLOGY_FILE}'. Cannot load LTM.")
        log_error(f"LTM Ontology file not found at {LTM_ONTOLOGY_FILE}")
        return None, None # Cannot proceed without the file

    world = World() # Create a new world for this session
    onto = None
    try:
        # **UPDATED:** Add the directory containing the ontology file to owlready2's search path
        ontology_dir = os.path.dirname(LTM_ONTOLOGY_FILE)
        if ontology_dir not in onto_path:
            print(f"Adding ontology directory to owlready2.onto_path: {ontology_dir}")
            onto_path.append(ontology_dir)
        else:
             print(f"Ontology directory already in owlready2.onto_path: {ontology_dir}")

        # **UPDATED:** Load using the Identity IRI. owlready2 will search onto_path.
        print(f"Attempting to load ontology using identity IRI: {LTM_ONTOLOGY_IDENTITY_IRI}")
        onto = world.get_ontology(LTM_ONTOLOGY_IDENTITY_IRI).load()

        if onto is None:
             # This could happen if the file is empty or corrupted, or identity IRI mismatch
             raise ValueError(f"Ontology loaded as None using identity IRI {LTM_ONTOLOGY_IDENTITY_IRI}. Check file content and internal IRI declaration.")

        print(f"✅ Long-Term Memory Ontology '{onto.name}' (Base IRI: {onto.base_iri}) loaded successfully using identity IRI and onto_path.")
        return onto, world

    # Catch potential errors during loading (e.g., file parsing errors)
    except Exception as e:
        print(f"❌ ERROR loading LTM Ontology using Identity IRI {LTM_ONTOLOGY_IDENTITY_IRI} (searched in {onto_path}): {e}")
        log_error(f"Error loading LTM Ontology: {e}", exc_info=True)
        traceback.print_exc()

    # Fallback if any error occurred during loading
    print("⚠️ Creating new in-memory LTM ontology for this session as fallback (loading failed).")
    # Ensure world exists even if loading failed mid-way
    if 'world' not in locals():
        world = World()
    onto = world.get_ontology(LTM_ONTOLOGY_IDENTITY_IRI + "_load_error_fallback")
    return onto, world


def save_ltm_ontology(world, onto=None):
    """
    Saves the current state of the specified LTM ontology object within the world
    to the file path defined in config.
    """
    if owlready2 is None:
        print("⚠️ Owlready2 not available. Cannot save LTM Ontology.")
        return
    if world is None:
        print("⚠️ World object is None. Cannot save LTM Ontology.")
        return

    if onto is None:
        onto = world.get_ontology(LTM_ONTOLOGY_IDENTITY_IRI)
        if not onto:
            fallback_iris_to_check = [
                LTM_ONTOLOGY_IDENTITY_IRI + "_load_error_fallback",
                LTM_ONTOLOGY_IDENTITY_IRI + "_missing_file_fallback",
                LTM_ONTOLOGY_IDENTITY_IRI + "_runtime_fallback"
            ]
            for iri in fallback_iris_to_check:
                onto = world.get_ontology(iri)
                if onto:
                    print(f"⚠️ Saving ontology found with fallback IRI: {iri}")
                    break
        if not onto:
            log_error(f"Could not find ontology object with IRI {LTM_ONTOLOGY_IDENTITY_IRI} or known fallback IRIs in the world to save.")
            active_iris = [o.base_iri for o in world.ontologies]
            log_error(f"Ontologies currently in world: {active_iris}")
            return

    if onto:
        try:
            os.makedirs(os.path.dirname(LTM_ONTOLOGY_FILE), exist_ok=True)
            print(f"Attempting to save ontology '{onto.name}' (IRI: {onto.base_iri}) to {LTM_ONTOLOGY_FILE}")
            onto.save(file=LTM_ONTOLOGY_FILE, format="rdfxml")
            print(f"✅ Long-Term Memory Ontology saved.")
        except Exception as e:
            log_error(f"Error saving LTM Ontology to {LTM_ONTOLOGY_FILE}: {e}", exc_info=True)
            traceback.print_exc()
    else:
         log_error("No valid ontology object provided or found to save.")