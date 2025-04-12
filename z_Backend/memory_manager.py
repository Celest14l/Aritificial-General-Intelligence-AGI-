# Encapsulates the logic for interacting with Short-Term Memory (STM cache) and Long-Term Memory (LTM dict/Ontology).
# memory_manager.py
import json
import time
import storage # Needs access to save_ltm
from utils import log_error

# --- Owlready2 Check (Similar to storage.py) ---
try:
    # Import necessary components from owlready2
    import owlready2
    from owlready2 import World, Thing, DataProperty, ObjectProperty, onto_path, sync_reasoner_pellet
    owlready2_available = True
    print("Owlready2 components imported successfully in memory_manager.")
except ImportError:
    log_error("Owlready2 library not found in memory_manager. LTM functionality will be disabled.")
    owlready2_available = False
    # Define dummy classes/functions if owlready2 is not available, so the rest of the code doesn't break
    class Thing: pass
    class DataProperty: pass
    class ObjectProperty: pass


# --- Short-Term Memory (STM) ---

def add_to_stm(stm_cache, last_stm_key_tracker, user_input):
    """Adds user input to the STM cache and updates the last key tracker."""
    current_timestamp = time.time()
    # Ensure cache object exists (defensive check)
    if stm_cache is None:
        log_error("STM cache object is None in add_to_stm.")
        return
    stm_cache[current_timestamp] = user_input
    if last_stm_key_tracker is None:
        log_error("last_stm_key_tracker object is None in add_to_stm.")
        return
    last_stm_key_tracker['key'] = current_timestamp # Use a mutable dict to track
    print(f"Stored in STM (Key: {current_timestamp}): '{user_input}'")

def forget_last_stm(stm_cache, last_stm_key_tracker):
    """Forgets the last explicitly tracked item from STM."""
    response_text = ""
    # Ensure objects exist
    if stm_cache is None: return "STM cache not initialized."
    if last_stm_key_tracker is None: return "STM tracker not initialized."

    last_key = last_stm_key_tracker.get('key')

    if last_key and last_key in stm_cache:
        try:
            forgotten_item = stm_cache.pop(last_key)
            response_text = f"Okay, I've forgotten that I stored: \"{forgotten_item[:50]}...\""
            last_stm_key_tracker['key'] = None # Reset tracker

            # Find the next most recent key if needed
            if stm_cache:
                try:
                    # Find the max key among remaining items
                    last_stm_key_tracker['key'] = max(stm_cache.keys())
                except ValueError:
                     last_stm_key_tracker['key'] = None # Cache became empty
        except KeyError:
             # Handle rare case where key disappears between check and pop
             response_text = "Looks like that memory expired just now."
             last_stm_key_tracker['key'] = None
    else:
        response_text = "There's nothing specific for me to forget right now, or the last item has expired."

    return response_text

def recall_stm_time(stm_cache, value, unit):
    """Recalls an item from STM based on a time delta."""
    # Ensure cache exists
    if stm_cache is None: return "STM cache not initialized."

    delta_seconds = value * 60 if unit == "minute" else value * 3600
    found_item = None
    found_time_ago = ""
    response_text = ""

    try:
        # Iterate through sorted keys (newest first)
        # Make a copy of keys for safe iteration if cache might change
        sorted_keys = sorted(list(stm_cache.keys()), reverse=True)
        now = time.time()
        for key_ts in sorted_keys:
             # Check if key still exists before accessing (it might expire during loop)
             if key_ts not in stm_cache: continue

             time_diff = now - key_ts
             # Allow +/- 20% margin, maybe slightly more for practical use
             # Ensure delta_seconds is not zero to avoid division issues
             if delta_seconds > 0 and time_diff >= (delta_seconds * 0.75) and time_diff <= (delta_seconds * 1.25):
                found_item = stm_cache.get(key_ts) # Use get for safety
                if found_item: # Check if item still exists after get
                    minutes_ago = round(time_diff / 60)
                    found_time_ago = f"About {minutes_ago} minute{'s' if minutes_ago != 1 else ''} ago"
                    break # Found the first match in the timeframe
    except Exception as e:
        log_error(f"Error during timed STM recall: {e}", exc_info=True)
        return "An error occurred while trying to recall from memory."


    if found_item:
         response_text = f"{found_time_ago}, you said: \"{found_item}\""
    else:
         response_text = f"Sorry, I don't have a specific memory from around {value} {unit}s ago."
    return response_text

def recall_stm_general(stm_cache, current_input_timestamp):
    """Recalls the most recent item in STM *before* the current input."""
     # Ensure cache exists
    if stm_cache is None: return "STM cache not initialized."

    found_item = None
    found_time_ago = ""
    response_text = ""

    try:
        # Find the most recent item *excluding* the current input's timestamp
        # Make a copy of keys for safe iteration/filtering
        previous_keys = sorted([k for k in list(stm_cache.keys()) if k < current_input_timestamp], reverse=True)

        if previous_keys:
            previous_key = previous_keys[0] # The most recent *previous* key
            found_item = stm_cache.get(previous_key) # Use get for safety
            if found_item: # Check if item still exists
                time_diff = time.time() - previous_key
                minutes_ago = max(1, round(time_diff / 60)) # Show at least 1 minute
                found_time_ago = f"A moment ago ({minutes_ago} minute{'s' if minutes_ago != 1 else ''})"
            else:
                 # Item expired between getting keys and accessing item
                 found_item = None # Ensure it's None if access failed
    except Exception as e:
        log_error(f"Error during general STM recall: {e}", exc_info=True)
        return "An error occurred while trying to recall the last thing you said."


    if found_item:
         response_text = f"{found_time_ago}, you said: \"{found_item}\""
    else:
         response_text = "I don't have a specific recent memory to recall (besides what you just said)."
    return response_text


# --- Long-Term Memory (LTM) - Ontology Interaction ---

# Helper to safely define necessary OWL constructs if they don't exist
def ensure_owl_constructs(onto):
    """Ensures base classes and properties exist in the ontology."""
    if not owlready2_available or onto is None:
        log_error("Cannot ensure OWL constructs: Owlready2 not available or ontology is None.")
        return False

    # It's safer to define within the ontology's context
    with onto:
        # Define UserFact class if it doesn't exist
        if getattr(onto, "UserFact", None) is None:
             print(f"Ontology '{onto.name}': Defining class 'UserFact'")
             class UserFact(Thing): pass
             # Add labels or comments if desired
             # UserFact.label = ["User Fact"]

        # Define hasValue property if it doesn't exist
        # *** IMPORTANT: Ensure this property name ('hasValue') matches your OWL file ***
        # *** OR matches what you intend to use consistently.                 ***
        if getattr(onto, "hasValue", None) is None:
             print(f"Ontology '{onto.name}': Defining DataProperty 'hasValue'")
             class hasValue(DataProperty):
                 # Define domain/range if needed, though often optional for basic use
                 # domain = [onto.UserFact] # Apply to UserFact individuals
                 range = [str]         # Assume string values initially
             # Add labels if desired
             # hasValue.label = ["has value"]

        # Optional: Define hasFactKey property if you want to store the original key
        # if getattr(onto, "hasFactKey", None) is None:
        #     print(f"Ontology '{onto.name}': Defining DataProperty 'hasFactKey'")
        #     class hasFactKey(DataProperty):
        #         domain = [onto.UserFact]
        #         range = [str]
        #     hasFactKey.label = ["has fact key"]

    return True # Indicate constructs should now exist


def store_ltm_fact(onto, world, key, value):
    """Stores a fact in the LTM ontology and saves it."""
    # Renamed `world` parameter as it's needed for saving via storage.py
    if not owlready2_available or onto is None or world is None:
        log_error(f"LTM store failed: Owlready2 available={owlready2_available}, onto is None={onto is None}, world is None={world is None}")
        return "LTM (Ontology) is not available or not properly initialized."

    response_text = ""
    # Sanitize key for use as individual name (basic cleaning)
    # Consider more robust cleaning if keys can be complex
    individual_name = key.lower().replace(" ", "_").replace("'", "").replace("?", "").replace(".", "").replace(",", "")
    # Prevent overly long or empty names
    individual_name = individual_name[:50] # Limit length
    if not individual_name:
        return "Invalid key provided for LTM (results in empty name)."

    clean_value = value.strip()
    if not clean_value:
         return "Cannot store an empty value in LTM."

    try:
        # Ensure the class and property exist (uses the helper)
        if not ensure_owl_constructs(onto):
             return "Failed to ensure necessary ontology structure."

        # --- Create or access the individual ---
        # Option 1: Using the class constructor (cleaner)
        individual = onto.UserFact(individual_name)
        print(f"Created/Accessed LTM individual: {individual.iri}")

        # Option 2: Searching first (if you want to update existing instead of overwriting)
        # existing_individual = onto.search_one(iri=f"*{individual_name}", _is_a=onto.UserFact)
        # if existing_individual:
        #     individual = existing_individual
        #     print(f"Found existing LTM individual: {individual.iri}")
        # else:
        #     individual = onto.UserFact(individual_name)
        #     print(f"Created new LTM individual: {individual.iri}")


        # --- Assign the value using the defined property ---
        # Ensure property exists on the ontology object before assigning
        value_property = getattr(onto, "hasValue", None)
        if value_property is None:
             # This shouldn't happen if ensure_owl_constructs worked
             log_error("Ontology property 'hasValue' not found after ensuring constructs.")
             return "Internal error: Ontology property 'hasValue' missing."

        # Owlready2 properties often expect lists, even for single values
        setattr(individual, "hasValue", [clean_value])
        print(f"Set '{individual_name}.hasValue' = '{clean_value}'")

        # --- Optional: Store original key ---
        # key_property = getattr(onto, "hasFactKey", None)
        # if key_property:
        #     setattr(individual, "hasFactKey", [key])
        #     print(f"Set '{individual_name}.hasFactKey' = '{key}'")


        # --- Save the changes ---
        # Pass the world object to the saving function in storage.py
        storage.save_ltm_ontology(world, onto) # Pass both world and specific onto object
        response_text = f"Okay, I've remembered that '{key}' is '{clean_value}' in the knowledge base."

    except Exception as e:
        log_error(f"Error storing LTM fact (key='{key}', individual='{individual_name}'): {e}", exc_info=True)
        response_text = "Sorry, I encountered an error trying to remember that."

    return response_text

def recall_ltm_fact(onto, key):
    """Recalls a fact from the LTM ontology."""
    if not owlready2_available or onto is None:
        return "LTM (Ontology) is not available."

    response_text = ""
    # Use the same cleaning logic as in store_ltm_fact
    individual_name = key.lower().replace(" ", "_").replace("'", "").replace("?", "").replace(".", "").replace(",", "")
    individual_name = individual_name[:50] # Limit length
    if not individual_name:
        return "Invalid key provided for LTM recall."

    try:
        # Ensure the property we want to query exists (class assumed needed too)
        if not ensure_owl_constructs(onto):
              return "Failed to ensure necessary ontology structure for recall."

        value_property_name = "hasValue" # Must match the property used for storing

        # --- Find the individual by name ---
        # Search within the specific ontology's namespace for the individual
        # Make sure UserFact class exists before using it in search
        user_fact_class = getattr(onto, "UserFact", None)
        if user_fact_class is None:
             return "Internal Error: UserFact class not defined in ontology for searching."

        # Search for an individual of the correct type and name
        individual = onto.search_one(iri=f"*{individual_name}", is_a=user_fact_class)

        if individual:
            print(f"Found LTM individual for recall: {individual.iri}")
            # Check if the individual has the specific property we're looking for
            if hasattr(individual, value_property_name):
                value_list = getattr(individual, value_property_name)
                if value_list: # Check if the list is not empty
                    # Retrieve the value (taking the first item)
                    value = value_list[0]
                    response_text = f"Based on the knowledge base, '{key}' is '{value}'."
                else:
                    # Property exists but has no value assigned
                    response_text = f"I have an entry for '{key}', but no specific value is stored for it."
            else:
                # Individual exists but doesn't have the 'hasValue' property
                 response_text = f"I know about '{key}', but I don't have that specific detail ('{value_property_name}') stored."
        else:
            print(f"Individual '{individual_name}' not found in LTM.")
            response_text = f"Sorry, I don't have specific information stored for '{key}' in the knowledge base."

    except Exception as e:
        log_error(f"Error recalling LTM fact (key='{key}', individual='{individual_name}'): {e}", exc_info=True)
        response_text = "Sorry, I encountered an error trying to recall that information."

    return response_text


# --- LTM Summary for Prompt ---
def get_ltm_summary(onto):
    """Generates a simple JSON string summary of LTM facts for the LLM prompt."""
    # Return empty JSON string if unavailable or None
    if not owlready2_available or onto is None:
        log_error("LTM summary requested but ontology not available.")
        return "{}"

    summary_facts = {}
    try:
        # Ensure the UserFact class and hasValue property exist
        if not ensure_owl_constructs(onto):
             log_error("Failed to ensure ontology constructs for LTM summary.")
             return json.dumps({"error": "Ontology structure missing"})

        user_fact_class = getattr(onto, "UserFact", None)
        value_property_name = "hasValue" # Must match property used in store/recall

        if user_fact_class is None:
             log_error("UserFact class not found for LTM summary.")
             return json.dumps({"status": "UserFact class not defined"})

        print(f"Generating LTM summary from instances of {user_fact_class} using property '{value_property_name}'...")
        instances_processed = 0
        # Iterate through all individuals of type UserFact
        for individual in user_fact_class.instances():
            instances_processed += 1
            # Check if the individual has the value property
            if hasattr(individual, value_property_name):
                value_list = getattr(individual, value_property_name)
                if value_list: # Ensure there's a value
                    # --- Reconstruct Key (Potential Fragility) ---
                    # This assumes the individual.name is the cleaned key.
                    # Using a separate hasFactKey property would be better.
                    key_name = individual.name.replace("_", " ")
                    # --- Get Value ---
                    value = value_list[0]
                    summary_facts[key_name] = value

        print(f"Processed {instances_processed} UserFact instances for LTM summary.")
        if not summary_facts and instances_processed > 0:
             log_error(f"Processed {instances_processed} UserFacts but extracted no key/value pairs for summary.")


    except Exception as e:
        log_error(f"Error generating LTM summary for prompt: {e}", exc_info=True)
        return json.dumps({'error': 'Could not retrieve LTM summary'})

    # Return summary as a JSON string (empty JSON string if no facts)
    return json.dumps(summary_facts)