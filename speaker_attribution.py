import pandas as pd
import re
import string
import logging
import json

from collections import defaultdict
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification

logging.basicConfig(filename="./logs/attribution.debug.log", filemode="w", encoding="utf-8",level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

# Load a pre-trained NLP model for Named Entity Recognition (NER)
model_name = "dbmdz/bert-large-cased-finetuned-conll03-english"
# This will download and cache the model/tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name,cache_dir="./hf_cache")
model = AutoModelForTokenClassification.from_pretrained(model_name)

ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer)

# Track speaker frequency
speaker_memory = defaultdict(int)
last_speaker = None  # Store the last identified speaker
previous_speaker = None  # Store the speaker before last
narrator = Constants.NARRATOR  # Global default narrator
flashback_mode = False  # Track the flashback state
question_followed = False  # Track if a question was followed by another question

# Define a custom pronoun-to-speaker mapping
pronoun_map = {
    "i": narrator,
    "me": narrator,
    "my": narrator,
    "he": None,  # Will be resolved dynamically
    "she": None,  # Will be resolved dynamically
    "they": None,  # Will be resolved dynamically
}

def detect_flashback(text):
    """Detect flashbacks using temporal cues."""
    return bool(re.search(r"\b(years ago|back then|he remembered|she recalled)\b", text, re.IGNORECASE))

def check_pronoun(word):
    temp = word.lower().strip()
    if temp in pronoun_map:
        return pronoun_map[temp]
    else:
        return word


def update_speaker_history(speaker):
    """Update last and previous speaker only if conditions are met."""
    global last_speaker, previous_speaker

    if speaker and speaker != last_speaker:
        previous_speaker = last_speaker
        last_speaker = speaker

def extract_dialogue(text):
    """Extract spoken sentences from a chapter using regex."""
    return re.findall(r'\"(.*?)\"', text)

def is_narration(text):
    """Determine if a line is narration (unquoted)."""
    return not re.search(r"\".*?\"", text)

def clean_action_text(text):
    """Trim leading/trailing whitespace and remove excess punctuation."""
    text = text.strip()  # Remove surrounding whitespace
    text = text.lstrip(string.punctuation).rstrip(string.punctuation)  # Strip leading & trailing punctuation
    text = text.strip()  # Remove surrounding whitespace
    return text

def clean_string(s):
    return re.sub(r'"|<|>|\[|]|/b|/p', '', s)

def split_line(text):
    """Merge multiple quoted texts while ensuring correct grammar separation."""
    match_double = re.match(r'^(\".*?\")\s*(.*?)(\".*?\")\s*(.*)$', text)  # Detect two quotes with an action beat
    if match_double:
        first_quote = match_double.group(1).strip() if match_double.group(1) else ""
        action_between_quotes = match_double.group(2).strip() if match_double.group(2) else ""
        second_quote = match_double.group(3).strip() if match_double.group(3) else ""
        trailing_action = match_double.group(4).strip() if match_double.group(4) else ""

        # Merge quoted texts using appropriate grammar
        if action_between_quotes.endswith(","):  # If the action uses a comma, use `, `
            merged_quotes = f"{first_quote}, {second_quote}"
        else:  # Otherwise, use `. `
            merged_quotes = f"{first_quote}. {second_quote}"

        merged_action = clean_action_text(f"{action_between_quotes} {trailing_action}")

        return clean_string(merged_quotes.strip('"').strip()), clean_string(merged_action)

    match_single = re.match(r'^(\".*?\")\s*(.*)$', text)  # Detect single quoted text + action beat
    if match_single:
        quoted_text = match_single.group(1).strip('"').strip() if match_single.group(1) else ""
        action_beat = match_single.group(2).strip() if match_single.group(2) else ""
        return clean_string(quoted_text), clean_string(action_beat)

    return '', clean_string(clean_action_text(text.strip()))  # If no quotes found, treat the entire text as narration

def extract_speaker(quoted_text, action_beat):
    # Determine speaker based on dialogue and action beat attribution.
    global flashback_mode, last_speaker, previous_speaker, speaker_memory, narrator, question_followed


    logging.debug(f"QUOTED TEXT: {repr(quoted_text)}")
    logging.debug(f"ACTION BEAT: {repr(action_beat)}")


    # Detect flashback mode
    if detect_flashback(action_beat):
        flashback_mode = True
        return "Flashback Narrator"
    elif re.search(r"\b(now|present day|suddenly)\b", action_beat, re.IGNORECASE):
        flashback_mode = False

    # If the line is narration, return Narrator
    if not quoted_text:
        return narrator

    # Handle "I" action beats explicitly (e.g., "I sigh.", "I hesitate.")
    if re.search(r"^I\s(?:gush|counter|yelp|say|inform|reply|fake|lie|finish|remind|chime in|require|burble|interject|interrupt|report|murmur|mutter|sigh|agree|continue|cry|respond|shout|nod|frown|hesitate|pause|shrug|laugh|smile|ask)", action_beat, re.IGNORECASE):
        return check_pronoun("i")

    # Check explicit speaker attribution in dialogue
    match = re.search(r"(\w+)\s(?:sighs|passes|sniffles|accelerates|admits|spits|fakes|smiles|hmmms|snaps|gasps|blurts|starts|murmurs|informs|yells|mumbles|burbles|gushes|mutters|repeats|adds|said|says|replies|replied|cries|slurs|cry|cried|shouts|respond|comments|asked|asks|continued|continues|remarked|shouted|whispered|slips in|chirps in)", action_beat)
    if match:
        speaker = check_pronoun(match.group(1))
        if speaker is not None:
            speaker_memory[speaker] += 1
            return speaker.strip()
        else:
            rematch = re.search(r"(he|she|they)", match.group(1), re.IGNORECASE)
            if rematch:
                speaker = previous_speaker if previous_speaker != narrator else last_speaker
                speaker = speaker if speaker else "PFY"
                return speaker.strip()

    # Detect structured introductions like: "Hi, I'm Dave, Sales Director for X."
    intro_match = re.search( r'^\s*(Hi|Hello|Hey)[,\s]+(I\'m|I am|My name is|This is)\s+([A-Z][a-zA-Z\-]+)\b(?:,.*)?', quoted_text, re.IGNORECASE )
    if intro_match:
        speaker = intro_match.group(3).strip()
        speaker_memory[speaker] += 1
        return speaker.strip()

    # Detect speaker shift caused by an interjection
    if re.match(r'^\.\.+', quoted_text):
        return previous_speaker  # Assign the speaker who was cut off

    # Detect response-following behavior
    response_trigger = re.search(r"\b(yes|no|right|exactly|true|false|hmm|ah|okay|well|but|anyway|actually)\b", quoted_text, re.IGNORECASE)
    if response_trigger and previous_speaker:
        return previous_speaker  # Maintain response attribution

    # Explicit attribution using name detection in action beats
    match = re.search(r"\b(\w+)\s(the|his|her|their)\s(?:secretary|mate|friend|colleague|assistant)\b", action_beat, re.IGNORECASE)
    if match:
        speaker = match.group(1)
        speaker_memory[speaker] += 1
        return speaker.strip()

    # Detect consecutive questions and set state
    if quoted_text.endswith("?"):
        if question_followed:
            return previous_speaker if previous_speaker else last_speaker  # Maintain correct transition
        question_followed = True
    else:
        question_followed = False  # Reset state for non-question responses
        return previous_speaker


    # Process action beat attribution for third-person pronouns
    pronoun_match = re.search(r"\b(he|she|they)\s(?:burbles|gasps|murmured|blurts|sighed|cries|cried|shouted|paused|continued|remarked|slipped in|added|interrupted|muttered|chimed in)\b", action_beat, re.IGNORECASE)
    if pronoun_match:
        return previous_speaker if previous_speaker else last_speaker

    # Detect action-based speaker attribution
    action_match = re.search(r'\[.*?(\b[A-Z][a-z]+.*?)\b', action_beat.strip())
    if action_match:
        possible_speaker = action_match.group(1).strip()

        # If this speaker isn't PFY/BOFH, update attribution
        if possible_speaker.lower() != narrator.lower():
            speaker_memory[possible_speaker] += 1
            return possible_speaker

    return max(speaker_memory, key=speaker_memory.get, default=narrator)

def starts_with( prologue, line ):
    if line and prologue:
        return line.strip().lower().startswith(prologue.lower())
    else:
        return False

def validate_content(text):
    # Returns empty string if the input contains no letters or digits.
    # Otherwise, returns the original string.
    if not re.search(r'[a-zA-Z0-9]', text):
        return ""
    return text

def process_chapter(chapter_text):
    # Process a book chapter and attribute speakers dynamically while keeping line numbering."
    lines = chapter_text.split("\n")

    for index, line in enumerate(lines):
        quoted_text, action_beat = split_line(line)
        speaker = extract_speaker(quoted_text, action_beat)
        update_speaker_history(speaker)
        # Skip prologue/epilogue
        if not starts_with("Episode", action_beat):
            conversation_history.append(f"index {index}: speaker {speaker}: quote {validate_content(quoted_text)}: action {validate_content(action_beat)}")

    return conversation_history

def conversation_to_dataframe(history):
    # Convert conversation history into a structured DataFrame.
    data = []

    for line in history:
        # Extract line number, speaker, quoted text, and action beat
        match = re.match(r"index (\d+): speaker([^:]+): quote (.*): action (.*)", line)
        if match:
            line_number, speaker, quoted_text, action_text = match.groups()

            # Append structured data
            data.append({
                "Line Number": int(line_number),
                "Speaker": speaker,
                "Spoken": quoted_text.strip(),
                "Action": action_text.strip(),
                "Full" : quoted_text.strip() + " [" + action_text.strip() + "]",
                "Original": line.strip(),
            })

    # Convert to DataFrame
    df = pd.DataFrame(data)
    return df


#---------------------------------------------------------------------------------------------------
articles = Constants.get_data_path(Constants.DS_ARTICLES)
records = [] # Initialize raw records
conversation_history = [] # Initialize conversation history

with open(articles, "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        if "Content" in obj:
            raw = obj["Content"].replace("\\n", "\n")
            obj["Content"] = Constants.normalize_eol(raw)

        records.append(obj)

frame = pd.DataFrame( records )
print(f"BOFH articles loaded from {articles}")

# extract content, iterate on /n
total_articles = len(frame)
for counter, row in frame.iterrows():
    last_speaker = None  # reset last identified speaker
    previous_speaker = None  # reset the speaker before last
    content = row["Content"]
    # Process the chapter
    conversation = process_chapter(content)
    Constants.print_progress(counter, total_articles)

# Convert conversation history into DataFrame
conversation_df = conversation_to_dataframe(conversation_history)

# Clean up: drop empty rows, reset index
cleaned_df = (
    conversation_df
    .replace('', pd.NA)
    .dropna(subset=['Spoken', 'Action'], how='all')
    .reset_index(drop=True)
)

cleaned_df['Spoken'] = cleaned_df['Spoken'].apply(Constants.fix_mojibake_safe)
cleaned_df['Action'] = cleaned_df['Action'].apply(Constants.fix_mojibake_safe)

# Optionally, reset the index
cleaned_df.reset_index(drop=True, inplace=True)

ATTRIBUTED = Constants.get_data_path(Constants.DS_ATTRIBUTED)
cleaned_df.to_json(ATTRIBUTED, orient="records", lines=True, force_ascii=False)

print(f"BOFH speaker attributed saved to {ATTRIBUTED}")

