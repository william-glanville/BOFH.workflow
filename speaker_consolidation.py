import pandas as pd
import re
import json
import logging
import common.constants as constants
from tqdm import tqdm

ATTRIBUTED = constants.get_data_path(constants.DS_ATTRIBUTED)
CONSOLIDATED = constants.get_data_path(constants.DS_CONSOLIDATED)
INCLUDE_ACTION_BEAT = True
CHAIN_PREVIOUS_RESPONSE = True

# Utility: Normalize whitespace and punctuation
def clean_text(text):
    return re.sub(r'\s+', ' ', str(text).strip())

def consolidate():
    last_bofh_response = None
    context_buffer = []
    results = []

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(constants.get_log_path("consolidation.debug.log")),
            logging.StreamHandler()
        ]
    )

    # Load speaker-attributed data
    with open(ATTRIBUTED, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    df = pd.DataFrame(records)
    logging.info(f"BOFH Speaker Attributions loaded from {ATTRIBUTED} with {len(df)} entries.")

    # Preprocess fields
    df[['Speaker', 'Spoken', 'Action']] = df[['Speaker', 'Spoken', 'Action']].fillna('')
    df['Speaker'] = df['Speaker'].apply(clean_text)
    df['Line'] = df['Spoken'].apply(clean_text)
    df['Beat'] = df['Action'].apply(clean_text)

    for i in tqdm(range(len(df))):
        speaker = df.loc[i, 'Speaker']
        line = df.loc[i, 'Line']
        beat = df.loc[i, 'Beat']

        if not line and beat:
            beat_only_buffer = f"*({beat})*"
            continue

        line_with_beat = f"{line} *({beat})*" if INCLUDE_ACTION_BEAT and beat.strip() else line

        if 'beat_only_buffer' in locals():
            line_with_beat = f"{beat_only_buffer} {line_with_beat}"
            del beat_only_buffer

        if speaker.lower() != constants.PRINCIPLE.lower():
            context_buffer.append(line_with_beat)
        else:
            if not context_buffer:
                continue

            context = " ".join(context_buffer)
            if CHAIN_PREVIOUS_RESPONSE and last_bofh_response:
                context = f"{last_bofh_response} {context}"

            results.append({
                "context": context,
                "response": line_with_beat,
                "source_line": i
            })

            last_bofh_response = line_with_beat
            context_buffer = []

    # Save as JSONL
    with open(CONSOLIDATED, "w", encoding="utf-8") as f:
        for r in results:
            if isinstance(r, dict):
                json.dump(r, f, ensure_ascii=False)
                f.write("\n")
            else:
                logging.warning("Skipping non-dict entry: %s", type(r))

    logging.info(f"Wrote {len(results)} context–response pairs to: {CONSOLIDATED}")

def main():
    consolidate()

if __name__ == "__main__":
    main()