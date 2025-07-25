# -*- coding: utf-8 -*-
import json
import constants
from pathlib import Path
from llama_cpp import Llama

# Paths – adjust if needed
INPUT_LEX_PATH  = Path(constants.get_data_path( constants.DS_TONAL_ANALYSIS ) ) 
OUTPUT_CHAT_LEX = Path(constants.get_data_path( constants.DS_TONAL_LEXICON ) )
MODEL_PATH      = "models/sarcasm_categorizer.gguf"

# 1) Load your existing tonal lexicon
with open(INPUT_LEX_PATH, "r", encoding="utf-8") as f:
    lex = json.load(f)["tones"]

# 2) Spin up Llama (we only need its tokenizer)
llm = Llama(model_path=MODEL_PATH, n_ctx=1, verbose=False)

def tokenize_text(text):
    """Return a list of token IDs for any input string."""
    res = llm.tokenize(text)
    return res["tokens"]

# 3) Build chat-ready lexicon
chat_lex = {}
for tone, info in lex.items():
    examples    = info["examples"]     # sample responses
    desc        = info["description"]  # human-readable description

    # Tokenize the tone label itself and its description
    tone_tokens = tokenize_text(tone)
    desc_tokens = tokenize_text(desc)

    # We don’t have a single sarcasm score per tone,
    # but you could infer one (e.g. average level from your records).
    # For now set it to a placeholder or a computed value:
    sarcasm_level = info.get("sarcasm_level", None)

    chat_lex[tone] = {
        "tone_tokens": tone_tokens,
        "description_tokens": desc_tokens,
        "sarcasm_level": sarcasm_level,
        "examples": examples
    }

# 4) Write out the chat lexicon
with open(OUTPUT_CHAT_LEX, "w", encoding="utf-8") as out:
    json.dump(chat_lex, out, indent=2, ensure_ascii=False)

print(f"Chat lexicon written: {OUTPUT_CHAT_LEX} ({len(chat_lex)} tones)")