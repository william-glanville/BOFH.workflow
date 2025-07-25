import json
import constants
import logging
from jinja2 import Environment, FileSystemLoader
from collections import defaultdict
from llama_cpp import Llama
from constants import print_progress

logging.basicConfig(filename=constants.get_log_path("tonal_analysis.debug.log"), filemode="w", encoding="utf-8",level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

MAX_TOKENS = 512

CONSOLIDATED_PATH = constants.get_data_path(constants.DS_CONSOLIDATED)
LEXICON_OUTPUT_PATH = constants.get_data_path(constants.DS_TONAL_LEXICON)
RAW_CACHE_PATH = constants.get_data_path(constants.DS_TONAL_ANALYSIS)

MODEL_PATH = constants.get_model_path(constants.MODEL_SARCASM_CATEGORIZER_LLAMA)

# Load GGUF model
llm = Llama(model_path=MODEL_PATH, n_ctx=2048, n_threads=8, n_gpu_layers=32, verbose=False)

# Prompt construction
PROMPT_INSTRUCTION = constants.load_text_file( constants.get_prompt_path( constants.PROMPT_TONE_ANALYSIS ) )

env = Environment(
    loader=FileSystemLoader(constants.get_prompt_path("")),
    keep_trailing_newline=True,
    autoescape=False
)

template = env.get_template("falcon_template.j2")  # the file you posted

def make_prompt(instruction, context, response):
    messages = [
        {"role": "system", "content": f"{instruction}"},
        {"role": "user", "content": f'Context: "{context}"\nResponse: "{response}"'}
    ]
    prompt = template.render(
        messages=messages,
        custom_tools=None,
        tools=None,
        tools_in_user_message=False,
        date_string="07 Jul 2025",
        add_generation_prompt=True
    )
    return prompt

def parse_keyvalue_output(text):
    keywords, descriptions, sarcasm = [], {}, None
    for line in text.strip().splitlines():
        # split on the first '=' then on the first ';'
        if '=' not in line:
            continue
        left, rest = line.split("=",1)
        if left == "sarcasm_level":
            sarcasm = int(rest)
        else:
            # left == "keyword"
            key, desc = rest.split(";description=",1)
            keywords.append(key)
            descriptions[key] = desc
    return {
        "Tone_Keywords":     keywords,
        "Tone_Descriptions": descriptions,
        "Sarcasm_Level":     sarcasm
    }

def main():
    # 1) Count total records for progress tracking
    with open(CONSOLIDATED_PATH, "r", encoding="utf-8") as src_file:
        total = sum(1 for _ in src_file)

    lexicon = defaultdict(list)
    descriptions = {}

    # 2) Process each line and report progress
    with open(CONSOLIDATED_PATH, "r", encoding="utf-8") as src, \
         open(RAW_CACHE_PATH, "w", encoding="utf-8") as out:
        for idx, line in enumerate(src, start=1):
            # call your progress printer
            print_progress(idx, total)

            record = json.loads(line)
            context  = record.get("context", "").strip()
            response = record.get("response", "").strip()
            if not response:
                continue

            prompt = make_prompt(PROMPT_INSTRUCTION, context, response)
            logging.info("PROMPT")
            logging.info(prompt)
            try:
                result = llm(
                    prompt,
                    max_tokens=MAX_TOKENS,
                    temperature=0.8,
                    top_k=40,
                    top_p=0.95,
                    repeat_penalty=1.0,
                    stop=["<|eot_id|>"],    # stop on the Jinja‐injected end-of-assistant tag
                )

                logging.info("RESULT")
                logging.info(result)
                output = result["choices"][0]["text"].strip()
            except Exception as e:
                output = f"[ERROR] {e}"
                print(f"Error on input: {context[:30]} | {response[:30]} >> {e}")

            parsed = parse_keyvalue_output(output)
            record.update(parsed)
            record["Raw_Tone_Description"] = output
            out.write(json.dumps(record, ensure_ascii=False) + "\n")

            for tone in parsed["Tone_Keywords"]:
                lexicon[tone].append(response)
                if tone not in descriptions:
                    descriptions[tone] = parsed["Tone_Descriptions"].get(tone, "")

    # === BUILD FINAL LEXICON ===
    with open(LEXICON_OUTPUT_PATH, "w", encoding="utf-8") as lex_out:
        json.dump({
            "tones": {
                tone: {
                    "examples": lexicon[tone][:5],
                    "description": descriptions[tone]
                } for tone in sorted(lexicon)
            }
        }, lex_out, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()
