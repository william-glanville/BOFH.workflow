import time
import subprocess
import threading
import signal
import os
import re
import codecs
import ftfy
import json

from pathlib import Path

ANACONDA = "C:/storage/tools/anaconda3/Scripts/conda.exe"
CONDA_PREFIX = "C:/storage/tools/anaconda3"

ENV_LLAMA = "cuda-llama-env"
ENV_SARCASM = "sarcasm-env"
ENV_LLAMA_TRAIN = "llama-train"

ROOT = Path(__file__).resolve().parent

DS_ARTICLES = "bofh_articles.jsonl"
DS_ATTRIBUTED = "attributed.jsonl"
DS_CONSOLIDATED = "consolidated.jsonl"
DS_TONAL_LEXICON = "tonal_lexicon.jsonl"
DS_TONAL_ANALYSIS = "tonal_analysis.jsonl"
DS_CONSOLIDATED_OUTLIER = "consolidated_outliers.jsonl"
DS_CONSOLIDATED_STANDARD = "consolidated.standard.jsonl"
DS_CLASSIFIED = "classified.jsonl"
DS_CLASSIFIED_STANDARD = "classified.standard.jsonl"
DS_TOKENIZED = "tokenized.jsonl"
DS_TOKENIZATION_REPORT = "tokenization_report.md"
DS_TOKENIZATION_REPORT_TEMPLATE = "tokenization_validation_report.template"

DS_MARKDOWN_THEME = "report_theme.css"
DS_CONSOLE_THEME = "console.css"
DS_CONSOLE_TEMPLATE = "console.html"

TONAL_TOKENS = ["<s_tone>", "</s_tone>", "<s_reg>", "</s_reg>"]

MODEL_SARCASM_CATEGORIZER_LLAMA = "llama-3.1-8b-sarcasm.Q4_K_S.gguf"
MODEL_FALCON_INSTRUCT = "tiiuae-falcon-7b-instruct-Q4_K_S.gguf"

MODEL_SARCASM_DETECTOR ="mrm8488/t5-base-finetuned-sarcasm-twitter"
MODEL_SENTIMENT_ANALYSIS ="cardiffnlp/twitter-roberta-base-sentiment-latest"
MODEL_TEXT_CLASSIFICATION ="ynie/roberta-large_conv_contradiction_detector_v0"

PROMPT_TONE_ANALYSIS = "tone_analysis_prompt.txt"
PROMPT_TONE_REPAIR = "tone_json_repair_prompt.txt"

DIR_COMMON = f"{ROOT}/common"
DIR_DATA = f"{ROOT}/data"
DIR_PROMPTS = f"{ROOT}/prompts"
DIR_LOGS = f"{ROOT}/logs"
DIR_CHECKPOINTS = f"{ROOT}/checkpoints"
DIR_MODELS = f"{ROOT}/models"
DIR_TEST = f"{ROOT}/test"
DIR_GUFF = f"{ROOT}/gguf"
DIR_MISC = f"{ROOT}/misc"
DIR_MERGED_FOR_EXPORT = f"{ROOT}/merged_model_for_export"

DIR_TRAINING_CORPUS = f"{DIR_DATA}/training.corpus"

DIR_LLAMA_CPP = "C:/storage/development/llama/llama.cpp"
DIR_GGML_LLAMA = "C:/storage/tools/ggml.org/llama.cpp"

NARRATOR = "BOFH"
PRINCIPLE = "BOFH"

PROGRESS_PATTERN = re.compile(r"PROGRESS:\s*(\d+)/(\d+)", re.IGNORECASE)

CREATION_FLAGS = 0
PRE_EXEC_FN = None

LABEL_DIAGNOSTICS = "Diagnostics"
LABEL_COLLECTOR = "Collector"
LABEL_ATTRIBUTION = "Speaker Attribution"
LABEL_CONSOLIDATION = "Consolidation"
LABEL_TONAL_ANALYSIS = "Tonal Analysis"
LABEL_TOKENIZATION = "Training Corpus Builder"
LABEL_TOKEN_VALIDATION = "Validation"
LABEL_TRAINING = "Training"
LABEL_QUANTIZATION = "Quantization"
LABEL_CHAT = "Chat"
LABEL_EVALUATION = "Evaluation"

GUI_MAX_DIVS = 1000

if os.name == "nt":
    CREATION_FLAGS = subprocess.CREATE_NEW_PROCESS_GROUP
else:
    PRE_EXEC_FN = os.setsid

def get_Llama_cpp_path(filename):
    return f"{DIR_LLAMA_CPP}/{filename}"

def fix_mojibake_safe(text):
    return ftfy.fix_text(text) if isinstance(text, str) else text


def decode_unicode(text):
    return codecs.decode(text, 'unicode_escape')

def normalize_eol(text):
    return text.replace('\r\n', '\n').replace('\r', '\n')

def print_progress(current, total):
    print(f"PROGRESS: {current}/{total}", flush=True)

def get_path( root, filename ):
    return f"{root}/{filename}"

def get_base_path(filename):
    return f"{ROOT}/{filename}"

def get_common_path(filename):
    return f"{DIR_COMMON}/{filename}"

def get_data_path(filename):
    return f"{DIR_DATA}/{filename}"

def get_misc_path(filename):
    return f"{DIR_MISC}/{filename}"

def get_prompt_path(filename):
    return f"{DIR_PROMPTS}/{filename}"

def get_log_path(filename):
    return f"{DIR_LOGS}/{filename}"

def get_checkpoint_path(filename):
    return f"{DIR_CHECKPOINTS}/{filename}"

def get_model_path(filename):
    return f"{DIR_MODELS}/{filename}"

def get_guff_path(filename):
    return f"{DIR_GUFF}/{filename}"

def safe_text(text):
    try:
        return str(text).encode("utf-8", errors="replace").decode("utf-8")
    except Exception as e:
        return f"[ERROR sanitizing output: {e}]"

def load_json_schema(candidate: str) -> dict:
    path = Path(candidate)
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")
    with open(path, "r", encoding="utf-8" ) as f:
        schema = json.load(f)
    text = load_text_file(candidate)
    return schema,text

def load_text_file(candidate: str) -> str:
    path = Path(candidate)
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()

    return text

class ProgressTracker:
    def __init__(self, description="Progress", minimum=0, maximum=100, display_fn=print, progressbar=None):
        self.description = description
        self.minimum = minimum
        self.maximum = maximum
        self.current = minimum
        self.timestamps = []
        self.start_time = time.time()
        self.display_fn = display_fn
        self.progressbar = progressbar

    def update(self, value):
        now = time.time()
        self.current = value
        self.timestamps.append((now, value))
        if len(self.timestamps) > 10:
            self.timestamps.pop(0)

        percent = 100 * (self.current - self.minimum) / (self.maximum - self.minimum)
        if self.progressbar:
            self.progressbar["value"] = percent
        self._display_progress(self._calculate_eta())

    def _calculate_eta(self):
        if len(self.timestamps) < 2:
            return None
        t0, v0 = self.timestamps[0]
        t1, v1 = self.timestamps[-1]
        if v1 == v0:
            return None
        rate = (v1 - v0) / (t1 - t0)
        remaining = self.maximum - self.current
        return remaining / rate if rate > 0 else None

    def _display_progress(self, eta):
        percent = 100 * (self.current - self.minimum) / (self.maximum - self.minimum)
        bar = "|" * int(percent / 5) + "-" * (20 - int(percent / 5))

        elapsed = time.time() - self.start_time
        eta_str = self._format_time(eta) if eta else "Calculating..."
        elapsed_str = self._format_time(elapsed)

        self.display_fn( "ProgressTracker", f"{self.description}: [{bar}] {percent:.1f}% | Elapsed: {elapsed_str} | ETA: {eta_str}")

    def _format_time(self, seconds):
        minutes, sec = divmod(int(seconds), 60)
        hours, minutes = divmod(minutes, 60)
        return f"{hours}h {minutes}m {sec}s" if hours else f"{minutes}m {sec}s" if minutes else f"{sec}s"

class SimpleRunner:
    def __init__(self, command, description, display_fn, tracker, on_finish):
        self.command = command
        self.description = description
        self.display_fn = display_fn
        self.tracker = tracker
        self.on_finish = on_finish
        self.proc = None
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.pid     = None
    def start(self):
        self.thread.start()

    def _run(self):

        try:            
            self.proc = subprocess.Popen(
                self.command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",  # or "ignore"
                preexec_fn = PRE_EXEC_FN,
                creationflags = CREATION_FLAGS,
                env = self._conda_subproc_env(),
                bufsize=1             # line-buffered
            )
            
            self.pid = self.proc.pid
            
            result = self.proc.wait()
            

            if result != 0:
                self.display_fn("SimpleRunner", f"[{self.description}] Exit code: {result}")

        except Exception as e:
            self.display_fn("SimpleRunner", f"[{self.description}] Exception: {type(e).__name__}: {e}")
        finally:
            if self.on_finish:
                self.on_finish()

    def safe_decode(self, data, source="unknown"):
        try:
            return data.decode("utf-8")
        except UnicodeDecodeError as e:
            print(f"🚨 Unicode error from {source}: {e}")
            print("Raw bytes:", repr(data[:20]))
            return data.decode("utf-8", errors="replace")

    def _conda_subproc_env(self):
        # 1. Start from the current environment
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        # 3. Prepend the Conda executables
        conda_paths = [
            str(Path(CONDA_PREFIX)/"condabin"),
            str(Path(CONDA_PREFIX)/"Scripts"),   # Windows
            str(Path(CONDA_PREFIX)/"Library"/"bin"),  # common for DLLs on Windows
            str(Path(CONDA_PREFIX)/"bin"),       # Linux/macOS
        ]
        # Filter out non-existent paths
        conda_paths = [p for p in conda_paths if Path(p).exists()]
    
        # 4. Update PATH
        env["PATH"] = os.pathsep.join(conda_paths + [env.get("PATH", "")])
        return env

    def cancel(self):
        if self.proc and self.proc.poll() is None:
            self.display_fn( "SimpleRunner", f"[{self.description}] Termination requested...")

            try:
                if os.name == "nt":
                    self.proc.send_signal(signal.CTRL_BREAK_EVENT)
                else:
                    os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)

                self.display_fn( "SimpleRunner", f"[{self.description}] Termination signal sent.")
            except Exception as e:
                self.display_fn( "SimpleRunner", f"[{self.description}] Error during cancel: {type(e).__name__}: {e}")


# class SubprocessRunner:
#     _active_instance = None  # Class-level lock

#     def __init__(self, command, max_steps, description, display_fn, progressbar=None, on_complete=None):
#         self.command = command
#         self.max_steps = max_steps
#         self.description = description
#         self.display_fn = display_fn
#         self.progressbar = progressbar
#         self.on_complete = on_complete
#         self.tracker = ProgressTracker(description, 0, max_steps, display_fn, progressbar)
#         self._process = None
#         self._thread = None
#         self.cancelled = False

#     def start(self):
#         if SubprocessRunner._active_instance:
#             self.display_fn("SubprocessRunner",f"[{self.description}] Error: Another process is already running.")
#             return
#         SubprocessRunner._active_instance = self
#         self._thread = threading.Thread(target=self._run, daemon=True)
#         self._thread.start()

#     def _run(self):
#         try:
#             self._process = subprocess.Popen(
#                 self.command,
#                 shell=True,
#                 stdout=subprocess.PIPE,
#                 stderr=subprocess.PIPE,
#                 text=True,
#                 encoding="utf-8",     # force UTF-8 decoding
#                 errors="replace",     # replace malformed bytes with �
#                 bufsize=1             # line-buffered
#             )
#             for line in self._process.stdout:
#                 if self.cancelled:
#                     break
#                 line = line.strip()
#                 self.display_fn(line)
#                 match = PROGRESS_PATTERN.search(line)
#                 if match:
#                     current, total = map(int, match.groups())
#                     self.tracker.maximum = total  # Update if it changes dynamically
#                     self.tracker.update(current)
#             self._process.wait()
#         finally:
#             if self.cancelled:
#                 self.display_fn("SubprocessRunner",f"[{self.description}] Cancelled by user.")
#             elif self.on_complete:
#                 self.on_complete()

#     def cancel(self):
#         if self._process and self._process.poll() is None:
#             self.cancelled = True
#             try:
#                 # Works on Windows and Unix
#                 if os.name == 'nt':
#                     self._process.send_signal(signal.CTRL_BREAK_EVENT)
#                 else:
#                     self._process.terminate()
#                 self.display_fn("SubprocessRunner",f"[{self.description}] Termination signal sent.")
#             except Exception as e:
#                 self.display_fn("SubprocessRunner",f"[{self.description}] Failed to cancel: {e}")
