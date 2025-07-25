import os
import constants
import logging
import json
import chardet
import datasets
import TrainingConfig
import model_loader
from model_loader import ModelRetriever
from datasets import Dataset, Features, Value, List
from telemetry import SocketTelemetrySender, ConsoleTelemetrySender

datasets.disable_progress_bar()
datasets.disable_caching()
os.environ["PYTHONIOENCODING"] = "utf-8"

logging.basicConfig(
    filename=constants.get_log_path("generate_training_corpus.debug.log"),
    filemode="w",
    encoding="utf-8",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

CONFIG = TrainingConfig.TrainerConfig()
TONAL_ANALYSIS_PATH = constants.get_data_path(constants.DS_TONAL_ANALYSIS)
TRAINING_CORPUS_PATH = constants.DIR_TRAINING_CORPUS

class CorpusBuilder:
    def __init__(self, config, corpus_path, output_path):
        self.config = config
        self.tokenizer = None
        self.corpus_path = corpus_path
        self.output_path = output_path
        self.monitor = SocketTelemetrySender() 
        #self.monitor = ConsoleTelemetrySender() 
        self.modelloader = ModelRetriever( model_loader.MODEL_NAME, constants.TONAL_TOKENS )        
        self.monitor.display("Corpus Builder", "Started")

    def setup(self):
        self.monitor.display("Corpus Builder", "Setup")
        self.modelloader.retrieve()
        self.tokenizer = self.modelloader.get_tokenizer()
        self.monitor.display("Corpus Builder", "Ready")
        
    def load_raw_entries(self):
        self.monitor.display("Corpus Builder", f"Loading {self.corpus_path}")
        with open(self.corpus_path, "rb") as raw:
            encoding = chardet.detect(raw.read())["encoding"]
            self.monitor.display("Corpus Builder", f"Detected encoding: {encoding}")
        entries = []
        with open(self.corpus_path, encoding=encoding, errors="replace") as f:
            for i, line in enumerate(f):
                try:
                    entries.append(json.loads(line))
                except Exception as e:
                    self.monitor.display("Corpus Builder", f"[Line {i}] JSON decode failed: {e}")
        return entries

    def normalize(self, s):
        import unicodedata
        return unicodedata.normalize("NFKC", s).encode("utf-8", errors="replace").decode("utf-8")

    def preprocess_entry(self, entry):
        try:
            tone = ", ".join(entry["Tone_Keywords"])
            sarcasm = str(entry["Sarcasm_Level"])
            context = self.normalize(entry["context"].strip())
            response = self.normalize(entry["response"].strip())
    
            # Construct stylized prefix
            style_prefix = f"<|tone:{tone}|><|sarcasm:{sarcasm}|> "
    
            # Full prompt with stylistic control
            full_text = (
                f"<|system|>\nYou are BOFH — sharp, sarcastic, and ruthlessly insightful.\n"
                f"{style_prefix}"
                f"<|user|>\n{context}\n"
                f"<|assistant|>\n{response} <|eos|>"
            )
    
            encoding = self.tokenizer(
                full_text,
                truncation=True,
                padding=False,
                max_length=self.config.max_length,
                return_attention_mask=True
            )
    
            if len(encoding["input_ids"]) == self.config.max_length:
                self.monitor.display("Corpus Builder", "[Truncation] Entry hit max_length cutoff")
    
            return {
                "input_ids": encoding["input_ids"],
                "attention_mask": encoding["attention_mask"],
                "labels": encoding["input_ids"].copy()
            }
    
        except Exception as e:
            self.monitor.display("Corpus Builder", f"[Tokenizer error] Skipping: {entry['context'][:60]} → {e}")
            return None
    
    def semantic_align(self, example):
        EOS_ID = self.tokenizer.convert_tokens_to_ids("<|eos|>")
        ids, labels, mask = example["input_ids"], example["labels"], example["attention_mask"]

        try:
            eos_index = ids.index(EOS_ID) + 1
        except ValueError:
            decoded = self.tokenizer.decode(ids, skip_special_tokens=True)
            for mark in [".", "?", "!", "…”", ".”"]:
                if mark in decoded:
                    cut = decoded[:decoded.rfind(mark) + len(mark)]
                    eos_index = len(self.tokenizer(cut)["input_ids"])
                    break
            else:
                eos_index = self.config.max_seq_length

        eos_index = min(eos_index, self.config.max_seq_length)
        pad = self.config.max_seq_length - eos_index
        
        if eos_index == self.config.max_seq_length:
            self.monitor.display("Corpus Builder", "[Aligner] No semantic boundary found — using max_seq_length cutoff")
            
        return {
            "input_ids": ids[:eos_index] + [self.tokenizer.pad_token_id] * pad,
            "attention_mask": mask[:eos_index] + [0] * pad,
            "labels": labels[:eos_index] + [self.tokenizer.pad_token_id] * pad
        }

    def build_dataset(self):
        self.monitor.display("Corpus Builder", "Building dataset")
        raw = self.load_raw_entries()
        processed = []
        for e in raw:
            item = self.preprocess_entry(e)
            if item:
                processed.append(item)

        dataset = Dataset.from_list(processed).cast(
        Features({
            "input_ids": List(Value("int32")),
            "attention_mask": List(Value("int32")),
            "labels": List(Value("int32"))
        })

        )
        self.monitor.display("Corpus Builder", "Mapping")
        return dataset.map(self.semantic_align)

    def profile(self, dataset):
        from tqdm import tqdm
        import numpy as np
        lengths = [len(ex["input_ids"]) for ex in tqdm(dataset, desc="Token length scan")]
        stats = {
            "max": max(lengths),
            "median": np.median(lengths),
            "90%": np.percentile(lengths, 90),
            "95%": np.percentile(lengths, 95),
            "99%": np.percentile(lengths, 99),
            "count": len(lengths)
        }
        
        self.monitor.display_dict("Corpus Builder Profile", stats)

    def save(self, dataset):
        self.monitor.display("Corpus Builder", f"Saving tokenized dataset to {self.output_path}")
        dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
        dataset.save_to_disk(self.output_path)
        self.monitor.display("Corpus Builder", f"Saved tokenized dataset to {self.output_path}")

def main():
    print("R0")
    builder = CorpusBuilder(CONFIG, TONAL_ANALYSIS_PATH, TRAINING_CORPUS_PATH)
    print("R1")
    builder.setup()
    print("R2")
    dataset = builder.build_dataset()
    builder.profile(dataset)
    builder.save(dataset)

if __name__ == "__main__":
    main()