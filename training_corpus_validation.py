import os
import logging
import json
import constants
import torch
from datasets import load_from_disk
from collections import Counter
from TokenizationReport import generate_tokenization_report  # if separate
from telemetry import SocketTelemetrySender
from memory_monitor import MemoryMonitor

telemetry = SocketTelemetrySender()
memory = MemoryMonitor()

TRAINING_CORPUS_PATH = constants.DIR_TRAINING_CORPUS
LOG_PATH = constants.get_log_path("validate_corpus.log")
SUMMARY_PATH = constants.get_log_path("validate_corpus.summary.json")

logging.basicConfig(
    filename=LOG_PATH,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def validate(dataset):
    lengths = []
    mismatch_count = 0
    sarcasm_freq = Counter()
    tone_seq_lengths = Counter()
    
    rec_count = len(dataset)
    
    for i, item in enumerate(dataset):        
        if telemetry and i%10==0:
            telemetry.report_progress("Validation", i, rec_count )
        # Ensure tensors are on CPU and properly cast
        input_ids = item["input_ids"].cpu() if isinstance(item["input_ids"], torch.Tensor) else torch.tensor(item["input_ids"])
        labels    = item["labels"].cpu()    if isinstance(item["labels"], torch.Tensor)    else torch.tensor(item["labels"])

        if input_ids.size(0) != labels.size(0):
            mismatch_count += 1
            logging.warning(f"[Mismatch @ {i}] input_ids: {input_ids.size(0)} vs labels: {labels.size(0)}")
            continue

        lengths.append(input_ids.size(0))

        # Slice tone label section
        mask_indices = (labels == -100).nonzero(as_tuple=True)[0]
        tone_section = labels[mask_indices[-1]+1:] if len(mask_indices) > 0 else labels
        tone_seq_lengths[tone_section.size(0)] += 1

        if tone_section.numel() > 0:
            sarcasm_token = tone_section[-1].item()
            sarcasm_freq[sarcasm_token] += 1

    # Stats block
    stats = {
        "total_samples": len(dataset),
        "avg_input_length": round(sum(lengths)/len(lengths), 2) if lengths else 0,
        "label_mismatches": mismatch_count,
        "sarcasm_token_distribution": dict(sarcasm_freq),
        "tone_sequence_lengths": dict(tone_seq_lengths)
    }

    with open(SUMMARY_PATH, "w", encoding="utf-8") as fout:
        json.dump(stats, fout, indent=2)
    logging.info(f"✅ Validation complete. Summary written to: {SUMMARY_PATH}")

def main():
    if not os.path.exists(TRAINING_CORPUS_PATH):
        telemetry.display("Validation", f"❌ Corpus not found: {TRAINING_CORPUS_PATH}")
        return
    try:
        dataset = load_from_disk(TRAINING_CORPUS_PATH)
        validate(dataset)
        generate_tokenization_report()
    except Exception as e:
        telemetry.display("Validation",f"🚨 Validation error: {e}")
        logging.error(f"Validation failure: {e}")

if __name__ == "__main__":
    main()