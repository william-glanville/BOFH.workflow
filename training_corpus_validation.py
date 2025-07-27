import os
import json
import logging
import torch
import webview
from collections import Counter
from datasets import load_from_disk
from reporting import ReportRenderer
from telemetry import SocketTelemetrySender
import constants

class CorpusValidator:
    def __init__(self):
        self.telemetry = SocketTelemetrySender()
        self.corpus_path = constants.DIR_TRAINING_CORPUS
        self.log_path = constants.get_log_path("validate_corpus.log")
        self.summary_path = constants.get_log_path("validate_corpus.summary.json")

        logging.basicConfig(
            filename=self.log_path,
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s"
        )

    def validate(self, dataset):
        lengths, mismatch_count = [], 0
        sarcasm_freq = Counter()
        tone_seq_lengths = Counter()
        total = len(dataset)

        for i, item in enumerate(dataset):
            if self.telemetry and i % 10 == 0:
                self.telemetry.report_progress("Validation", i, total)

            input_ids = item["input_ids"].cpu() if isinstance(item["input_ids"], torch.Tensor) else torch.tensor(item["input_ids"])
            labels = item["labels"].cpu() if isinstance(item["labels"], torch.Tensor) else torch.tensor(item["labels"])

            if input_ids.size(0) != labels.size(0):
                mismatch_count += 1
                logging.warning(f"[Mismatch @ {i}] input_ids: {input_ids.size(0)} vs labels: {labels.size(0)}")
                continue

            lengths.append(input_ids.size(0))
            mask_indices = (labels == -100).nonzero(as_tuple=True)[0]
            tone_section = labels[mask_indices[-1]+1:] if len(mask_indices) > 0 else labels
            tone_seq_lengths[tone_section.size(0)] += 1

            if tone_section.numel() > 0:
                sarcasm_token = tone_section[-1].item()
                sarcasm_freq[sarcasm_token] += 1

        stats = {
            "total_samples": total,
            "avg_input_length": round(sum(lengths) / len(lengths), 2) if lengths else 0,
            "label_mismatches": mismatch_count,
            "sarcasm_token_distribution": dict(sarcasm_freq),
            "tone_sequence_lengths": dict(tone_seq_lengths)
        }

        with open(self.summary_path, "w", encoding="utf-8") as fout:
            json.dump(stats, fout, indent=2)

        self.telemetry.display("Validation", f"Validation complete. Summary written to: {self.summary_path}")
        self.telemetry.display("Validation", "Generating report")

    def generate_tokenization_report(self):
        path = constants.get_log_path(constants.DS_TOKENIZATION_REPORT)
        if not os.path.exists(path):
            self.telemetry.display("Validation", "Tokenization Report not found.")
            return

        with open(path, encoding="utf-8") as file:
            content = file.read()
        template_path = constants.get_misc_path(constants.DS_TOKENIZATION_REPORT_TEMPLATE)
        html = ReportRenderer(template_path).render(content)
        self.launch_report("Tokenization Report", html)

    def launch_report(self, title, html, width=1000, height=800):
        window = webview.create_window(title, html=html, width=width, height=height)
        webview.start()

    def run(self):
        if not os.path.exists(self.corpus_path):
            self.telemetry.display("Validation", f"Corpus not found: {self.corpus_path}")
            return

        try:
            dataset = load_from_disk(self.corpus_path)
            self.validate(dataset)
            self.generate_tokenization_report()
        except Exception as e:
            self.telemetry.display("Validation", f"Validation error: {e}")
            logging.error(f"Validation failure: {e}")

if __name__ == "__main__":
    CorpusValidator().run()