# -*- coding: utf-8 -*-
class Metrics:
    def __init__(self):
        self.outputs = []

    def add(self, prompt, response):
        self.outputs.append((prompt, response))

    def print_summary(self):
        for i, (p, r) in enumerate(self.outputs):
            print(f"\n🔹 Prompt {i+1}: {p}\n📎 Response:\n{r}\n")

    # Optional: add BLEU, ROUGE, BERTScore evaluations here
