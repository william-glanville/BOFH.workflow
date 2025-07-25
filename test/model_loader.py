# -*- coding: utf-8 -*-
from transformers import AutoModelForCausalLM, AutoTokenizer

class ModelLoader:
    def __init__(self, config):
        self.model = AutoModelForCausalLM.from_pretrained(config.model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_path)
        self.pad_token_id = config.pad_token_id

    def encode(self, prompt):
        return self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

    def decode(self, output):
        return self.tokenizer.decode(output[0], skip_special_tokens=True)
