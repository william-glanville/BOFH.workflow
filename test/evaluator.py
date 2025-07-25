# -*- coding: utf-8 -*-
class Evaluator:
    def __init__(self, model_loader, config):
        self.model = model_loader.model
        self.tokenizer = model_loader.tokenizer
        self.config = config
        self.encode = model_loader.encode
        self.decode = model_loader.decode

    def generate_response(self, prompt):
        self.model.eval()
        inputs = self.encode(prompt)
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                repetition_penalty=self.config.repetition_penalty,
                pad_token_id=self.config.pad_token_id
            )
        return self.decode(output)
