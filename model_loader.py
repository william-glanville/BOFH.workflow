import torch
import chardet
import json

from telemetry import TelemetryProxy
from collections import defaultdict
from torch.utils.checkpoint import checkpoint
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorWithPadding,
    AutoConfig,
    MistralConfig, 
    MistralForCausalLM
)


HF_USER_TOKEN = "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"
MODEL_NAME = "NousResearch/Nous-Hermes-2-Mistral-7B-DPO"

class ModelRetriever:
    def __init__(self, model_id, use_checkpoint=True, tonal_tokens=None, corpus_path=None, dtype=torch.bfloat16):
        self.model_id = model_id
        self.use_checkpoint = use_checkpoint
        self.tonal_tokens = tonal_tokens or []
        self.corpus_path = corpus_path
        self.dtype = dtype
        self.model = None
        self.tokenizer = None
        self.data_collator = None
        self.monitor = TelemetryProxy()
        self.seed_map = {
            "<s_tone>": ["sarcasm", "tone", "style", "attitude"],
            "</s_tone>": ["tone"],
            "<s_reg>": ["formal", "register", "polite"],
            "</s_reg>": ["register"]
        }
    def retrieve(self):
        self.monitor.display("ModelRetriever", f"Checking model: {self.model_id}")

        # Tokenizer
        self.monitor.display("ModelRetriever", "Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, token=HF_USER_TOKEN)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        style_tokens = [
            "<|tone:angry|>", "<|tone:ironic|>", "<|tone:witty|>","<|tone:helpful|>"
            "<|sarcasm:none|>", "<|sarcasm:low|>", "<|sarcasm:high|>"
        ]
        self.tokenizer.add_tokens(style_tokens)
        combinations = self._scan_style_combinations(self.corpus_path);
        self.tokenizer.add_tokens(combinations)
        self.tokenizer.add_tokens(list(self.tonal_tokens))
        self.monitor.display("ModelRetriever", "Tokenizer loaded and extended.")

        # Model
        self.monitor.display("ModelRetriever", "Downloading model weights...")        
        self.model = CausalLMWithCheckpointing.from_pretrained(
            self.model_id,
            token=HF_USER_TOKEN,
            torch_dtype=self.dtype,
            device_map="cuda",
            low_cpu_mem_usage=False
        )

        self.model.resize_token_embeddings(len(self.tokenizer), mean_resizing=False)
        
        self.monitor.display("ModelRetriever", f"Model loaded and resized for {len(self.tokenizer)} tokens.")
        
        new_token_count = self.tokenizer.add_tokens(list(self.tonal_tokens))
        self.monitor.display("ModelRetriever", f"Added {new_token_count} new tokens.")



        self.monitor.display("ModelRetriever", "Seeding custom embeddings")
        self.seed_custom_embeddings(self.seed_map, self.model, self.tokenizer)
        
        self.monitor.display("ModelRetriever", "Building Data Collator")
        # Collator
        self.data_collator = DataCollatorWithPadding(
            tokenizer=self.tokenizer,
            pad_to_multiple_of=8,
            return_tensors="pt"
        )

        # Report memory and completion
        total_params = sum(p.numel() for p in self.model.parameters())
        self.monitor.display("ModelRetriever", f"Model total parameters: {total_params:,}")
        self.monitor.report_gpu_memory()

    def seed_custom_embeddings(self, seed_map, model, tokenizer):
    
        embeddings = model.get_input_embeddings().weight
    
        with torch.no_grad():
            for target_token, anchor_tokens in seed_map.items():
                anchor_ids = [
                    tokenizer.convert_tokens_to_ids(t)
                    for t in anchor_tokens
                    if tokenizer.convert_tokens_to_ids(t) != tokenizer.unk_token_id
                ]
                if not anchor_ids:
                    continue
    
                # Average embeddings of anchors
                anchor_vecs = torch.stack([embeddings[i] for i in anchor_ids])
                mean_vec = anchor_vecs.mean(dim=0)
    
                target_id = tokenizer.convert_tokens_to_ids(target_token)
                embeddings[target_id] = mean_vec
         
                if target_id == tokenizer.unk_token_id:
                   self.monitor.display("ModelRetriever", f"[Warning] Token {target_token} not recognized.")
                   continue
   
    def get_tokenizer(self):
        return self.tokenizer
    
    def get_components(self):
        return self.model, self.tokenizer, self.data_collator
    
    def _generate_style_histogram(self,corpus_path):
        with open(corpus_path, "rb") as raw:
            encoding = chardet.detect(raw.read())["encoding"]
    
        histogram = defaultdict(int)
    
        with open(corpus_path, encoding=encoding, errors="replace") as f:
            for i, line in enumerate(f):
                try:
                    entry = json.loads(line)
                    tone = ", ".join(entry["Tone_Keywords"])
                    sarcasm = str(entry["Sarcasm_Level"])
                    key = f"<|tone:{tone}|><|sarcasm:{sarcasm}|>"
                    histogram[key] += 1
                except Exception as e:
                    self.monitor.display("Model Loader", f"[Line {i}] JSON decode failed: {e}")
    
        self.monitor.display_dict("Model Loader", dict(histogram) )

    def _scan_style_combinations(self,corpus_path):
        
        combos = set()

        if corpus_path:        
            with open(corpus_path, "rb") as raw:
                encoding = chardet.detect(raw.read())["encoding"]
            
            with open(corpus_path, encoding=encoding, errors="replace") as f:
                for i, line in enumerate(f):
                    try:
                        entry = json.loads(line)
                        tone = ", ".join(entry["Tone_Keywords"])
                        sarcasm = str(entry["Sarcasm_Level"])
                        combo = f"<|tone:{tone}|><|sarcasm:{sarcasm}|>"
                        combos.add(combo)
                    except Exception as e:
                        print(f"[Line {i}] JSON decode failed: {e}")

        return sorted(combos)
    
    def _dynamic_lora_report(self, model, targets=None):
        if targets is None:
            targets = ["q_proj", "k_proj", "v_proj", "o_proj", "out_proj", "lm_head"]
    
        actual_model = getattr(model, 'model', None) or getattr(model, 'base_model', None) or model
    
        if not hasattr(actual_model, 'named_modules'):
            raise TypeError(f"Expected a model with 'named_modules', got {type(actual_model)}")
    
        report = defaultdict(list)
    
        

        try:
            for name, module in actual_model.named_modules():
                #self.monitor.display( "LoRA Report" , f"Module: {name}, parameters: {getattr(module, 'parameters', None)}, type: {type(module)}")
                if any(target in name for target in targets):
                    try:
                        param_count = sum(p.numel() for p in module.parameters())
                    except TypeError:
                        param_count = 0  # if module.parameters is not callable or iterable
    
                    module_type = type(module).__name__
                    matched_target = next((t for t in targets if t in name), "unknown")
    
                    report[module_type].append({
                        "path": name,
                        "params": param_count,
                        "target_key": matched_target
                    })
        except Exception as e:
            raise RuntimeError(f"Failed during module inspection: {e}")
    
        return dict(report)



    def lora_report(self):
        if self.model:
            results = self._dynamic_lora_report(self.model)
            self.monitor.display("Model LoRA report", results)
        else:
            self.monitor.display("Model LoRA report", "Model {self.model_id} not loaded")

class CausalLMWithCheckpointing(MistralForCausalLM):
    def __init__(self, config, use_checkpoint=True, **kwargs):
        super().__init__(config)
        self.use_checkpoint = use_checkpoint
        
    @classmethod
    def from_pretrained(cls, model_id, **kwargs):
        config = AutoConfig.from_pretrained(model_id, token=kwargs.get("token"))
        use_checkpoint = kwargs.pop("use_checkpoint", True)
        base_model = MistralForCausalLM.from_pretrained(model_id, config=config, **kwargs)
    
        model = cls(config=config, use_checkpoint=use_checkpoint)
        model.load_state_dict(base_model.state_dict())
    
        return model


    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        # Get embeddings
        inputs_embeds = self.transformer.wte(input_ids)
        hidden_states = inputs_embeds

        # Apply transformer blocks
        for block in self.transformer.h:
            if self.use_checkpoint:
                hidden_states = checkpoint(block, hidden_states, attention_mask)
            else:
                hidden_states = block(hidden_states, attention_mask)[0]

        # Final layer norm + LM head
        hidden_states = self.transformer.ln_f(hidden_states)
        logits = self.lm_head(hidden_states)

        return {"logits": logits}
