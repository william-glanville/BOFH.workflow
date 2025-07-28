import os, subprocess
import constants

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from checkpointing import find_latest_checkpoint
from telemetry import TelemetryProxy
from TrainingConfig import QuantizationConfig


class ModelQuantizer:
    def __init__(self, config: QuantizationConfig):
        self.cfg = config
        self.telemetry = TelemetryProxy()
        os.makedirs(self.cfg.output_dir, exist_ok=True)

    def get_file_size_mb(self, path):
        return os.path.getsize(path) / (1024 * 1024)

    def load_and_merge_model(self):
        self.telemetry.display( "Quantizer loading","🔍 Locating checkpoint...")
        _, checkpoint = find_latest_checkpoint(self.cfg.checkpoint_dir)

        if not os.path.isdir(self.cfg.adapter_dir):
            raise FileNotFoundError(f"❌ Adapter directory missing: {self.cfg.adapter_dir}")
        if not checkpoint:
            raise FileNotFoundError("❌ No checkpoint found to evaluate.")

        self.telemetry.display("ModelQuantizer", f"🧠 Evaluating model from: {checkpoint}")
        base = AutoModelForCausalLM.from_pretrained(self.cfg.base_model).to(self.cfg.device)

        tokenizer = AutoTokenizer.from_pretrained(self.cfg.base_model)
        tokenizer.add_tokens( constants.TONAL_TOKENS )
        
        base.resize_token_embeddings(len(tokenizer))
        base.eval()
        
        adapter = PeftModel.from_pretrained(base, self.cfg.adapter_dir)
        merged = adapter.merge_and_unload()
        merged.save_pretrained(self.cfg.output_dir)

        tokenizer.save_pretrained(self.cfg.output_dir)

        self.telemetry.display("ModelQuantizer", "Merge complete")

    def convert_to_gguf(self):
        self.telemetry.display("ModelQuantizer" , f"⚙️ Converting to GGUF: {self.cfg.gguf_fp16}")
        result = subprocess.run([
            "python", self.cfg.convert_script,
            self.cfg.output_dir,
            "--outfile", self.cfg.gguf_fp16
            ], 
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding="utf-8"
        )
        self.telemetry.display("ModelQuantizer" , f"Converted file {self.cfg.gguf_fp16}, \nresult:{result.stdout}")

    def quantize(self):
        self.telemetry.display("ModelQuantizer" , f"🔧 Quantizing to {self.cfg.quant_type}: {self.cfg.gguf_quant}")
        result = subprocess.run([
            self.cfg.quantize_bin,
            self.cfg.gguf_fp16,
            self.cfg.gguf_quant,
            self.cfg.quant_type
            ], 
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding="utf-8"
        )
        self.telemetry.display("ModelQuantizer" , f"🔧 Quantizing result {result.stdout}")
        
        unquantized_size = self.get_file_size_mb(self.cfg.gguf_fp16)
        quantized_size = self.get_file_size_mb(self.cfg.gguf_quant)
        self.telemetry.display("ModelQuantizer" ,f"📦 Unquantized: {unquantized_size:.2f} MB")
        self.telemetry.display("ModelQuantizer" ,f"🎯 Quantized ({self.cfg.quant_type}): {quantized_size:.2f} MB")
        self.telemetry.display("ModelQuantizer" , {
            "tag": "Quantize",
            "unquantized_mb": unquantized_size,
            "quantized_mb": quantized_size,
            "type": self.cfg.quant_type
        })


def main():
    cfg = QuantizationConfig()
    quantizer = ModelQuantizer(cfg)

    quantizer.load_and_merge_model()
    quantizer.convert_to_gguf()
    quantizer.quantize()

    quantizer.telemetry.display("ModelQuantizer" ,"✅ Quantization pipeline complete.")


if __name__ == "__main__":
    main()