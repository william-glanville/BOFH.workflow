import os
import constants
from dataclasses import dataclass
from pathlib import Path

@dataclass
class TrainerConfig:
    batch_size: int = 32  # Increased to more typical value
    accum_steps: int = 24
    learning_rate: float = 5e-5
    num_epochs: int = 3
    max_length: int = 512
    max_seq_length: int = 320
    label_seq_len: int = 32
    loss_threshold: float = 2.0
    PAD_TOKEN_ID = -100
    log_dir: Path = Path(constants.get_log_path("bofh_lora"))
    checkpoint_dir: Path = Path(constants.DIR_CHECKPOINTS)
    adapter_path: Path = Path(constants.get_checkpoint_path("bofh_adapter/"))
    sample_prompt: str = (
        "<|system|>\nYou are BOFH.\n"
        "<|user|>\nWhat’s the worst subprocess mistake?\n"
        "<|assistant|>\n"
    )

    def __post_init__(self):
        # Validate and create directories if they don't exist
        for path in [self.log_dir, self.checkpoint_dir, self.adapter_path]:
            path.mkdir(parents=True, exist_ok=True)
            if not os.access(path, os.W_OK):
                raise PermissionError(f"No write permission for directory: {path}")

        # Validate batch size
        if self.batch_size < 1:
            raise ValueError("Batch size must be positive")

        # Validate sequence lengths
        if self.max_length < 1 or self.label_seq_len < 1:
            raise ValueError("Sequence lengths must be positive")
          
@dataclass
class QuantizationConfig:
    base_model = "HuggingFaceH4/zephyr-7b-alpha"
    adapter_dir = Path(constants.get_checkpoint_path("bofh_adapter"))
    checkpoint_dir = Path(constants.DIR_CHECKPOINTS)
    output_dir = Path(constants.DIR_MERGED_FOR_EXPORT)
    convert_script = constants.get_Llama_cpp_path("convert_hf_to_gguf.py")
    quantize_bin = constants.get_Llama_cpp_path( "build/bin/Release/llama-quantize.exe" )
    quant_type = "Q4_K_M"
    gguf_fp16 = constants.get_guff_path("bofh-unquantized.gguf")
    gguf_quant = constants.get_guff_path("bofh-q4_k_m.gguf")
    device = "cuda"  # Change to 'cuda' if needed

            