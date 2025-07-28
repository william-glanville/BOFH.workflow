import os, gc, time, logging, torch
from datasets import load_from_disk
from transformers import get_linear_schedule_with_warmup
from accelerate import Accelerator
from torch.optim import AdamW
from torch.utils.data import DataLoader
from peft import get_peft_model, LoraConfig, TaskType

import constants
import TrainingConfig
import utilities as utils
import model_loader
import multiprocessing

from checkpointing import find_latest_checkpoint, load_checkpoint, save_checkpoint
from telemetry import TelemetryProxy
from model_loader import ModelRetriever
os.environ["PYTHONIOENCODING"] = "utf-8"

logging.basicConfig(
    filename=constants.get_log_path("train.debug.log"),
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(message)s"
)

class BOFHTrainer:
    def __init__(self, config: TrainingConfig.TrainerConfig):
        self.config = config
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        self.model = None
        self.tokenizer = None
        self.data_collator = None
        self.train_ds = None
        self.dataloader = None
        self.optimizer = None
        self.scheduler = None
        self.accelerator = None
        self.total_steps = None
        self.start_epoch = 0
        self.monitor = TelemetryProxy()        
        self.modelloader = ModelRetriever( model_loader.MODEL_NAME, constants.TONAL_TOKENS )        
        self.num_workers = multiprocessing.cpu_count() // 2
        self.start_time = time.time()
        
    def setup(self):
        try:
            self.monitor.report_gpu_memory()
            self.monitor.display("Training", "Loading model and tokenizer...")
            self.monitor.display("Training", "Model Retriever Setup")
            self.modelloader.retrieve()
            self.monitor.display("Training", "Model Retriever Ready")
            self.model, self.tokenizer, self.data_collator = self.modelloader.get_components()
            self.monitor.display("Training", "Tokenizer - add tokens")
            self.tokenizer.add_tokens(constants.TONAL_TOKENS)
            self.model.resize_token_embeddings(len(self.tokenizer))
            #self.monitor.display("Training", "LoRA report")
            #self.modelloader.lora_report()
            
        except Exception as e:
            self.monitor.display( "Training", f"Setup failed with {e}")
            raise

        self.monitor.display( "Training", f"Loading train dataset: {constants.DIR_TRAINING_CORPUS}")

        self.train_ds = load_from_disk(constants.DIR_TRAINING_CORPUS)
        self.train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

        try:        
            self.dataloader = DataLoader(
                self.train_ds,
                batch_size=self.config.batch_size,
                shuffle=True,
                collate_fn=self.data_collator,
                num_workers = self.num_workers,
                pin_memory=True, # faster host->device transfer 
                drop_last=True, # keep batch sizes consistent
            )
    
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=8,
                lora_alpha=32,
                lora_dropout=0.1,
                bias="none",
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "lm_head"]
            )
            self.model = get_peft_model(self.model, peft_config)
            self.model.gradient_checkpointing_enable()
        
            self.monitor.display( "Training", "Setting up optimizer, accelerator")
            self.accelerator = Accelerator(mixed_precision="fp16")
            self.optimizer = AdamW(self.model.parameters(), lr=self.config.learning_rate)
            self.model, self.dataloader, self.optimizer = self.accelerator.prepare(
                self.model, self.dataloader, self.optimizer
            )
            
        except Exception as e:
            self.monitor.display( "Training", f"[Dataloading Fail] {e}")
            raise


            
        self.total_steps = len(self.dataloader) * self.config.num_epochs
        effective_steps = self.total_steps // self.config.accum_steps
        num_warmup = int(0.05 * effective_steps)

        self.monitor.display( "Training", "Setting up scheduler")
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup,
            num_training_steps=self.total_steps
        )

        self.monitor.display( "Training", "Locating last checkpoint")
        _, resume_path = find_latest_checkpoint()
        if resume_path:
            self.start_epoch = load_checkpoint(self.model, self.optimizer, resume_path)
        else:
            self.monitor.display( "Training", "No checkpoint found — starting from scratch.")
        
        self.monitor.report_gpu_memory()

    def train(self):
        self.start_time = time.time()
        self.monitor.report_gpu_memory()
        self.monitor.display( "Training", "Beginning training...")

        self.model.train()

        self.monitor.display( "Training", "Step 1")
        for epoch in range(self.start_epoch, self.config.num_epochs):
            self.monitor.display( "Training", "Step 2")
            for step, batch in enumerate(self.dataloader):
                #self.monitor.display( "Training", "Step 3")
                global_step = epoch * len(self.dataloader) + step
                with self.accelerator.accumulate(self.model):
                    #self.monitor.display( "Training", f"Step 4.0 - Batch Keys {batch.keys()}")
                    outputs = self.model(**batch)
                    #self.monitor.display( "Training", f"Step 4.1 - Outputs.loss {outputs.loss}")
                    loss = outputs.loss / self.config.accum_steps
                    
                    #self.monitor.display( "Training", f"Step 4.2, loss = {loss}")
                    self.accelerator.backward(loss)                    
                    #self.monitor.display( "Training", "Step 5")
                    if self.accelerator.sync_gradients:
                        #self.monitor.display( "Training", "Step 5.1")
                        # Clip gradients and get their norm
                        grad_norm = torch.nn.utils.clip_grad_norm_( self.model.parameters(), max_norm=1.0 )
                        self.monitor.report_gradnorm(global_step, grad_norm)
                        #self.monitor.display( "Training", "Step 5.2")
                        self.optimizer.step()
                        # Log the current learning rate
                        current_lr = self.optimizer.param_groups[0]['lr']
                        #self.monitor.display( "Training", "Step 5.3")
                        self.monitor.report_learningrate(global_step, current_lr)        
                        self.scheduler.step()
                        #self.monitor.display( "Training", "Step 5.4")
                        self.optimizer.zero_grad()
                        #self.monitor.display( "Training", "Step 5.5")

                try:
                    #self.monitor.display( "Training", "Step 5.6")
                    # Send raw loss, lr, grad_norm every 10 steps
                    self.monitor.report_progress( "Training", global_step, self.total_steps)
                    self.monitor.report_loss( global_step, loss )
                    self.log_step( epoch, step, loss, global_step )
                except Exception as e:
                    self.monitor.display("Progress Exception", f"{e}")
                        
            self.monitor.display( "Training", "Epoch {epoch} Completed")
            self.save_and_sample(epoch)
            gc.collect()
            torch.cuda.empty_cache()
            self.monitor.report_gpu_memory()

        self.monitor.report_gpu_memory()
        self.model.save_pretrained(self.config.adapter_path)
        self.monitor.display( "Training", "Finetuning complete. Adapter saved.")

    def save_and_sample(self, epoch):
        path = os.path.join(self.config.checkpoint_dir, f"bofh_epoch_{epoch}.pth")
        save_checkpoint(self.model, self.optimizer, epoch, path)
        self.generate_sample(self.config.sample_prompt,epoch)

    def generate_sample(self, prompt, epoch, max_new_tokens=100):
        self.model.eval()
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.15
            )
        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True).split("<|assistant|>\n")[-1].strip()
        self.monitor.display( "Training", f"\n Sample (Epoch {epoch}):\n{text}\n")
        self.model.train()

    def log_step(self, epoch, step, loss, global_step):
        percent = (global_step + 1) / self.total_steps * 100
        elapsed = time.time() - self.start_time
        eta = utils.format_eta((self.total_steps - global_step - 1) * (elapsed / (global_step + 1)))
        self.monitor.display( "Training", f"[Epoch {epoch}] Step {step} | Loss: {loss:.4f} | {percent:.2f}% | ETA: {eta}")
        
        
