# 🖥️ BOFH Workflow Companion

A sarcastically sentient AI training pipeline & telemetry-powered GUI for debugging, diagnostics, and delightfully hostile automation.

## 🤖 About This Project

This repo unites modular AI training utilities with real-time telemetry feedback inside a fully themed Tkinter GUI. Whether you're fine-tuning sarcastic inference engines, visualizing gradient norms live, or tracking loss convergence without losing your mind — BOFH Workflow Companion brings transparency, control, and unapologetic attitude to the process.

Features include:

- 🧠 **Training Framework** — Modular `BOFHTrainer` class supports LoRA adapters, gradient accumulation, checkpointing, and semantic corpus alignment.
- 🎛️ **Live Telemetry System** — Displays GPU usage, RAM consumption, learning rate, loss, and gradient norms with customizable console output.
- 📊 **Graph View** — Embedded Matplotlib figures with auto-scaling for real-time metric trends.
- 💬 **Chat + Console Views** — A multi-tab GUI with HTML-driven logs and a stylized chat panel for inference.
- 🧬 **Corpus Builder** — Preprocesses raw tone & sarcasm-annotated JSONL entries into tokenized Hugging Face datasets with embedded style control.

## 🚀 Quickstart

1. Clone the repo:
   ```bash
   git clone https://github.com/william-glanville/bofh-workflow.git
   cd bofh-workflow
   ```

2. Run the GUI:
   ```bash
   python main.py
   ```

3. Run training (after configuring paths):
   ```bash
   python train.py
   ```

## 🔍 Telemetry Tags

| Metric     | Description                     |
|------------|----------------------------------|
| `LOSS`     | Smoothed training loss           |
| `GRADNORM` | Gradient clipping magnitude      |
| `LR`       | Learning rate per step           |
| `PROGRESS` | Training completion percentage   |
| `MEMORY`   | GPU and RAM usage snapshot       |

## 🧠 Style-Aware Corpus Training

Training samples embed stylistic metadata directly:

```
<|tone:hostile, ironic|><|sarcasm:high|> 
<|user|>
Why did the sysadmin reformat my soul?
<|assistant|>
To improve your startup time — but not your personality. <|eos|>
```

## ⚙️ Requirements

- Python 3.9+
- PyTorch
- Hugging Face Transformers
- Accelerate
- TkinterModernThemes
- Matplotlib
- tkinterweb

---

Built to make AI training as expressive, insightful, and sarcastically aware as your favorite sysadmin. 💾💣
