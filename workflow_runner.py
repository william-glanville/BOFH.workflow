import time, shutil
import tkinter.simpledialog as sd


class WorkflowRunner:
    def __init__(self, app, constants):
        self.app = app
        self.constants = constants
        self.active_runner = None
        self.last_task_name = None

    def create_command(self, env_name, script_path, *script_args):
        conda = shutil.which("conda") or "conda"
        cmd = [
            conda, "run",
            "--no-capture-output",
            "-n", env_name,
            "python", "-u", script_path
        ]
        cmd += list(script_args)
        return cmd

    def run_tracked_subprocess(self, command, description):
        self.app.set_workflow_active(True)
        self.app.display("System", f"executing {description}")

        try:
            runner = self.constants.SimpleRunner(
                command=command,
                description=description,
                display_fn=self.app.display,
                on_finish=self.on_finish,
            )
            self.active_runner = runner
            runner.start()

            while runner.pid is None:
                time.sleep(0.01)

        except Exception as e:
            self.app.display("SYSTEM",f"[{description}] Failed to start: {type(e).__name__}: {e}")
            self.app.master.after(0, self.cleanup)

    def on_finish(self):
        self.app.display("SYSTEM","Task complete. Nothing crashed... surprisingly.")      
        self.cleanup()
        #self.app.generate_reports(self.last_task_name)

    def cleanup(self):
        self.app.set_workflow_active(False)
        self.active_runner = None

    def cancel_active_task(self):
        if self.active_runner:
            try:
                self.active_runner.cancel()
            finally:
                self.active_runner = None
                self.last_task_name = None
                self.cleanup()

    def try_launch(self, name, command, description):
        self.app.display("SYSTEM",f"[{name.upper()}] Launching subprocess...")
        self.run_tracked_subprocess(command, description)

    # Entry points for each task
    def launch_collector(self):
        start = sd.askstring("Start Date", "Enter start date (YYYY-MM-DD):", initialvalue="2000-05-01")
        end = sd.askstring("End Date", "Enter end date (YYYY-MM-DD):", initialvalue="2025-06-27")
        if not start or not end:
            self.app.display("SYSTEM","COLLECTOR Launch canceled due to missing dates.")
            return
        cmd = self.create_command(self.constants.ENV_SARCASM, self.constants.get_base_path("collector.py"), "--start", start, "--end", end)
        self.try_launch("Collector", cmd, f"Data Collection from {start} to {end}")

    def launch_task(self, name, env, script):
        self.last_task_name = name
        cmd = self.create_command(env, self.constants.get_base_path(script))
        self.try_launch(name, cmd, name)

    def launch_diagnostic(self): 
        self.launch_task("Diagnostics", self.constants.ENV_LLAMA_TRAIN, "diagnostics.py")
    def launch_attribution(self): 
        self.launch_task("Attribution", self.constants.ENV_SARCASM, "speaker_attribution.py")
    def launch_consolidation(self): 
        self.launch_task("Consolidation", self.constants.ENV_SARCASM, "speaker_consolidation.py")
    def launch_generate_tonal_lexicon(self): 
        self.launch_task("Tonal Analysis", self.constants.ENV_LLAMA, "tonal_analysis.py")
    def launch_tokenizer(self): 
        self.launch_task("Training Corpus Builder", self.constants.ENV_LLAMA_TRAIN, "training_corpus_builder.py")
    def launch_validation(self): 
        self.launch_task("Validation", self.constants.ENV_LLAMA_TRAIN, "training_corpus_validation.py")
    def launch_training(self): 
        self.launch_task("Training", self.constants.ENV_LLAMA_TRAIN, "Training.py")
    def launch_quantization(self): 
        self.launch_task("Quantization", self.constants.ENV_LLAMA_TRAIN, "quantizer.py")