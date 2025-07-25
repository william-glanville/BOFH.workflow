import os
import constants
import webview
import telemetry
import tkinter as tk
from gui_core import WorkflowApp
from workflow_runner import WorkflowRunner
from reporting import ReportRenderer

os.environ["PYTHONIOENCODING"] = "utf-8"

class BOFHApp(WorkflowApp):
    def __init__(self):
        super().__init__()

        self.runner = WorkflowRunner(self, constants )

        # Bind action buttons
        for label, method in [
            ( constants.LABEL_DIAGNOSTICS, self.runner.launch_diagnostic),
            ( constants.LABEL_COLLECTOR, self.runner.launch_collector),
            ( constants.LABEL_ATTRIBUTION, self.runner.launch_attribution),
            ( constants.LABEL_CONSOLIDATION, self.runner.launch_consolidation),
            ( constants.LABEL_TONAL_ANALYSIS, self.runner.launch_generate_tonal_lexicon),
            ( constants.LABEL_TOKENIZATION, self.runner.launch_tokenizer),
            ( constants.LABEL_TOKEN_VALIDATION, self.runner.launch_validation),
            ( constants.LABEL_TRAINING, self.runner.launch_training),
            ( constants.LABEL_QUANTIZATION, self.runner.launch_quantization),
            ( constants.LABEL_EVALUATION, self.cmd_placeholder),
            ( constants.LABEL_CHAT, self.cmd_placeholder),
            ("Close App", self.close_app),
        ]:
            b = self.build_button(label, method)
            b.pack(pady=2, anchor="w")

        self.report_map = {
            constants.LABEL_TOKEN_VALIDATION: lambda x: self.generate_tokenization_report(x),
        }
        self.cancel_button.config(command=self.runner.cancel_active_task)
        self.clear_button.config(command=self.clear_console)

    def display( self, tag, message ):
        candidate = {
            "type": telemetry.TYPE_DISPLAY,
            "data": {
                "tag" : tag,
                "message" : message,
            }
         }
        self.process_telemetry( candidate )
    
    def build_button(self, text, command):
        return tk.Button(self.button_frame, text=text, command=command, width=20)

    def cmd_placeholder(self):
        self.display("SYSTEM","That button’s not wired yet. Feels like management, doesn’t it?")

    def undefined(self,task):
        self.display("SYSTEM",f" No {task} report defined yet. Yeah Nah, Out of sight out of mind Bru!")
        
    def generate_reports(self, last ):
        self.report_map.get(last,self.undefined)(last)
        
    def generate_tokenization_report(self,last):
        path = constants.get_log_path(constants.DS_TOKENIZATION_REPORT)
        if not os.path.exists(path):
            self.display("SYSTEM","Tokenization Report not found.")
            return

        with open(path, encoding="utf-8") as file:
            content = file.read()
        template_path = constants.get_misc_path(constants.DS_TOKENIZATION_REPORT_TEMPLATE)
        html = ReportRenderer(template_path).render(content)        
        self.start_report_async("Tokenization Report", html)
        
    def launch_report(self,title, html, width=1000, height=800):
        window = webview.create_window(title, html=html, width=width, height=height)
        webview.start()
    
    def start_report_async(self, title, html):
        def queue_launch():
            self.launch_report(title, html)
        self.master.after(0, queue_launch)
        
if __name__ == "__main__":
    app = BOFHApp()
    app.run()