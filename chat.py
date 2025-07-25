import subprocess
import threading
import requests
import constants
from tkinter import Frame, Entry, Button
from tkinterweb import HtmlFrame

LLAMA_SERVER = constants.get_path( constants.DIR_GGML_LLAMA, "llama-server.exe")
MODEL_PATH = constants.get_guff_path("bofh-q4_k_m.gguf")
#MODEL_PATH = constants.get_model_path("llama-3.1-8b-sarcasm.Q4_K_S.gguf")

print(LLAMA_SERVER)
print(MODEL_PATH)

class BOFHChatController:
    def __init__(self, model_engine=None):
        self.console_renderer = None
        self.model = model_engine
        self.config = {
            "temperature": 0.8,
            "repeat_penalty": 1.1,
            "max_tokens": 128,
            "top_p": 0.95
        }
        self.chat_history = []

    def format_prompt(self):
        prompt = "### System:\nYou are BOFH, the Bastard Operator From Hell. Respond sarcastically in short exchanges like a helpdesk chat.\n"
        for msg in self.chat_history[-10:]:
            role = "User" if msg["role"] == "user" else "BOFH"
            prompt += f"\n### {role}:\n{msg['message']}"
        prompt += "\n### BOFH:\nThe following is a brutally honest reply:\n"
        return prompt

    def send_user_message(self, message):
        self.chat_history.append({"role": "user", "message": message})
        prompt = self.format_prompt()

        def handle_response(response):
            self.chat_history.append({"role": "bot", "message": response})
            self.console_renderer(response)

        self.model.ask(prompt, handle_response)
        
class BOFHChatGUI(Frame):
    def __init__(self, root, controller):
        super().__init__(root)
        self.max_divs = constants.GUI_MAX_DIVS
        self.counter = 0
        self.controller = controller
        self.divs = []  # Stores div strings
        
        # Response viewer
        self.html_frame = HtmlFrame(self,messages_enabled = False)
        self.html_frame.pack(fill="both", expand=True,padx=2, pady=2)
        self.template = self.load_console_template()

        # Floating input bar
        self.input_frame = Frame(self, bd=5, relief="sunken", bg="lightblue")
        self.input_frame.pack(side="bottom", fill="x",padx=2, pady=2)

        self.entry = Entry(self.input_frame)
        self.entry.pack(side="left", fill="x", expand=True)

        self.send_btn = Button(self.input_frame, text="Send", command=self.on_send)
        self.send_btn.pack(side="right")
    
    def load_console_template(self):    
        with open(constants.get_misc_path(constants.DS_CONSOLE_THEME), "r", encoding="utf-8") as css_file:
            css = css_file.read()

        style = f"<style>\n{css}\n</style>"
        
        with open(constants.get_misc_path(constants.DS_CONSOLE_TEMPLATE), "r", encoding="utf-8") as html_template:
            html = html_template.read()
    
        #set the stylesheet
        html = html.replace("{style}", style)
        return html
    
    def append_message(self, context: str, content: str, css_class="log"):
        self.counter += 1
        div_id = f"msg_{self.counter}"
        div_html = f'<div id="{div_id}" class="{css_class}">{context} : {content}</div>'
        self.divs.append(div_html)

        # Enforce div limit
        if len(self.divs) > self.max_divs:
            self.divs.pop(0)

        # Update display
        full_html = self.template.replace( "{content}",''.join(self.divs))        

        #print(f"HTML = {full_html}")
        self.html_frame.load_html(full_html)
        
    def on_send(self):
        user_message = self.entry.get()
        self.append_message( "USER", user_message, css_class="user")
        self.controller.send_user_message(user_message)

    def append_response(self, message):
        self.append_message( "BOFH", message, css_class="assistant")

    def clear(self):
        self.divs.clear()
        self.counter = 0
        self.html_frame.load_html("<html><body></body></html>")
        
class BOFHInferenceEngine:
    def __init__(self, model_path, server_bin, port=9199, config=None):
        self.model_path = model_path
        self.server_bin = server_bin
        self.port = port
        self.config = config or {
            "temperature": 0.8,
            "repeat_penalty": 1.1,
            "max_tokens": 512,
            "top_p": 0.95
        }
        self.proc = None

    def start(self):
        if self.proc and self.proc.poll() is None:
            return  # Already running
        args = [
            self.server_bin,
            "--model", self.model_path,
            "--port", str(self.port),
            "--n_gpu_layers", "35",
            "--log-disable"
        ]
        # self.proc = subprocess.Popen(
        #     args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        # )
        self.proc = subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        try:
            outs, errs = self.proc.communicate(timeout=5)
            if self.proc.returncode != 0:
                print("[Error] Server failed to start.")
                print("STDERR:", errs)
        except subprocess.TimeoutExpired:
            # If it's still running, good — swallow logs later
            print("[Info] Server appears to be alive.")
        



    def stop(self):
        if self.proc:
            self.proc.terminate()
            self.proc.wait()
        self.proc = None

    def is_alive(self):
        try:
            r = requests.get(f"http://127.0.0.1:{self.port}/health", timeout=2)
            return r.ok
        except:
            return False

    def ask(self, prompt: str, callback):
        def run():
            payload = {
                "prompt": prompt,
                "temperature": self.config["temperature"],
                "top_p": self.config["top_p"],
                "repeat_penalty": self.config["repeat_penalty"],
                "n_predict": self.config["max_tokens"],
                "stream": False,
                "stop":["###"]
            }
            try:
                response = requests.post(
                    f"http://127.0.0.1:{self.port}/completion",
                    json=payload, timeout=30
                )
                result = response.json().get("content", "")
            except Exception as e:
                result = f"[Error] BOFH server failed: {e}"
            print(f"CHAT result = {result}")
            callback(result)

        threading.Thread(target=run, daemon=True).start()