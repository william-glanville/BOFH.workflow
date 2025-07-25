import time
import tkinter as tk
import TKinterModernThemes as Theme
import matplotlib
import telemetry
import constants
import queue
import html
import chat

from tkinter import ttk, messagebox, Button
from utilities import get_emoji_font
from telemetry import TelemetryTCPServer

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinterweb import HtmlFrame
from chat import BOFHChatGUI, BOFHChatController, BOFHInferenceEngine

matplotlib.use("TkAgg")

TELEMETRY_MESSAGE = """<div class="telemetry_message"><strong>{tag}</strong>: {message}</div>"""

class WorkflowApp(Theme.ThemedTKinterFrame):
    def __init__(self, title="BOFH WORKFLOW", theme="azure", mode="dark"):
        super().__init__(title, theme, mode)
        
        self.master.protocol("WM_DELETE_WINDOW", self.close_app)
        self.telemetry_queue = queue.Queue()
        self.master.after(100, self.check_telemetry_queue)
        self.telemetry_server = TelemetryTCPServer(callback=self.process_telemetry)
        self.telemetry_server.start()

        # Create the model engine
        self.engine = BOFHInferenceEngine(chat.MODEL_PATH, chat.LLAMA_SERVER)
        self.engine.start()

        # Instantiate the chat controller
        self.chat_controller = BOFHChatController( model_engine=self.engine )

        # Frames
        self.content_frame = tk.Frame(self.master)
        self.content_frame.pack(fill="both", expand=True)

        self.status_frame = tk.Frame(self.master)
        self.status_frame.pack(fill="x", padx=10)

        # Left Panel (Workflow Buttons)
        self.button_frame = tk.LabelFrame(
            self.content_frame,
            text="Workflow",
            padx=10,
            pady=10,
            bg="#2e2e2e",
            fg="white"
        )        
        self.button_frame.pack(side="left", fill="y", padx=(10, 5), pady=10)

        self.main_view = MainView(self.content_frame,self.chat_controller)
        self.main_view.pack(side="right", fill="both", expand=True)
        self.chat_controller.console_renderer = self.main_view.chat().append_response
        
        # Status + Controls
        ttk.Style().configure("thick.Horizontal.TProgressbar", thickness=20)

        self.progress_label = tk.Label(self.status_frame, text="Idle", anchor="w")
        self.progress_label.pack(fill="x")

        self.progress = ttk.Progressbar(self.status_frame, orient='horizontal', mode='determinate')
        self.progress.pack(fill="x", pady=(0, 10))


        self.telemetry_frame = tk.Frame(self.status_frame, bg="#1e1e1e")
        self.telemetry_frame.pack(fill="x", pady=(4, 0), anchor="e")  # Align to right


        self.gpu_status_label = tk.Label(
            self.telemetry_frame,
            text="GPU: --",
            font=( get_emoji_font(), 10),
            fg="#cccccc",
            bg="#1e1e1e"
        )
        self.gpu_status_label.pack(side="left", padx=(10, 0))

        self.mem_bar = tk.Label(
            self.telemetry_frame,
            text="🧠 GPU 0.00 GB | 🖥️ RAM 0.00 GB @ --:--:--",
            anchor="w",
            font=(get_emoji_font(), 10),
            bg="#1e1e1e",
            fg="#cccccc"
        )
        self.mem_bar.pack(side="left", padx=(10, 0))

        self.telemetry_led = tk.Canvas(self.telemetry_frame, width=16, height=16, highlightthickness=0, bg="#2e2e2e")
        self.telemetry_led_id = self.telemetry_led.create_oval(2, 2, 14, 14, fill="red", outline="black")
        self.telemetry_led.pack(side="right", padx=(5, 10))

        self.health_bar = tk.Canvas(self.telemetry_frame, width=120, height=16, bg="#222222", highlightthickness=0)
        self.health_rect = self.health_bar.create_rectangle(0, 0, 120, 16, fill="red", outline="")
        self.health_bar.pack(side="right", padx=(0, 0))

        self.health_label = tk.Label(self.telemetry_frame, text="Telemetry", fg="#cccccc", bg="#1e1e1e")
        self.health_label.pack(side="right", padx=(0, 10))

        self.cancel_button = tk.Button(self.master, text="Cancel", bg="red", fg="white")
        self.cancel_button.pack(pady=(0, 5))

        self.clear_button = tk.Button(self.master, text="Clear Console", bg="seagreen", fg="white")
        self.clear_button.pack(pady=(0, 10))
        


        self.last_telemetry_time = time.time()

    def update_health_bar(self):
        # Fade over 5 minutes (300 seconds)
        max_age = 300
        age = time.time() - self.last_telemetry_time
        age = min(age, max_age)
    
        # Compute fade color from green → red
        green = max(0, int(255 * (1 - age / max_age)))
        red = min(255, int(255 * (age / max_age)))
        color = f"#{red:02x}{green:02x}00"
    
        self.health_bar.itemconfig(self.health_rect, fill=color)
        self.master.after(1000, self.update_health_bar)

    def _set_telemetry_led(self, status: dict):
        color = {
            "online": "green",
            "offline": "red",
            "warning": "orange"
        }.get(status["state"], "red")
    
        self.telemetry_led.itemconfig(self.telemetry_led_id, fill=color)
        
    def process_telemetry(self, packet):
        self.telemetry_queue.put(packet)

    def check_telemetry_queue(self):
        try:
            while True:
                packet = self.telemetry_queue.get_nowait()
                self._handle_packet(packet)  # Your existing logic inside dispatch()
        except queue.Empty:
            pass
        self.master.after(100, self.check_telemetry_queue)  # Keep polling

    def _handle_packet(self, packet):
        self.last_telemetry_time = time.time()            
        if packet.get("type") == telemetry.TYPE_MEMORY:
            self._update_memory_bar(packet["data"])
        elif packet.get("type") == telemetry.TYPE_CONNECTION:
            self._set_telemetry_led(packet["data"])
        elif packet.get("type") == "loss":
            self.progress_label.config(text=f"📉 Loss: {packet['data']:.4f}")
        elif packet.get("type") == telemetry.TYPE_ALERT:
            messagebox.showwarning("Trainer Alert", packet["data"])
        elif packet.get("type") == telemetry.TYPE_GPU:
            self._update_gpu_status(packet["data"])
        elif packet.get("type") == telemetry.TYPE_BANNER:
            self._log( self.get_banner_html(packet["data"]["message"]) )
        elif packet.get("type") == telemetry.TYPE_PROGRESS:
            self._update_progress_bar(packet["data"])
        elif packet.get("type") == telemetry.TYPE_TRAINING:
            self._on_training_telemetry(packet["data"])
        elif packet.get("type") in [telemetry.TYPE_DISPLAY, telemetry.TYPE_DIAGNOSTICS, telemetry.TYPE_ENVIRONMENT]:
            self._log(packet["data"])
        else:
            self._log(f"No telemetry handling for type {packet.get('type')}")

    def on_display(self, msg: dict):
        tag = msg.get("tag")
        if not tag:
            return
        
        packet = {
            "type": telemetry.TYPE_DISPLAY,
            "data": msg
        }
        self.process_telemetry( packet )
        
    def _on_training_telemetry(self, msg: dict):
        tag = msg.get("tag")
        if not tag:
            return
        self._plot_training_telemetry(tag,msg)
    
    def _plot_training_telemetry(self, tag, msg: dict):
        x = msg.get("step", len(self.main_view.graph().data[tag]["x"]))
        y = None
        for k, v in msg.items():
            if k in ("tag", "step"):
                continue
            if isinstance(v, (int, float)):
                y = v
                break
        if y is None:
            return

        self.main_view.graph().update_graph(tag, x, y)
        
    def _update_progress_bar(self, stats: dict):
        tag = stats.get("tag", "")
        progress = stats.get("progress", 0)
        total = stats.get("total", 100)
        value = (progress/total)*100        
        self.progress["value"] = value
        self.progress_label.config(text=f"{tag} Progress: {value:.1f}%")
        
    def _update_memory_bar(self, stats: dict):
        timestamp = time.strftime("%H:%M:%S")  # Generate live timestamp
        tag = stats.get("tag", "")
    
        gpua = stats.get("gpu_allocated", 0)
        gpur = stats.get("gpu_reserved", 0)
        ram = stats.get("ram_used", 0)
    
        formatted = (
            f"🧠 GPU Allocated {gpua:.2f} GB | "
            f"🧠 GPU Reserved {gpur:.2f} GB | "
            f"🖥️ RAM {ram:.2f} GB @ {timestamp} | "
            f"{tag}"
        )
    
        self.mem_bar.config(text=formatted)


    def _update_gpu_status(self, data):
        status_text = "Available ✅" if data.get("enabled") else "Unavailable ❌"
        self.gpu_status_label.config(text=f"GPU: {status_text}")
    
        if "error" in data:
            self._log(f"[GPU Error] {data['error']}")
        else:
            self._log(f"[GPU Status] {status_text}")

    def close_app(self):
        if messagebox.askyesno("Exit Confirmation", "Are you quite sure you want to shut down this indispensable masterpiece of automation?"):
            self.master.destroy()
            self.master.quit()

    def set_buttons_state(self, state="normal"):
        for child in self.button_frame.winfo_children():
            if isinstance(child, tk.Button) and child != self.cancel_button:
                child.configure(state=state)

    def clear_console(self):
        self.main_view.console().clear()


    def _log(self, message):
        if isinstance(message, dict):
            text = self._process_dictionary(message)
        else:
            text = self.sanitize_html(str(message))
        
        self.main_view.console().append_message(text,css_class="log")

    def _process_dictionary( self, data: dict ):
        keys = set(data.keys())
        text = ""
        if keys == {"tag", "message"}:
            tag = data["tag"]
            message = data["message"]
            text = TELEMETRY_MESSAGE.format(tag=tag,message=message)
        else:
            text = self.dict_to_html_table(data)
        return text
    
    def sanitize_html(self, text: str ):
        return text.encode("utf-8", errors="replace").decode("utf-8")

    def dict_to_html_table(self, data: dict):
        rows = []
        for key, value in data.items():
            rows.append(
                f"<tr><td><strong>{html.escape(str(key))}</strong></td><td>{html.escape(str(value))}</td></tr>"
            )
    
        return f"""
            <table id="info_table" border="1" cellspacing="0" cellpadding="6" style="width:100%; border-collapse: collapse;">
                <thead>
                    <tr><th>Key</th><th>Value</th></tr>
                </thead>
                <tbody>
                    {''.join(rows)}
                </tbody>
            </table>
        """.strip()
        
    def get_banner_html(self, message: str):
        return f"<div class='banner'>{message}</div>".strip()


class MainView(tk.Frame):
    def __init__(self, root, chat_controller):
        super().__init__(root, bg="#2e2e2e", bd=2, relief="groove")
        
        self.pack(fill="both", expand=True)

        # Initialize all views but only show one at a time
        self.console_view = HTMLConsole(self)
        self.graph_view = GraphView(self, [telemetry.TAG_LOSS, telemetry.TAG_LR, telemetry.TAG_GRADNORM, telemetry.TAG_VALLOSS])
        self.chat_view = BOFHChatGUI(self, controller=chat_controller)

        chat_controller.console_renderer = self.chat_view.append_response
        
        # Button bar
        self.button_bar = tk.Frame(self)
        self.button_bar.pack(side="bottom", fill="x")

        Button(self.button_bar, text="Console", command=lambda: self.switch_view(self.console_view)).pack(side="left")
        Button(self.button_bar, text="Graph", command=lambda: self.switch_view(self.graph_view)).pack(side="left")
        Button(self.button_bar, text="Chat", command=lambda: self.switch_view(self.chat_view)).pack(side="left")

        # Default: show console
        self.active_view = self.console_view
        #self.active_view.pack(fill="both", expand=True)
        #self.view_frame = tk.Frame(self.content_frame, bg="#2e2e2e", bd=2, relief="groove")
        self.active_view.pack(side="right", fill="both", expand=True, padx=(10, 10), pady=(10, 10))

    def console(self):
        return self.console_view

    def graph(self):
        return self.graph_view

    def chat(self):
        return self.chat_view
    
    def switch_view(self, target_view):
        if self.active_view == target_view:
            return  # No change needed

        self.active_view.pack_forget()
        self.active_view = target_view
        #self.active_view.pack(fill="both", expand=True)
        self.active_view.pack(side="right", fill="both", expand=True, padx=(10, 10), pady=(10, 10))

class GraphView(ttk.Frame):
    def __init__(self, parent, graph_names, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.graph_names = graph_names
        self.data = {name: {'x': [], 'y': []} for name in graph_names}
        self.figures = {}
        self.axes = {}
        self.lines = {}

        # Create a vertical scrollable canvas
        canvas = tk.Canvas(self)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        self.scrollable_frame = ttk.Frame(canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Create a matplotlib figure & canvas for each graph
        for name in graph_names:
            frame = ttk.LabelFrame(self.scrollable_frame, text=name)
            frame.pack(fill="both", expand=True, padx=10, pady=5)

            fig = Figure(figsize=(5, 2))
            ax = fig.add_subplot(111)
            ax.set_title(name)
            ax.set_xlabel("Step")
            ax.set_ylabel(name)
            line, = ax.plot([], [], lw=2)

            canvas_fig = FigureCanvasTkAgg(fig, master=frame)
            canvas_fig.get_tk_widget().pack(fill="both", expand=True)
            canvas_fig_widget = canvas_fig.get_tk_widget()
            canvas_fig_widget.pack(fill="both", expand=True)
            self.bind_scroll(canvas, canvas_fig_widget)

            self.figures[name] = fig
            self.axes[name] = ax
            self.lines[name] = line

    def bind_scroll(self, canvas, canvas_widget):
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
    
        def _on_linux_scroll(event, direction):
            canvas.yview_scroll(direction, "units")
    
        # Windows and macOS
        canvas_widget.bind("<Enter>", lambda e: canvas_widget.focus_set())
        canvas_widget.bind("<MouseWheel>", _on_mousewheel)
    
        # Linux
        canvas_widget.bind("<Button-4>", lambda e: _on_linux_scroll(e, -1))  # scroll up
        canvas_widget.bind("<Button-5>", lambda e: _on_linux_scroll(e, 1))   # scroll down)

    def update_graph(self, name: str, x_val: float, y_val: float):
        """Add a point to the named graph and redraw."""
        if name not in self.lines:
            # Optionally, create new graph on the fly
            self._create_graph(name)

        d = self.data.setdefault(name, {'x': [], 'y': []})
        d['x'].append(x_val)
        d['y'].append(y_val)

        line = self.lines[name]
        line.set_data(d['x'], d['y'])
        ax = self.axes[name]
        ax.relim()
        ax.autoscale_view()

        self.figures[name].canvas.draw_idle()

    def _create_graph(self, name):
        """Dynamically create a new graph if a new tag arrives."""
        frame = ttk.LabelFrame(self.scrollable_frame, text=name)
        frame.pack(fill="both", expand=True, padx=10, pady=5)

        fig = Figure(figsize=(5, 2))
        ax = fig.add_subplot(111)
        ax.set_title(name)
        ax.set_xlabel("Step")
        ax.set_ylabel(name)
        line, = ax.plot([], [], lw=2)

        canvas_fig = FigureCanvasTkAgg(fig, master=frame)
        canvas_fig.get_tk_widget().pack(fill="both", expand=True)

        self.figures[name] = fig
        self.axes[name] = ax
        self.lines[name] = line

class HTMLConsole(tk.Frame):
    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.max_divs = constants.GUI_MAX_DIVS
        self.counter = 0
        self.divs = []

        self.html_frame = HtmlFrame(
            self,
            horizontal_scrollbar="auto",
            vertical_scrollbar="auto",
            messages_enabled=False
        )
        self.html_frame.pack(fill="both", expand=True)

        # load raw template and css
        raw = open( constants.get_misc_path(constants.DS_CONSOLE_TEMPLATE), encoding="utf-8" ).read()
        css = open( constants.get_misc_path(constants.DS_CONSOLE_THEME), encoding="utf-8" ).read()

        style_tag = f"<style>\n{css}\n</style>"

        # Inject a flex‐container with reversed column flow.
        # Notice `{content}` remains as a placeholder.
        self.template = (
            raw
            .replace("{style}", style_tag)
            .replace(
                "{content}",
                # This wrapper will reverse the vertical order.
                '<div id="logframe" '
                'style="height:100%; overflow-y:auto; '
                      'display:flex; flex-direction:column-reverse;">'
                "{content}"
                "</div>"
            )
        )
        self.bind_scroll_events()
        
    def bind_scroll_events(self):
        def _on_mousewheel(event):
            self.html_frame.yview_scroll(int(-1*(event.delta/120)), "units")
    
        def _on_linux_scroll(event, direction):
            self.html_frame.yview_scroll(direction, "units")
    
        self.html_frame.bind("<Enter>", lambda e: self.html_frame.focus_set())
        self.html_frame.bind("<MouseWheel>", _on_mousewheel)
        self.html_frame.bind("<Button-4>", lambda e: _on_linux_scroll(e, -1))
        self.html_frame.bind("<Button-5>", lambda e: _on_linux_scroll(e, 1))

    def append_message(self, content: str, css_class="log"):
        self.counter += 1
        #safe = self.sanitize_html(content)
        safe = content
        # Build each div normally; order doesn’t matter,
        # flex-direction: column-reverse flips it visually.
        div = (
            f'<div id="msg_{self.counter}" '
            f'class="{css_class}">{safe}</div>'
        )
        self.divs.append(div)

        if len(self.divs) > self.max_divs:
            self.divs.pop(0)

        # Join in the literal order; flex will reverse it.
        body_html = "".join(self.divs)
        html_doc = self.template.replace("{content}", body_html)

        # Wrap load in try/except to catch any lingering decode issues
        try:
            self.html_frame.load_html(html_doc)
        except UnicodeDecodeError:
            clean = html_doc.encode("utf-8", "replace").decode("utf-8")
            self.html_frame.load_html(clean)

    def clear(self):
        self.divs.clear()
        self.counter = 0
        self.html_frame.load_html("<html><body></body></html>")

    @staticmethod
    def sanitize_html(text: str) -> str:
        # Escape anything that could break your markup,
        # and strip out characters that tkinterweb can’t decode.
        return html.escape(text).encode("utf-8", "ignore").decode("utf-8")
    