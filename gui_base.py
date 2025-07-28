import dearpygui.dearpygui as dpg
import os, sys
import constants
import gui_utils
import gui_display_panels
import queue
import time
import telemetry
import threading

from gui_utils import StatusCircle
from EditThemePlugin import EditThemePlugin
from telemetry import TelemetryTCPServer
from workflow_runner import WorkflowRunner

os.environ["PYTHONIOENCODING"] = "utf-8"

BUTTON_WIDTH = 180

class BOFHDashboard:
    def __init__(self):
        
        self.telemetry_queue = queue.Queue()
        self.telemetry_server = TelemetryTCPServer(callback=self._enqueue_telemetry)
        self.telemetry_server.start()
        
        dpg.create_context()
        gui_utils.bind_font()
        dpg.create_viewport(title="BOFH Dashboard", width=1000, height=700)
        dpg.setup_dearpygui()        
                
        self.display_panel = gui_display_panels.DisplayPanel()

        self.runner = WorkflowRunner(self, constants )
        self._busy = False
        self.progress = ProgressTracker(telemetry_fn=self._enqueue_telemetry)
        self.status = None
        
    def _schedule_next(self):
        threading.Timer(0.2, self._process_telemetry).start()

    def launch(self):
        
        with dpg.window( tag="main_window", label="BOFH Dashboard", autosize=True, no_resize=False ):    
            # Top-level grouping
            with dpg.group(horizontal=False):
    
                # Row: Info Panel
                with dpg.child_window(height=40,autosize_x=True):
                    with dpg.group(horizontal=True) as info_grp:
                        dpg.add_text("Info Panel")
                        dpg.add_input_text(label="Console")
                        dpg.add_checkbox(label="Enable Metrics")
                        dpg.add_separator()
                        self.status = StatusCircle(info_grp,tag="telemetry_led")
                        
                with dpg.child_window(tag="progress_panel", autosize_x=True, height=60):
                    dpg.add_text("Operation Progress")
                    dpg.add_progress_bar(
                        tag="progress_bar",
                        default_value=0.0,
                        width=-1,             # stretch to full width
                        overlay="0%"          # initial text overlay
                    )

                # Row: Workflow Buttons (Left) + Display Panel (Body)
                with dpg.group(horizontal=True):
                    with dpg.child_window(tag="workflow_buttons", width=200, autosize_y=True):
                        dpg.add_text("Workflow Buttons")
                        self._build_buttons()
    
                    with dpg.child_window(autosize_x=True, autosize_y=True):
                        self.display_panel.render()
        
        dpg.set_primary_window("main_window", True)
        
        self._last_telemetry = time.time()
        self.progress.setup(dpg)
        self._schedule_next()
        
        dpg.show_viewport()
        dpg.start_dearpygui()        
        dpg.destroy_context()

    def _enqueue_telemetry(self, packet):
        self.telemetry_queue.put(packet)

    def _process_telemetry(self):
        try:
            while True:
                packet = self.telemetry_queue.get_nowait()
                self._handle_packet(packet)  # Your existing logic inside dispatch()
            self._last_telemetry = time.time()
        except queue.Empty:
            pass
        self._schedule_next()

        
    def _handle_packet(self, packet):
        self.last_telemetry_time = time.time()            
        if packet.get("type") == telemetry.TYPE_MEMORY:
            self._update_memory(packet["data"])
        elif packet.get("type") == telemetry.TYPE_CONNECTION:
            self._set_telemetry_led(packet["data"])
        elif packet.get("type") == telemetry.TYPE_GPU:
            self._update_gpu_status(packet["data"])
        elif packet.get("type") == telemetry.TYPE_BANNER:
            self._log( packet["data"]["message"] ) 
        elif packet.get("type") == telemetry.TYPE_PROGRESS:
            self._update_progress(packet["data"])
        elif packet.get("type") == telemetry.TYPE_TRAINING:
            self._on_training_telemetry(packet["data"])
        elif packet.get("type") in [telemetry.TYPE_DISPLAY, telemetry.TYPE_DIAGNOSTICS, telemetry.TYPE_ENVIRONMENT,telemetry.TYPE_ALERT]:
            self._log(packet["data"])
        else:
            self._log(f"No telemetry handling for type {packet.get('type')}")

    def _on_training_telemetry(self, data:dict):
        self.display_panel.graph_panel().updateData( data )
         
    def _update_memory(self,data:dict):
        self.display_panel.graph_panel().updateData( data )
        
    def _set_telemetry_led(self, data: dict):
        self.status.setStatus(data["state"])
        
    def _update_gpu_status( self, data: dict):
        if data:
            # simple display
            if "error" in data.keys():
                self.display(data["tag"], f"GPU {data['enbaled']} - Error = {data['error']}")
            else:
                self.display(data["tag"], f"GPU {data['enabled'] }")
            
    def _update_progress(self, data: dict):
        if data:
            self.progress.update( data["tag"], int(data["progress"]), int(data["total"]) )
            
    def _confirm_action(self, label, callback):
        def launch_popup():
            if not dpg.does_item_exist("confirmation_popup"):
                with dpg.window(
                    tag="confirmation_popup",
                    modal=True,
                    popup=True,
                    no_title_bar=True,
                    width=340,
                    height=160,
                    pos=(dpg.get_viewport_width()//2 - 170, dpg.get_viewport_height()//2 - 80),
                    no_move=True,
                    menubar=False
                ):
                    # Dialog style
                    dpg.bind_item_theme("confirmation_popup", gui_utils.get_dialog_theme())
    
                    dpg.add_text(f"Confirm Execute {label}?")
                    dpg.add_spacer(height=20)
    
                    # Centered horizontal buttons
                    with dpg.group(horizontal=True):
                        dpg.add_button(label="Proceed", width=140, callback=lambda: [callback(), dpg.delete_item("confirmation_popup")])
                        dpg.add_spacer(width=10)
                        dpg.add_button(label="Cancel", width=140, callback=lambda: dpg.delete_item("confirmation_popup"))
            else:
                dpg.configure_item("confirmation_popup", show=True)

        launch_popup()

    def _block_action(self,label):
        def launch_popup():
            if not dpg.does_item_exist("busy_modal"):
                with dpg.window(tag="busy_modal", modal=True, show=True, no_title_bar=True, autosize=True):
                    dpg.add_text("System is busy. Please wait until the current task finishes.")
                    dpg.add_button(label="OK",width=75,callback=lambda s, a: dpg.configure_item("busy_modal", show=False))
            else:
                dpg.configure_item("busy_modal", show=True)

        launch_popup()

        
    def _build_buttons(self):            
        actions = [
            (constants.LABEL_DIAGNOSTICS, self.runner.launch_diagnostic),
            (constants.LABEL_COLLECTOR, self.runner.launch_collector),
            (constants.LABEL_ATTRIBUTION, self.runner.launch_attribution),
            (constants.LABEL_CONSOLIDATION, self.runner.launch_consolidation),
            (constants.LABEL_TONAL_ANALYSIS, self.runner.launch_generate_tonal_lexicon),
            (constants.LABEL_TOKENIZATION, self.runner.launch_tokenizer),
            (constants.LABEL_TOKEN_VALIDATION, self.runner.launch_validation),
            (constants.LABEL_TRAINING, self.runner.launch_training),
            (constants.LABEL_QUANTIZATION, self.runner.launch_quantization),
            (constants.LABEL_EVALUATION, self.cmd_placeholder),
            ("Close App", self.close_app),
        ]
        EditThemePlugin()
        for label, method in actions:
            dpg.add_button(label=label, width=BUTTON_WIDTH, callback=self._make_callback(label, method))
        dpg.add_separator()
        dpg.add_button(label="Cancel Running Process", width=BUTTON_WIDTH, callback=self._make_cancel_running_process())
            
    def _make_cancel_running_process(self):
        def kill():
            if not self.runner:
                self.display("SYSTEM", "No active sub-process to cancel")
                return

            try:
                self.display("SYSTEM", "Cancel active sub-process")
                self.runner.cancel_active_task()
            except Exception as e:
                    self.display("SYSTEM", f"Error cancelling sub-process: {e}")

        return kill
    
    def _make_callback(self, label, method):
        def cb(sender, app_data):
            # if we’re mid‐flight, pop up a warning
            if self._busy:
                self._block_action(label)
                return
        
            # otherwise go ahead and confirm + launch
            self.progress.reset()
            self._confirm_action(label, method)
            
        return cb
        #return lambda s, a: self._confirm_action(label, method)

    def cmd_placeholder(self):
        self.display("SYSTEM", "That button’s not wired yet. Feels like management, doesn’t it?")

    def display(self, tag, message):
        candidate = {
            "type": telemetry.TYPE_DISPLAY,
            "data": {
                "tag" : tag,
                "message" : message,
            }
         }
        self._enqueue_telemetry( candidate )
        
    def close_app(self):
        self.display("SYSTEM", "Closing app. Wrath level: Satisfied.")
        dpg.stop_dearpygui()

    def set_workflow_active(self, enabled: bool):
        self._busy = enabled

    def _log(self, message):        
        self.display_panel.messages_panel().add_message( message )
    
        
class ProgressTracker:
    def __init__( self, telemetry_fn=None ):
        self.gui = None
        self.telemetry_fn = telemetry_fn

    def setup(self, gui):
        self.gui = gui
        self.reset()
        
    def reset(self):
        self.start_time = time.time()
        self.gui.set_value("progress_bar", 0.0)
        self.gui.configure_item("progress_bar", overlay="0.0%")
        
    def update(self, tag: str, current: int, total: int):
        now = time.time()
        fraction = current / total if total else 0.0
        percent  = int(fraction * 100)

        self.gui.set_value("progress_bar", fraction)
        self.gui.configure_item("progress_bar", overlay=f"{percent}%")

        elapsed = now - self.start_time
        eta = sys.maxsize
        if current > 0 :
            eta = ((elapsed/current)*total)-elapsed
        
        eta_str = self._format_time(eta) if eta else "Calculating..."
        elapsed_str = self._format_time(elapsed)
                
        candidate = {
            "type": telemetry.TYPE_DISPLAY,
            "data": {
                "tag" : f"{tag}",
                "message" : f"Elapsed: {elapsed_str} ETA: {eta_str}",
            }
         }
        self.telemetry_fn( candidate )
        
    def _format_time(self, seconds):
        minutes, sec = divmod(int(seconds), 60)
        hours, minutes = divmod(minutes, 60)
        return f"{hours}h {minutes}m {sec}s" if hours else f"{minutes}m {sec}s" if minutes else f"{sec}s"

def main():
    gui = BOFHDashboard()
    gui.launch()
    
if __name__ == "__main__":
    main()
    