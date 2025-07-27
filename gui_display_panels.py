import dearpygui.dearpygui as dpg
from datetime import datetime
from collections import deque
import gui_utils
import constants

class DisplayPanel:
    def __init__(self):
        
        self._messages_panel =  MessagesPanel(parent="active_content")
        self._graph_panel = GraphPanel()
        self._chat_panel = ChatPanel()
        
        self.panels = {
            "Messages": self._messages_panel,
            "Graphs": self._graph_panel,
            "Chat": self._chat_panel
        }
        
        self.active_panel = "Messages"

    def render(self):
        with dpg.group(horizontal=False):

            # Navigation buttons for switching panels
            with dpg.group(horizontal=True):
                for name in self.panels:
                    def make_callback(panel_name):
                        def callback(sender, app_data):
                            self.switch_panel(panel_name)
                        return callback
                
                    dpg.add_button(
                        label=name,
                        width=140,
                        tag=f"panel_btn_{name}",
                        callback=make_callback(name)
                    )

            
            dpg.add_separator()

            # Pre-create active content container
            dpg.add_child_window(tag="active_content", autosize_x=True, autosize_y=True)

            # Initial panel render
            self._render_active_panel()

    def switch_panel(self, panel_name):
        if panel_name not in self.panels:
            print(f"[WARN] Unknown panel: {panel_name}")
            return

        self.active_panel = panel_name
        dpg.delete_item("active_content", children_only=True)        
        self._render_active_panel()

    def _render_active_panel(self):
        with dpg.group(parent="active_content"):            
            self.panels[self.active_panel].render()

    def messages_panel(self):
        return self._messages_panel
    def graph_panel(self):
        return self._graph_panel
            
class MessagesPanel:
    def __init__(self, parent):
        self.parent          = parent
        self.MAX_RECORDS     = 1000
        self.RECORDS         = deque(maxlen=self.MAX_RECORDS)
        self.container_tag   = "messages_panel_container"
        self.table_tag       = "message_table"
        self.scroll_tag       = "table_scroll"
        
    def render(self):

        with dpg.group(tag=self.container_tag, parent=self.parent):
            with dpg.child_window(tag=self.scroll_tag, width=600, height=300, autosize_x=True, autosize_y=True, horizontal_scrollbar=True):
                with dpg.table(
                    tag=self.table_tag,
                    header_row=True,
                    resizable=True,
                    policy=dpg.mvTable_SizingStretchProp,
                    row_background=True,
                    borders_innerV=True
                ):
                    dpg.add_table_column(
                        label="Timestamp",
                        init_width_or_weight=150,
                        width_fixed=True
                    )
                    dpg.add_table_column(
                        label="Message",
                        init_width_or_weight=1,
                        width_fixed=False
                    )

            # Populate any pre-existing records
            for entry in self.RECORDS:
                self._append_row(entry)

    def _append_row(self, entry):
        """Helper to add exactly one row to the existing table."""
        with dpg.table_row(parent=self.table_tag):
            # Timestamp
            dpg.add_text(entry["time"])
            # Content: dict or scalar
            content = entry["content"]
            if isinstance(content, dict):
                if "tag" in content and "message" in content:
                    dpg.add_text(f"{content['tag']} : {content['message']}")
                else:
                    # flatten all key/value pairs
                    combined = " | ".join(f"{k}:{v}" for k, v in content.items())
                    dpg.add_text(combined)
            else:
                dpg.add_text(str(content))

        dpg.set_frame_callback(dpg.get_frame_count() + 2, self._scroll_to_bottom)

        
    def _scroll_to_bottom(self):
        dpg.set_y_scroll(self.scroll_tag, dpg.get_y_scroll_max(self.scroll_tag))
        
    def add_message(self, message):
        """
        Append a new entry (timestamp + content).  
        Updates the UI immediately if visible, 
        and deletes the oldest row if at capacity.
        """
        timestamp = datetime.now().strftime("%H:%M:%S")
        entry     = {"time": timestamp, "content": message}

        # Will the oldest record be dropped?
        will_drop = len(self.RECORDS) == self.RECORDS.maxlen
        self.RECORDS.append(entry)

        # Only touch the UI if the container exists and is shown
        if (dpg.does_item_exist(self.container_tag)
        and dpg.is_item_shown(self.container_tag)):

            # Handle overflow: delete first (oldest) row
            if will_drop:
                rows = dpg.get_item_children(self.table_tag, slot=1) or []
                if rows:
                    dpg.delete_item(rows[0])

            # Append the new row
            self._append_row(entry)

    def clear(self):
        """Clear all in-memory records and remove all table rows."""
        self.RECORDS.clear()
        if dpg.does_item_exist(self.table_tag):
            rows = dpg.get_item_children(self.table_tag, slot=1) or []
            for row in rows:
                dpg.delete_item(row)

    
class ChatPanel:
    def render(self):
        with dpg.group(horizontal=True):
            dpg.add_input_text(label="", hint="Ask a question...", tag="chat_input", width=400)
            dpg.add_button(label="Submit", callback=self.submit_chat)

    def submit_chat(self):
        text = dpg.get_value("chat_input")
        self.respond_to_query(text)
        
    def respond_to_query( self,text):
        print(f"PLACE HOLDER {text}")
        
        
        
        
class GraphPanel:
    def __init__(self):
        self.cache = gui_utils.GraphDataCache()
        self.animation_enabled = False
        self.metrics = [
            (constants.SERIES_LOSS, "Loss", "Step", "Loss"),
            (constants.SERIES_GRADNORM, "GradNorm", "Step", "Grad Norm"),
            (constants.SERIES_LEARNINGRATE, "Learning Rate", "Step", "LR"),
            (constants.SERIES_GPUALLOCATED, "GPU Allocated", "Time", "MB"),
            (constants.SERIES_GPURESERVED, "GPU Reserved", "Time", "MB"),
            (constants.SERIES_RAMUSED, "RAM Usage", "Time", "MB")
        ]
        
    def render(self):
        with dpg.child_window(tag="GraphPanel", autosize_x=True, autosize_y=True, border=False):
            for series, title, x_label, y_label in self.metrics:
                xtag = f"x{x_label}{series}"
                ytag = f"y{y_label}{series}"
                ptag = f"{title}_series"
                with dpg.collapsing_header(label=title, default_open=True):
                    with dpg.plot(height=250, width=-1):
                        if series in constants.SERIES_TIME:
                            x_axis = dpg.add_plot_axis(dpg.mvXAxis, label=x_label, tag=xtag,time=True)
                        else:
                            x_axis = dpg.add_plot_axis(dpg.mvXAxis, label=x_label, tag=xtag)
                        y_axis = dpg.add_plot_axis(dpg.mvYAxis, label=y_label, tag=ytag)

                        x_data, y_data = self.cache.get_series(series)
                        dpg.add_line_series(x_data, y_data, label=title, parent=y_axis, tag=ptag)

    def updateData(self, data: dict):
        if data["tag"]:
            self.cache.addData( data )
        self._update()
    
    def _update(self):
        for series, title, x_label, y_label in self.metrics:
            xtag = f"x{x_label}{series}"
            ytag = f"y{y_label}{series}"
            ptag = f"{title}_series"
            try:
                if dpg.does_item_exist(ptag):
                    x_data, y_data = self.cache.get_series(series)
                    dpg.set_value(ptag, [x_data, y_data])
                    dpg.fit_axis_data(xtag)
                    dpg.fit_axis_data(ytag)
            except Exception as e:
                print(f"_update failed with {e}")
                
