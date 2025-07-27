import time
import constants
import traceback
import dearpygui.dearpygui as dpg
from collections import deque
from typing import Dict, Any


BASE_THEME = [
    {"bg_color"     :(30,30,40)}, 
    {"text_color"   :(210,210,220)}, 
    {"btn_color"    :(60,90,120)},
    {"accent_color" :(80,110,140)}
]
    
class GraphDataCache:


    STALE_THRESHOLD_SECONDS = 600  # 10 minutes

    def __init__(self, max_length=500):
        self.max_length = max_length
        self.data: Dict[str, deque] = {
            # Step-based
            constants.SERIES_LOSS: deque(maxlen=max_length),
            constants.SERIES_GRADNORM: deque(maxlen=max_length),
            constants.SERIES_LEARNINGRATE: deque(maxlen=max_length),
            # Timestamp-based
            constants.SERIES_GPUALLOCATED: deque(maxlen=max_length),
            constants.SERIES_GPURESERVED: deque(maxlen=max_length),
            constants.SERIES_RAMUSED: deque(maxlen=max_length)
        }

    def addData(self, data: dict):
        if data["tag"]:
            series = data["tag"]
            if series in constants.SERIES_STEPS:
                self.data[series].append((data["step"], data["value"]))
                self._purge_stale(series)
            elif series in constants.SERIES_TIME:
                self.data[series].append((data["timestamp"], data["value"]))
                self._purge_stale(series)
            else:
                print(f"Graph Cache - ignoring series {series}")

    def _purge_stale(self, series: str):
        # Removes entries older than the set thresholdf in seconds from timestamp-based series.
        now = time.time()
        trimmed = deque((ts, val) for ts, val in self.data[series]
                        if now - ts <= self.STALE_THRESHOLD_SECONDS)
        self.data[series] = deque(trimmed, maxlen=self.max_length)

    def get_series(self, series: str) -> list[tuple[Any, float]]:
        result = self.data.get(series)
        if not result:
            result = [(0,0.0)]
        
        try:
            x,y=zip(*result)
        except ValueError:
            x,y=[0],[0.0]
            
        return x,y

    def set_max_length(self, new_length: int):
        self.max_length = new_length
        for key in self.data:
            old_data = list(self.data[key])
            self.data[key] = deque(old_data[-new_length:], maxlen=new_length)
            
class StatusCircle:
    """
    A simple drawable status indicator that renders a colored circle
    and lets you switch between fault/ok/warn/unknown states.
    """
    STATUS_FAULT = "fault"
    STATUS_OK = "ok"
    STATUS_WARN = "warn"
    STATUS_UNKNOWN = "unknown"
    
    # RGBA colors for each status
    STATUS_COLORS = {
        STATUS_FAULT:   (255,   0,   0, 255),   # red
        STATUS_OK:      (  0, 255,   0, 255),   # green
        STATUS_WARN:    (255, 165,   0, 255),   # orange
        STATUS_UNKNOWN: (128, 128, 128, 255),   # grey
    }

    def __init__(self, parent, size=20, status="unknown", tag=None):
        """
        parent : int
            DearPyGui container (window, group, etc.) to attach the drawlist.
        size : int
            Width and height of the draw area (a square).
        status : str
            Initial status; one of "fault", "ok", "warn", "unknown".
        tag : str, optional
            Unique tag for the circle item (auto‐generated if omitted).
        """
        self.size = size
        self.radius = size // 2
        self.center = (self.radius, self.radius)
        self.status = status
        self.tag = tag or f"status_circle_{id(self)}"

        # 1) Create a small drawlist for our circle
        self.drawlist = dpg.add_drawlist(
            width=size, height=size, parent=parent, tag=f"{self.tag}_dl"
        )

        # 2) Draw the circle, save its item ID
        col = StatusCircle.STATUS_COLORS.get(status, StatusCircle.STATUS_COLORS["unknown"])
        self.circle_id = dpg.draw_circle(
            self.center,
            self.radius,
            color=col,
            fill=col,
            thickness=1,
            parent=self.drawlist,
            tag=self.tag
        )

    def setFault(self):
        self.setStatus(self.STATUS_FAULT)
    def setOk(self):
        self.setStatus(self.STATUS_OK)
    def setWarn(self):
        self.setStatus(self.STATUS_WARN)
    def setUnknown(self):
        self.setStatus(self.STATUS_UNKNOWN)
        
    def setStatus(self, status):
        """
        Change the circle’s color to reflect a new status.
        status : str
            One of "fault", "ok", "warn", "unknown".
        """
        col = StatusCircle.STATUS_COLORS.get(status, StatusCircle.STATUS_COLORS["unknown"])
        dpg.configure_item(self.circle_id, color=col, fill=col)
        self.status = status

def create_theme(bg_color=(30,30,40), text_color=(210,210,220), btn_color=(60,90,120), accent_color=(80,110,140)):
    with dpg.theme() as theme:
        with dpg.theme_component(dpg.mvAll):
            dpg.add_theme_color(dpg.mvThemeCol_WindowBg, bg_color)
            dpg.add_theme_color(dpg.mvThemeCol_Text, text_color)
            dpg.add_theme_color(dpg.mvThemeCol_Button, btn_color)
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, accent_color)
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, [c - 20 for c in accent_color])
            dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 6)
            dpg.add_theme_style(dpg.mvStyleVar_ItemSpacing, 8, 4)
            dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 10, 10)
    return theme

def set_theme( theme ):
    dpg.bind_theme( theme )
    
def get_dialog_theme():
    with dpg.theme() as theme:
        with dpg.theme_component(dpg.mvAll):
            dpg.add_theme_color(dpg.mvThemeCol_WindowBg, (40, 44, 52, 240))  # Charcoal overlay
            dpg.add_theme_color(dpg.mvThemeCol_Border, (250, 200, 0, 255))    # Bold amber border
            dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 15, 15)
            dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 6)
            dpg.add_theme_style(dpg.mvStyleVar_ItemSpacing, 10, 10)
    return theme    

def bind_font( font="SEGUIEMJ.TTF"):
    from pathlib import Path

    #font_path = constants.get_font_path("NotoColorEmoji.ttf")
    font_path = constants.get_font_path(font)
    
    assert Path(font_path).is_file(), f"Font {font_path} not found or broken!"
    
    try:
        with dpg.font_registry():
            emoji_font = dpg.add_font( font_path, 14 )
            dpg.bind_font( emoji_font )            
    except Exception as e:
        print(f"Font error: {e}")
        traceback.print_exc()   # <- This prints the C-API error message


def emoji_range():
    blocks = [
        (0x1F300, 0x1F5FF),
        (0x1F600, 0x1F64F),
        (0x1F680, 0x1F6FF),
        (0x1F700, 0x1F77F),
        (0x1F780, 0x1F7FF),
        (0x1F900, 0x1F9FF),
        (0x1FA00, 0x1FA6F),
        (0x2600,  0x26FF),
        (0x2700,  0x27BF),
    ]
    glyphs = []
    for start, end in blocks:
        glyphs += list(range(start, end + 1))
    return glyphs

