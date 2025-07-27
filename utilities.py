import unicodedata
import platform
from datetime import timedelta

def format_eta(seconds):
    delta = timedelta(seconds=int(seconds))
    days = delta.days
    hours, rem = divmod(delta.seconds, 3600)
    mins, secs = divmod(rem, 60)
    return f"{days}d {hours:02d}:{mins:02d}:{secs:02d}" if days else f"{hours:02d}:{mins:02d}:{secs:02d}"

def sanitize(text):
    # Normalize and replace rogue bytes early
    return (
        unicodedata.normalize("NFKC", text)
        .encode("utf-8", "replace")
        .decode("utf-8")
        .replace("\x8d", "")  # kill known decoder bombs
        .replace("\x8f", "")
    )


def get_emoji_font() -> str:
    return {
        "Windows": "Segoe UI Emoji",
        "Darwin": "Apple Color Emoji",
        "Linux": "Noto Color Emoji"
    }.get(platform.system(), "Segoe UI")

def safe_font_test(text_area):
    for font in ["Segoe UI Emoji", "Arial", "Noto Color Emoji", "Symbola"]:
        try:
            text_area.config(font=(font, 12))
            text_area.insert("end", f"Testing font: {font} 🚀😎🔥\n", "emoji")
        except Exception as e:
            print(f"{font} not usable: {e}")