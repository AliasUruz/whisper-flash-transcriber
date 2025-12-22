from pathlib import Path
import sys

# Constants
COLOR_BG = "#202028"          # Mica Alt Dark
COLOR_SECTION_BG = "#2A2A35"  # Subtle Lighter Block
COLOR_TEXT_PRIMARY = "#FFFFFF"
COLOR_TEXT_SECONDARY = "#AAAAAA"
COLOR_ACCENT = "#00F0FF"      # Neon Cyan
COLOR_BORDER = "#3A3A45"
BORDER_RADIUS = 10

def get_asset_path(filename: str) -> str:
    if hasattr(sys, "_MEIPASS"):
        base = Path(sys._MEIPASS)
    else:
        # Assuming src/ui/theme.py -> src/assets
        base = Path(__file__).parent.parent.parent / "src"
    return str(base / "assets" / filename)
