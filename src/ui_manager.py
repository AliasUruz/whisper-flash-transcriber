import pystray
from PIL import Image, ImageDraw
import tkinter as tk
import threading
import logging
from typing import Callable

STATE_IDLE = "IDLE"
STATE_LOADING_MODEL = "LOADING_MODEL"
STATE_RECORDING = "RECORDING"
STATE_TRANSCRIBING = "TRANSCRIBING"

ICON_COLORS = {
    STATE_IDLE: ('green', 'white'),
    STATE_LOADING_MODEL: ('gray', 'yellow'),
    STATE_RECORDING: ('red', 'white'),
    STATE_TRANSCRIBING: ('blue', 'white'),
}
DEFAULT_ICON_COLOR = ('gray', 'white')

class UIManager:
    def __init__(self, on_exit: Callable):
        self.on_exit = on_exit
        self.icon = None
        self.root = tk.Tk()
        self.root.withdraw()

    def create_image(self, width: int, height: int, color1, color2=None):
        image = Image.new('RGB', (width, height), color1)
        if color2:
            dc = ImageDraw.Draw(image)
            dc.rectangle((width//4, height//4, width*3//4, height*3//4), fill=color2)
        return image

    def start_tray(self, initial_state: str, menu: pystray.Menu):
        color1, color2 = ICON_COLORS.get(initial_state, DEFAULT_ICON_COLOR)
        image = self.create_image(64, 64, color1, color2)
        tooltip = f"Whisper ({initial_state})"
        self.icon = pystray.Icon("whisper", image, tooltip, menu)
        threading.Thread(target=self.icon.run, daemon=True, name="TrayIcon").start()
        logging.info("Ícone da bandeja iniciado")

    def update_state(self, state: str):
        if not self.icon:
            return
        color1, color2 = ICON_COLORS.get(state, DEFAULT_ICON_COLOR)
        image = self.create_image(64, 64, color1, color2)
        self.icon.icon = image
        self.icon.title = f"Whisper ({state})"

    def stop(self):
        if self.icon:
            self.icon.stop()
        self.root.quit()
