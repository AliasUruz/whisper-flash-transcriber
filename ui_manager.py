import logging
import customtkinter as ctk

class UIManager:
    """Gerencia janelas da interface usando CustomTkinter."""

    def __init__(self, main_tk_root):
        self.main_tk_root = main_tk_root
        self.live_window = None
        self.live_textbox = None

    def show_live_transcription_window(self):
        """Exibe uma janela semi-transparente com um textbox para transcrição ao vivo."""
        if self.live_window and self.live_window.winfo_exists():
            return

        self.live_window = ctk.CTkToplevel(self.main_tk_root)
        self.live_window.overrideredirect(True)
        self.live_window.geometry("400x150+50+50")
        self.live_window.attributes("-alpha", 0.85)
        self.live_window.attributes("-topmost", True)

        self.live_textbox = ctk.CTkTextbox(self.live_window, wrap="word", activate_scrollbars=True)
        self.live_textbox.pack(fill="both", expand=True)
        self.live_textbox.insert("end", "Ouvindo...")

    def update_live_transcription(self, new_text):
        if self.live_textbox and self.live_window.winfo_exists():
            if self.live_textbox.get("1.0", "end-1c") == "Ouvindo...":
                self.live_textbox.delete("1.0", "end")
            self.live_textbox.insert("end", new_text + " ")
            self.live_textbox.see("end")

    def update_live_transcription_threadsafe(self, new_text):
        if self.main_tk_root:
            self.main_tk_root.after(0, lambda: self.update_live_transcription(new_text))

    def close_live_transcription_window(self):
        if self.live_window:
            try:
                self.live_window.destroy()
            except Exception as e:
                logging.error(f"Erro ao fechar janela ao vivo: {e}")
            finally:
                self.live_window = None
                self.live_textbox = None
