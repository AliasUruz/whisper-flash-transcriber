import customtkinter as ctk
import tkinter as tk


BaseToplevel = getattr(ctk, "CTkToplevel", tk.Toplevel)

class ToastNotification(BaseToplevel):
    def __init__(self, master, message, duration=2000):
        super().__init__(master)

        self.overrideredirect(True)
        self.lift()
        self.attributes("-topmost", True)
        self.attributes("-alpha", 0.9)

        self.label = ctk.CTkLabel(
            self,
            text=message,
            corner_radius=10,
            fg_color=("#333333", "#444444"),
            text_color=("#FFFFFF", "#FFFFFF"),
            padx=20,
            pady=10,
            font=("Segoe UI", 14)
        )
        self.label.pack(expand=True, fill="both")

        self.duration = duration

        # Position the toast notification
        parent = getattr(self, "master", None)
        master_width = master_height = master_x = master_y = 0

        if parent is not None:
            try:
                parent.update_idletasks()
                master_width = parent.winfo_width()
                master_height = parent.winfo_height()
                master_x = parent.winfo_x()
                master_y = parent.winfo_y()
            except tk.TclError:
                # Parent window might be destroyed or withdrawn; fall back
                parent = None

        self.update_idletasks()
        width = self.winfo_width()
        height = self.winfo_height()

        # Default to bottom-right of the primary screen
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        x = screen_width - width - 40
        y = screen_height - height - 60

        # If the master window has a meaningful geometry, anchor the toast
        # relative to it while clamping inside the visible screen area.
        if parent is not None and master_width > 1 and master_height > 1:
            x = master_x + master_width - width - 20
            y = master_y + master_height - height - 20
            x = max(0, min(x, screen_width - width))
            y = max(0, min(y, screen_height - height))

        self.geometry(f"+{int(x)}+{int(y)}")

        self.after(self.duration, self.destroy)
