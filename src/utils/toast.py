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
        self.master.update_idletasks()
        master_width = self.master.winfo_width()
        master_height = self.master.winfo_height()
        master_x = self.master.winfo_x()
        master_y = self.master.winfo_y()

        self.update_idletasks()
        width = self.winfo_width()
        height = self.winfo_height()

        # Default to bottom-right of the primary screen
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        x = screen_width - width - 40
        y = screen_height - height - 60

        self.geometry(f"+{int(x)}+{int(y)}")

        self.after(self.duration, self.destroy)
