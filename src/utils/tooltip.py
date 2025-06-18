import customtkinter as ctk


class Tooltip:
    """Exibe texto informativo quando o cursor passa sobre um widget."""

    def __init__(self, widget, text: str, delay: int = 500) -> None:
        self.widget = widget
        self.text = text
        self.delay = delay
        self.tooltip_win = None
        self.after_id = None

        self.widget.bind("<Enter>", self._schedule)
        self.widget.bind("<Leave>", self._hide)

    def _schedule(self, _event=None) -> None:
        self._unschedule()
        self.after_id = self.widget.after(self.delay, self._show)

    def _unschedule(self) -> None:
        if self.after_id:
            self.widget.after_cancel(self.after_id)
            self.after_id = None

    def _show(self) -> None:
        if self.tooltip_win or not self.text:
            return
        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 1
        self.tooltip_win = ctk.CTkToplevel(self.widget)
        self.tooltip_win.wm_overrideredirect(True)
        self.tooltip_win.wm_geometry(f"+{x}+{y}")
        label = ctk.CTkLabel(
            self.tooltip_win,
            text=self.text,
            fg_color="gray15",
            text_color="white",
            corner_radius=5,
            padx=5,
            pady=3,
        )
        label.pack()

    def _hide(self, _event=None) -> None:
        self._unschedule()
        if self.tooltip_win:
            self.tooltip_win.destroy()
            self.tooltip_win = None
