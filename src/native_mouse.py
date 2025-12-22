import ctypes
from ctypes import wintypes
import threading
import logging

# --- Win32 Constants ---
WH_MOUSE_LL = 14
WM_LBUTTONDOWN = 0x0201
WM_LBUTTONUP = 0x0202
WM_RBUTTONDOWN = 0x0204
WM_RBUTTONUP = 0x0205
WM_QUIT = 0x0012

# --- Win32 Structures ---
class MSLLHOOKSTRUCT(ctypes.Structure):
    _fields_ = [
        ("pt", wintypes.POINT),
        ("mouseData", wintypes.DWORD),
        ("flags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ctypes.c_ulong)
    ]

# --- Win32 Function Prototypes ---
user32 = ctypes.windll.user32
kernel32 = ctypes.windll.kernel32

# HHOOK SetWindowsHookExA(int idHook, HOOKPROC lpfn, HINSTANCE hMod, DWORD dwThreadId);
SetWindowsHookEx = user32.SetWindowsHookExA
SetWindowsHookEx.argtypes = [ctypes.c_int, ctypes.c_void_p, wintypes.HINSTANCE, wintypes.DWORD]
SetWindowsHookEx.restype = wintypes.HHOOK

# LRESULT CallNextHookEx(HHOOK hhk, int nCode, WPARAM wParam, LPARAM lParam);
LRESULT = ctypes.c_long
CallNextHookEx = user32.CallNextHookEx
CallNextHookEx.argtypes = [wintypes.HHOOK, ctypes.c_int, wintypes.WPARAM, wintypes.LPARAM]
CallNextHookEx.restype = LRESULT

# BOOL UnhookWindowsHookEx(HHOOK hhk);
UnhookWindowsHookEx = user32.UnhookWindowsHookEx
UnhookWindowsHookEx.argtypes = [wintypes.HHOOK]
UnhookWindowsHookEx.restype = wintypes.BOOL

# BOOL GetMessageW(LPMSG lpMsg, HWND hWnd, UINT wMsgFilterMin, UINT wMsgFilterMax);
GetMessage = user32.GetMessageW
GetMessage.argtypes = [ctypes.POINTER(wintypes.MSG), wintypes.HWND, wintypes.UINT, wintypes.UINT]
GetMessage.restype = wintypes.BOOL

# SHORT GetAsyncKeyState(int vKey);
GetAsyncKeyState = user32.GetAsyncKeyState
GetAsyncKeyState.argtypes = [ctypes.c_int]
GetAsyncKeyState.restype = ctypes.c_short

# BOOL PostThreadMessageW(DWORD idThread, UINT Msg, WPARAM wParam, LPARAM lParam);
PostThreadMessage = user32.PostThreadMessageW
PostThreadMessage.argtypes = [wintypes.DWORD, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
PostThreadMessage.restype = wintypes.BOOL

# Define the callback type
HOOKPROC = ctypes.WINFUNCTYPE(LRESULT, ctypes.c_int, wintypes.WPARAM, wintypes.LPARAM)

class NativeMouseHook:
    def __init__(self, core_service):
        self.core = core_service
        self.hook_id = None
        self.thread_id = None
        self.thread = None
        self.suppress_next_up = False
        self.hook_proc = HOOKPROC(self._low_level_mouse_proc) # Keep reference alive!
        logging.info("NativeMouseHook initialized.")

    def start(self):
        if self.thread and self.thread.is_alive():
            return
        
        self.thread = threading.Thread(target=self._thread_run, daemon=True)
        self.thread.start()
        logging.info("NativeMouseHook thread started.")

    def stop(self):
        if self.thread_id:
            logging.info("Stopping NativeMouseHook...")
            # Post WM_QUIT to the hook thread to break the GetMessage loop
            PostThreadMessage(self.thread_id, WM_QUIT, 0, 0)
            if self.thread:
                self.thread.join(timeout=1.0)
            self.hook_id = None
            self.thread_id = None
            logging.info("NativeMouseHook stopped.")

    def _thread_run(self):
        # Store thread ID for PostThreadMessage
        self.thread_id = kernel32.GetCurrentThreadId()
        
        # Install Hook (hMod=None works for low-level hooks in Python)
        self.hook_id = SetWindowsHookEx(WH_MOUSE_LL, self.hook_proc, None, 0)
        
        if not self.hook_id:
            logging.error(f"Failed to install mouse hook. Error: {kernel32.GetLastError()}")
            return

        logging.info("Mouse hook installed successfully.")

        # Message Pump
        msg = wintypes.MSG()
        # GetMessage blocks until a message arrives. Returns 0 on WM_QUIT.
        while GetMessage(ctypes.byref(msg), None, 0, 0) > 0:
            user32.TranslateMessage(ctypes.byref(msg))
            user32.DispatchMessageW(ctypes.byref(msg))

        # Cleanup
        UnhookWindowsHookEx(self.hook_id)
        logging.info("Mouse hook uninstalled.")

    def _low_level_mouse_proc(self, nCode, wParam, lParam):
        if nCode >= 0:
            # wParam contains the message ID (e.g., WM_RBUTTONDOWN)
            
            if wParam == WM_RBUTTONDOWN:
                # Check if Left Mouse Button is currently held down
                # 0x01 is VK_LBUTTON. 0x8000 mask checks the "currently down" bit.
                lmb_down = GetAsyncKeyState(0x01) & 0x8000
                
                if lmb_down:
                    logging.info("Native Hook: Chord Detected (LMB+RMB). Suppressing.")
                    self.suppress_next_up = True
                    
                    # Trigger the action in a separate thread to not block the hook
                    threading.Thread(target=self._safe_toggle, daemon=True).start()
                    
                    # Return 1 to BLOCK the event
                    return 1
            
            elif wParam == WM_RBUTTONUP:
                if self.suppress_next_up:
                    logging.info("Native Hook: Suppressing RMB Up.")
                    self.suppress_next_up = False
                    return 1

        # Pass to next hook
        return CallNextHookEx(self.hook_id, nCode, wParam, lParam)

    def _safe_toggle(self):
        try:
            self.core.toggle_recording()
        except Exception as e:
            logging.error(f"Toggle failed: {e}")
