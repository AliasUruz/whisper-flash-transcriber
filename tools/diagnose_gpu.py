import logging
import sys
import traceback
from faster_whisper import WhisperModel

logging.basicConfig(level=logging.INFO)

def check_gpu():
    print("--- DIAGNOSTIC START ---")
    try:
        import ctranslate2
        print(f"CTranslate2 version: {ctranslate2.__version__}")
        print(f"CUDA available (ctranslate2): {ctranslate2.get_cuda_device_count() > 0}")
    except Exception as e:
        print(f"Error checking ctranslate2: {e}")

    print("Attempting to init WhisperModel with device='cuda'...")
    try:
        # Tenta carregar um modelo pequeno ou apenas inicializar a classe se possível (mas precisa de modelo)
        # Vamos tentar carregar o tiny para ser rápido, apenas para ver se o backend sobe.
        # Se falhar, vai falhar antes de baixar se for erro de DLL.
        model = WhisperModel("tiny", device="cuda", compute_type="float16")
        print("SUCCESS: WhisperModel initialized on CUDA.")
    except Exception as e:
        print("FAILURE: Could not initialize on CUDA.")
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Message: {e}")
        traceback.print_exc()
    print("--- DIAGNOSTIC END ---")

if __name__ == "__main__":
    check_gpu()
