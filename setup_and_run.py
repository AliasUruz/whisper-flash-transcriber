import json
import os
from pathlib import Path
import subprocess
import sys

def setup_and_run():
    # 1. Configurar o caminho do modelo
    config_dir = Path.home() / ".whisper_flash_transcriber"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "config.json"
    
    settings = {}
    if config_path.exists():
        try:
            with open(config_path, "r") as f:
                settings = json.load(f)
        except:
            pass
            
    # Atualiza com o pedido do usuário
    settings["model_path"] = "D:\\WhisperModels"
    
    # Garante defaults se estiver vazio
    if "hotkey" not in settings: settings["hotkey"] = "f3"
    if "auto_paste" not in settings: settings["auto_paste"] = True
    
    print(f"Updating config at {config_path}...")
    with open(config_path, "w") as f:
        json.dump(settings, f, indent=2)
        
    # 2. Criar diretório se não existir
    try:
        os.makedirs("D:\\WhisperModels", exist_ok=True)
        print("Created/Verified D:\\WhisperModels")
    except Exception as e:
        print(f"Warning: Could not create D:\\WhisperModels: {e}")

    # 3. Rodar o app
    print("Launching app...")
    subprocess.run([sys.executable, "src/main.py"])

if __name__ == "__main__":
    setup_and_run()
