from pathlib import Path
import wave
import struct
from playwright.sync_api import sync_playwright


def _cria_wav(path: Path) -> None:
    """Gera um pequeno arquivo WAV silencioso para uso no teste."""
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16_000)
        frames = struct.pack("<h", 0) * 16_000
        wf.writeframes(frames)


def test_headless_smoke(tmp_path: Path) -> None:
    """Verifica o fluxo básico de upload e resposta em modo headless."""
    wav_path = tmp_path / "sample.wav"
    _cria_wav(wav_path)

    html_path = Path(__file__).parent / "mock_chatgpt.html"

    with sync_playwright() as p:
        navegador = p.chromium.launch(headless=True)
        pagina = navegador.new_page()
        pagina.goto(html_path.as_uri(), timeout=60_000)

        with pagina.expect_file_chooser() as fc_info:
            pagina.locator("button[data-testid='composer-plus-btn']").click()
        seletor_arquivo = fc_info.value
        seletor_arquivo.set_files(str(wav_path))

        pagina.wait_for_selector("button[data-testid='send-button']:not([disabled])", timeout=20_000)
        pagina.click("button[data-testid='send-button']")

        pagina.locator("div[data-message-author-role='assistant']").wait_for(timeout=120_000)
        navegador.close()
