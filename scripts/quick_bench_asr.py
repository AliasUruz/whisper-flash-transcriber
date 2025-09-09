"""Quick benchmarking utility for ASR backends."""

from __future__ import annotations

import time
import numpy as np

from src.config_manager import ConfigManager
from src.transcription_handler import TranscriptionHandler


def main() -> None:
    cfg = ConfigManager()

    handler = TranscriptionHandler(
        cfg,
        gemini_api_client=None,
        on_model_ready_callback=lambda: None,
        on_model_error_callback=lambda e: print(f"error: {e}"),
        on_transcription_result_callback=lambda text, _orig: print(text),
        on_agent_result_callback=None,
        on_segment_transcribed_callback=None,
        is_state_transcribing_fn=lambda: False,
    )

    handler.reload_asr()
    audio = np.zeros(int(16000 * 15), dtype="float32")
    times = []
    for _ in range(3):
        start = time.perf_counter()
        handler._asr_backend.transcribe(audio, chunk_length_s=30, batch_size=1)
        times.append(time.perf_counter() - start)
    median = sorted(times)[len(times) // 2]
    print(f"median_time={median:.2f}s")


if __name__ == "__main__":
    main()
