# Future Changes

This file records ideas for potential improvements.

* **In-memory transcription** – Avoid writing temporary WAV files by passing audio arrays directly into the pipeline. Update `_save_and_transcribe_task` accordingly while keeping file-based processing available for debugging.
