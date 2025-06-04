Item 5 - Possible future change:
- Implement in-memory transcription to avoid writing temporary WAV files. Modify
  `_save_and_transcribe_task` to pass audio arrays directly into the pipeline as
  inputs, keeping file-based processing as an optional fallback for debugging.
  
# Future Changes

This file records ideas for potential improvements.

* **In-memory transcription** – Avoid writing temporary WAV files by passing audio arrays directly into the pipeline. Update `_save_and_transcribe_task` accordingly while keeping file-based processing available for debugging.