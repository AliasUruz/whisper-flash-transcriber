Item 5 - Possible future change:
- Implement in-memory transcription to avoid writing temporary WAV files. Modify
  `_save_and_transcribe_task` to pass audio arrays directly into the pipeline as
  inputs, keeping file-based processing as an optional fallback for debugging.
