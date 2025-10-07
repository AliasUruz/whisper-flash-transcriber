# Deployment Notes

## Maintaining custom installation directories

Whisper Flash Transcriber now persists several heavyweight resources outside of the application tree. The directories are
configurable (`storage_root_dir`, `python_packages_dir`, `vad_models_dir`, and `hf_cache_dir`) and may live on fast secondary
drives or shared network volumes. When packaging the application for distribution or preparing backups:

- Include the configured directories in your backup strategy. They contain cached ASR models, third-party Python wheels, Silero
  VAD weights, and Hugging Face metadata that can take hours to rebuild over slow connections.
- If you relocate the directories to a new machine, update the corresponding paths in `config.json` before launching the
  application. The bootstrap step validates the paths and recreates missing folders, but it will not infer new locations.
- When using deployment automation, ensure that the process exports `PYTHONPATH` with the `python_packages_dir` value so that the
  interpreter resolves the installed wheels correctly.

Keeping these directories consistent across deployments shortens cold starts and avoids redundant downloads during maintenance
windows.
