# Changelog

## 2025-10-29
- Corrigido o registro da seção avançada na janela de configurações para que o
  painel de IA obedeça ao alternador "Mostrar avançado" sem gerar conflitos de
  ordenação entre frames.
- Atualizado o requisito opcional de `bitsandbytes` para restringir a instalação
  a Linux x86_64 e alinhar o intervalo suportado (`>=0.45,<0.46`).
- Implementado fallback automático para `float16` quando a quantização
  solicitada no Hugging Face não possui branch disponível, evitando falhas de
  download e carga do backend.

## 2025-02-15
- Hardened the hotkey teardown path to avoid referencing stale debounce
  variables, ensuring driver failures during unregister are logged cleanly.

## 2025-02-14
- Replaced the invalid `torch==2.7.1` pin with the Windows-compatible `torch==2.5.1` CPU wheel and clarified GPU installation steps.
- Removed the conflicting `pydantic==2.7.1` constraint, keeping `pydantic>=2.9,<3` as the single policy across the project.
- Moved `bitsandbytes` to the optional dependency set with a platform guard so Windows installations no longer fail out of the box.

## 2025-09-19
- Stabilized the Silero VAD pipeline with shape validation, float32 normalization, and JSON logging to `logs/vad_failure.jsonl`.
- Stripped ANSI escape sequences from console output and disabled transformer progress colors by default.
- Simplified the ASR settings panel with a `Show advanced` toggle that hides backend and quantization options until requested.

## 2025-10-27
- Added `--skip-bootstrap` to bypass preflight and dependency audits during troubleshooting sessions.
- Instrumented bootstrap with `bootstrap.step.*` markers to pinpoint initialization stalls in logs.
- Added `scripts/diagnostics.ps1` to collect environment checks on Windows before launching.
- Expanded README/AGENTS documentation com novas orientações de startup, comportamento de bandeja e playbooks de troubleshooting.
