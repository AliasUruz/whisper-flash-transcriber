# Plano de Remediação Pós-Auditoria — Branch `alpha`

## Contexto
- Auditoria de 24/09/2025 identificou regressões críticas que impedem a inicialização do app, quebram o pipeline de VAD/ASR e comprometem interações com a UI.
- Prioridade imediata: restaurar módulos estruturantes (ConfigManager, Core, TranscriptionHandler, VAD) antes de evoluir funcionalidades.
- Todo o trabalho deve ser realizado em pequenas etapas validadas com `flake8` e testes direcionados já existentes (sem criar novos arquivos de teste).

## 1. Reestabelecer a infraestrutura de configuração
- [ ] Reintroduzir a declaração `class ConfigManager:` no topo de `src/config_manager.py` e corrigir toda a indentação da classe.
- [ ] Garantir que `messagebox` seja importado explicitamente (`from tkinter import messagebox`).
- [ ] Revisar métodos públicos para assegurar que as assinaturas originais e retornos sejam restaurados.
- [ ] Alinhar defaults de `vad_pre_speech_padding_ms`/`vad_post_speech_padding_ms` com `src/config_schema.py` (decidir valores finais e documentar).
- [ ] Executar `flake8 src/config_manager.py src/config_schema.py` e resolver pendências.

## 2. Corrigir inicialização do núcleo e orquestração
- [ ] Revisar `AppCore.__init__` removendo/reordenando a segunda criação do `ActionOrchestrator` para que use a instância já inicializada com dependências completas.
- [ ] Garantir que `TranscriptionHandler` seja criado antes de qualquer referência a ele no orquestrador.
- [ ] Repassar referências obrigatórias (clipboard, callbacks, state manager) após os ajustes e validar via leitura do fluxo em `core.py`.
- [ ] Implementar ou eliminar o stub `start_key_detection_thread`, assegurando que a UI não invoque código vazio.
- [ ] Rodar `flake8 src/core.py src/action_orchestrator.py` para confirmar ausência de regressões de estilo.

## 3. Estabilizar carregamento de modelos e transcrição
- [ ] Refatorar `_initialize_model_and_processor` e `_load_model_task` em `src/transcription_handler.py` para que retornem `(model, processor)` de forma consistente em todos os caminhos.
- [ ] Substituir chamadas a `core_ref._set_state` por interações válidas com `state_manager` ou outros métodos existentes.
- [ ] Validar que `reload_asr` propaga corretamente o novo backend e atualiza o estado do app sem levantar exceções.
- [ ] Confirmar compatibilidade com configurações (modelos suportados, device) e ajustar logs para facilitar debug futuro.
- [ ] Executar `flake8 src/transcription_handler.py` após os reparos.

## 4. Reconstruir o pipeline de VAD e captura de áudio
- [ ] Corrigir o construtor de `VADManager` para usar atributos definidos (`self.config_manager.get("vad_pre_speech_padding_ms")`, etc.).
- [ ] Fazer `process_chunk` retornar explicitamente uma tupla `(is_speech: bool, frames: list[bytes])` (ou estrutura equivalente) e documentar o contrato.
- [ ] Atualizar `AudioHandler._process_audio_queue` para consumir o novo retorno sem depender de variáveis inexistentes.
- [ ] Revisar manipulação de exceções/logs no pipeline para evitar estados travados quando o VAD falhar.
- [ ] Ajustar `tests/test_vad_pipeline.py` para instanciar `VADManager` de forma realista (sem `__new__`) e cobrir o contrato corrigido.
- [ ] Rodar `flake8 src/audio_handler.py src/vad_manager.py tests/test_vad_pipeline.py` e executar `pytest tests/test_vad_pipeline.py`.

## 5. Realinhar interações da UI e validações finais
- [ ] Atualizar `UIManager` para usar APIs públicas do core (`state_manager.get_current_state()` ou equivalente) em vez de atributos inexistentes.
- [ ] Revisar métodos que dependem de `start_key_detection_thread` e garantir fluxo funcional após a correção da seção 2.
- [ ] Validar manualmente a jornada hotkey → gravação → transcrição → colagem em ambiente de desenvolvimento.
- [ ] Rodar uma passagem completa de `flake8 src/ui_manager.py` e `pytest` (subconjunto existente relevante) para confirmar estabilidade.
- [ ] Documentar no README (em inglês) um resumo das correções críticas e instruções de validação pós-fix.

---

**Checklist de Encerramento Geral**
- [ ] `flake8` sem erros nos módulos tocados.
- [ ] `pytest` passando para suites relacionadas (sem criação de novos arquivos de teste).
- [ ] Revisão manual do fluxo principal concluída e registrada em changelog/README.
