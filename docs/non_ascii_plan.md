# Non ASCII remediation checklist

## Logging strings (high priority)

### src/audio_handler.py
- [ ] L156: "Duracao da gravacao excedeu {self.current_max_memory_seconds}s. Moving from RAM to disk." -> "Recording duration exceeded {self.current_max_memory_seconds}s; moving from RAM to disk."
- [ ] L189: "Erro no processamento da fila de audio: %s" -> "Error processing audio queue: %s"
- [ ] L398: "Erro ao fechar arquivo temporario: %s" -> "Failed to close temporary recording file: %s"
- [ ] L410: "Duracao gravada {recording_duration:.2f}s abaixo do minimo configurado {self.min_record_duration}s. Descartando." -> "Recorded duration {recording_duration:.2f}s below configured minimum {self.min_record_duration}s; discarding segment."
- [ ] L413: "Gravacao muito curta (< {self.min_record_duration}s) ou vazia. Descartando." -> "Recording shorter than {self.min_record_duration}s or empty; discarding segment."
- [ ] L436: "Gravacao temporaria salva em %s" -> "Temporary recording saved at %s"
- [ ] L438: "Falha ao salvar gravacao temporaria: %s" -> "Failed to save temporary recording: %s"
- [ ] L593: "Erro ao remover arquivo temporario: %s" -> "Failed to remove temporary recording file: %s"
- [ ] L606: "Erro ao fechar stream de audio: %s" -> "Failed to close audio stream: %s"

### src/core.py
- [ ] L548: "AppCore[%s]: diretorio de cache de ASR indisponivel (%s, motivo=%s)." -> "AppCore[%s]: ASR cache directory unavailable (%s, reason=%s)."
- [ ] L596: "AppCore: UI ainda nao esta pronta; %d tooltip(s) still pending." -> "AppCore: UI not ready yet; %d tooltip(s) pending."
- [ ] L673: "Nao foi possivel obter duracao do audio: %s" -> "Unable to read audio duration: %s"
- [ ] L679: "Segmento de audio ({duration_seconds:.2f}s) e mais curto que o minimo configurado ({min_duration}s). Skipping." -> "Audio segment {duration_seconds:.2f}s shorter than configured minimum {min_duration}s; skipping."
- [ ] L694: "AppCore: Segmento de audio pronto ({duration_seconds:.2f}s). Dispatching to TranscriptionHandler (agent_mode={is_agent_mode})." -> "AppCore: Audio segment ready ({duration_seconds:.2f}s). Dispatching to TranscriptionHandler (agent_mode={is_agent_mode})."
- [ ] L1117: "Falha apos {max_attempts} reload attempts. Ultimo erro: {last_error}" -> "Failed after {max_attempts} reload attempts. Last error: {last_error}"
- [ ] L1581: "AppCore: configuracao rejeitada por validacao: %s" -> "AppCore: configuration rejected by validation: %s"
- [ ] L1671: "Configuracao '{key}' alterada para: {value}" -> "Configuration '{key}' changed to: {value}"
- [ ] L1680: "Configuracao 'agent_auto_paste' (unificada) alterada para: %s" -> "Configuration 'agent_auto_paste' (unified) changed to: %s"
- [ ] L1717: "Erro ao atualizar configuracoes do TranscriptionHandler: %s" -> "Failed to update TranscriptionHandler settings: %s"
- [ ] L1782: "Nenhuma configuracao alterada." -> "No configuration changes detected."
- [ ] L1791: "Configuracao '{key}' ja possui o valor '{value}'. Nenhuma alteracao necessaria." -> "Configuration '{key}' already has value '{value}'; no update needed."
- [ ] L1798: "Configuracao '{key}' alterada para: {value}" -> (same as L1671, consider helper)
- [ ] L1825: "TranscriptionHandler: Configuracoes de transcricao atualizadas via update_setting para '{key}'." -> "TranscriptionHandler: transcription settings updated via update_setting for '{key}'."
- [ ] L1832: "Falha ao iniciar recarregamento do modelo apos update_setting ({key}): {error}" -> "Failed to trigger model reload after update_setting ({key}): {error}"
- [ ] L1844: "AudioHandler: Configuracoes atualizadas via update_setting para '{key}'." -> "AudioHandler: settings updated via update_setting for '{key}'."
- [ ] L1882: "Configuracao '{key}' atualizada e propagada com sucesso." -> "Configuration '{key}' updated and propagated successfully."

### src/keyboard_hotkey_manager.py
- [ ] L52: "Erro ao carregar configuracao: %s" -> "Failed to load configuration: %s"
- [ ] L67: "Erro ao salvar configuracao: %s" -> "Failed to save configuration: %s"
- [ ] L136: "Configuracao atualizada: record_key=%s, agent_key=%s, record_mode=%s" -> "Configuration updated: record_key=%s, agent_key=%s, record_mode=%s"
- [ ] L140: "Erro ao atualizar configuracao: %s" -> "Failed to update configuration: %s"
- [ ] L177: "Falha especifica ao limpar hooks do teclado: %s" -> "Specific failure while clearing keyboard hooks: %s"
- [ ] L203: "Falha especifica ao registrar hotkey release: %s" -> "Specific failure registering release hotkey: %s"
- [ ] L218: "Falha especifica ao registrar hotkey de gravacao: %s" -> "Specific failure registering recording hotkey: %s"
- [ ] L223: "Erro ao registrar hotkey de gravacao: %s" -> "Failed to register recording hotkey: %s"
- [ ] L242: "Falha especifica ao registrar hotkey de comando: %s" -> "Specific failure registering command hotkey: %s"
- [ ] L359: "Iniciando deteccao de tecla com timeout de {timeout} segundos..." -> "Starting key detection with timeout of {timeout} seconds..."

### src/transcription_handler.py
- [ ] L609: "Nenhuma chave de API encontrada para o provedor {provider}. Pulando correcao de texto." -> "No API key available for provider {provider}; skipping text correction."
- [ ] L643: "Transcricao corrigida: {text}" -> "Corrected transcription: {text}"
- [ ] L997: "Auto-selecao de GPU (maior VRAM total): {device} ({memory})" -> "Auto-selected GPU with highest total VRAM: {device} ({memory})"
- [ ] L1039: "Falha ao escolher GPU por memoria livre: {error}" -> "Failed to select GPU by free memory: {error}"
- [ ] L1218: "Erro durante a transcricao via backend unificado: {error}" -> "Error during transcription via unified backend: {error}"
- [ ] L1310: "Iniciando transcricao de segmento com batch_size={batch}" -> "Starting segment transcription with batch_size={batch}"
- [ ] L1411: "OOM detectado. Reduzindo batch_size de {old_bs} para {new_bs} para proximas solicitacoes." -> "OOM detected. Reducing batch_size from {old_bs} to {new_bs} for the next submissions."
- [ ] L1425: "OOM persistente. Reduzindo chunk_length_sec de {old_chunk} para {new_chunk} para proximas solicitacoes." -> "Persistent OOM. Reducing chunk_length_sec from {old_chunk} to {new_chunk} for the next submissions."
- [ ] L1432: "Falha ao ajustar parametros apos OOM: {error}" -> "Failed to adjust parameters after OOM: {error}"
- [ ] L1458: "Erro durante a transcricao de segmento: {error}" -> "Error while transcribing segment: {error}"
- [ ] L1473: "Transcricao bruta: {text}" -> "Raw transcription: {text}"
- [ ] L1603: "Erro ao encerrar o executor de transcricao: {error}" -> "Failed to shut down transcription executor: {error}"

### src/config_manager.py
- [ ] L300: "Falha ao criar diretorio de cache ASR '{asr_cache_dir}': {error}" -> "Failed to create ASR cache directory '{asr_cache_dir}': {error}"

## System dialogs/tooltips
- [ ] Atualizar mensagens de UI (tooltips, dialogos, menus) conforme inventario do scanner (pendente mapear com ajuda do --all-strings).

## Docstrings/comments to revisit (lower priority)
- [ ] Revisar docstrings com portugues em src/audio_handler.py, src/core.py, src/ui_manager.py, src/utils/* (manter apenas se necessario para onboarding interno).

## Encoding clean-up helpers
- [ ] Atualizar scripts/scan_non_ascii.py para substituir uso de ast.Str (remover warnings).
- [ ] Considerar helper central (ex.: module logging_messages.py) para frases compartilhadas.
- [ ] Documentar instrucoes de encoding no README/CHANGELOG apos ajustes.
