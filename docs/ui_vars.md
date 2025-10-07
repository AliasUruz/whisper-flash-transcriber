# Inventário de variáveis do Settings UI

Este documento consolida todas as instâncias de `ctk.*Var` usadas na janela de configurações criada em `UIManager.run_settings_gui`. Cada linha relaciona a variável com o widget que a consome e a chave de configuração persistida pela `ConfigManager`. Os nomes dos widgets correspondem às variáveis locais usadas ao instanciá-los.

| Seção | Variável | Tipo | Widget(s) associado(s) | Chave(s) de configuração | Observações |
| --- | --- | --- | --- | --- | --- |
| Geral | `auto_paste_var` | `BooleanVar` | `CTkSwitch` (`paste_switch`) | `"auto_paste"` | Alterna o recurso de auto-colagem após a transcrição. |
| Geral | `mode_var` | `StringVar` | `CTkRadioButton` (`toggle_rb`, `hold_rb`) | `"record_mode"` | Seleciona entre os modos *toggle* e *hold* de gravação. |
| Geral | `detected_key_var` | `StringVar` | `CTkLabel` (`key_display`) | `"record_key"` | Exibe a tecla atual e recebe o valor detectado pelo listener de hotkeys. |
| Geral | `agent_key_var` | `StringVar` | `CTkLabel` (`agent_key_display`) | `"agent_key"` | Espelha a tecla do modo agente; atualizado pelo detector dedicado. |
| Correção de texto | `agent_model_var` | `StringVar` | `CTkOptionMenu` (`agent_model_menu`) | `"gemini_agent_model"` | Seleciona o modelo Gemini específico para o modo agente. |
| Geral | `hotkey_stability_service_enabled_var` | `BooleanVar` | `CTkSwitch` (`stability_switch`) | `"hotkey_stability_service_enabled"` | Liga/desliga o serviço de estabilização dos hotkeys. |
| Geral | `launch_at_startup_var` | `BooleanVar` | `CTkSwitch` (`startup_switch`) | `"launch_at_startup"` | Habilita a inicialização automática no Windows. |
| Armazenamento | `storage_root_dir_var` | `StringVar` | `CTkEntry` (`storage_root_entry`) | `"storage_root_dir"` | Diretório base usado para modelos e outros artefatos pesados. Alterações disparam `_maybe_migrate_storage_paths()` para mover cache e gravações quando os caminhos específicos não foram sobrescritos. |
| Armazenamento | `recordings_dir_var` | `StringVar` | `CTkEntry` (`recordings_dir_entry`) | `"recordings_dir"` | Pasta onde arquivos WAV temporários e salvos são armazenados; quando não há override, acompanha a migração disparada pelo `storage_root_dir`. |
| Som | `sound_enabled_var` | `BooleanVar` | `CTkSwitch` (`sound_switch`) | `"sound_enabled"` | Ativa os bipes de início/fim de gravação. |
| Som | `sound_frequency_var` | `StringVar` | `CTkEntry` (`freq_entry`) | `"sound_frequency"` | Entrada textual convertida para inteiro ao aplicar as configurações. |
| Som | `sound_duration_var` | `StringVar` | `CTkEntry` (`duration_entry`) | `"sound_duration"` | Texto convertido para float durante a validação. |
| Som | `sound_volume_var` | `DoubleVar` | `CTkSlider` (`volume_slider`) | `"sound_volume"` | Slider de 0.0 a 1.0 para volume dos sinais sonoros. |
| Correção de texto | `text_correction_enabled_var` | `BooleanVar` | `CTkSwitch` (`correction_switch`) | `"text_correction_enabled"` | Controla a habilitação geral dos serviços de correção. |
| Correção de texto | `text_correction_service_var` | `StringVar` | Atualizado pelo `CTkOptionMenu` (`service_menu`) | `"text_correction_service"` | Recebe o valor efetivo do serviço selecionado via `service_display_map`. |
| Correção de texto | `text_correction_service_label_var` | `StringVar` | `CTkOptionMenu` (`service_menu`) | `"text_correction_service"` (via `service_display_map`) | Mantém o rótulo exibido ao usuário; sincroniza com `text_correction_service_var`. |
| Correção de texto | `openrouter_api_key_var` | `StringVar` | `CTkEntry` (`openrouter_key_entry`) | `"openrouter_api_key"` | Entrada mascarada para a chave OpenRouter (persistida em `secrets.json`). |
| Correção de texto | `openrouter_model_var` | `StringVar` | `CTkEntry` (`openrouter_model_entry`) | `"openrouter_model"` | Modelo usado nos pedidos ao OpenRouter. |
| Correção de texto | `gemini_api_key_var` | `StringVar` | `CTkEntry` (`gemini_key_entry`) | `"gemini_api_key"` | Entrada mascarada para a chave Gemini (persistida em `secrets.json`). |
| Correção de texto | `gemini_model_var` | `StringVar` | `CTkOptionMenu` (`gemini_model_menu`) | `"gemini_model"` | Seleciona o modelo padrão de correção do Gemini. |
| Transcrição | `batch_size_var` | `StringVar` | `CTkEntry` (`batch_entry`) | `"batch_size"` | Valor numérico interpretado como inteiro antes de salvar. |
| Transcrição | `chunk_length_mode_var` | `StringVar` | `CTkOptionMenu` (`chunk_mode_menu`) | `"chunk_length_mode"` | Define se o tamanho de *chunk* é manual ou automático. |
| Transcrição | `chunk_length_sec_var` | `DoubleVar` | `CTkEntry` (`chunk_len_entry`) | `"chunk_length_sec"` | Valor em segundos usado quando o modo é manual. |
| Transcrição | `min_transcription_duration_var` | `DoubleVar` | `CTkEntry` (`min_transcription_duration_entry`) | `"min_transcription_duration"` | Limite mínimo para descartar segmentos muito curtos. |
| Transcrição | `min_record_duration_var` | `DoubleVar` | `CTkEntry` (`min_record_duration_entry`) | `"min_record_duration"` | Descarta gravações breves antes de enviar ao Whisper. |
| Transcrição | `use_vad_var` | `BooleanVar` | `CTkCheckBox` (`vad_checkbox`) | `"use_vad"` | Alterna o uso do VAD Silero (é desativado se o modelo não estiver disponível). |
| Transcrição | `vad_threshold_var` | `DoubleVar` | `CTkEntry` (`vad_threshold_entry`) | `"vad_threshold"` | Probabilidade mínima para considerar fala. |
| Transcrição | `vad_silence_duration_var` | `DoubleVar` | `CTkEntry` (`vad_silence_entry`) | `"vad_silence_duration"` | Tempo de silêncio antes de cortar o áudio. |
| Transcrição | `vad_pre_speech_padding_ms_var` | `IntVar` | — (controle ainda não exposto) | `"vad_pre_speech_padding_ms"` | Variável preparada para configurar o *padding* de pré-fala; enquanto não há widget associado, o valor permanece sincronizado via payload salvo manualmente. |
| Transcrição | `vad_post_speech_padding_ms_var` | `IntVar` | `CTkEntry` (`vad_post_padding_entry`) | `"vad_post_speech_padding_ms"` | Define a janela extra após o silêncio detectado; o `AudioHandler` e o `VadManager` sincronizam essa margem ao persistir a configuração. |
| Transcrição | `save_temp_recordings_var` | `BooleanVar` | `CTkSwitch` (`temp_recordings_switch`) | `"save_temp_recordings"` | Mantém ou remove arquivos temporários após o processamento; quando ativo, aplica a cota `record_storage_limit` (MB) e remove gravações antigas para liberar espaço. |
| Transcrição | `record_storage_mode_var` | `StringVar` | `CTkOptionMenu` (`storage_mode_menu`) | `"record_storage_mode"` | Seleciona memória, disco ou modo automático; também define `new_record_to_memory` no `apply_settings`. |
| Transcrição | `max_memory_seconds_var` | `DoubleVar` | `CTkEntry` (`mem_time_entry`) | `"max_memory_seconds"` | Limite máximo de áudio em memória antes de migrar para disco. |
| Transcrição | `max_memory_seconds_mode_var` | `StringVar` | `CTkOptionMenu` (`mem_mode_menu`) | `"max_memory_seconds_mode"` | Alterna entre cálculo manual e automático do limite de memória. |
| Transcrição | `display_transcripts_var` | `BooleanVar` | `CTkSwitch` (`display_switch`) | `"display_transcripts_in_terminal"` | Define se o texto final é impresso no terminal. |
| ASR | `asr_backend_var` | `StringVar` | `CTkOptionMenu` (`asr_backend_menu`) | `"asr_backend"` | Permite sobrepor o backend inferido pelo modelo selecionado. |
| ASR | `asr_model_id_var` | `StringVar` | Alimentado por `CTkOptionMenu` (`asr_model_menu`) via `asr_model_display_var` | `"asr_model_id"` | Guarda o identificador interno do modelo escolhido. |
| ASR | `asr_model_display_var` | `StringVar` | `CTkOptionMenu` (`asr_model_menu`) | `"asr_model_id"` (mapeado por `display_to_id`) | Variável de exibição; converte nomes amigáveis para `asr_model_id_var`. |
| ASR | `asr_compute_device_var` | `StringVar` | `CTkOptionMenu` (`asr_device_menu`) | `"asr_compute_device"` | Representa a seleção textual (*Auto*, *Force CPU*, *Force CUDA*), traduzida diretamente para o backend. |
| ASR | `asr_dtype_var` | `StringVar` | `CTkOptionMenu` (`asr_dtype_menu`) | `"asr_dtype"` | Define a precisão dos tensores do backend Torch. |
| ASR | `asr_ct2_compute_type_var` | `StringVar` | `CTkOptionMenu` (`asr_ct2_menu`) | `"asr_ct2_compute_type"` | Ajusta o compute type quando o backend é CTranslate2. |
| ASR | `models_storage_dir_var` | `StringVar` | `CTkEntry` (`models_dir_entry`) | `"models_storage_dir"` | Diretório raiz usado para armazenar modelos e demais artefatos pesados; também serve de destino padrão quando o `storage_root_dir` muda sem override explícito. |
| ASR | `asr_cache_dir_var` | `StringVar` | `CTkEntry` (`asr_cache_entry`) | `"asr_cache_dir"` | Diretório raiz usado para armazenar modelos baixados. |

## Notas adicionais

- A configuração `record_storage_limit` ainda não possui `ctk.*Var` associada. Quando definida manualmente em `config.json`, o `AudioHandler` limita o total combinado de `temp_recording_*.wav` e `recording_*.wav` no diretório configurado por `recordings_dir`, removendo os arquivos mais antigos assim que o teto (em MiB) é excedido.

## Duplicatas ou variáveis potencialmente obsoletas

- `asr_model_display_var`: é instanciada inicialmente sem valor e, alguns blocos adiante, recebe nova instância preenchida com os rótulos do catálogo. Não há efeito funcional, mas a criação anterior é redundante.

Nenhuma outra `ctk.*Var` aparece duplicada ou sem uso na interface atual.
