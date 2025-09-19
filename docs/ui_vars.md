# Inventário de variáveis do Settings UI

Este documento consolida todas as instâncias de `ctk.*Var` usadas na janela de configurações criada em `UIManager.run_settings_gui`. Cada linha relaciona a variável com o widget que a consome e a chave de configuração persistida pela `ConfigManager`. Os nomes dos widgets correspondem às variáveis locais usadas ao instanciá-los.

| Seção | Variável | Tipo | Widget(s) associado(s) | Chave(s) de configuração | Observações |
| --- | --- | --- | --- | --- | --- |
| Geral | `auto_paste_var` | `BooleanVar` | `CTkSwitch` (`paste_switch`) | `"auto_paste"` | Alterna o recurso de auto-colagem após a transcrição. |
| Geral | `mode_var` | `StringVar` | `CTkRadioButton` (`toggle_rb`, `hold_rb`) | `"record_mode"` | Seleciona entre os modos *toggle* e *hold* de gravação. |
| Geral | `detected_key_var` | `StringVar` | `CTkLabel` (`key_display`) | `"record_key"` | Exibe a tecla atual e recebe o valor detectado pelo listener de hotkeys. |
| Geral | `agent_key_var` | `StringVar` | `CTkLabel` (`agent_key_display`) | `"agent_key"` | Espelha a tecla do modo agente; atualizado pelo detector dedicado. |
| Geral | `agent_model_var` | `StringVar` | — | `"gemini_agent_model"` | Mantém o valor configurado mas não há widget para alterá-lo; apenas reaplicado no `apply_settings` e em *restore defaults*. |
| Geral | `hotkey_stability_service_enabled_var` | `BooleanVar` | `CTkSwitch` (`stability_switch`) | `"hotkey_stability_service_enabled"` | Liga/desliga o serviço de estabilização dos hotkeys. |
| Geral | `launch_at_startup_var` | `BooleanVar` | `CTkSwitch` (`startup_switch`) | `"launch_at_startup"` | Habilita a inicialização automática no Windows. |
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
| Transcrição | `enable_torch_compile_var` | `BooleanVar` | `CTkSwitch` (`torch_compile_switch`) | `"enable_torch_compile"` | Ativa/desativa o `torch.compile` experimental. |
| Transcrição | `chunk_length_mode_var` | `StringVar` | `CTkOptionMenu` (`chunk_mode_menu`) | `"chunk_length_mode"` | Define se o tamanho de *chunk* é manual ou automático. |
| Transcrição | `chunk_length_sec_var` | `DoubleVar` | `CTkEntry` (`chunk_len_entry`) | `"chunk_length_sec"` | Valor em segundos usado quando o modo é manual. |
| Transcrição | `min_transcription_duration_var` | `DoubleVar` | `CTkEntry` (`min_transcription_duration_entry`) | `"min_transcription_duration"` | Limite mínimo para descartar segmentos muito curtos. |
| Transcrição | `min_record_duration_var` | `DoubleVar` | `CTkEntry` (`min_record_duration_entry`) | `"min_record_duration"` | Descarta gravações breves antes de enviar ao Whisper. |
| Transcrição | `use_vad_var` | `BooleanVar` | `CTkCheckBox` (`vad_checkbox`) | `"use_vad"` | Alterna o uso do VAD Silero (é desativado se o modelo não estiver disponível). |
| Transcrição | `vad_threshold_var` | `DoubleVar` | `CTkEntry` (`vad_threshold_entry`) | `"vad_threshold"` | Probabilidade mínima para considerar fala. |
| Transcrição | `vad_silence_duration_var` | `DoubleVar` | `CTkEntry` (`vad_silence_entry`) | `"vad_silence_duration"` | Tempo de silêncio antes de cortar o áudio. |
| Transcrição | `save_temp_recordings_var` | `BooleanVar` | `CTkSwitch` (`temp_recordings_switch`) | `"save_temp_recordings"` | Mantém ou remove arquivos temporários após o processamento. |
| Transcrição | `record_storage_mode_var` | `StringVar` | `CTkOptionMenu` (`storage_mode_menu`) | `"record_storage_mode"` | Seleciona memória, disco ou modo automático; também define `new_record_to_memory` no `apply_settings`. |
| Transcrição | `max_memory_seconds_var` | `DoubleVar` | `CTkEntry` (`mem_time_entry`) | `"max_memory_seconds"` | Limite máximo de áudio em memória antes de migrar para disco. |
| Transcrição | `max_memory_seconds_mode_var` | `StringVar` | `CTkOptionMenu` (`mem_mode_menu`) | `"max_memory_seconds_mode"` | Alterna entre cálculo manual e automático do limite de memória. |
| Transcrição | `display_transcripts_var` | `BooleanVar` | `CTkSwitch` (`display_switch`) | `"display_transcripts_in_terminal"` | Define se o texto final é impresso no terminal. |
| ASR | `asr_backend_var` | `StringVar` | `CTkOptionMenu` (`asr_backend_menu`) | `"asr_backend"` | Permite sobrepor o backend inferido pelo modelo selecionado. |
| ASR | `asr_model_id_var` | `StringVar` | Alimentado por `CTkOptionMenu` (`asr_model_menu`) via `asr_model_display_var` | `"asr_model_id"` | Guarda o identificador interno do modelo escolhido. |
| ASR | `asr_model_display_var` | `StringVar` | `CTkOptionMenu` (`asr_model_menu`) | `"asr_model_id"` (mapeado por `display_to_id`) | Variável de exibição; converte nomes amigáveis para `asr_model_id_var`. |
| ASR | `asr_compute_device_var` | `StringVar` | `CTkOptionMenu` (`asr_device_menu`) | `"asr_compute_device"` + `"gpu_index"` | Representa a seleção textual (*Auto*, *Force CPU*, *GPU X*), traduzida para backend e índice ao aplicar. |
| ASR | `asr_dtype_var` | `StringVar` | `CTkOptionMenu` (`asr_dtype_menu`) | `"asr_dtype"` | Define a precisão dos tensores do backend Torch. |
| ASR | `asr_ct2_compute_type_var` | `StringVar` | `CTkOptionMenu` (`asr_ct2_menu`) | `"asr_ct2_compute_type"` | Ajusta o compute type quando o backend é CTranslate2. |
| ASR | `asr_cache_dir_var` | `StringVar` | `CTkEntry` (`asr_cache_entry`) | `"asr_cache_dir"` | Diretório raiz usado para armazenar modelos baixados. |

## Duplicatas ou variáveis potencialmente obsoletas

- `agent_model_var`: inicializada com `"gemini_agent_model"`, aplicada e restaurada, porém nenhum widget atual permite modificar esse valor. Sugere-se incluir um seletor explícito ou remover a variável se o parâmetro for fixo. 
- `asr_model_display_var`: é instanciada inicialmente sem valor e, alguns blocos adiante, recebe nova instância preenchida com os rótulos do catálogo. Não há efeito funcional, mas a criação anterior é redundante.

Nenhuma outra `ctk.*Var` aparece duplicada ou sem uso na interface atual.
