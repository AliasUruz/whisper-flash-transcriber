# Plano de Ação Técnico para o Codex: Transcrição de Áudio em Memória

## Objetivo Principal
Modificar o fluxo de processamento de áudio do Whisper Recorder para que os dados gravados (arrays NumPy) sejam passados diretamente para o pipeline de transcrição do Whisper, eliminando a necessidade de salvar arquivos `.wav` temporários em disco. Isso aumentará a velocidade e a eficiência, especialmente para transcrições longas. Uma nova opção de configuração será adicionada para permitir o salvamento opcional de arquivos de áudio para fins de depuração.

## Racional da Mudança
A operação de escrita e leitura de arquivos em disco (I/O) é um gargalo de desempenho conhecido, especialmente para gravações de áudio mais longas. O salvamento e posterior carregamento de arquivos `.wav` introduz latência e sobrecarga desnecessária. Ao processar os dados de áudio diretamente na memória, podemos:
- **Aumentar a Velocidade:** Reduzir significativamente o tempo entre o fim da gravação e o início da transcrição, eliminando as operações de I/O de disco.
- **Aumentar a Eficiência:** Diminuir o uso do disco, prolongar a vida útil do SSD/HDD e evitar a fragmentação de arquivos temporários.
- **Melhorar a Robustez:** Eliminar potenciais falhas relacionadas a permissões de escrita de arquivo, espaço em disco insuficiente ou problemas de disco.
- **Simplificar o Fluxo:** Reduzir a complexidade do gerenciamento de arquivos temporários.

Esta mudança está alinhada com as melhorias planejadas descritas em [`FUTURE_CHANGES.md`](FUTURE_CHANGES.md:10), especificamente a "In-memory transcription".

## Plano de Ação Detalhado

### Tarefa 1: Adicionar e Gerenciar a Configuração `save_audio_for_debug`

Esta tarefa foca na introdução de uma nova configuração que permitirá ao usuário escolher se deseja salvar os arquivos de áudio gravados em disco para fins de depuração. Por padrão, esta opção estará desativada para garantir o benefício de desempenho da transcrição em memória.

**Contexto:**
- **Arquivo:** [`whisper_tkinter.py`](whisper_tkinter.py)
- **Estruturas:**
    - `DEFAULT_CONFIG` (dicionário de configurações padrão)
    - Constantes de configuração (para chaves de dicionário)
    - `WhisperCore.__init__` (inicialização de variáveis de instância)
    - `_load_config` (método para carregar configurações do arquivo `config.json`)
    - `_save_config` (método para salvar configurações no arquivo `config.json`)
    - `apply_settings_from_external` (método para aplicar configurações da GUI)
    - `run_settings_gui` (função que constrói a interface de configurações)

**Passos de Implementação (Sub-tarefas):**

#### 1.1 Adicionar a Nova Chave ao `DEFAULT_CONFIG`

**Objetivo:** Definir o valor padrão para a nova configuração `save_audio_for_debug` como `False`.

**Contexto:**
- **Arquivo:** [`whisper_tkinter.py`](whisper_tkinter.py)
- **Local:** Linhas 85-132, dentro do dicionário `DEFAULT_CONFIG`.

**Passos de Implementação:**
1.  Localize o dicionário `DEFAULT_CONFIG`.
2.  Adicione a chave `"save_audio_for_debug"` com o valor `False` após a chave `"gemini_model_options"`.

**Bloco de Código Relevante:**

##### ### Código Antigo
```python
# whisper_tkinter.py:126
    "auto_reregister_hotkeys": True,
    "gemini_model_options": [
        "gemini-1.5-flash",
        "gemini-1.5-pro",
        "gemini-2.0-pro"
    ]
}
```

##### ### Código Novo
```python
# whisper_tkinter.py:126
    "auto_reregister_hotkeys": True,
    "gemini_model_options": [
        "gemini-1.5-flash",
        "gemini-1.5-pro",
        "gemini-2.0-pro"
    ],
    "save_audio_for_debug": False # Nova configuração para transcrição em memória
}
```

#### 1.2 Adicionar Constante para a Nova Chave de Configuração

**Objetivo:** Criar uma constante para a chave `"save_audio_for_debug"` para melhorar a legibilidade e evitar erros de digitação.

**Contexto:**
- **Arquivo:** [`whisper_tkinter.py`](whisper_tkinter.py)
- **Local:** Linhas 146-150, entre as outras constantes de configuração.

**Passos de Implementação:**
1.  Localize a seção de constantes.
2.  Adicione a constante `SAVE_AUDIO_FOR_DEBUG_CONFIG_KEY = "save_audio_for_debug"`.

**Bloco de Código Relevante:**

##### ### Código Antigo
```python
# whisper_tkinter.py:146
# Batch size and GPU index configuration keys
BATCH_SIZE_CONFIG_KEY = "batch_size"
GPU_INDEX_CONFIG_KEY = "gpu_index"
# Agent key configuration
AGENT_KEY_CONFIG_KEY = "agent_key"
```

##### ### Código Novo
```python
# whisper_tkinter.py:146
# Batch size and GPU index configuration keys
BATCH_SIZE_CONFIG_KEY = "batch_size"
GPU_INDEX_CONFIG_KEY = "gpu_index"
# Save audio for debug configuration key
SAVE_AUDIO_FOR_DEBUG_CONFIG_KEY = "save_audio_for_debug" # Nova constante
# Agent key configuration
AGENT_KEY_CONFIG_KEY = "agent_key"
```

#### 1.3 Inicializar a Variável de Instância em `WhisperCore.__init__`

**Objetivo:** Garantir que a instância `WhisperCore` tenha um atributo `self.save_audio_for_debug` inicializado com o valor padrão.

**Contexto:**
- **Arquivo:** [`whisper_tkinter.py`](whisper_tkinter.py)
- **Local:** Linhas 209-253, dentro do método `__init__` da classe `WhisperCore`.

**Passos de Implementação:**
1.  Localize o método `__init__` da classe `WhisperCore`.
2.  Adicione a linha `self.save_audio_for_debug = DEFAULT_CONFIG[SAVE_AUDIO_FOR_DEBUG_CONFIG_KEY]` junto com as outras inicializações de configuração.

**Bloco de Código Relevante:**

##### ### Código Antigo
```python
# whisper_tkinter.py:242
        self.gemini_agent_model = DEFAULT_CONFIG["gemini_agent_model"]
        self.agent_auto_paste = DEFAULT_CONFIG["agent_auto_paste"]
        self.gemini_model_options = []
        self.sound_lock = RLock()  # Lock for sound playback
```

##### ### Código Novo
```python
# whisper_tkinter.py:242
        self.gemini_agent_model = DEFAULT_CONFIG["gemini_agent_model"]
        self.agent_auto_paste = DEFAULT_CONFIG["agent_auto_paste"]
        self.gemini_model_options = []
        self.sound_lock = RLock()  # Lock for sound playback
        self.save_audio_for_debug = DEFAULT_CONFIG[SAVE_AUDIO_FOR_DEBUG_CONFIG_KEY] # Inicializa a nova configuração
```

#### 1.4 Carregar e Validar a Configuração em `_load_config`

**Objetivo:** Ler o valor de `save_audio_for_debug` do arquivo `config.json` (se existir) e aplicá-lo à instância, com tratamento de erros e fallback para o padrão.

**Contexto:**
- **Arquivo:** [`whisper_tkinter.py`](whisper_tkinter.py)
- **Local:** Linhas 448-719, dentro do método `_load_config`.

**Passos de Implementação:**
1.  Localize o método `_load_config`.
2.  Após a validação do `gpu_index`, adicione um novo bloco `try-except` para carregar e validar `save_audio_for_debug`.
3.  Certifique-se de que o valor seja convertido para booleano.
4.  Adicione uma linha de log para confirmar o valor carregado.

**Bloco de Código Relevante:**

##### ### Código Antigo
```python
# whisper_tkinter.py:647
            self.gpu_index = -1
            self.gpu_index_specified = False

        # Load and validate Gemini mode
        try:
            gemini_mode_val = str(self.config.get("gemini_mode", DEFAULT_CONFIG["gemini_mode"])).lower()
            if gemini_mode_val in ["correction", "general"]:
                self.gemini_mode = gemini_mode_val
            else:
                logging.warning(f"Invalid gemini_mode '{self.config.get('gemini_mode')}' in config. Falling back to '{DEFAULT_CONFIG['gemini_mode']}'.")
                self.gemini_mode = DEFAULT_CONFIG["gemini_mode"]
        except (ValueError, TypeError):
            logging.warning(f"Invalid gemini_mode type in config. Falling back to '{DEFAULT_CONFIG['gemini_mode']}'.")
            self.gemini_mode = DEFAULT_CONFIG["gemini_mode"]
```

##### ### Código Novo
```python
# whisper_tkinter.py:647
            self.gpu_index = -1
            self.gpu_index_specified = False

        # Load and validate save_audio_for_debug
        try:
            self.save_audio_for_debug = bool(self.config.get(SAVE_AUDIO_FOR_DEBUG_CONFIG_KEY, DEFAULT_CONFIG[SAVE_AUDIO_FOR_DEBUG_CONFIG_KEY]))
        except (ValueError, TypeError):
            logging.warning(f"Invalid '{SAVE_AUDIO_FOR_DEBUG_CONFIG_KEY}' value in config. Falling back to default.")
            self.save_audio_for_debug = DEFAULT_CONFIG[SAVE_AUDIO_FOR_DEBUG_CONFIG_KEY]
            self.config[SAVE_AUDIO_FOR_DEBUG_CONFIG_KEY] = self.save_audio_for_debug # Ensure config dict is updated

        # Load and validate Gemini mode
        try:
            gemini_mode_val = str(self.config.get("gemini_mode", DEFAULT_CONFIG["gemini_mode"])).lower()
            if gemini_mode_val in ["correction", "general"]:
                self.gemini_mode = gemini_mode_val
            else:
                logging.warning(f"Invalid gemini_mode '{self.config.get('gemini_mode')}' in config. Falling back to '{DEFAULT_CONFIG['gemini_mode']}'.")
                self.gemini_mode = DEFAULT_CONFIG["gemini_mode"]
        except (ValueError, TypeError):
            logging.warning(f"Invalid gemini_mode type in config. Falling back to '{DEFAULT_CONFIG['gemini_mode']}'.")
            self.gemini_mode = DEFAULT_CONFIG["gemini_mode"]
```

#### 1.5 Salvar a Configuração em `_save_config`

**Objetivo:** Incluir a nova configuração `save_audio_for_debug` ao salvar o arquivo `config.json`, garantindo que as preferências do usuário sejam persistidas.

**Contexto:**
- **Arquivo:** [`whisper_tkinter.py`](whisper_tkinter.py)
- **Local:** Linhas 719-755, dentro do método `_save_config`.

**Passos de Implementação:**
1.  Localize o método `_save_config`.
2.  Adicione a chave `SAVE_AUDIO_FOR_DEBUG_CONFIG_KEY` ao dicionário `config_to_save`.

**Bloco de Código Relevante:**

##### ### Código Antigo
```python
# whisper_tkinter.py:752
            BATCH_SIZE_CONFIG_KEY: self.batch_size,
            GPU_INDEX_CONFIG_KEY: self.gpu_index,
            AUTO_REREGISTER_CONFIG_KEY: self.auto_reregister_hotkeys
        }
```

##### ### Código Novo
```python
# whisper_tkinter.py:752
            BATCH_SIZE_CONFIG_KEY: self.batch_size,
            GPU_INDEX_CONFIG_KEY: self.gpu_index,
            AUTO_REREGISTER_CONFIG_KEY: self.auto_reregister_hotkeys,
            SAVE_AUDIO_FOR_DEBUG_CONFIG_KEY: self.save_audio_for_debug # Salva a nova configuração
        }
```

#### 1.6 Atualizar `apply_settings_from_external`

**Objetivo:** Permitir que a janela de configurações externa passe o valor da nova configuração para a instância `WhisperCore`.

**Contexto:**
- **Arquivo:** [`whisper_tkinter.py`](whisper_tkinter.py)
- **Local:** Linhas 2054-2369, dentro do método `apply_settings_from_external`.

**Passos de Implementação:**
1.  Adicione `new_save_audio_for_debug=None` à assinatura da função.
2.  Dentro da função, adicione um bloco para aplicar o novo valor a `self.save_audio_for_debug` e marcar `config_needs_saving` se o valor for diferente.
3.  Atualize a chamada `_save_config()` para incluir a nova configuração.

**Bloco de Código Relevante:**

##### ### Código Antigo
```python
# whisper_tkinter.py:2079
        new_auto_reregister=None,
        new_agent_key=None,
        new_gemini_model_options=None,
    ):
        """Applies settings passed from the external settings window/thread."""
        logging.info("Applying new configuration from external source.")
        key_changed = False
        mode_changed = False
        config_needs_saving = False
        gemini_changed = False # Initialize gemini_changed here
```

##### ### Código Novo
```python
# whisper_tkinter.py:2079
        new_auto_reregister=None,
        new_agent_key=None,
        new_gemini_model_options=None,
        new_save_audio_for_debug=None, # Novo parâmetro
    ):
        """Applies settings passed from the external settings window/thread."""
        logging.info("Applying new configuration from external source.")
        key_changed = False
        mode_changed = False
        config_needs_saving = False
        gemini_changed = False # Initialize gemini_changed here
```

##### ### Código Antigo (continuação)
```python
# whisper_tkinter.py:2192
                else:
                    self.stop_reregister_event.set()
                    self.stop_health_check_event.set()

        # Keyboard library is always Win32
        self.keyboard_library = KEYBOARD_LIB_WIN32
```

##### ### Código Novo (continuação)
```python
# whisper_tkinter.py:2192
                else:
                    self.stop_reregister_event.set()
                    self.stop_health_check_event.set()

        # Apply save_audio_for_debug setting
        if new_save_audio_for_debug is not None:
            save_audio_bool = bool(new_save_audio_for_debug)
            if save_audio_bool != self.save_audio_for_debug:
                self.save_audio_for_debug = save_audio_bool
                config_needs_saving = True
                logging.info(f"Save audio for debug changed to: {self.save_audio_for_debug}")

        # Keyboard library is always Win32
        self.keyboard_library = KEYBOARD_LIB_WIN32
```

##### ### Código Antigo (continuação)
```python
# whisper_tkinter.py:2359
                    new_batch_size=batch_size_to_apply,
                    new_gpu_index=gpu_index_to_apply,
                    new_auto_reregister=auto_reregister_to_apply
                ) # Fechar parênteses da chamada da função
            else:
                logging.critical("CRITICAL: apply_settings_from_external method not found on core_instance!")
```

##### ### Código Novo (continuação)
```python
# whisper_tkinter.py:2359
                    new_batch_size=batch_size_to_apply,
                    new_gpu_index=gpu_index_to_apply,
                    new_auto_reregister=auto_reregister_to_apply,
                    new_save_audio_for_debug=save_audio_for_debug_to_apply # Passa a nova configuração
                ) # Fechar parênteses da chamada da função
            else:
                logging.critical("CRITICAL: apply_settings_from_external method not found on core_instance!")
```

#### 1.7 Adicionar `CTkSwitch` na Interface de Configurações

**Objetivo:** Fornecer uma opção na GUI para o usuário controlar a configuração `save_audio_for_debug`.

**Contexto:**
- **Arquivo:** [`whisper_tkinter.py`](whisper_tkinter.py)
- **Local:** Linhas 2560-3366, dentro da função `run_settings_gui`.

**Passos de Implementação:**
1.  Localize a função `run_settings_gui`.
2.  Adicione uma nova `ctk.BooleanVar` para `save_audio_for_debug`.
3.  Crie um novo `CTkFrame` para a seção "Debug Settings".
4.  Dentro deste frame, adicione um `CTkSwitch` para controlar a variável `save_audio_var`.
5.  No método `apply_settings`, obtenha o valor de `save_audio_var` e passe-o para `core_instance.apply_settings_from_external`.

**Bloco de Código Relevante:**

##### ### Código Antigo
```python
# whisper_tkinter.py:2647
    sound_volume_var = ctk.DoubleVar(value=core_instance.sound_volume); settings_vars.append(sound_volume_var)
    text_correction_enabled_var = ctk.BooleanVar(value=core_instance.text_correction_enabled); settings_vars.append(text_correction_enabled_var)
    text_correction_service_var = ctk.StringVar(value=core_instance.text_correction_service); settings_vars.append(text_correction_service_var)
    openrouter_api_key_var = ctk.StringVar(value=core_instance.openrouter_api_key); settings_vars.append(openrouter_api_key_var)
    openrouter_model_var = ctk.StringVar(value=core_instance.openrouter_model); settings_vars.append(openrouter_model_var)
    gemini_api_key_var = ctk.StringVar(value=core_instance.gemini_api_key); settings_vars.append(gemini_api_key_var)
    gemini_model_var = ctk.StringVar(value=core_instance.gemini_model); settings_vars.append(gemini_model_var)
    gemini_mode_var = ctk.StringVar(value=core_instance.gemini_mode); settings_vars.append(gemini_mode_var)  # Variável para o modo Gemini
    batch_size_var = ctk.IntVar(value=core_instance.batch_size); settings_vars.append(batch_size_var)
    gpu_index_var = ctk.IntVar(value=core_instance.gpu_index); settings_vars.append(gpu_index_var)
    sound_enabled_var = ctk.BooleanVar(value=core_instance.sound_enabled)
    sound_frequency_var = ctk.StringVar(value=str(core_instance.sound_frequency))
    sound_duration_var = ctk.StringVar(value=str(core_instance.sound_duration))
    text_correction_enabled_var = ctk.BooleanVar(value=core_instance.text_correction_enabled)
    text_correction_service_var = ctk.StringVar(value=core_instance.text_correction_service)
    openrouter_api_key_var = ctk.StringVar(value=core_instance.openrouter_api_key)
    openrouter_model_var = ctk.StringVar(value=core_instance.openrouter_model)
    gemini_api_key_var = ctk.StringVar(value=core_instance.gemini_api_key)
    gemini_model_var = ctk.StringVar(value=core_instance.gemini_model)
    gemini_mode_var = ctk.StringVar(value=core_instance.gemini_mode) # Variável para o modo Gemini
    batch_size_var = ctk.StringVar(value=str(core_instance.batch_size))
    gpu_index_var = ctk.StringVar(value=str(core_instance.gpu_index))
    # keyboard_library_var removida pois não é mais usada
```

##### ### Código Novo
```python
# whisper_tkinter.py:2647
    sound_volume_var = ctk.DoubleVar(value=core_instance.sound_volume); settings_vars.append(sound_volume_var)
    text_correction_enabled_var = ctk.BooleanVar(value=core_instance.text_correction_enabled); settings_vars.append(text_correction_enabled_var)
    text_correction_service_var = ctk.StringVar(value=core_instance.text_correction_service); settings_vars.append(text_correction_service_var)
    openrouter_api_key_var = ctk.StringVar(value=core_instance.openrouter_api_key); settings_vars.append(openrouter_api_key_var)
    openrouter_model_var = ctk.StringVar(value=core_instance.openrouter_model); settings_vars.append(openrouter_model_var)
    gemini_api_key_var = ctk.StringVar(value=core_instance.gemini_api_key); settings_vars.append(gemini_api_key_var)
    gemini_model_var = ctk.StringVar(value=core_instance.gemini_model); settings_vars.append(gemini_model_var)
    gemini_mode_var = ctk.StringVar(value=core_instance.gemini_mode); settings_vars.append(gemini_mode_var)  # Variável para o modo Gemini
    batch_size_var = ctk.IntVar(value=core_instance.batch_size); settings_vars.append(batch_size_var)
    gpu_index_var = ctk.IntVar(value=core_instance.gpu_index); settings_vars.append(gpu_index_var)
    save_audio_var = ctk.BooleanVar(value=core_instance.save_audio_for_debug); settings_vars.append(save_audio_var) # Nova variável para salvar áudio
    sound_enabled_var = ctk.BooleanVar(value=core_instance.sound_enabled)
    sound_frequency_var = ctk.StringVar(value=str(core_instance.sound_frequency))
    sound_duration_var = ctk.StringVar(value=str(core_instance.sound_duration))
    text_correction_enabled_var = ctk.BooleanVar(value=core_instance.text_correction_enabled)
    text_correction_service_var = ctk.StringVar(value=core_instance.text_correction_service)
    openrouter_api_key_var = ctk.StringVar(value=core_instance.openrouter_api_key)
    openrouter_model_var = ctk.StringVar(value=core_instance.openrouter_model)
    gemini_api_key_var = ctk.StringVar(value=core_instance.gemini_api_key)
    gemini_model_var = ctk.StringVar(value=core_instance.gemini_model)
    gemini_mode_var = ctk.StringVar(value=core_instance.gemini_mode) # Variável para o modo Gemini
    batch_size_var = ctk.StringVar(value=str(core_instance.batch_size))
    gpu_index_var = ctk.StringVar(value=str(core_instance.gpu_index))
    # keyboard_library_var removida pois não é mais usada
```

##### ### Código Antigo (continuação)
```python
# whisper_tkinter.py:3346
    ctk.CTkEntry(openrouter_model_row, textvariable=openrouter_model_var).pack(side="left", fill="x", expand=True, padx=5)
 
    # --- Action Buttons ---
    # Note: The button_frame was defined earlier in the original code but is placed outside the scrollable frame in CTk
    button_frame = ctk.CTkFrame(settings_win, fg_color="#222831", corner_radius=12) # Recreate outside scrollable
    button_frame.pack(side="bottom", fill="x", padx=10, pady=10)
    ctk.CTkButton(button_frame, text="Apply", command=lambda: apply_settings(), width=120, fg_color="#00a0ff", hover_color="#0078d7").pack(side="right", padx=5) # Already English
    ctk.CTkButton(button_frame, text="Cancel", command=lambda: close_settings(), width=120, fg_color="#393E46", hover_color="#444444").pack(side="right", padx=5) # Already English
```

##### ### Código Novo (continuação)
```python
# whisper_tkinter.py:3346
    ctk.CTkEntry(openrouter_model_row, textvariable=openrouter_model_var).pack(side="left", fill="x", expand=True, padx=5)
 
    # --- Debug Settings Section (NOVA SEÇÃO) ---
    debug_section_frame = ctk.CTkFrame(scrollable, fg_color="#222831", corner_radius=12)
    debug_section_frame.pack(fill="x", pady=(0, 10), padx=0)
    ctk.CTkLabel(debug_section_frame, text="Debug Settings", font=("Segoe UI", 13, "bold"), text_color="#00a0ff").pack(anchor="w", padx=5)
    ctk.CTkSwitch(debug_section_frame, text="Save Audio for Debug", variable=save_audio_var, onvalue=True, offvalue=False).pack(anchor="w", padx=5, pady=(5, 0))

    # --- Action Buttons ---
    # Note: The button_frame was defined earlier in the original code but is placed outside the scrollable frame in CTk
    button_frame = ctk.CTkFrame(settings_win, fg_color="#222831", corner_radius=12) # Recreate outside scrollable
    button_frame.pack(side="bottom", fill="x", padx=10, pady=10)
    ctk.CTkButton(button_frame, text="Apply", command=lambda: apply_settings(), width=120, fg_color="#00a0ff", hover_color="#0078d7").pack(side="right", padx=5) # Already English
    ctk.CTkButton(button_frame, text="Cancel", command=lambda: close_settings(), width=120, fg_color="#393E46", hover_color="#444444").pack(side="right", padx=5) # Already English
```

##### ### Código Antigo (continuação)
```python
# whisper_tkinter.py:3059
                    new_batch_size=batch_size_to_apply,
                    new_gpu_index=gpu_index_to_apply,
                    new_auto_reregister=auto_reregister_to_apply
                ) # Fechar parênteses da chamada da função
            else:
                logging.critical("CRITICAL: apply_settings_from_external method not found on core_instance!")
                messagebox.showerror("Internal Error", "Cannot apply settings: Core method missing.", parent=settings_win) # Already English
                return
        except Exception as e:
            logging.error(f"Error calling apply_settings_from_external from settings thread: {e}", exc_info=True)
            messagebox.showerror("Error", f"Failed to apply settings:\n{e}", parent=settings_win) # Already English
            return

        new_record_key_temp = None
        new_agent_key_temp = None # Resetar também a variável temporária da agent key
        close_settings()
```

##### ### Código Novo (continuação)
```python
# whisper_tkinter.py:3059
                    new_batch_size=batch_size_to_apply,
                    new_gpu_index=gpu_index_to_apply,
                    new_auto_reregister=auto_reregister_to_apply,
                    new_save_audio_for_debug=save_audio_for_debug_to_apply # Passa a nova configuração
                ) # Fechar parênteses da chamada da função
            else:
                logging.critical("CRITICAL: apply_settings_from_external method not found on core_instance!")
                messagebox.showerror("Internal Error", "Cannot apply settings: Core method missing.", parent=settings_win) # Already English
                return
        except Exception as e:
            logging.error(f"Error calling apply_settings_from_external from settings thread: {e}", exc_info=True)
            messagebox.showerror("Error", f"Failed to apply settings:\n{e}", parent=settings_win) # Already English
            return

        new_record_key_temp = None
        new_agent_key_temp = None # Resetar também a variável temporária da agent key
        close_settings()
```

### Tarefa 2: Modificar o Fluxo de Gravação e Transcrição para In-Memory

Esta tarefa é o cerne da mudança, alterando como os dados de áudio são tratados após a gravação e antes da transcrição.

**Contexto:**
- **Arquivo:** [`whisper_tkinter.py`](whisper_tkinter.py)
- **Métodos:**
    - `stop_recording` (chama a função de processamento de áudio)
    - `_save_and_transcribe_task` (será renomeada e modificada para `_process_audio_task`)
    - `_transcribe_audio_task` (será modificada para aceitar dados em memória)
    - `_delete_audio_file` (será chamada condicionalmente)

**Passos de Implementação (Sub-tarefas):**

#### 2.1 Renomear `_save_and_transcribe_task` para `_process_audio_task`

**Objetivo:** Refletir a nova responsabilidade da função, que agora não apenas salva, mas processa o áudio para transcrição.

**Contexto:**
- **Arquivo:** [`whisper_tkinter.py`](whisper_tkinter.py)
- **Local:** Linha 1814 (definição da função) e todas as chamadas a ela.

**Passos de Implementação:**
1.  Altere o nome da função de `_save_and_transcribe_task` para `_process_audio_task`.
2.  Atualize a chamada a esta função em `stop_recording`.

**Bloco de Código Relevante:**

##### ### Código Antigo
```python
# whisper_tkinter.py:1814
    def _save_and_transcribe_task(self, audio_data, agent_mode=False):
        """Saves audio data and starts transcription. When agent_mode is True, the result triggers the agent prompt."""
        logging.info("Save and transcribe task started.")
        self._set_state(STATE_SAVING)
```

##### ### Código Novo
```python
# whisper_tkinter.py:1814
    def _process_audio_task(self, audio_data, agent_mode=False): # Renomeado
        """Processes audio data, optionally saves it for debug, and starts transcription."""
        logging.info("Process audio task started.")
        self._set_state(STATE_TRANSCRIBING) # Mudar estado diretamente para TRANSCRIBING
```

##### ### Código Antigo (chamada em `stop_recording`)
```python
# whisper_tkinter.py:1738
            # self._set_state(STATE_SAVING)
            # Start save task in thread
            threading.Thread(target=self._save_and_transcribe_task, args=(audio_data_copy, agent_mode), daemon=True, name="SaveTranscribeThread").start()
```

##### ### Código Novo (chamada em `stop_recording`)
```python
# whisper_tkinter.py:1738
            # self._set_state(STATE_SAVING) # Removido, estado será definido em _process_audio_task
            # Start process task in thread
            threading.Thread(target=self._process_audio_task, args=(audio_data_copy, agent_mode), daemon=True, name="ProcessAudioThread").start() # Renomeado
```

#### 2.2 Modificar a Lógica em `_process_audio_task`

**Objetivo:** Implementar a lógica condicional de salvamento de arquivo para depuração e passar o array NumPy diretamente para a transcrição.

**Contexto:**
- **Arquivo:** [`whisper_tkinter.py`](whisper_tkinter.py)
- **Local:** Linhas 1814-1869, dentro do método `_process_audio_task` (antigo `_save_and_transcribe_task`).

**Passos de Implementação:**
1.  Altere o estado inicial de `STATE_SAVING` para `STATE_TRANSCRIBING`, pois o foco principal agora é a transcrição.
2.  Mova toda a lógica de salvamento do arquivo WAV para dentro de um bloco `if self.save_audio_for_debug:`.
3.  Ajuste a chamada para `_transcribe_audio_task` para passar o `audio_data` (array NumPy) e, condicionalmente, o `final_filename` para exclusão posterior.

**Bloco de Código Relevante:**

##### ### Código Antigo
```python
# whisper_tkinter.py:1814
    def _save_and_transcribe_task(self, audio_data, agent_mode=False):
        """Saves audio data and starts transcription. When agent_mode is True, the result triggers the agent prompt."""
        logging.info("Save and transcribe task started.")
        self._set_state(STATE_SAVING)

        timestamp = int(time.time())
        temp_filename = f"temp_recording_{timestamp}.wav"
        final_filename = f"recording_{timestamp}.wav"
        saved_successfully = False

        try:
            # --- Save Audio ---
            logging.info(f"Saving audio to {temp_filename}")

            # Converter para formato compatível com int16 antes de salvar
            if audio_data.dtype != np.int16:
                # Check max/min values before scaling to prevent clipping/overflow
                max_val = np.max(np.abs(audio_data))
                if max_val > 1.0:
                    logging.warning(f"Audio data exceeds expected range [-1.0, 1.0] (max abs: {max_val}). Clipping may occur.")
                    audio_data = np.clip(audio_data, -1.0, 1.0)
                audio_data_int16 = (audio_data * (2**15 - 1)).astype(np.int16)
            else:
                audio_data_int16 = audio_data # Already int16

            # Salvar usando wave para garantir compatibilidade
            with wave.open(temp_filename, 'wb') as wf:
                wf.setnchannels(AUDIO_CHANNELS)
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(AUDIO_SAMPLE_RATE)
                wf.writeframes(audio_data_int16.tobytes())

            # Verificar se o arquivo foi salvo corretamente
            if not os.path.exists(temp_filename) or os.path.getsize(temp_filename) == 0:
                raise ValueError("Arquivo WAV vazio ou não criado após gravação")

            os.rename(temp_filename, final_filename)
            logging.info(f"Audio saved as {final_filename} (size: {os.path.getsize(final_filename)} bytes)")
            saved_successfully = True

        except Exception as e:
            logging.error(f"Error saving audio: {e}", exc_info=True)
            self._set_state(STATE_ERROR_AUDIO) # Or a new ERROR_SAVE state?
            self._log_status(f"Error saving audio: {e}", error=True)
            if os.path.exists(temp_filename): self._delete_audio_file(temp_filename) # Cleanup temp
            # Stay in error state - DO NOT proceed to transcription

        # --- Trigger Transcription Task ONLY if save was successful ---
        if saved_successfully:
            self._set_state(STATE_TRANSCRIBING)
            # Run transcription in a new thread
            threading.Thread(target=self._transcribe_audio_task, args=(final_filename, agent_mode), daemon=True, name="TranscriptionThread").start()
        else:
             logging.error("Skipping transcription because audio save failed.")
             # State should already be ERROR_AUDIO
```

##### ### Código Novo
```python
# whisper_tkinter.py:1814
    def _process_audio_task(self, audio_data, agent_mode=False): # Renomeado
        """Processes audio data, optionally saves it for debug, and starts transcription."""
        logging.info("Process audio task started.")
        self._set_state(STATE_TRANSCRIBING) # Mudar estado diretamente para TRANSCRIBING

        final_filename = None # Inicializa como None
        if self.save_audio_for_debug:
            try:
                timestamp = int(time.time())
                temp_filename = f"temp_recording_{timestamp}.wav"
                final_filename = f"recording_{timestamp}.wav"
                
                # Converter para formato compatível com int16 antes de salvar
                if audio_data.dtype != np.int16:
                    # Check max/min values before scaling to prevent clipping/overflow
                    max_val = np.max(np.abs(audio_data))
                    if max_val > 1.0:
                        logging.warning(f"Audio data exceeds expected range [-1.0, 1.0] (max abs: {max_val}). Clipping may occur.")
                        audio_data = np.clip(audio_data, -1.0, 1.0)
                    audio_data_int16 = (audio_data * (2**15 - 1)).astype(np.int16)
                else:
                    audio_data_int16 = audio_data # Already int16

                # Salvar usando wave para garantir compatibilidade
                with wave.open(temp_filename, 'wb') as wf:
                    wf.setnchannels(AUDIO_CHANNELS)
                    wf.setsampwidth(2)  # 16-bit
                    wf.setframerate(AUDIO_SAMPLE_RATE)
                    wf.writeframes(audio_data_int16.tobytes())

                # Verificar se o arquivo foi salvo corretamente
                if not os.path.exists(temp_filename) or os.path.getsize(temp_filename) == 0:
                    raise ValueError("Arquivo WAV vazio ou não criado após gravação")

                os.rename(temp_filename, final_filename)
                logging.info(f"Audio saved for debugging as {final_filename} (size: {os.path.getsize(final_filename)} bytes)")
            except Exception as e:
                logging.error(f"Error saving audio for debug: {e}", exc_info=True)
                # Não impede a transcrição, apenas loga o erro de salvamento
                final_filename = None 

        # Inicia a transcrição com os dados em memória
        # Passa o nome do arquivo para exclusão opcional em _transcribe_audio_task
        thread_args = (audio_data, agent_mode, final_filename) 
        threading.Thread(target=self._transcribe_audio_task, args=thread_args, daemon=True, name="TranscriptionThread").start()
```

#### 2.3 Alterar a Assinatura de `_transcribe_audio_task`

**Objetivo:** Modificar a função para aceitar diretamente o array NumPy de dados de áudio, em vez de um nome de arquivo. Adicionar um parâmetro opcional para o nome do arquivo a ser excluído (se salvo para depuração).

**Contexto:**
- **Arquivo:** [`whisper_tkinter.py`](whisper_tkinter.py)
- **Local:** Linha 1881 (definição da função) e todas as chamadas a ela.

**Passos de Implementação:**
1.  Altere a assinatura da função de `_transcribe_audio_task(self, audio_filename, agent_mode=False)` para `_transcribe_audio_task(self, audio_input, agent_mode=False, audio_filename_to_delete=None)`.

**Bloco de Código Relevante:**

##### ### Código Antigo
```python
# whisper_tkinter.py:1881
    def _transcribe_audio_task(self, audio_filename, agent_mode=False):
        """Transcribes a single audio file. If agent_mode, send result to agent prompt."""
        start_process_time = time.time()
        logging.info(f"Transcription task started for {audio_filename}")
        text_result = None
        transcription_error = None
```

##### ### Código Novo
```python
# whisper_tkinter.py:1881
    def _transcribe_audio_task(self, audio_input, agent_mode=False, audio_filename_to_delete=None): # Assinatura alterada
        """Transcribes audio from a NumPy array. If agent_mode, send result to agent prompt."""
        start_process_time = time.time()
        logging.info(f"Transcription task started for in-memory audio data.") # Log atualizado
        text_result = None
        transcription_error = None
```

#### 2.4 Adaptar `_transcribe_audio_task` para Processamento em Memória

**Objetivo:** Remover a dependência de arquivos WAV e usar o array NumPy diretamente com o pipeline do Whisper. Gerenciar a exclusão do arquivo de depuração.

**Contexto:**
- **Arquivo:** [`whisper_tkinter.py`](whisper_tkinter.py)
- **Local:** Linhas 1881-2052, dentro do método `_transcribe_audio_task`.

**Passos de Implementação:**
1.  Remova todo o bloco de verificação de integridade do arquivo WAV (linhas 1888-1924).
2.  Altere a chamada ao `self.pipe` para usar `audio_input` em vez de `audio_filename`.
3.  Modifique a seção `finally` para chamar `_delete_audio_file` apenas se `audio_filename_to_delete` for fornecido.

**Bloco de Código Relevante:**

##### ### Código Antigo
```python
# whisper_tkinter.py:1888
        # Verificar integridade do arquivo antes de transcrever
        try:
            with wave.open(audio_filename, 'rb') as wf:
                n_channels = wf.getnchannels()
                framerate = wf.getframerate()
                sampwidth = wf.getsampwidth()
                n_frames = wf.getnframes()
                logging.debug(f"WAV check: Channels={n_channels}, Rate={framerate}, Width={sampwidth}, Frames={n_frames}")
                if n_channels != AUDIO_CHANNELS:
                    raise ValueError(f"Invalid channels: {n_channels} (expected {AUDIO_CHANNELS})")
                if framerate != AUDIO_SAMPLE_RATE:
                    raise ValueError(f"Invalid sample rate: {framerate} (expected {AUDIO_SAMPLE_RATE})")
                if sampwidth != 2:  # 16-bit
                    raise ValueError(f"Invalid sample width: {sampwidth} (expected 2)")
                if n_frames == 0:
                    raise ValueError("WAV file has zero frames.")
        except wave.Error as e:
            logging.error(f"Invalid WAV file format for {audio_filename}: {e}")
            transcription_error = e
            text_result = f"[Transcription Error: Invalid WAV format]"
        except ValueError as e:
            logging.error(f"Invalid WAV file properties for {audio_filename}: {e}")
            transcription_error = e
            text_result = f"[Transcription Error: Invalid WAV properties]"
        except Exception as e:
            logging.error(f"Error opening/checking WAV file {audio_filename}: {e}", exc_info=True)
            transcription_error = e
            text_result = f"[Transcription Error: Cannot read WAV]"

        # If WAV check failed, set error state and return early
        if transcription_error:
            with self.transcription_lock:
                self.transcription_in_progress = False # Ensure flag is cleared
            self._set_state(STATE_ERROR_TRANSCRIPTION)
            self._log_status(f"Error: Invalid audio file - {transcription_error}", error=True)
            self._delete_audio_file(audio_filename)
            return
```

##### ### Código Novo
```python
# whisper_tkinter.py:1888 (Este bloco será removido)
        # A verificação de integridade do arquivo WAV é removida, pois o áudio é processado em memória.
        # O áudio_input já é um array NumPy validado pela gravação.
```

##### ### Código Antigo (chamada ao pipeline)
```python
# whisper_tkinter.py:1953
            # Default behavior (language detection):
            # ATENÇÃO: Revertido para 'inputs' (parâmetro posicional) e removida a lógica de soundfile/attention_mask
            # devido a TypeError na versão atual do transformers.
            # A correção para 'input_features' e 'attention_introduzida
            # após a atualização da biblioteca transformers para uma versão compatível.
            result = self.pipe(audio_filename, chunk_length_s=30, batch_size=self.batch_size, return_timestamps=False)
```

##### ### Código Novo (chamada ao pipeline)
```python
# whisper_tkinter.py:1953
            # O pipeline do transformers aceita diretamente um array NumPy
            result = self.pipe(audio_input, chunk_length_s=30, batch_size=self.batch_size, return_timestamps=False) # Usa audio_input
```

##### ### Código Antigo (bloco `finally`)
```python
# whisper_tkinter.py:2047
            # --- Cleanup ---
            self._delete_audio_file(audio_filename) # Delete file after processing
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logging.debug("Cleared GPU cache after transcription task.")
```

##### ### Código Novo (bloco `finally`)
```python
# whisper_tkinter.py:2047
            # --- Cleanup ---
            # Exclui o arquivo de áudio APENAS se ele foi salvo para depuração
            if audio_filename_to_delete:
                self._delete_audio_file(audio_filename_to_delete)
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logging.debug("Cleared GPU cache after transcription task.")
```

### Tarefa 3: Atualizar a Interface Gráfica de Configurações

Esta tarefa garante que a nova opção `save_audio_for_debug` seja visível e configurável através da janela de configurações do aplicativo.

**Contexto:**
- **Arquivo:** [`whisper_tkinter.py`](whisper_tkinter.py)
- **Função:** `run_settings_gui` (responsável por construir a janela de configurações).
- **Método:** `apply_settings` (dentro de `run_settings_gui`, responsável por coletar os valores da GUI e passá-los para o core).

**Passos de Implementação (Sub-tarefas):**

#### 3.1 Adicionar a Variável `save_audio_var`

**Objetivo:** Criar uma variável `BooleanVar` para vincular ao `CTkSwitch` na GUI.

**Contexto:**
- **Arquivo:** [`whisper_tkinter.py`](whisper_tkinter.py)
- **Local:** Linhas 2633-2670, dentro da função `run_settings_gui`, na seção de declaração de variáveis.

**Passos de Implementação:**
1.  Adicione `save_audio_var = ctk.BooleanVar(value=core_instance.save_audio_for_debug); settings_vars.append(save_audio_var)` junto com as outras variáveis.

**Bloco de Código Relevante:**

##### ### Código Antigo
```python
# whisper_tkinter.py:2647
    sound_volume_var = ctk.DoubleVar(value=core_instance.sound_volume); settings_vars.append(sound_volume_var)
    text_correction_enabled_var = ctk.BooleanVar(value=core_instance.text_correction_enabled); settings_vars.append(text_correction_enabled_var)
    text_correction_service_var = ctk.StringVar(value=core_instance.text_correction_service); settings_vars.append(text_correction_service_var)
    openrouter_api_key_var = ctk.StringVar(value=core_instance.openrouter_api_key); settings_vars.append(openrouter_api_key_var)
    openrouter_model_var = ctk.StringVar(value=core_instance.openrouter_model); settings_vars.append(openrouter_model_var)
    gemini_api_key_var = ctk.StringVar(value=core_instance.gemini_api_key); settings_vars.append(gemini_api_key_var)
    gemini_model_var = ctk.StringVar(value=core_instance.gemini_model); settings_vars.append(gemini_model_var)
    gemini_mode_var = ctk.StringVar(value=core_instance.gemini_mode); settings_vars.append(gemini_mode_var)  # Variável para o modo Gemini
    batch_size_var = ctk.IntVar(value=core_instance.batch_size); settings_vars.append(batch_size_var)
    gpu_index_var = ctk.IntVar(value=core_instance.gpu_index); settings_vars.append(gpu_index_var)
    sound_enabled_var = ctk.BooleanVar(value=core_instance.sound_enabled)
    sound_frequency_var = ctk.StringVar(value=str(core_instance.sound_frequency))
    sound_duration_var = ctk.StringVar(value=str(core_instance.sound_duration))
    text_correction_enabled_var = ctk.BooleanVar(value=core_instance.text_correction_enabled)
    text_correction_service_var = ctk.StringVar(value=core_instance.text_correction_service)
    openrouter_api_key_var = ctk.StringVar(value=core_instance.openrouter_api_key)
    openrouter_model_var = ctk.StringVar(value=core_instance.openrouter_model)
    gemini_api_key_var = ctk.StringVar(value=core_instance.gemini_api_key)
    gemini_model_var = ctk.StringVar(value=core_instance.gemini_model)
    gemini_mode_var = ctk.StringVar(value=core_instance.gemini_mode) # Variável para o modo Gemini
    batch_size_var = ctk.StringVar(value=str(core_instance.batch_size))
    gpu_index_var = ctk.StringVar(value=str(core_instance.gpu_index))
    # keyboard_library_var removida pois não é mais usada
```

##### ### Código Novo
```python
# whisper_tkinter.py:2647
    sound_volume_var = ctk.DoubleVar(value=core_instance.sound_volume); settings_vars.append(sound_volume_var)
    text_correction_enabled_var = ctk.BooleanVar(value=core_instance.text_correction_enabled); settings_vars.append(text_correction_enabled_var)
    text_correction_service_var = ctk.StringVar(value=core_instance.text_correction_service); settings_vars.append(text_correction_service_var)
    openrouter_api_key_var = ctk.StringVar(value=core_instance.openrouter_api_key); settings_vars.append(openrouter_api_key_var)
    openrouter_model_var = ctk.StringVar(value=core_instance.openrouter_model); settings_vars.append(openrouter_model_var)
    gemini_api_key_var = ctk.StringVar(value=core_instance.gemini_api_key); settings_vars.append(gemini_api_key_var)
    gemini_model_var = ctk.StringVar(value=core_instance.gemini_model); settings_vars.append(gemini_model_var)
    gemini_mode_var = ctk.StringVar(value=core_instance.gemini_mode); settings_vars.append(gemini_mode_var)  # Variável para o modo Gemini
    batch_size_var = ctk.IntVar(value=core_instance.batch_size); settings_vars.append(batch_size_var)
    gpu_index_var = ctk.IntVar(value=core_instance.gpu_index); settings_vars.append(gpu_index_var)
    save_audio_var = ctk.BooleanVar(value=core_instance.save_audio_for_debug); settings_vars.append(save_audio_var) # Nova variável para salvar áudio
    sound_enabled_var = ctk.BooleanVar(value=core_instance.sound_enabled)
    sound_frequency_var = ctk.StringVar(value=str(core_instance.sound_frequency))
    sound_duration_var = ctk.StringVar(value=str(core_instance.sound_duration))
    text_correction_enabled_var = ctk.BooleanVar(value=core_instance.text_correction_enabled)
    text_correction_service_var = ctk.StringVar(value=core_instance.text_correction_service)
    openrouter_api_key_var = ctk.StringVar(value=core_instance.openrouter_api_key)
    openrouter_model_var = ctk.StringVar(value=core_instance.openrouter_model)
    gemini_api_key_var = ctk.StringVar(value=core_instance.gemini_api_key)
    gemini_model_var = ctk.StringVar(value=core_instance.gemini_model)
    gemini_mode_var = ctk.StringVar(value=core_instance.gemini_mode) # Variável para o modo Gemini
    batch_size_var = ctk.StringVar(value=str(core_instance.batch_size))
    gpu_index_var = ctk.StringVar(value=str(core_instance.gpu_index))
    # keyboard_library_var removida pois não é mais usada
```

#### 3.2 Adicionar a Seção "Debug Settings" com o `CTkSwitch`

**Objetivo:** Criar a interface visual para a nova configuração.

**Contexto:**
- **Arquivo:** [`whisper_tkinter.py`](whisper_tkinter.py)
- **Local:** Linhas 3346-3359, após a seção "OpenRouter API Section".

**Passos de Implementação:**
1.  Crie um novo `CTkFrame` para a seção "Debug Settings".
2.  Adicione um `CTkLabel` para o título da seção.
3.  Adicione um `ctk.CTkSwitch` com o texto "Save Audio for Debug" e vincule-o à `save_audio_var`.

**Bloco de Código Relevante:**

##### ### Código Antigo
```python
# whisper_tkinter.py:3346
    ctk.CTkEntry(openrouter_model_row, textvariable=openrouter_model_var).pack(side="left", fill="x", expand=True, padx=5)
 
    # --- Action Buttons ---
    # Note: The button_frame was defined earlier in the original code but is placed outside the scrollable frame in CTk
    button_frame = ctk.CTkFrame(settings_win, fg_color="#222831", corner_radius=12) # Recreate outside scrollable
    button_frame.pack(side="bottom", fill="x", padx=10, pady=10)
    ctk.CTkButton(button_frame, text="Apply", command=lambda: apply_settings(), width=120, fg_color="#00a0ff", hover_color="#0078d7").pack(side="right", padx=5) # Already English
    ctk.CTkButton(button_frame, text="Cancel", command=lambda: close_settings(), width=120, fg_color="#393E46", hover_color="#444444").pack(side="right", padx=5) # Already English
```

##### ### Código Novo
```python
# whisper_tkinter.py:3346
    ctk.CTkEntry(openrouter_model_row, textvariable=openrouter_model_var).pack(side="left", fill="x", expand=True, padx=5)
 
    # --- Debug Settings Section (NOVA SEÇÃO) ---
    debug_section_frame = ctk.CTkFrame(scrollable, fg_color="#222831", corner_radius=12)
    debug_section_frame.pack(fill="x", pady=(0, 10), padx=0)
    ctk.CTkLabel(debug_section_frame, text="Debug Settings", font=("Segoe UI", 13, "bold"), text_color="#00a0ff").pack(anchor="w", padx=5)
    ctk.CTkSwitch(debug_section_frame, text="Save Audio for Debug", variable=save_audio_var, onvalue=True, offvalue=False).pack(anchor="w", padx=5, pady=(5, 0))

    # --- Action Buttons ---
    # Note: The button_frame was defined earlier in the original code but is placed outside the scrollable frame in CTk
    button_frame = ctk.CTkFrame(settings_win, fg_color="#222831", corner_radius=12) # Recreate outside scrollable
    button_frame.pack(side="bottom", fill="x", padx=10, pady=10)
    ctk.CTkButton(button_frame, text="Apply", command=lambda: apply_settings(), width=120, fg_color="#00a0ff", hover_color="#0078d7").pack(side="right", padx=5) # Already English
    ctk.CTkButton(button_frame, text="Cancel", command=lambda: close_settings(), width=120, fg_color="#393E46", hover_color="#444444").pack(side="right", padx=5) # Already English
```

#### 3.3 Passar o Valor da Nova Configuração em `apply_settings`

**Objetivo:** Coletar o estado do `CTkSwitch` e passá-lo para o método `apply_settings_from_external` do core.

**Contexto:**
- **Arquivo:** [`whisper_tkinter.py`](whisper_tkinter.py)
- **Local:** Linhas 2949-3062, dentro da função `apply_settings`.

**Passos de Implementação:**
1.  Obtenha o valor de `save_audio_var` e armazene-o em uma variável local, por exemplo, `save_audio_for_debug_to_apply`.
2.  Passe esta nova variável como argumento para `core_instance.apply_settings_from_external`.

**Bloco de Código Relevante:**

##### ### Código Antigo
```python
# whisper_tkinter.py:3039
                return

        models_text = gemini_models_textbox.get("1.0", "end-1c")
        new_models_list = [line.strip() for line in models_text.split("\n") if line.strip()]
        if not new_models_list:
            messagebox.showwarning("Invalid Value", "The model list cannot be empty. Please add at least one model.", parent=settings_win)
            return


        try:
            if hasattr(core_instance, 'apply_settings_from_external'):
                core_instance.apply_settings_from_external(
                    # Pass all relevant settings to the core instance method
                    new_key=key_to_apply,
                    new_mode=mode_to_apply,
                    new_auto_paste=auto_paste_to_apply,
                    new_sound_enabled=sound_enabled_to_apply,
                    new_sound_frequency=sound_freq_to_apply,
                    new_sound_duration=sound_duration_to_apply,
                    new_sound_volume=sound_volume_to_apply,
                    new_agent_key=agent_key_to_apply,
                    new_text_correction_enabled=text_correction_enabled_var.get(),
                    new_text_correction_service=text_correction_service_var.get(),
                    new_openrouter_api_key=openrouter_api_key_var.get(),
                    new_openrouter_model=openrouter_model_var.get(),
                    new_gemini_api_key=gemini_api_key_var.get(),
                    new_gemini_model=gemini_model_var.get(),
                    new_gemini_agent_model=model_to_apply,
                    new_agent_auto_paste=paste_to_apply,
                    new_gemini_model_options=new_models_list,
                    new_batch_size=batch_size_to_apply,
                    new_gpu_index=gpu_index_to_apply,
                    new_auto_reregister=auto_reregister_to_apply
                ) # Fechar parênteses da chamada da função
            else:
                logging.critical("CRITICAL: apply_settings_from_external method not found on core_instance!")
                messagebox.showerror("Internal Error", "Cannot apply settings: Core method missing.", parent=settings_win) # Already English
                return
        except Exception as e:
            logging.error(f"Error calling apply_settings_from_external from settings thread: {e}", exc_info=True)
            messagebox.showerror("Error", f"Failed to apply settings:\n{e}", parent=settings_win) # Already English
            return

        new_record_key_temp = None
        new_agent_key_temp = None # Resetar também a variável temporária da agent key
        close_settings()
```

##### ### Código Novo
```python
# whisper_tkinter.py:3039
                return

        models_text = gemini_models_textbox.get("1.0", "end-1c")
        new_models_list = [line.strip() for line in models_text.split("\n") if line.strip()]
        if not new_models_list:
            messagebox.showwarning("Invalid Value", "The model list cannot be empty. Please add at least one model.", parent=settings_win)
            return

        save_audio_for_debug_to_apply = save_audio_var.get() # Obtém o valor da nova configuração

        try:
            if hasattr(core_instance, 'apply_settings_from_external'):
                core_instance.apply_settings_from_external(
                    # Pass all relevant settings to the core instance method
                    new_key=key_to_apply,
                    new_mode=mode_to_apply,
                    new_auto_paste=auto_paste_to_apply,
                    new_sound_enabled=sound_enabled_to_apply,
                    new_sound_frequency=sound_freq_to_apply,
                    new_sound_duration=sound_duration_to_apply,
                    new_sound_volume=sound_volume_to_apply,
                    new_agent_key=agent_key_to_apply,
                    new_text_correction_enabled=text_correction_enabled_var.get(),
                    new_text_correction_service=text_correction_service_var.get(),
                    new_openrouter_api_key=openrouter_api_key_var.get(),
                    new_openrouter_model=openrouter_model_var.get(),
                    new_gemini_api_key=gemini_api_key_var.get(),
                    new_gemini_model=gemini_model_var.get(),
                    new_gemini_agent_model=model_to_apply,
                    new_agent_auto_paste=paste_to_apply,
                    new_gemini_model_options=new_models_list,
                    new_batch_size=batch_size_to_apply,
                    new_gpu_index=gpu_index_to_apply,
                    new_auto_reregister=auto_reregister_to_apply,
                    new_save_audio_for_debug=save_audio_for_debug_to_apply # Passa a nova configuração
                ) # Fechar parênteses da chamada da função
            else:
                logging.critical("CRITICAL: apply_settings_from_external method not found on core_instance!")
                messagebox.showerror("Internal Error", "Cannot apply settings: Core method missing.", parent=settings_win) # Already English
                return
        except Exception as e:
            logging.error(f"Error calling apply_settings_from_external from settings thread: {e}", exc_info=True)
            messagebox.showerror("Error", f"Failed to apply settings:\n{e}", parent=settings_win) # Already English
            return

        new_record_key_temp = None
        new_agent_key_temp = None # Resetar também a variável temporária da agent key
        close_settings()
```

## Critérios de Verificação e Teste

Para garantir que as modificações foram implementadas corretamente e não introduziram regressões, os seguintes testes devem ser realizados:

- [ ] **Verificação 1: Transcrição em Memória (Padrão)**
    - **Cenário:** Iniciar o aplicativo com a configuração `save_audio_for_debug` desativada (valor padrão).
    - **Passos:**
        1.  Certifique-se de que o `config.json` não contenha a chave `"save_audio_for_debug"` ou que ela esteja definida como `false`.
        2.  Inicie o `whisper_tkinter.py`.
        3.  Realize uma gravação de áudio (curta ou longa).
        4.  Observe o diretório do aplicativo.
    - **Resultado Esperado:**
        - A transcrição deve ocorrer normalmente e ser colada (se `auto_paste` estiver ativado).
        - **Nenhum arquivo `.wav` temporário ou final deve ser criado** no diretório do aplicativo durante ou após a transcrição.
        - Verifique os logs para confirmar que não há mensagens de "Saving audio to..." ou "Audio saved as...".

- [ ] **Verificação 2: Transcrição com Salvamento para Depuração**
    - **Cenário:** Ativar a configuração `save_audio_for_debug` e verificar o salvamento e exclusão do arquivo.
    - **Passos:**
        1.  Abra a janela de configurações do aplicativo.
        2.  Ative a opção "Save Audio for Debug" (o novo `CTkSwitch`).
        3.  Clique em "Apply" e feche a janela de configurações.
        4.  Realize uma gravação de áudio.
        5.  Observe o diretório do aplicativo.
    - **Resultado Esperado:**
        - A transcrição deve ocorrer normalmente e ser colada.
        - Um arquivo `recording_<timestamp>.wav` **deve ser criado** no diretório do aplicativo durante o processo.
        - Após a conclusão da transcrição, este arquivo `recording_<timestamp>.wav` **deve ser automaticamente excluído**.
        - Verifique os logs para confirmar as mensagens de "Saving audio to..." e "Deleted audio file...".

- [ ] **Verificação 3: Persistência da Configuração**
    - **Cenário:** Verificar se a configuração `save_audio_for_debug` é persistida corretamente entre as sessões.
    - **Passos:**
        1.  Inicie o aplicativo.
        2.  Abra a janela de configurações e ative a opção "Save Audio for Debug".
        3.  Clique em "Apply" e feche a janela.
        4.  Feche o aplicativo completamente (via ícone da bandeja -> Exit).
        5.  Reabra o aplicativo.
        6.  Abra a janela de configurações novamente.
    - **Resultado Esperado:** A opção "Save Audio for Debug" deve estar **ativada** na interface, refletindo a última configuração salva.

- [ ] **Verificação 4: Teste de Regressão Geral**
    - **Cenário:** Assegurar que nenhuma funcionalidade existente foi quebrada pelas modificações.
    - **Passos:**
        1.  Testar o modo de gravação "Toggle" (iniciar/parar com a mesma tecla).
        2.  Testar o modo de gravação "Press/Hold" (gravar enquanto a tecla está pressionada).
        3.  Testar a funcionalidade de "Auto Paste".
        4.  Testar a funcionalidade de "Agent Mode" (gravação de comando e processamento pelo Gemini).
        5.  Testar a correção de texto com OpenRouter (se configurado).
        6.  Testar a correção de texto com Gemini (se configurado, em ambos os modos "Correction" e "General").
        7.  Testar os sons de feedback (início/fim de gravação).
        8.  Testar a detecção e re-registro de hotkeys.
        9.  Testar a funcionalidade de "Test Sound" na janela de configurações.
        10. Testar a alteração de outras configurações (ex: `batch_size`, `gpu_index`, `min_record_duration`).
    - **Resultado Esperado:** Todas as funcionalidades devem operar como esperado, sem erros ou comportamentos inesperados.