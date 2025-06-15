# Plano de Ação Técnico: Otimização Avançada do Whisper Tkinter (v2.0)

## 1. Objetivo Principal

Transformar a aplicação `whisper_tkinter` em uma ferramenta de transcrição de alto desempenho, focada em qualidade, velocidade e eficiência para áudios longos (1 a 10+ minutos). Aprimorar drasticamente a inteligência do sistema no gerenciamento de recursos de hardware (GPU) e introduzir feedback em tempo real para o usuário, criando uma experiência fluida, automatizada e profissional.

## 2. Racional da Mudança

A arquitetura atual, embora funcional, apresenta gargalos significativos que limitam seu desempenho e a qualidade da experiência do usuário em casos de uso mais exigentes:

1.  **Latência em Áudios Longos:** O modelo "gravar tudo, depois transcrever" cria uma longa espera para o usuário. Para uma gravação de 10 minutos, o usuário aguarda os 10 minutos da gravação e, em seguida, um tempo adicional significativo para o processamento, sem nenhum feedback em tempo real.
2.  **Uso Ineficiente de Recursos:** O sistema não se adapta dinamicamente às condições atuais do sistema. O `batch_size` da GPU é definido no início e não considera a VRAM *disponível* no momento da transcrição, o que pode levar a erros de `OutOfMemory` (OOM) ou a um subaproveitamento da capacidade da GPU.
3.  **Qualidade de Transcrição Estagnada:** Para áudios muito longos, a falta de uma detecção de atividade de voz (VAD) mais sofisticada significa que silêncios são processados desnecessariamente. A ausência de diarização (identificação de falantes) limita a clareza em transcrições de conversas.
4.  **Intervenção Manual do Usuário:** O usuário precisa selecionar manualmente o índice da GPU e adivinhar um `batch_size` apropriado, o que exige conhecimento técnico e é propenso a erros.
5.  **Falta de Feedback:** O usuário não tem visibilidade sobre o que está sendo transcrito até o final do processo, o que pode ser frustrante e ineficiente.

Este plano de ação visa resolver esses problemas, introduzindo uma arquitetura de streaming, gerenciamento dinâmico de recursos, feedback em tempo real e melhorias na pipeline de processamento de áudio.

## 3. Plano de Ação Detalhado

---

### **Épico 1: Otimização de Performance e Gerenciamento de GPU**

**Racional:** Automatizar e otimizar o uso da GPU para maximizar a velocidade de transcrição e evitar erros de memória, proporcionando uma experiência "plug-and-play" para o usuário.

#### **Tarefa 1.1: Implementar Gerenciamento Dinâmico de `batch_size`**

*   **Contexto:** Modificar a classe `WhisperCore` no arquivo [`whisper_tkinter.py`](whisper_tkinter.py) para calcular o `batch_size` dinamicamente antes de cada transcrição, com base na VRAM *livre* no momento.
*   **Passos de Implementação:**
    1.  Criar um novo método na classe `WhisperCore` chamado `_get_dynamic_batch_size()`.
    2.  Este método deve verificar se a CUDA está disponível (`torch.cuda.is_available()`) e se um índice de GPU válido (`>= 0`) está selecionado. Se não, retorna um `batch_size` seguro para CPU (ex: 4).
    3.  Se a CUDA estiver disponível, ele deve usar `torch.cuda.mem_get_info(self.gpu_index)` para obter a memória VRAM livre (`free_memory_bytes`).
    4.  Implementar uma lógica de mapeamento (tabela/if-elif-else) para a VRAM livre para um `batch_size` apropriado. A lógica deve ser conservadora para evitar erros de OOM.
    5.  Adicionar logs detalhados para informar a VRAM livre, a VRAM total e qual `batch_size` foi escolhido e por quê.
    6.  Modificar o método `_transcribe_audio_task` para chamar `_get_dynamic_batch_size()` no início da tarefa e usar o valor retornado na chamada da `pipeline` do Whisper, sobrepondo-se ao valor da configuração estática.

*   **Bloco de Código Relevante:**

    **Local:** [`whisper_tkinter.py`](whisper_tkinter.py), dentro da classe `WhisperCore`.

    **### Código Novo (Adicionar este método):**
    ```python
    def _get_dynamic_batch_size(self) -> int:
        """
        Calcula um batch_size apropriado dinamicamente com base na VRAM livre.
        Retorna um valor seguro para CPU se a CUDA não estiver disponível.
        """
        if not torch.cuda.is_available() or self.gpu_index < 0:
            logging.info("GPU não disponível ou não selecionada, usando batch size de CPU (4).")
            return 4

        try:
            device = torch.device(f"cuda:{self.gpu_index}")
            free_memory_bytes, total_memory_bytes = torch.cuda.mem_get_info(device)
            free_memory_gb = free_memory_bytes / (1024**3)
            total_memory_gb = total_memory_bytes / (1024**3)
            logging.info(f"Verificando VRAM para GPU {self.gpu_index}: {free_memory_gb:.2f}GB livres de {total_memory_gb:.2f}GB.")

            # Lógica para determinar o batch size. Estes valores são para o 'large-v3' e podem ser ajustados.
            # A lógica é baseada na VRAM LIVRE.
            if free_memory_gb >= 10.0:
                bs = 32
            elif free_memory_gb >= 6.0:
                bs = 16
            elif free_memory_gb >= 4.0:
                bs = 8
            elif free_memory_gb >= 2.0:
                bs = 4
            else:
                bs = 2
            
            logging.info(f"VRAM livre ({free_memory_gb:.2f}GB) -> Batch size dinâmico selecionado: {bs}")
            return bs

        except Exception as e:
            logging.error(f"Erro ao calcular batch size dinâmico: {e}. Usando valor da configuração: {self.batch_size}", exc_info=True)
            return self.batch_size # Retorna ao valor configurado em caso de erro.
    ```

    **Local:** [`whisper_tkinter.py`](whisper_tkinter.py), método `_transcribe_audio_task`.

    **### Código Antigo:**
    ```python
    result = self.pipe(audio_input, chunk_length_s=30, batch_size=self.batch_size, return_timestamps=False)
    ```

    **### Código Novo:**
    ```python
    # Calcula o batch size dinamicamente antes da transcrição
    dynamic_batch_size = self._get_dynamic_batch_size()

    logging.info(f"Iniciando transcrição com batch_size={dynamic_batch_size}...")
    result = self.pipe(audio_input, chunk_length_s=30, batch_size=dynamic_batch_size, return_timestamps=False)
    ```

#### **Tarefa 1.2: Aprimorar a Interface de Configuração da GPU**

*   **Contexto:** Melhorar a janela de configurações em [`whisper_tkinter.py`](whisper_tkinter.py) para que a seleção de GPU seja mais intuitiva e menos propensa a erros.
*   **Passos de Implementação:**
    1.  Criar uma função auxiliar, `get_available_devices_for_ui()`, que detecta as GPUs disponíveis usando `torch.cuda.device_count()` e `torch.cuda.get_device_name()`.
    2.  A função deve retornar uma lista de strings formatadas, como `["Auto-selecionar (Recomendado)", "CPU", "GPU 0: NVIDIA GeForce RTX 4090"]`.
    3.  Na função `run_settings_gui`, chamar essa função auxiliar para obter a lista de dispositivos.
    4.  Substituir o `CTkEntry` para o `gpu_index` por um `CTkOptionMenu` (dropdown) populado com a lista de dispositivos.
    5.  Na função `apply_settings`, implementar a lógica para mapear a string selecionada (ex: "GPU 0: ...") de volta para um índice numérico (`-1` para Auto, `0`, `1`, etc., e um valor especial como `-2` para CPU, ou simplesmente usar a string). O mais simples é salvar a string e interpretá-la na hora de carregar o modelo. Para este plano, vamos converter para índice.

*   **Bloco de Código Relevante:**

    **Local:** [`whisper_tkinter.py`](whisper_tkinter.py), adicionar como função global ou método estático.

    **### Código Novo (Adicionar esta função):**
    ```python
    def get_available_devices_for_ui():
        """Retorna uma lista de dispositivos para a UI de configurações."""
        devices = ["Auto-selecionar (Recomendado)"]
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            for i in range(num_gpus):
                try:
                    device_name = torch.cuda.get_device_name(i)
                    total_mem_gb = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                    devices.append(f"GPU {i}: {device_name} ({total_mem_gb:.1f}GB)")
                except Exception as e:
                    devices.append(f"GPU {i}: Erro ao obter nome")
                    logging.error(f"Não foi possível obter nome da GPU {i}: {e}")
        devices.append("Forçar CPU")
        return devices
    ```

    **Local:** [`whisper_tkinter.py`](whisper_tkinter.py), na função `run_settings_gui`.

    **### Código Antigo (GPU Index Entry):**
    ```python
    gpu_index_row = ctk.CTkFrame(gpu_section_frame, fg_color="#222831")
    gpu_index_row.pack(fill="x", padx=0, pady=(5, 0))
    ctk.CTkLabel(gpu_index_row, text="GPU Index:", width=120).pack(side="left", padx=5)
    ctk.CTkEntry(gpu_index_row, textvariable=gpu_index_var, width=80).pack(side="left", padx=5)
    ```

    **### Código Novo (GPU Index OptionMenu):**
    ```python
    # Obter a lista de GPUs disponíveis
    available_devices = get_available_devices_for_ui()
    
    # Mapear o índice atual para a string da UI
    current_device_selection = "Auto-selecionar (Recomendado)" # Default
    if core_instance.gpu_index_specified:
        if core_instance.gpu_index >= 0:
            for device_str in available_devices:
                if device_str.startswith(f"GPU {core_instance.gpu_index}"):
                    current_device_selection = device_str
                    break
        elif core_instance.gpu_index == -1: # Se foi especificado como -1, pode ser auto ou CPU forçado
            current_device_selection = "Forçar CPU" # Assumimos que -1 especificado significa CPU

    gpu_selection_var = ctk.StringVar(value=current_device_selection)

    gpu_index_row = ctk.CTkFrame(gpu_section_frame, fg_color="#222831")
    gpu_index_row.pack(fill="x", padx=0, pady=(5, 0))
    ctk.CTkLabel(gpu_index_row, text="Dispositivo:", width=120).pack(side="left", padx=5)
    ctk.CTkOptionMenu(gpu_index_row, variable=gpu_selection_var, values=available_devices).pack(side="left", fill="x", expand=True, padx=5)
    ```

    **Local:** [`whisper_tkinter.py`](whisper_tkinter.py), na função `apply_settings`.

    **### Código Novo (Lógica de conversão da UI para índice):**
    ```python
    # ... dentro de apply_settings() ...
    
    # Converter seleção da UI de dispositivo para índice numérico
    selected_device_str = gpu_selection_var.get()
    gpu_index_to_apply = -1 # Default para "Auto-selecionar"
    
    if "Forçar CPU" in selected_device_str:
        gpu_index_to_apply = -1 # Usamos -1 para CPU forçada
    elif selected_device_str.startswith("GPU"):
        try:
            # Extrai o número do início da string "GPU X: ..."
            gpu_index_to_apply = int(selected_device_str.split(":")[0].replace("GPU", "").strip())
        except (ValueError, IndexError):
            logging.error(f"Não foi possível parsear o índice da GPU da string: '{selected_device_str}'. Usando auto-seleção.")
            gpu_index_to_apply = -1 # Fallback para auto
            
    # Na chamada para core_instance.apply_settings_from_external, passe `new_gpu_index=gpu_index_to_apply`
    # ...
    ```

---

### **Épico 2: Arquitetura de Transcrição por Streaming**

**Racional:** Mudar fundamentalmente o paradigma de "gravar-depois-transcrever" para um fluxo contínuo, reduzindo drasticamente a latência percebida pelo usuário e permitindo o processamento de áudios de duração virtualmente ilimitada.

#### **Tarefa 2.1: Integrar VAD (Voice Activity Detection) com `silero-vad`**

*   **Contexto:** Esta é uma tarefa preparatória crucial para o streaming. Vamos integrar um VAD para detectar segmentos de fala em tempo real.
*   **Passos de Implementação:**
    1.  Adicionar `onnx`, `onnxruntime` e `torchaudio` ao `requirements.txt`.
    2.  Instruir o usuário (ou criar um script de setup) para baixar o modelo `silero_vad.onnx` do repositório do Silero (`snakers4/silero-vad`).
    3.  Criar uma nova classe, `VADManager`, em um novo arquivo `vad_manager.py`.
    4.  A classe `VADManager` deve carregar o modelo VAD e fornecer um método simples, `is_speech(chunk)`, que recebe um chunk de áudio (tensor PyTorch) e retorna `True` se houver fala.
    5.  Modificar o método `_audio_callback` em [`whisper_tkinter.py`](whisper_tkinter.py). Em vez de simplesmente adicionar dados a `self.recording_data`, ele agora deve alimentar os chunks de áudio em uma `queue.Queue` para processamento em outra thread.

*   **Bloco de Código Relevante:**

    **Local:** Novo arquivo `vad_manager.py`.

    **### Código Novo (Completo):**
    ```python
    import torch
    import onnxruntime
    import numpy as np
    import logging
    from typing import Tuple

    class VADManager:
        """Gerencia a Detecção de Atividade de Voz (VAD) usando o modelo Silero."""
        def __init__(self, model_path='silero_vad.onnx', threshold=0.5, sampling_rate=16000):
            """
            Inicializa o VAD.
            Args:
                model_path (str): Caminho para o arquivo do modelo .onnx.
                threshold (float): Limiar de probabilidade para considerar como fala.
                sampling_rate (int): Taxa de amostragem esperada (deve ser 16000).
            """
            try:
                self.model = onnxruntime.InferenceSession(model_path)
                self.threshold = threshold
                self.sampling_rate = sampling_rate
                # Inicializa estados ocultos para o modelo LSTM
                self._h = np.zeros((2, 1, 64), dtype=np.float32)
                self._c = np.zeros((2, 1, 64), dtype=np.float32)
                logging.info(f"Modelo VAD carregado de '{model_path}'.")
            except Exception as e:
                logging.error(f"Falha ao carregar o modelo VAD de '{model_path}': {e}", exc_info=True)
                raise

        def reset_states(self):
            """Reseta os estados internos do modelo VAD."""
            self._h = np.zeros((2, 1, 64), dtype=np.float32)
            self._c = np.zeros((2, 1, 64), dtype=np.float32)

        def is_speech(self, audio_chunk: np.ndarray) -> bool:
            """
            Verifica se um chunk de áudio contém fala.
            O chunk deve ser um array numpy float32 com sample rate de 16000.
            """
            if not isinstance(audio_chunk, np.ndarray):
                raise TypeError("O chunk de áudio deve ser um numpy array.")
            
            if audio_chunk.dtype != np.float32:
                audio_chunk = audio_chunk.astype(np.float32) / 32768.0 # Converte de int16 para float32 se necessário

            if len(audio_chunk.shape) > 1:
                audio_chunk = audio_chunk.mean(axis=1) # Garante que é mono

            # O modelo Silero espera um tensor PyTorch, mas onnxruntime aceita numpy
            audio_tensor = torch.from_numpy(audio_chunk)
            
            ort_inputs = {
                'input': audio_tensor.unsqueeze(0).numpy(),
                'h': self._h,
                'c': self._c,
                'sr': np.array([self.sampling_rate], dtype=np.int64)
            }
            # Executa a inferência
            ort_outs = self.model.run(None, ort_inputs)
            speech_prob = ort_outs[0][0][0]
            
            # Atualiza os estados internos
            self._h, self._c = ort_outs[1], ort_outs[2]

            return speech_prob > self.threshold
    ```

#### **Tarefa 2.2: Refatorar para Processamento em Fila (Queue)**

*   **Contexto:** Substituir a lógica de gravação monolítica por um sistema produtor-consumidor usando `queue.Queue`.
*   **Passos de Implementação:**
    1.  Em `WhisperCore.__init__`, inicializar `self.audio_queue = queue.Queue(maxsize=100)` (para evitar consumo excessivo de memória).
    2.  Inicializar o `VADManager`: `self.vad_manager = VADManager()`.
    3.  Criar uma nova thread consumidora, `_process_audio_queue_task`, que será iniciada por `start_recording`.
    4.  O loop desta thread deve:
        a. Obter um chunk de áudio da `self.audio_queue` com um timeout (ex: `self.audio_queue.get(timeout=1)`).
        b. Passá-lo para `self.vad_manager.is_speech()`.
        c. Manter um buffer de áudio de fala (`speech_buffer`). Se for fala, adiciona ao buffer.
        d. Manter um contador de silêncio (`silence_frames_count`). Se for silêncio, incrementa o contador.
        e. Se o contador de silêncio exceder um limiar (ex: 1 segundo de silêncio) E o `speech_buffer` tiver dados, considerar o buffer um "segmento completo".
        f. Empacotar o `speech_buffer` e enviá-lo para uma nova thread de transcrição.
        g. Limpar o `speech_buffer` e resetar o contador de silêncio.
    5.  O `_audio_callback` se torna o produtor, simplesmente colocando os `indata` na `self.audio_queue`.
    6.  A lógica de `start_recording` e `stop_recording` muda: `start` inicia a thread `_process_audio_queue_task`. `stop` envia um sinal (ex: `None`) para a fila para que a thread termine seu loop e processe qualquer áudio remanescente no `speech_buffer`.

*   **Bloco de Código Relevante:**

    **Local:** [`whisper_tkinter.py`](whisper_tkinter.py), na classe `WhisperCore`.

    **### Código Novo (Lógica da thread consumidora):**
    ```python
    def _process_audio_queue_task(self):
        """
        Consome áudio da fila, usa VAD para detectar fala e envia segmentos para transcrição.
        """
        self.vad_manager.reset_states()
        speech_buffer = []
        
        # Configurações para detecção de segmento
        # Processa um segmento após 1.0 segundo de silêncio.
        SILENCE_THRESHOLD_S = 1.0 
        # Tamanho do chunk em segundos (depende do InputStream, mas podemos estimar)
        CHUNK_S = 0.5 
        silence_chunks_needed = int(SILENCE_THRESHOLD_S / CHUNK_S)
        silence_counter = 0

        logging.info("Thread de processamento de áudio (VAD) iniciada.")

        while self.is_recording:
            try:
                chunk = self.audio_queue.get(timeout=0.5)
                if chunk is None: # Sinal de parada
                    break

                is_speech = self.vad_manager.is_speech(chunk)

                if is_speech:
                    silence_counter = 0
                    speech_buffer.append(chunk)
                else: # É silêncio
                    silence_counter += 1
                    if silence_counter > silence_chunks_needed and speech_buffer:
                        # Silêncio detectado após fala, segmento completo!
                        segment_to_transcribe = np.concatenate(speech_buffer)
                        logging.info(f"Segmento de fala detectado ({len(segment_to_transcribe)/self.AUDIO_SAMPLE_RATE:.2f}s). Enviando para transcrição.")
                        
                        # Envia para transcrição em uma nova thread
                        threading.Thread(
                            target=self._transcribe_audio_task, 
                            args=(segment_to_transcribe,), 
                            daemon=True,
                            name="StreamingTranscriptionThread"
                        ).start()
                        
                        speech_buffer.clear()
                        silence_counter = 0

            except queue.Empty:
                # Timeout, continua o loop para verificar self.is_recording
                continue
        
        # Processar qualquer áudio remanescente no buffer quando a gravação para
        if speech_buffer:
            final_segment = np.concatenate(speech_buffer)
            logging.info(f"Processando segmento final de fala ({len(final_segment)/self.AUDIO_SAMPLE_RATE:.2f}s).")
            threading.Thread(
                target=self._transcribe_audio_task, 
                args=(final_segment,), 
                daemon=True,
                name="FinalTranscriptionThread"
            ).start()
            speech_buffer.clear()

        logging.info("Thread de processamento de áudio (VAD) finalizada.")
    ```

---

### **Épico 3: Refatoração e Melhorias Gerais**

**Racional:** Modernizar a base de código, melhorar a manutenibilidade e adicionar funcionalidades que aprimoram a experiência geral.

#### **Tarefa 3.1: Dividir `whisper_tkinter.py` em Módulos**

*   **Contexto:** O arquivo [`whisper_tkinter.py`](whisper_tkinter.py) tem mais de 3600 linhas, tornando-o difícil de manter.
*   **Passos de Implementação:**
    1.  Criar um novo diretório `src`.
    2.  Mover [`whisper_tkinter.py`](whisper_tkinter.py) para `src/main.py`.
    3.  **Módulo de Configuração:** Criar `src/config_manager.py`. Mover toda a lógica de `_load_config` e `_save_config`, e as constantes relacionadas a configuração (ex: `CONFIG_FILE`, `DEFAULT_CONFIG`) para uma classe `ConfigManager`. O `__init__` da classe principal receberá uma instância de `ConfigManager`.
    4.  **Módulo de Áudio:** Criar `src/audio_handler.py`. Mover a lógica de gravação de áudio (`_audio_callback`, `_record_audio_task`, a nova `_process_audio_queue_task`), a integração com o `VADManager`, e as constantes `AUDIO_SAMPLE_RATE`, `AUDIO_CHANNELS` para uma classe `AudioHandler`.
    5.  **Módulo de Transcrição:** Criar `src/transcription_handler.py`. Mover a lógica de `_load_model_task`, `_transcribe_audio_task`, `_get_dynamic_batch_size`, a interação com a `pipeline` do Whisper e a lógica de correção de texto (`_correct_text_with_...`) para uma classe `TranscriptionHandler`.
    6.  **Módulo de UI:** Criar `src/ui_manager.py`. Mover toda a lógica de criação da janela de configurações (`run_settings_gui`) e do ícone da bandeja (`pystray`, `create_dynamic_menu`, `update_tray_icon`, etc.) para uma classe `UIManager`.
    7.  **Módulo Core:** A classe `WhisperCore` será movida para `src/core.py` e renomeada para `AppCore`. Ela se tornará um orquestrador, inicializando e conectando os novos módulos. Ela manterá o gerenciamento de estado (`self.current_state`) e os locks principais, passando callbacks entre os módulos.
    8.  **Ponto de Entrada:** O arquivo `src/main.py` se tornará muito mais simples, responsável apenas por inicializar o `AppCore` e o `UIManager` e iniciar o loop principal.

*   **Observação para o Codex:** Esta é uma refatoração significativa. Faça-a passo a passo, garantindo que as importações sejam corrigidas a cada arquivo movido. Use `from . import module_name` para importações relativas dentro do pacote `src`.

---

### **Épico 4: Experiência do Usuário e Feedback em Tempo Real**

**Racional:** Fornecer ao usuário visibilidade imediata sobre o processo de transcrição, transformando a experiência de "caixa preta" em uma interação transparente e responsiva.

#### **Tarefa 4.1: Implementar Janela de Transcrição ao Vivo**

*   **Contexto:** Criar uma pequena janela não intrusiva que exibe o texto transcrito à medida que os segmentos são processados pela nova arquitetura de streaming.
*   **Passos de Implementação:**
    1.  Na classe `UIManager` (`src/ui_manager.py`), criar um método `show_live_transcription_window()`.
    2.  Este método criará uma janela `CTkToplevel`, sem bordas (`overrideredirect(True)`), pequena e semi-transparente (`attributes('-alpha', 0.8)`).
    3.  A janela conterá um widget `CTkTextbox` para exibir o texto.
    4.  O `AppCore` precisará de um novo callback, `on_segment_transcribed(text)`.
    5.  O método `_transcribe_audio_task` (agora em `TranscriptionHandler`), após transcrever um segmento, chamará este callback.
    6.  O `UIManager` registrará uma função para este callback que atualiza o `CTkTextbox` na janela ao vivo. A atualização deve ser feita de forma segura para a thread da GUI (usando `main_tk_root.after(0, ...)`).
    7.  `start_recording` deve chamar `show_live_transcription_window()`, e `stop_recording` deve fechá-la após um pequeno delay.

*   **Bloco de Código Relevante:**

    **Local:** `src/ui_manager.py`, dentro da classe `UIManager`.

    **### Código Novo (Lógica da janela ao vivo):**
    ```python
    class UIManager:
        def __init__(self, main_tk_root):
            # ...
            self.live_window = None
            self.live_textbox = None

        def show_live_transcription_window(self):
            if self.live_window and self.live_window.winfo_exists():
                return
            
            self.live_window = ctk.CTkToplevel(self.main_tk_root)
            self.live_window.overrideredirect(True)
            self.live_window.geometry("400x150+50+50") # Posição inicial
            self.live_window.attributes("-alpha", 0.85)
            self.live_window.attributes("-topmost", True)

            self.live_textbox = ctk.CTkTextbox(self.live_window, wrap="word", activate_scrollbars=True)
            self.live_textbox.pack(fill="both", expand=True)
            self.live_textbox.insert("end", "Ouvindo...")

        def update_live_transcription(self, new_text):
            if self.live_textbox and self.live_window.winfo_exists():
                # Limpa o "Ouvindo..." inicial
                if self.live_textbox.get("1.0", "end-1c") == "Ouvindo...":
                    self.live_textbox.delete("1.0", "end")
                
                self.live_textbox.insert("end", new_text + " ")
                self.live_textbox.see("end") # Auto-scroll

        def close_live_transcription_window(self):
            if self.live_window:
                self.live_window.destroy()
                self.live_window = None
                self.live_textbox = None
    ```

## 5. Critérios de Verificação e Teste

### Testes de GPU e Performance:
- [ ] Verificar se a aplicação inicia corretamente em uma máquina sem GPU (modo CPU).
- [ ] Em uma máquina com GPU, verificar se a UI de configurações exibe corretamente a lista de GPUs disponíveis com nome e VRAM.
- [ ] Selecionar uma GPU específica na UI, salvar, reiniciar e confirmar se a GPU correta é usada (verificar logs).
- [ ] Com a GPU selecionada, realizar uma transcrição e verificar nos logs se o `batch_size` dinâmico foi calculado e utilizado com base na VRAM livre.
- [ ] Abrir outro programa que consuma VRAM (ex: um jogo) e realizar uma transcrição para confirmar que o `batch_size` dinâmico se ajusta para um valor menor.

### Testes de Streaming e Áudio Longo:
- [ ] Iniciar a gravação e falar por 15 segundos. Verificar se a janela de transcrição ao vivo aparece e é atualizada com o texto.
- [ ] Ficar em silêncio por 5 segundos durante uma gravação e verificar se o VAD ignora corretamente o silêncio (verificar logs) e não envia segmentos vazios para transcrição.
- [ ] Realizar uma gravação contínua de 5 minutos, alternando entre fala e silêncio, para testar a robustez do sistema de segmentação e a concatenação do texto na janela ao vivo.
- [ ] Pressionar o botão de parar no meio de uma frase e verificar se o segmento final é processado corretamente e adicionado à transcrição final.
- [ ] Verificar se a transcrição final (colada ou copiada) contém o texto completo de todos os segmentos.

### Testes de Experiência do Usuário (UX):
- [ ] Verificar se a janela de transcrição ao vivo aparece em uma posição razoável da tela.
- [ ] Verificar se a janela de transcrição ao vivo se fecha automaticamente após a gravação parar.
- [ ] Verificar se o texto na janela ao vivo tem auto-scroll para a parte mais recente.

### Testes de Regressão:
- [ ] Verificar se o modo "toggle" e "press" ainda funcionam como esperado após a refatoração.
- [ ] Verificar se a tecla de atalho do Agente Gemini ainda funciona.
- [ ] Verificar se a correção de texto com Gemini e OpenRouter ainda é aplicada corretamente (na transcrição final).
- [ ] Verificar se salvar o áudio para debug ainda funciona quando a opção está ativada.
- [ ] Verificar se todas as opções da janela de configurações (som, auto-paste, etc.) ainda são aplicadas corretamente após a refatoração.
- [ ] Verificar se a aplicação encerra de forma limpa, sem erros de threads pendentes.