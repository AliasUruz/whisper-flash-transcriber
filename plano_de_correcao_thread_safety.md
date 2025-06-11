# Plano de Ação: Correção de Violação de Thread-Safety em whisper_tkinter.py

## 1. Objetivo Estratégico

Refatorar o mecanismo de abertura da janela de configurações para que todas as operações de GUI (criação, verificação, manipulação de widgets) sejam executadas exclusivamente pela `MainThread`, eliminando a `RuntimeError: main thread is not in main loop` e garantindo a estabilidade da aplicação.

## 2. Princípio Técnico Chave

A `PystrayThread` (ou qualquer outra thread secundária) NUNCA deve invocar diretamente uma função do `Tkinter`. A comunicação deve ser feita de forma assíncrona, delegando a tarefa para a `MainThread` através do sistema de fila de eventos (`ui_event_queue`) já existente.

## 3. Fases de Implementação

### Fase 1: Desacoplamento da Lógica de GUI do Callback de Evento

Nesta fase, iremos remover a lógica de GUI da função que é diretamente chamada pela thread insegura.

*   **Passo 1.1: Localização**
    *   **Arquivo**: `whisper_tkinter.py`
    *   **Função**: `on_settings_menu_click(*_)`

*   **Passo 1.2: Análise do Código Legado (Problemático)**
    ```python
    def on_settings_menu_click(*_):
        """Abre a janela de configurações ou foca nela se já estiver aberta."""
        global settings_window_instance

        # ANÁLISE: A linha abaixo é o ponto de falha. `winfo_exists()` é uma chamada de GUI.
        # Ela está sendo executada pela `PystrayThread`, o que é proibido.
        if settings_window_instance and settings_window_instance.winfo_exists():
            logging.info("Settings window already exists. Focusing.")
            try:
                # ANÁLISE: `lift()` e `focus_force()` também são operações de GUI.
                settings_window_instance.lift()
                settings_window_instance.focus_force()
            except Exception as e:
                logging.warning(f"Could not focus existing settings window, allowing recreation: {e}")
                settings_window_instance = None
        
        # ANÁLISE: A criação de uma nova thread que por sua vez manipula a GUI também é problemática
        # se não for gerenciada a partir da MainThread.
        if not (settings_window_instance and settings_window_instance.winfo_exists()):
            logging.info("Starting new settings window thread...")
            settings_thread = threading.Thread(
                target=run_settings_gui, daemon=True, name="SettingsGUIThread")
            settings_thread.start()
    ```

*   **Passo 1.3: Implementação da Delegação (Código-Alvo)**
    A função será reescrita para ter uma única responsabilidade: enviar uma mensagem para a `MainThread`.
    ```python
    def on_settings_menu_click(*_):
        """Enfileira um evento para abrir a janela de configurações na thread da UI."""
        global core_instance
        if core_instance:
            logging.info("Settings menu clicked. Enqueuing 'open_settings' event for UI thread.")
            event = {"type": "open_settings"}
            core_instance.ui_event_queue.put(event)
        else:
            logging.error("Cannot open settings: core_instance is not available.")
    ```

*   **Passo 1.4: Validação da Fase 1**
    Após a modificação, o corpo da função `on_settings_menu_click` deve conter apenas a lógica para acessar `core_instance` e usar `ui_event_queue.put()`. Nenhuma chamada a `winfo_exists`, `lift`, `focus_force` ou criação de `threading.Thread` para a GUI deve permanecer nesta função.

### Fase 2: Implementação do Manipulador de Evento na Thread Segura

Agora, vamos ensinar a `MainThread` a como agir quando receber a mensagem que criamos na Fase 1.

*   **Passo 2.1: Localização**
    *   **Arquivo**: `whisper_tkinter.py`
    *   **Função**: `process_ui_events(root, q)`

*   **Passo 2.2: Análise da Estrutura Existente**
    A função já possui uma estrutura `if/elif` para processar eventos como `"error"`, `"state_update"` e `"paste"`. Iremos adicionar uma nova condição a esta estrutura.

*   **Passo 2.3: Inserção do Novo Manipulador e Transplante da Lógica de GUI (Código-Alvo)**
    A lógica removida na Fase 1 será inserida aqui, dentro de um novo bloco `elif`, onde sua execução é segura.
    ```python
    def process_ui_events(root, q):
        """Verifica a fila de eventos e processa-os na thread da GUI."""
        try:
            event = q.get_nowait() # Pega um item sem bloquear
            if event["type"] == "error":
                import tkinter.messagebox as messagebox
                messagebox.showerror(event["title"], event["message"])

            elif event["type"] == "state_update":
                global core_instance
                if core_instance and hasattr(core_instance, 'set_state_update_callback'):
                    update_tray_icon(event["state"])
                else:
                    logging.warning("core_instance ou set_state_update_callback não disponível para state_update.")

            elif event["type"] == "paste":
                try:
                    pyautogui.hotkey('ctrl', 'v')
                    logging.info("Pasted transcription via UI event queue.")
                except Exception as e:
                    logging.error(f"Error pasting via UI event queue: {e}")

            # INÍCIO DA MODIFICAÇÃO
            elif event["type"] == "open_settings": 
                logging.info("Processing 'open_settings' event from UI queue.")
                global settings_window_instance
                if settings_window_instance and settings_window_instance.winfo_exists():
                    logging.info("Settings window already exists. Focusing.")
                    try:
                        settings_window_instance.lift()
                        settings_window_instance.focus_force()
                    except Exception as e:
                        logging.warning(f"Could not focus existing settings window, allowing recreation: {e}")
                        settings_window_instance = None 
                
                if not (settings_window_instance and settings_window_instance.winfo_exists()):
                    logging.info("Starting new settings window thread...")
                    settings_thread = threading.Thread(
                        target=run_settings_gui, daemon=True, name="SettingsGUIThread")
                    settings_thread.start()
            # FIM DA MODIFICAÇÃO

        except queue.Empty:
            pass 
        except Exception as e:
            logging.error(f"Erro ao processar evento da UI: {e}", exc_info=True)
        finally:
            root.after(100, process_ui_events, root, q)
    ```

*   **Passo 2.4: Validação da Fase 2**
    A função `process_ui_events` deve agora conter o bloco `elif event["type"] == "open_settings":` completo, com toda a lógica que foi removida da função `on_settings_menu_click`.