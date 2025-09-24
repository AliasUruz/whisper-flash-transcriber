# AGENTS.md - Documento Mestre de Contexto do Projeto

**Propósito:** Este documento serve como a fonte única e detalhada da verdade para qualquer desenvolvedor ou agente de IA que
precise entender, manter ou expandir o projeto **"Whisper Flash Transcriber"**. Ele descreve a arquitetura, as convenções e o fluxo
de trabalho do aplicativo.

ORDENS DO USUÁRIO: Whenever you create suggested tasks to the user, you've got to give as much details as possible. Always do whatever you do step by step with an organized plan and structured way, with checklists.

Não é necessário instalar todos os requisitos, apenas o necessário, o mínimo.

Do whatever the user tells you to. Sempre fale com o usuário em português do Brasil.

---

## 1. Visão Geral do Projeto

**Whisper Flash Transcriber** é um utilitário desktop para Windows voltado à produtividade, que oferece transcrição de áudio de
alta performance com um fluxo de trabalho discreto e centrado na privacidade.

### Filosofia e Objetivos

*   **Privacidade em Primeiro Lugar:** A transcrição de áudio (ASR) ocorre inteiramente na máquina do usuário. Esse é o ponto
    central de valor para usuários que lidam com informações sensíveis e não podem ou não querem enviar seus dados de áudio para
    a nuvem.
*   **Eficiência de Fluxo de Trabalho:** O aplicativo é projetado para ser um "assistente invisível". Ele vive na bandeja do
    sistema (`system tray`) e é operado por uma hotkey global. O objetivo é minimizar a troca de contexto do usuário, permitindo
    ditar texto diretamente em qualquer aplicativo.
*   **Flexibilidade e Poder:** Embora o núcleo seja local, o aplicativo oferece pontes para serviços de IA externos (Gemini,
    OpenRouter) para pós-processamento opcional, como correção de pontuação, gramática ou execução de comandos complexos (Modo
    Agente).

### Arquitetura e Componentes

O aplicativo utiliza uma arquitetura modular em Python orquestrada pelo `AppCore` (`src/core.py`), que integra os componentes
especializados descritos abaixo. O estado global da aplicação é propagado por um barramento dedicado (`StateManager`) e todos os
processos potencialmente bloqueantes rodam em threads auxiliares para manter a UI responsiva.

#### Componentes Principais

*   **`AppCore` (`src/core.py`):**
    *   **Responsabilidade:** Coordenar captura de áudio, transcrição, pós-processamento e sincronização com a UI. Ele instancia
        `AudioHandler`, `TranscriptionHandler`, `KeyboardHotkeyManager`, `GeminiAPI` e conecta-se ao catálogo de modelos via
        `model_manager`. O `UIManager` é criado externamente em `main.py` e conectado através do setter `app_core.ui_manager`.
    *   **Interações:** Recebe eventos de hotkey, aciona gravações, repassa áudio para transcrição, solicita downloads de modelos
        conforme necessário e publica notificações através do `StateManager` para manter tray e janelas sincronizadas.

*   **`StateManager` (`src/state_manager.py`):**
    *   **Responsabilidade:** Normalizar eventos e propagar transições de estado (`IDLE`, `RECORDING`, `TRANSCRIBING`, erros
        específicos, etc.) aos assinantes. Mantém histórico do último evento e evita notificações duplicadas.
    *   **Interações:** `AppCore`, `AudioHandler`, `TranscriptionHandler` e `UIManager` utilizam o `StateManager` para coordenar
        fluxos de longa duração. Chamadas são enfileiradas via `Tk.after` quando um root Tkinter é fornecido, garantindo que a UI
        seja atualizada na thread correta.

*   **`ConfigManager` (`src/config_manager.py`):**
    *   **Responsabilidade:** Fonte única da verdade para configurações (`config.json`) e credenciais (`secrets.json`). Expõe
        utilitários para persistir alterações, buscar valores com tipos adequados e resolver timeouts.
    *   **Interações:** Consultado por todos os módulos para parâmetros operacionais: modelo ASR, prompts, hotkeys, modo de
        armazenamento de áudio, serviços externos e limites de recursos.

*   **`model_manager` (`src/model_manager.py`):**
    *   **Responsabilidade:** Catalogar os modelos ASR suportados, listar instalações locais (incluindo cache compartilhado da
        Hugging Face) e gerenciar download/cancelamento com controle de timeout.
    *   **Interações:** Invocado pelo `AppCore` durante a inicialização e quando o usuário solicita instalação/remoção de modelos
        através da UI. Reporta progresso e erros para o `StateManager`.

*   **`AudioHandler` (`src/audio_handler.py`):**
    *   **Responsabilidade:** Capturar áudio com `sounddevice`, alternando entre buffer em memória e arquivo temporário conforme
        limites de RAM. Opcionalmente aplica VAD (`VADManager`) para encerrar capturas automaticamente.
    *   **Interações:** O `AppCore` controla início/fim das gravações. O `AudioHandler` publica estados (`RECORDING`,
        `TRANSCRIBING`, `ERROR_AUDIO`) via `StateManager` e entrega segmentos de áudio finalizados para o núcleo.

*   **`TranscriptionHandler` (`src/transcription_handler.py`):**
    *   **Responsabilidade:** Carregar o backend ASR selecionado (`whisper_flash`, `transformers`, `ct2` etc.), executar a
        transcrição em uma `ThreadPoolExecutor`, emitir callbacks por segmento e coordenar pós-processamento (correção de texto ou
        modo agente).
    *   **Interações:** Recebe áudio do `AppCore`, consulta `ConfigManager` para ajustes dinâmicos (batch size, device, prompts) e
        utiliza `GeminiAPI`/`OpenRouterAPI` quando habilitado. Ao finalizar, devolve o texto (bruto ou pós-processado) para o
        `AppCore`.

*   **`GeminiAPI` (`src/gemini_api.py`) e `OpenRouterAPI` (`src/openrouter_api.py`):**
    *   **Responsabilidade:** Encapsular clientes para correção de texto e modo agente. Validam presença de chaves, resolvem
        timeouts e expõem métodos de alto nível (`get_correction`, `get_agent_response`, `correct_text_async`).
    *   **Interações:** São instanciados pelo `AppCore`/`TranscriptionHandler`. Entram no pipeline somente quando correção de texto
        está habilitada e o serviço possui credenciais válidas; caso contrário, o resultado bruto é retornado sem bloqueios.

*   **`KeyboardHotkeyManager` (`src/keyboard_hotkey_manager.py`):**
    *   **Responsabilidade:** Registrar e monitorar hotkeys globais via biblioteca `keyboard`, com watchdog de saúde para
        re-registro periódico.
    *   **Interações:** Ao detectar eventos, invoca callbacks do `AppCore` (`toggle_recording`, `toggle_agent_mode`, etc.) e reporta
        falhas de registro ao `StateManager`.

*   **`UIManager` (`src/ui_manager.py`):**
    *   **Responsabilidade:** Controlar o ícone de bandeja (`pystray`), construir menus dinâmicos, abrir a janela de configurações
        (`customtkinter`) e refletir estados visuais (cores, tooltips, progresso de download).
    *   **Interações:** É criado em `main.py`, recebe uma referência ao `AppCore` e assina o `StateManager` para reagir a transições.
        Ao persistir configurações, chama métodos do núcleo para reaplicar parâmetros.

### Bootstrap e Ciclo de Vida

*   **`main.py`:** Ponto de entrada. Configura variáveis de ambiente, inicializa logging, executa diagnóstico de CUDA, aplica
    patches de `tkinter`, cria `AppCore` e `UIManager`, conecta ambos e inicia o `Tk.mainloop()` na thread principal.
*   **Gerenciamento de modelos:** Durante a inicialização, o `AppCore` sincroniza o cache de modelos (`model_manager.list_installed`).
    Se o modelo selecionado não estiver disponível localmente, o usuário é questionado antes de disparar `snapshot_download`.
*   **Limpeza:** `AppCore.shutdown()` garante que gravações pendentes sejam descartadas, hotkeys desregistradas e threads
    finalizadas. `atexit` complementa a limpeza da UI.

### Fluxo de Dados Típico

1.  **Usuário pressiona a hotkey configurada** (`F3` por padrão).
2.  `KeyboardHotkeyManager` detecta o evento e chama `AppCore.toggle_recording()`.
3.  `AppCore` solicita ao `StateManager` a transição para `RECORDING`, notifica a UI (ícone vermelho) e ordena que o `AudioHandler`
    inicie a captura.
4.  `AudioHandler` grava áudio, aplicando VAD se configurado, e armazena em RAM ou disco conforme o limite atual.
5.  **Usuário pressiona a hotkey novamente (ou solta, no modo Hold).**
6.  `AppCore` manda o `AudioHandler` finalizar a captura. Se a duração for abaixo do mínimo, a gravação é descartada e o estado
    volta para `IDLE`.
7.  Com áudio válido, `AudioHandler` envia o segmento ao `AppCore`, que transita para `TRANSCRIBING` (ícone azul) e delega o buffer
    ao `TranscriptionHandler`.
8.  `TranscriptionHandler` executa o backend ASR em background. Durante a execução, segmentos parciais podem ser reportados à UI.
9.  Após obter o texto bruto, `_process_ai_pipeline` avalia configurações:
    *   **Modo Agente:** Se ativo, encaminha o texto à `GeminiAPI` para gerar uma resposta; fallback para a transcrição original em
        caso de erro ou indisponibilidade.
    *   **Correção de Texto:** Quando habilitada e com credenciais válidas, envia o texto para Gemini ou OpenRouter (sincrono ou
        assíncrono). Se o serviço estiver indisponível, o resultado bruto é retornado.
10. `TranscriptionHandler` devolve o texto final ao `AppCore`.
11. `AppCore` retorna ao estado `IDLE`, atualiza o histórico, copia o texto para a área de transferência (`pyperclip`) e pode
    simular `Ctrl+V` (`pyautogui`) para colar no aplicativo ativo, conforme configuração.

---

## 2. Construindo e Executando o Projeto

### Gerenciamento de Dependências

O projeto utiliza múltiplos arquivos `requirements*.txt` para diferentes cenários. Use sempre um ambiente virtual (`venv`) para
isolar dependências.

```bash
# Criar e ativar um ambiente virtual
python -m venv venv
.\venv\Scripts\activate
```

*   **`requirements.txt`:** Dependências principais para execução do aplicativo em modo CPU. Instale com `pip install -r requirements.txt`.
*   **`requirements-optional.txt`:** Dependências opcionais (GPU, bibliotecas aceleradas). Instale apenas o necessário, ex.: `torch`
    com CUDA.
*   **`requirements-test.txt`:** Dependências para desenvolvimento e testes automatizados (`pytest`, linters, etc.).

### Executando a Aplicação

O ponto de entrada principal é `src/main.py`. Para executar em modo de desenvolvimento:

```bash
python src/main.py
```

Durante a primeira execução o aplicativo criará `config.json` e `hotkey_config.json` na raiz do projeto.

### Executando a Suíte de Testes

O projeto utiliza `pytest` para testes automatizados. Para executá-los:

```bash
pytest
```

---

## 3. Convenções de Desenvolvimento e Padrões

*   **Configuração como Fonte da Verdade:** Qualquer funcionalidade ajustável deve ser armazenada no `config.json`/`secrets.json`
    e acessada via `ConfigManager`. Evite valores codificados diretamente no código.
*   **Modelo de Threading:** A thread principal é dedicada à UI (`Tkinter mainloop`) e ao `pystray`. Qualquer operação bloqueante
    (download de modelos, inferência ASR, chamadas de rede) deve rodar em thread de fundo via `ThreadPoolExecutor` ou workers
    especializados do backend.
*   **Gerenciamento de Estado:** Utilize sempre o `StateManager` para publicar transições. Estados são observados pela UI, pelo
    `AudioHandler` e por outras rotinas de recuperação de erros. O `AppCore` apenas direciona eventos e não deve manipular ícones
    ou widgets diretamente.
*   **Tratamento de Erros:** Falhas esperadas (ex.: timeout ao chamar Gemini/OpenRouter) devem ser capturadas, registradas e causar
    degradação graciosa (usar texto bruto). Erros críticos (ex.: falha ao carregar o modelo ASR) devem acionar eventos específicos
    (`STATE_ERROR_MODEL`, etc.) para feedback imediato ao usuário.
*   **Download e Cache de Modelos:** Prefira as funções de alto nível do `model_manager` para listar, instalar ou remover modelos.
    Elas garantem consistência de cache e manipulação de exceções (`DownloadCancelledError`).
*   **Estilo de Código:** Seguir PEP 8. O projeto fornece `.flake8` e utilitários em `requirements-test.txt` para lint. Comentários
    e logs devem explicar decisões técnicas relevantes, evitando ruído.
