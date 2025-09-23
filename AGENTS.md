# AGENTS.md - Documento Mestre de Contexto do Projeto

**Propósito:** Este documento serve como uma fonte única e detalhada da verdade para qualquer desenvolvedor ou agente de IA que precise entender, manter ou expandir o projeto "Whisper Teste". Ele detalha a arquitetura, as convenções e o fluxo de trabalho do aplicativo.

ORDENS DO USUÁRIO: Whenever you create suggested tasks to the user, you've got to give as much details as possible. Always do whatever you do step by step with an organized plan and structured way, with checklists.

Não é necessário instalar todos os requisitos, apenas o necessário, o mínimo.

Do whatever the user tells you to. Sempre fale com o usuário em português do Brasil.

---

## 1. Visão Geral do Projeto

**Whisper Teste** é um utilitário de desktop para Windows focado em produtividade, que oferece transcrição de áudio de alta performance com um fluxo de trabalho discreto e centrado na privacidade.

### Filosofia e Objetivos

*   **Privacidade em Primeiro Lugar:** A principal proposta de valor é que a transcrição de áudio (ASR) ocorre inteiramente na máquina do usuário. Isso é fundamental para usuários que lidam com informações sensíveis e não podem ou não querem enviar seus dados de áudio para a nuvem.
*   **Eficiência de Fluxo de Trabalho:** O aplicativo é projetado para ser um "assistente invisível". Ele vive na bandeja do sistema (`system tray`) e é operado por uma hotkey global. O objetivo é minimizar a troca de contexto do usuário, permitindo ditar texto diretamente em qualquer aplicativo.
*   **Flexibilidade e Poder:** Embora o núcleo seja local, o aplicativo oferece pontes para serviços de IA externos (Gemini, OpenRouter) para tarefas de pós-processamento que não são críticas em termos de privacidade, como correção de pontuação ou execução de comandos complexos (Modo Agente), oferecendo o melhor dos dois mundos.

### Arquitetura Detalhada e Fluxo de Dados

O aplicativo utiliza uma arquitetura modular em Python, orquestrada por uma classe central (`AppCore`), que atua como o "sistema nervoso" do sistema. Abaixo está o detalhamento de cada componente e como eles interagem.

#### Componentes Principais

*   **`AppCore` (`src/core.py`):**
    *   **Responsabilidade:** É o coração da aplicação. Ele instancia todos os outros módulos, gerencia o estado geral da aplicação (ex: `IDLE`, `RECORDING`, `TRANSCRIBING`) e conecta os diferentes componentes através de um sistema de callbacks.
    *   **Interações:** Recebe eventos do `KeyboardHotkeyManager` (ex: "iniciar gravação"), comanda o `AudioHandler`, envia o áudio para o `TranscriptionHandler` e recebe o texto final para ser processado ou colado.

*   **`ConfigManager` (`src/config_manager.py`):**
    *   **Responsabilidade:** É a fonte única da verdade para todas as configurações. Ele gerencia a leitura e escrita de `config.json` (configurações gerais) e `secrets.json` (chaves de API), garantindo uma separação clara entre configuração e credenciais.
    *   **Interações:** É consultado por quase todos os outros módulos para obter parâmetros de operação (ex: qual modelo de ASR usar, qual hotkey registrar, qual prompt de IA utilizar).

*   **`AudioHandler` (`src/audio_handler.py`):**
    *   **Responsabilidade:** Gerencia tudo relacionado à captura de áudio. Utiliza a biblioteca `sounddevice` para interagir com o microfone.
    *   **Interações:** É comandado pelo `AppCore` para iniciar/parar a gravação. Durante a gravação, ele pode utilizar o `VADManager` para detectar atividade de voz e, ao final, empacota o áudio gravado (seja como um array NumPy em memória ou um arquivo `.wav` temporário) e o entrega ao `AppCore`.

*   **`TranscriptionHandler` (`src/transcription_handler.py`):**
    *   **Responsabilidade:** O componente de trabalho pesado. Ele carrega o modelo de ASR (ex: `whisper-large-v3-turbo`), gerencia uma `ThreadPoolExecutor` para executar a transcrição em uma thread de fundo (evitando que a UI congele) e orquestra o pós-processamento.
    *   **Interações:** Recebe o áudio do `AppCore`. Após a transcrição, ele pode, opcionalmente, chamar os clientes de API (`GeminiAPI`, `OpenRouterAPI`) para correção ou função agêntica antes de devolver o texto finalizado ao `AppCore`.

*   **`KeyboardHotkeyManager` (`src/keyboard_hotkey_manager.py`):**
    *   **Responsabilidade:** Registrar e escutar as hotkeys globais do sistema operacional usando a biblioteca `keyboard`.
    *   **Interações:** Quando uma hotkey registrada é pressionada, ele invoca um callback diretamente no `AppCore` (ex: `app_core.toggle_recording()`), que então inicia a cadeia de eventos.

*   **`UIManager` (`src/ui_manager.py`):**
    *   **Responsabilidade:** Gerenciar toda a interação visual com o usuário. Isso inclui o ícone na bandeja do sistema (`pystray`), seu menu de contexto dinâmico e a janela de configurações (`customtkinter`).
    *   **Interações:** Ele é notificado pelo `AppCore` sobre mudanças de estado para que possa atualizar a cor do ícone e a tooltip. Quando o usuário salva as configurações, o `UIManager` chama um método no `AppCore` para aplicar as novas configurações em toda a aplicação.

#### Fluxo de Dados Típico (Correção de Texto)

1.  **Usuário pressiona a hotkey `F3`**.
2.  `KeyboardHotkeyManager` detecta o evento e chama `app_core.toggle_recording()`.
3.  `AppCore` muda seu estado para `RECORDING`, notifica o `UIManager` (que troca a cor do ícone para vermelho) e comanda o `AudioHandler` para iniciar a gravação.
4.  `AudioHandler` começa a gravar o áudio do microfone.
5.  **Usuário pressiona `F3` novamente**.
6.  `KeyboardHotkeyManager` chama `app_core.toggle_recording()`.
7.  `AppCore` comanda o `AudioHandler` para parar.
8.  `AudioHandler` finaliza o arquivo de áudio e o entrega ao `AppCore` através de um callback.
9.  `AppCore` muda o estado para `TRANSCRIBING`, notifica a UI (ícone azul) e passa o áudio para o `TranscriptionHandler`.
10. `TranscriptionHandler` executa o modelo de ASR em uma thread de fundo. Ao obter o texto bruto, ele o envia para a `GeminiAPI`.
11. `GeminiAPI` envia o texto para a API do Google, recebe o texto corrigido e o retorna ao `TranscriptionHandler`.
12. `TranscriptionHandler` retorna o texto final corrigido para o `AppCore`.
13. `AppCore` muda o estado para `IDLE`, copia o texto para a área de transferência (`pyperclip`) e simula um `Ctrl+V` (`pyautogui`) para colar o texto no aplicativo ativo.

---

## 2. Construindo e Executando o Projeto

### Gerenciamento de Dependências

O projeto utiliza múltiplos arquivos `requirements.txt` para diferentes cenários. É crucial usar um **ambiente virtual** (`venv`) para isolar as dependências.

```bash
# Crie e ative um ambiente virtual
python -m venv venv
.\venv\Scripts\activate
```

*   **`requirements.txt`:** Contém as dependências principais para a execução do aplicativo em modo CPU. Instale com `pip install -r requirements.txt`.
*   **`requirements-optional.txt`:** Contém as dependências para habilitar a aceleração por GPU (NVIDIA), notadamente uma versão do `torch` compilada com suporte a CUDA. Instale com `pip install -r requirements-optional.txt`.
*   **`requirements-test.txt`:** Contém dependências adicionais para desenvolvimento e execução de testes, como o `pytest`. Instale com `pip install -r requirements-test.txt`.

### Executando a Aplicação

O ponto de entrada principal da aplicação é `src/main.py`. Para executar em modo de desenvolvimento:

```bash
python src/main.py
```

### Executando a Suíte de Testes

O projeto usa `pytest` para testes automatizados. A presença de um diretório `.pytest_cache` indica seu uso. Para executar os testes:

```bash
pytest
```

---

## 3. Convenções de Desenvolvimento e Padrões

*   **Configuração Como Fonte da Verdade:** Toda funcionalidade que pode ser ajustada pelo usuário **deve** ser definida no `config.json` e acessada através do `ConfigManager`. Evite valores "hard-coded".
*   **Modelo de Threading:** A thread principal é dedicada à UI (`Tkinter mainloop`) e ao `pystray`. **Qualquer operação bloqueante ou de longa duração** (carregamento de modelos, inferência de ASR, chamadas de API de rede) **deve obrigatoriamente ser executada em uma thread de fundo**. O padrão atual é usar `concurrent.futures.ThreadPoolExecutor`, como visto no `TranscriptionHandler`.
*   **Gerenciamento de Estado:** O estado da aplicação é explícito e centralizado (atualmente no `AppCore`). As mudanças de estado são o principal mecanismo para coordenar os diferentes módulos. Por exemplo, o `UIManager` não comanda o `AudioHandler` diretamente; ele reage a uma mudança de estado que o `AppCore` publicou.
*   **Tratamento de Erros:** Erros esperados (ex: falha de rede ao chamar a API Gemini) devem ser capturados, registrados em log, e a aplicação deve se recuperar graciosamente (ex: retornando o texto não corrigido em vez de falhar). Erros críticos (ex: falha ao carregar o modelo de ASR) devem levar a um estado de `ERROR_*`, que fornece feedback visual imediato ao usuário através do ícone da bandeja.
*   **Estilo de Código:** O código segue as convenções do PEP 8, com o auxílio do `flake8` (ver arquivo `.flake8` na raiz) para garantir a consistência.
