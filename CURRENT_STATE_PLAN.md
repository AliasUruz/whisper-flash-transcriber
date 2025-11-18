# Manual de Estado Atual: Whisper Flash Transcriber v2

**Document Version:** 1.0
**Date:** 2025-11-05

## 1. Visão Geral

Este documento descreve o estado de desenvolvimento atual do aplicativo Whisper Flash Transcriber. Ele serve como um reflexo preciso do código-fonte e das funcionalidades implementadas até a data deste documento.

## 2. Mandatos do Projeto (Conforme Implementado)

- **Linguagem:** O código-fonte, comentários e logs estão em inglês.
- **Sistema Operacional:** O aplicativo é desenvolvido e testado para Windows.
- **Dependências:** A lista de dependências reflete o arquivo `requirements.txt` atual.

## 3. Pilha de Tecnologia

| Componente | Tecnologia | Modelo/Versão Específica |
| :--- | :--- | :--- |
| **Linguagem** | Python | 3.11+ |
| **UI Framework** | Flet | (Versão do `requirements.txt`) |
| **Motor de Transcrição**| `faster-whisper` | `openai/whisper-large-v3-turbo` |
| **Hotkeys Globais** | `pynput` | (Versão do `requirements.txt`) |
| **Manuseio de Áudio**| `sounddevice`, `numpy`, `soundfile` | (Versão do `requirements.txt`) |
| **Interação com SO** | `pyperclip` | Para acesso à área de transferência |

## 4. Especificação da Configuração

A configuração é armazenada em `config.json` no diretório `~/.whisper_flash_transcriber/`.

```json
{
  "hotkey": "f3",
  "recording_mode": "toggle",
  "auto_paste": true
}
```

## 5. Comportamento de Primeira Execução

- Na primeira inicialização, o arquivo `config.json` é criado silenciosamente com os valores padrão se não existir.

## 6. Comportamento da UI e Feedback ao Usuário

A UI fornece feedback através do ícone na bandeja do sistema, principalmente pelo texto do tooltip.

### 6.1. Matriz de Estado da UI

| Estado do Sistema | Aparência do Ícone | Texto do Tooltip |
| :--- | :--- | :--- |
| `idle` | Ícone Padrão | "Ready to record. Press the hotkey." |
| `recording` | Ícone Padrão | "Recording... Press the hotkey to stop." |
| `transcribing`| Ícone Padrão | "Transcribing... Please wait." |
| `error` | Ícone Padrão | "Error: Check logs or hover for details." |

### 6.2. Notificação de Erro

- Erros críticos (ex: "microphone not found") **não geram um pop-up**. Eles atualizam o estado para `error` e exibem uma mensagem no tooltip do ícone da bandeja. Detalhes adicionais são registrados no console.

## 7. Arquitetura e Especificações dos Módulos

### 7.1. Estrutura de Diretórios
```
/
├── src/
│   ├── main.py
│   ├── ui.py
│   ├── core.py
│   └── hotkeys.py
└── requirements.txt
```

### 7.2. Especificações dos Módulos (Lógica Atual)

#### `src/main.py`
- **Responsabilidade:** Ponto de entrada da aplicação. Inicializa e conecta todos os componentes.
- **Lógica:**
  ```python
  # 1. Inicializa CoreService (que carrega ou cria a config).
  # 2. Inicializa AppUI.
  # 3. Define o callback de atualização da UI no CoreService.
  # 4. Adiciona os controles da UI à página Flet.
  # 5. Inicializa e inicia o HotkeyManager em uma thread separada.
  # 6. Esconde a janela da aplicação na inicialização.
  ```

#### `src/ui.py`
- **Responsabilidade:** Gerencia a janela de configurações e o ícone da bandeja.
- **Lógica:**
  ```python
  class AppUI:
      def __init__(self, page, core):
          # Armazena referências e define os controles da UI e o TrayIcon.

      def build_controls(self) -> ft.Column:
          # Retorna a coluna com os controles de configuração.

      def update_status(self, status: str, tooltip: str):
          # 1. Atualiza self.tray_icon.tooltip.
          # 2. A mudança de ícone NÃO está implementada.
          # 3. Chama self.page.update().

      def _save_settings(self, e):
          # 1. Coleta os valores dos campos da UI.
          # 2. Chama self.core.save_settings().
          # 3. Esconde a janela de configurações.
  ```

#### `src/core.py`
- **Responsabilidade:** Orquestra gravação, transcrição e ações de resultado.
- **Lógica de Gravação:** A gravação é feita **inteiramente na memória RAM**. O áudio é acumulado em uma lista de frames.
- **Lógica:**
  ```python
  class CoreService:
      def __init__(self):
          # Carrega/cria settings e carrega o modelo `faster-whisper`.

      def toggle_recording(self):
          # Alterna entre _start_recording e _stop_recording.

      def _start_recording(self):
          # 1. Define o estado como "recording" e atualiza a UI.
          # 2. Inicia a `sounddevice.InputStream`.
          # 3. O callback da stream simplesmente anexa os dados de áudio a uma lista (`self.audio_frames`).

      def _stop_recording(self):
          # 1. Para a `sounddevice.InputStream`.
          # 2. Define o estado como "transcribing" e atualiza a UI.
          # 3. Inicia _process_audio em uma nova thread.

      def _process_audio(self):
          # 1. Concatena os frames de áudio da lista em um array NumPy.
          # 2. Envia o array para transcrição.
          # 3. Chama _handle_result com o texto.
          # 4. Define o estado como "idle".

      def _handle_result(self, text: str):
          # 1. Copia o texto para a área de transferência com `pyperclip`.
          # 2. Se `auto_paste` for true, simula um Ctrl+V com `pynput`.

      def save_settings(self, settings: dict):
          # 1. Salva as configurações no arquivo config.json.
          # 2. Loga um aviso de que a aplicação precisa ser reiniciada para que a mudança de hotkey tenha efeito.
  ```

#### `src/hotkeys.py`
- **Responsabilidade:** Gerencia o listener de hotkey global.
- **Lógica:**
  ```python
  class HotkeyManager:
      def _on_press(self, key):
          # 1. Compara o nome da tecla pressionada com a hotkey das configurações.
          # 2. A verificação é simples e funciona para teclas únicas (ex: 'f3'), mas não para combinações.
          # 3. Se corresponder, chama `self.core.toggle_recording()`.

      def start_listening(self):
          # Inicia o listener do `pynput` em um loop de bloqueio.
  ```

## 8. Logging e Diagnósticos

- O logging é feito para o console (`stdout`) e cobre as principais ações da aplicação, como inicialização, carregamento de modelo, gravação e transcrição.
