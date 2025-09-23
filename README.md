# Whisper Teste: Transcritor de Áudio por Hotkey

Um assistente de transcrição de áudio de alta performance, executado localmente e ativado por uma simples tecla de atalho. Ideal para ditar textos, transcrever reuniões ou converter qualquer áudio em texto de forma rápida e privada.

## Funcionalidades Principais

- **Gravação por Hotkey:** Inicie e pare gravações instantaneamente com uma tecla de atalho customizável.
- **Modos de Gravação:** Suporta o modo "Toggle" (pressione para iniciar, pressione para parar) e "Hold" (grave apenas enquanto a tecla estiver pressionada).
- **Transcrição Local:** Utiliza o poder dos modelos Whisper da OpenAI para realizar transcrições de alta qualidade diretamente na sua máquina, garantindo privacidade.
- **Correção por IA (Opcional):** Conecte-se a serviços como Gemini ou OpenRouter para corrigir automaticamente a pontuação e a gramática do texto transcrito.
- **Modo Agente:** Execute comandos de texto complexos sobre o áudio gravado usando o poder da IA.
- **Colagem Automática:** O texto final pode ser colado automaticamente no aplicativo em foco, otimizando seu fluxo de trabalho.

## Instalação

### Pré-requisitos

- Python 3.9 ou superior.
- Git (para clonar o repositório).

### Passos de Instalação

1.  **Clone o repositório:**
    ```bash
    git clone https://github.com/seu_usuario/WhisperTeste.git
    cd WhisperTeste
    ```

2.  **Crie e ative um ambiente virtual:**
    ```bash
    # Windows
    python -m venv venv
    .\venv\Scripts\activate
    ```

3.  **Instale as dependências:**
    ```bash
    pip install -r requirements.txt
    ```

### Dependências Opcionais (Suporte a GPU)

Para uma performance de transcrição significativamente mais rápida, você pode usar sua placa de vídeo NVIDIA. Para isso, é necessário instalar o PyTorch com suporte a CUDA.

1.  Visite a página oficial do [PyTorch](https://pytorch.org/get-started/locally/).
2.  Use o configurador para selecionar a versão do PyTorch, sua plataforma e a versão do CUDA.
3.  Copie e execute o comando de instalação fornecido. Ele será algo parecido com:
    ```bash
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    ```
    **Nota:** O comando exato pode variar. Use sempre o comando fornecido pelo site oficial do PyTorch.

## Como Usar

### Primeira Execução

Execute a aplicação pela primeira vez com o comando:
```bash
python src/main.py
```
Na primeira execução, os arquivos de configuração `config.json` e `hotkey_config.json` serão criados automaticamente no diretório raiz.

### Configuração

- Um ícone da aplicação aparecerá na bandeja do sistema (ao lado do relógio).
- Clique com o botão direito no ícone e selecione "Settings".
- Na janela de configurações, você pode:
    - Definir a tecla de atalho para gravação ("Record Hotkey").
    - Escolher o modelo de transcrição (ASR Model). Se o modelo não estiver instalado, a aplicação oferecerá para baixá-lo.
    - Configurar serviços de IA, sons de feedback e outras opções.

### Gravando e Transcrevendo

- Pressione a tecla de atalho definida para começar a gravar.
- Pressione novamente (ou solte, dependendo do modo) para parar.
- A aplicação irá transcrever o áudio e, se configurado, copiar e colar o texto final automaticamente.

## Arquitetura (Breve Descrição)

- **`main.py`:** Ponto de entrada da aplicação. Inicializa o `AppCore` e a interface do usuário.
- **`core.py`:** O cérebro da aplicação. Orquestra todos os outros módulos e gerencia o estado geral.
- **`ui_manager.py`:** Gerencia a interface gráfica, incluindo a janela de configurações e o ícone da bandeja do sistema.
- **`audio_handler.py`:** Responsável por capturar o áudio do microfone.
- **`transcription_handler.py`:** Gerencia o carregamento do modelo Whisper e o processo de transcrição.
- **`config_manager.py`:** Lida com o carregamento e salvamento de todas as configurações do usuário.