# Whisper Transcription App

Uma aplicação desktop para transcrição de áudio em tempo real usando o modelo Whisper da OpenAI, com correção de texto opcional via OpenRouter API.

## Funcionalidades

- Transcrição de áudio em tempo real usando o modelo Whisper Large v3
- Ativação via tecla de atalho configurável (padrão: F3)
- Modo de gravação toggle (iniciar/parar com a mesma tecla)
- Colagem automática do texto transcrito
- Feedback sonoro configurável
- Correção de texto opcional via OpenRouter API (melhora pontuação e formatação)
- Interface gráfica para configurações

## Requisitos

- Python 3.8+
- PyTorch
- Transformers (Hugging Face)
- Sounddevice
- Tkinter
- Pystray
- Requests (para a API OpenRouter)

## Instalação

1. Clone este repositório:
   ```
   git clone https://github.com/seu-usuario/whisper-transcription-app.git
   cd whisper-transcription-app
   ```

2. Instale as dependências:
   ```
   pip install torch transformers sounddevice numpy pyautogui keyboard pystray pillow pyperclip requests
   ```

3. Execute o aplicativo:
   ```
   python whisper_tkinter.py
   ```

## Configuração

Na primeira execução, um arquivo `config.json` será criado automaticamente com configurações padrão. 

Para usar a correção de texto via OpenRouter:
1. Obtenha uma chave de API em [OpenRouter](https://openrouter.ai)
2. Abra as configurações do aplicativo
3. Ative a opção "Enable Text Correction with OpenRouter"
4. Insira sua chave de API
5. O modelo padrão é "deepseek/deepseek-chat-v3-0324:free"

## Uso

1. Pressione a tecla de atalho configurada (padrão: F3) para iniciar a gravação
2. Fale o texto que deseja transcrever
3. Pressione a tecla novamente para parar a gravação
4. O texto transcrito será automaticamente copiado para a área de transferência e colado no aplicativo ativo

## Licença

Este projeto está licenciado sob a licença MIT - veja o arquivo LICENSE para detalhes.
