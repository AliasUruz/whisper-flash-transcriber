# Whisper Transcription App

Uma aplicação desktop para transcrição de áudio em tempo real usando o modelo Whisper da OpenAI, com correção de texto opcional via OpenRouter API.

## Funcionalidades

- Transcrição de áudio em tempo real usando o modelo Whisper Large v3
- Ativação via tecla de atalho configurável (padrão: F3)
- Modo de gravação toggle (iniciar/parar com a mesma tecla)
- Colagem automática do texto transcrito
- Feedback sonoro configurável
- Correção de texto opcional via OpenRouter API ou Google Gemini API (melhora pontuação e formatação)
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

Para usar a correção de texto:
1. Abra as configurações do aplicativo
2. Ative a opção "Enable Text Correction"
3. Selecione o serviço desejado (OpenRouter ou Gemini)
4. Configure o serviço selecionado conforme instruções abaixo

Para configurar o OpenRouter:
1. Obtenha uma chave de API em [OpenRouter](https://openrouter.ai)
2. Insira sua chave de API no campo correspondente
3. O modelo padrão é "deepseek/deepseek-chat-v3-0324:free"

Para configurar o Google Gemini:
1. Obtenha uma chave de API em [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Insira sua chave de API no campo correspondente
3. O modelo padrão é "gemini-2.0-flash-001"

## Uso

1. Pressione a tecla de atalho configurada (padrão: F3) para iniciar a gravação
2. Fale o texto que deseja transcrever
3. Pressione a tecla novamente para parar a gravação
4. O texto transcrito será automaticamente copiado para a área de transferência e colado no aplicativo ativo

## Problemas Conhecidos e Soluções

### Bug da Biblioteca Keyboard no Windows 11

Em alguns sistemas Windows 11, a biblioteca Keyboard pode parar de responder após o primeiro uso da tecla de atalho. Para resolver este problema, o aplicativo inclui:

1. **Tecla de recarga (F4 por padrão)**: Pressione esta tecla para recarregar a biblioteca de teclado e restaurar a funcionalidade das teclas de atalho

2. **Opção no menu de contexto**: Clique com o botão direito no ícone da bandeja do sistema e selecione "Recarregar Teclado/Hotkey"

3. **Recarga automática periódica**: O aplicativo tenta recarregar automaticamente as teclas de atalho periodicamente para evitar problemas

Se as teclas de atalho pararem de funcionar, use um desses métodos para restaurar a funcionalidade.

## Licença

Este projeto está licenciado sob a licença MIT - veja o arquivo LICENSE para detalhes.
