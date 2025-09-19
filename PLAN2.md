# Plano detalhado de bugs

## 1. UI quebra quando `torch` nao esta instalado (mais facil)
- **Arquivo**: src/ui_manager.py:49
- **Problema**: O modulo importa `torch` no topo do arquivo. Em maquinas sem PyTorch instalado (cenario comum quando o usuario so usa CPU), a importacao gera `ModuleNotFoundError` e impede a aplicacao de inicializar, mesmo que a pessoa nao precise de listagem de GPUs.
- **Como reproduzir**: Em um ambiente Python sem PyTorch, rode `python src/main.py`. A execucao falha durante o import do `UIManager`.
- **Impacto**: Falha total de inicializacao para usuarios sem GPU ou que desejam evitar dependencias pesadas.
- **Correcao sugerida**: Envolver a importacao em `try/except`, guardar o modulo somente quando disponivel e ajustar `get_available_devices_for_ui()` para retornar apenas a opcao de CPU quando `torch` nao existir.

## 2. SDK Gemini obrigatorio derruba o aplicativo (facil)
- **Arquivo**: src/gemini_api.py:8
- **Problema**: O modulo importa `google.generativeai` de forma incondicional. Se o pacote nao estiver instalado, o erro aparece antes mesmo de a UI abrir, mesmo quando o usuario nao pretende usar a integracao com Gemini.
- **Como reproduzir**: Remova o pacote com `pip uninstall google-generativeai` e execute `python src/main.py`. A aplicacao encerra com `ModuleNotFoundError` durante o import.
- **Impacto**: Quem depende apenas de outros provedores (por exemplo, OpenRouter) nao consegue iniciar o aplicativo.
- **Correcao sugerida**: Transformar a importacao em opcional usando `try/except`, sinalizar que o cliente Gemini esta indisponivel e exibir um aviso amigavel quando o recurso for ativado sem a dependencia.

## 3. VAD exige PyTorch desnecessariamente (medio)
- **Arquivo**: src/vad_manager.py:5
- **Problema**: O VAD usa `torch.from_numpy` apenas para adicionar eixos ao array. Em maquinas sem PyTorch, a importacao quebra o modulo e o `AudioHandler` inteiro, impedindo gravacao mesmo com o VAD desativado.
- **Como reproduzir**: Remova PyTorch, inicialize o app (`python src/main.py`) e observe o erro na importacao do `VADManager`.
- **Impacto**: Usuarios de CPU ficam sem qualquer captura de audio e precisam instalar uma dependencia pesada sem necessidade real.
- **Correcao sugerida**: Substituir o uso de PyTorch por chamadas NumPy (`np.expand_dims`) ou carregar `torch` apenas quando presente, desabilitando o VAD caso contrario.

## 4. Timeout de download e convertido em cancelamento manual (mais dificil)
- **Arquivo**: src/model_manager.py:303-316
- **Problema**: A funcao `_check_abort()` verifica o deadline duas vezes. A primeira condicao levanta `DownloadCancelledError(..., by_user=True)` antes da segunda atribuir `timed_out=True`. Assim, timeouts sao registrados como se o usuario tivesse cancelado manualmente.
- **Como reproduzir**: Chame `ensure_download(..., timeout=1)` com rede lenta. O log resultante mostra "cancelled by caller" e o atributo `timed_out` fica falso.
- **Impacto**: UI, logs e metricas nao conseguem diferenciar timeouts de cancelamentos reais, bloqueando automatizacoes de retry.
- **Correcao sugerida**: Consolidar o check do deadline em um unico bloco que levante `DownloadCancelledError` com `timed_out=True` e `by_user=False`.
