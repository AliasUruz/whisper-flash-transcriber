# Plano Consolidado do Whisper Recorder (PLAN1)

## Resumo Integrado
Este documento une o plano tecnico original e o plano de bugs recentes. O foco imediato e fechar as pendencias que seguem em aberto e enderecar os bugs bloqueadores identificados durante a revisao.

## Foco Imediato em Pendencias

### Pendencias criticas do plano original
- [ ] Documentar e revisar o fluxo de estados do carregamento de modelos (itens 1.3a-1.3b).
- [ ] Blindar o prompt de instalacao de modelos e persistir a decisao do usuario (itens 1.4a-1.4b).
- [ ] Revisar `_start_model_download` garantindo reset de estado, mensagens e suporte a timeout/cancelamento (itens 1.5a-1.5c).
- [ ] Validar que TranscriptionHandler e ConfigManager respeitam as configuracoes atuais sem fallbacks inesperados (itens 1.6 e 1.7).
- [ ] Evitar downloads silenciosos e cobrir o ciclo com testes automatizados (itens 1.8 e 1.9).
- [ ] Inventariar e saneear variaveis da UI, incluindo sincronizacao e documentacao (itens 2.1, 2.3 e 2.5).
- [ ] Modularizar ajustes da UI/ASR para reduzir acoplamento (itens 3.5, 3.6 e 4.6).
- [ ] Consolidar melhorias no ConfigManager: chaves, reload e   limpeza de valores nulos (itens 5.1 a 5.5).
- [ ] Atualizar feedback da UI apos downloads e falhas, com mensagens claras em portugues (itens 6.1, 6.3 e 6.4).
- [ ] Entregar logs/metricas e suites de teste faltantes (itens O1-O4 e V1-V4).

### Bugs criticos recem-identificados (PLAN2)
- [ ] UI falha quando `torch` nao esta instalado (`src/ui_manager.py:49`).
- [ ] Falta de SDK Gemini derruba o aplicativo (`src/gemini_api.py:8`).
- [ ] VAD depende de PyTorch sem necessidade (`src/vad_manager.py:5`).
- [ ] Timeout de download registrado como cancelamento manual (`src/model_manager.py:303-316`).

## Plano Tecnico Detalhado (referencia completa)

# Plano de Correcoes do Whisper Recorder (Atualizado em 17/09/2025)

## Panorama Atual
O Whisper Recorder ja esta operando com uma camada de configuracao mais robusta, mas ainda convivemos com lacunas de observabilidade e consistencia de estado. A sincronizacao entre UI e nucleo cobre muitos cenarios recentes (GPU/CPU, tipo de modelo, quantizacao), contudo o pipeline de download/carregamento continua com trechos sem monitoramento e sem garantias de serializacao. As rotinas de aplicacao de configuracao evoluiram, mas persistem duplicacoes e caminhos silenciosos que dificultam diagnostico em producao.

## Backlog Tecnico Prioritario
As seções abaixo descrevem objetivos, motivacao tecnica e entregaveis esperados. Sempre que possivel foram referenciados arquivos e linhas atuais para orientar revisoes futuras.

### 1. AppCore e ciclo de download/carregamento de modelo
Contexto: `src/core.py` controla o estado principal da aplicacao. Hoje compreendemos que o erro primario ocorre durante a sincronizacao de modelos instalados e no fluxo que pede download ao usuario. Precisamos enxergar melhor essas transicoes e impedir que multiplas threads disparem prompts.

- [x] 1.1 Importar `list_installed` e `DownloadCancelledError` em `src/core.py` (concluido; ver `src/core.py:39` e `src/core.py:177`).
- [x] 1.2 Instrumentar observabilidade em torno de `list_installed(cache_dir)`:
  - Motivo: atualmente, quando `list_installed` falha, temos apenas um WARNING generico. Precisamos saber qual cache foi usado, em qual contexto (startup, sync manual, reload) e qual excecao ocorreu.
  - [x] 1.2.a Adicionar log DEBUG antes da chamada informando `cache_dir` resolvido, origem da chamada (`__init__`, `_sync_installed_models`) e thread atual (`threading.current_thread().name`).
  - [x] 1.2.b Registrar contagem de modelos retornados ou o erro original (`repr(e)`) em nivel WARNING/ERROR. Isso facilita correlacao com o log do `model_manager`.
  - [x] 1.2.c Avaliar retorno vazio: se `cache_dir` nao existir, exibir tooltip amigavel via UI Manager e manter estado consistente (`STATE_ERROR_MODEL`).
    - Notas 1.2: Logs e tratamento implementados em `src/core.py:224`, `src/core.py:272` e integração de tooltip/flush em `src/ui_manager.py:1318` + `src/main.py:141`.
- [ ] 1.3 Documentar e revisar o fluxo de estados (`STATE_LOADING_MODEL -> _prompt_model_install -> _start_model_download -> start_model_loading`):
  - Objetivo: garantir que a UI nunca receba estados confusos (ex.: LOADING_MODEL eterno). Criar um fluxograma simples (markdown/mermaid) no proprio arquivo ou anexar a este plano.
  - [ ] 1.3.a Mapear caminho feliz e bifurcacoes (cancelamento, erro de rede, cache invalido). Registar dependencias de cada funcao.
  - [ ] 1.3.b Assegurar que cada transicao chama `_set_state` apenas uma vez e que o estado final (IDLE/ERROR_MODEL) e informado a UI Manager (`self.state_update_callback`).
- [ ] 1.4 Blindar `_prompt_model_install` (`src/core.py:142`):
  - Problema atual: se o download e pedido duas vezes rapidamente (por chamadas consecutivas de `start_model_loading`), mais de um prompt pode aparecer.
  - [ ] 1.4.a Introduzir flag thread-safe (ex.: `self.model_prompt_active`) guardada por `RLock` para bloquear prompts simultaneos.
  - [ ] 1.4.b Persistir escolha do usuario (aceitar/adiar) no `ConfigManager` para evitar reaparecimento do prompt ate uma nova mudanca de backend/modelo.
- [ ] 1.5 Revisar `_start_model_download`:
  - Risco atual: mesmo apos download bem-sucedido, o estado pode permanecer em LOADING ate que `start_model_loading` finalize. Precisamos de logs e resets claros.
  - [ ] 1.5.a Adicionar tentativa de reset para `STATE_IDLE` quando o download conclui antes de chamar `start_model_loading` (importante para UI).
  - [ ] 1.5.b Consolidar mensagens ao usuario (usar `messagebox` com textos localizados) e registrar no log `model_id`, backend, tempo total e tamanho baixado (se conhecido).
  - [ ] 1.5.c Embutir suporte a timeout configuravel e cancelamento manual propagando para `ensure_download` (avaliar se `ensure_download` ja aceita sinal de cancelamento).
- [ ] 1.6 `TranscriptionHandler.start_model_loading` e `_load_model_task` devem honrar configuracoes atuais (`src/transcription_handler.py:440+`):
  - [ ] 1.6.a Confirmar que `self.config_manager` e reatribuido antes de cada load. Hoje `AppCore.apply_settings_from_external` chama `update_config`, mas precisamos validar se `self.transcription_handler.config_manager` foi atualizado antes do novo load.
  - [ ] 1.6.b Garantir que `self.backend_resolved`, `self.asr_compute_device` e `self.asr_model_id` respeitam escolhas da UI mesmo apos fallback `auto`. Se ocorrer fallback, registrar log com motivo.
  - [ ] 1.6.c Adicionar asserts/logs quando ocorrer fallback involuntario (ex.: usuario escolhe CPU mas heuristica ainda envia para GPU). Isso facilita debug com hardware heterogeneo.
- [ ] 1.7 Confirmar que `backend_registry` (`src/asr_backends.py`) e `_resolve_asr_settings` preservam escolhas da UI:
  - [ ] 1.7.a Criar testes unitarios simulando backend `ctranslate2` em CPU com `torch.cuda.is_available()` falso para garantir que o fluxo nao tenta GPU.
  - [ ] 1.7.b Validar conversao de string da UI ("Force CPU", "GPU 0: ...") em valor esperado (`cpu`, `cuda:0`) antes de salvar no config. Adicionar helper centralizado com cobertura de testes.
- [ ] 1.8 Garantir ausencia de downloads silenciosos:
  - [ ] 1.8.a Auditar chamadas de `ensure_download` fora de `_start_model_download`; qualquer execucao automatica deve ser precedida de prompt/confirmacao.
  - [ ] 1.8.b Adicionar indicador visual na UI (spinner ou botao desabilitado) enquanto o download ocorre. Coordenar com UI Manager para nao bloquear o mainloop.
- [ ] 1.9 Adicionar cobertura automatizada para regressao do ciclo:
  - [ ] 1.9.a Criar teste integrado simulando `apply_settings_from_external` alterando backend/modelo e verificando chamada de `_set_state` correta (usar mocks).
  - [ ] 1.9.b Criar teste para `DownloadCancelledError` garantindo que o estado final e `STATE_ERROR_MODEL` e que nao ocorre relancamento da excecao.

### 2. Mapeamento das variaveis da UI (`run_settings_gui` em `src/ui_manager.py`)
Contexto: A tela de configuracoes usa dezenas de `ctk.StringVar`/`BooleanVar`. Alguns problemas ja foram sanados (variaveis inexistentes), mas ainda ha dificuldades para auditar como cada campo escreve no ConfigManager.

- [ ] 2.1 Inventariar todas as instancias de `ctk.*Var`:
  - Entregavel: tabela (pode ser markdown dentro do repo) listando variavel, widget que consome, chave no `ConfigManager` e callbacks dependentes.
  - [ ] 2.1.a Construir tabela conforme acima. Pode ser no proprio `PLAN.md` ou arquivo auxiliar `docs/ui_vars.md`.
  - [ ] 2.1.b Validar valores iniciais versus `DEFAULT_CONFIG` (`src/config_manager.py`) para evitar divergencias quando o usuario restaura configuracoes.
- [x] 2.2 Variaveis inexistentes (`ct2_quant_var`, `gpu_selection_var`) substituidas (ver `src/ui_manager.py:410` e `src/ui_manager.py:438`).
- [ ] 2.3 Corrigir escopo das variaveis:
  - Problema: funcoes internas como `apply_settings` usam variaveis definidas mais acima, o que dificulta testes e manutenabilidade.
  - [ ] 2.3.a Variaveis usadas por funcoes internas devem ser atributos (`self.asr_ct2_compute_type_var`) ou passadas via closure controlada. Revisar cada uso.
  - [ ] 2.3.b Remover dependencias de variaveis globais do modulo (`model_manager` dummy) quando houver alternativa melhor.
- [x] 2.4 Widgets duplicados/menus redundantes para quantizacao removidos (UI ASR mais limpa).
- [ ] 2.5 Documentar sincronizacao UI -> ConfigManager:
  - [ ] 2.5.a Adicionar comentario ou docstring resumindo a ordem de aplicacao dos campos criticos (backend -> modelo -> device -> quantizacao) e justificativa (evitar race conditions de reload).
  - [ ] 2.5.b Criar rotina de validacao rapida (ex.: `validate_settings_inputs`) que destaque campos faltantes antes de chamar `core_instance_ref.apply_settings_from_external`.

### 3. Correcoes estruturais da UI
Contexto: Ajustes recentes resolveram parte do acoplamento, mas ainda existem pontos opcionais que melhorariam a manutencao e testabilidade.

- [x] 3.1 `self._gpu_selection_var` introduzida e sincronizada com `asr_compute_device_var`.
- [x] 3.2 `apply_settings` converte string de device/index antes de acionar o core (`src/ui_manager.py:525` aprox.).
- [x] 3.3 `ct2_quant_var` substituida por `asr_ct2_compute_type_var` e menus duplicados eliminados.
- [x] 3.4 Ordem de criacao de widgets/handlers ajustada para evitar uso antes da definicao.
- [ ] 3.5 (Opcional) Extrair montagem da secao ASR para funcao auxiliar reutilizavel (`_build_asr_section`). Objetivo: reduzir tamanho de `run_settings_gui` e permitir testes unitarios focados.
- [ ] 3.6 Modularizar validacoes numericas (`_safe_get_int`, `_safe_get_float`) criando helper em `src/utils/form_validation.py` (novo arquivo) com testes dedicados.

### 4. Selecao de modelo e estimativas
Contexto: A UI agora exibe informacoes ricas sobre modelos, mas podemos otimizar desempenho e reduzir chamadas repetidas ao backend de metadados.

- [x] 4.1 Uso de `display_to_id` em `_update_model_info` antes de consultar `model_manager` garantido.
- [x] 4.2 Tratamento de erros de rede em `get_model_download_size` com fallback amigavel (`src/core.py:150`).
- [x] 4.3 UI mostra tamanho instalado/necessario e status.
- [x] 4.4 `_update_install_button_state` e `_on_model_change` mantem backend/model alinhados.
- [x] 4.5 Tooltips/labels exibem backend sugerido e tamanho estimado.
- [ ] 4.6 Adicionar memoizacao temporaria das estimativas para evitar requisicoes repetidas. Implementar cache in-memory com TTL (ex.: 60s) no `model_manager` ou UI Manager.

### 5. Aplicacao de configuracoes (Core + ConfigManager)
Contexto: `AppCore.apply_settings_from_external` (`src/core.py:684`) e o ponto central de atualizacao. Ainda ha duplicacoes no mapa de chaves, condicoes com `None` e reloads sem controle fino.

- [ ] 5.1 Revisar cobertura de chaves:
  - [ ] 5.1.a Remover entradas duplicadas do `config_key_map` (`new_asr_backend`, `new_asr_model`, `new_ct2_quantization`) e documentar formato padrao (prefixo `new_`).
  - [ ] 5.1.b Incluir chaves ausentes que ja circulam na UI (`new_asr_dtype`, `new_chunk_length_mode`, `new_chunk_length_sec`, `new_clear_gpu_cache`). Adicionar comentarios apontando de onde vem cada parametro.
- [ ] 5.2 Evitar sobrescrita com `None` e reloads desnecessarios:
  - [ ] 5.2.a Ajustar loop para pular kwargs nulos ou iguais ao valor atual (reduz IO no `config.json`).
  - [ ] 5.2.b Centralizar logica de `config_changed` para chamar `save_config` apenas uma vez.
- [ ] 5.3 Atualizar estado apos aplicar configuracoes:
  - [ ] 5.3.a Se `reload_required` for verdadeiro mas `start_model_loading` falhar, garantir retorno a `STATE_ERROR_MODEL` com mensagem clara.
  - [ ] 5.3.b Publicar evento/notify para UI refletir `STATE_IDLE` quando o modelo ja esta pronto (evita spinner infinito na janela principal).
- [ ] 5.4 Expandir `reload_required`:
  - [ ] 5.4.a Acrescentar `ASR_COMPUTE_DEVICE_CONFIG_KEY` e `ASR_DTYPE_CONFIG_KEY` ao set `reload_keys` (mudanca de device/dtype exige reload).
  - [ ] 5.4.b Criar teste que altera device/dtype e verifica chamada de `start_model_loading` usando mock.
- [ ] 5.5 Simplificar atualizacao de `min_transcription_duration` e `min_record_duration`: mover logica para um helper e evitar set duplo com mesmo valor.

### 6. Feedback ao usuario
Contexto: Queremos que o usuario entenda o que esta acontecendo sem ter que abrir logs. Ainda existem lacunas em mensagens pos-download e banners de status.

- [ ] 6.1 Inserir banner/label persistente na janela de configuracoes com status atual do modelo (instalado, ausente, baixando). Pode reutilizar `ctk.CTkLabel` com cor conforme estado.
- [x] 6.2 Tamanho estimado ja aparece em prompts e botoes Install/Update.
- [ ] 6.3 Atualizar label/tooltip apos download:
  - [ ] 6.3.a Criar callback apos `ensure_download` que atualize `Tooltip` e texto do botao Install/Update.
  - [ ] 6.3.b Persistir resultado (sucesso/falha/cancelamento) para exibicao no proximo acesso (armazenar em `ConfigManager` ou atributo da UI Manager).
- [ ] 6.4 Melhorar mensagens de cancelamento/erro:
  - [ ] 6.4.a Detectar `DownloadCancelledError` e sugerir acao alternativa (ex.: trocar backend) em portugues claro.
  - [ ] 6.4.b Unificar mensagens de falha de rede e indicar arquivo de log (`logs/whisper_recorder.log`).

## Observabilidade e testes
A meta e padronizar logs e criar cobranca automatizada para os cenarios que hoje analisamos manualmente.

- [ ] O1 Criar logger dedicado `whisper_recorder.model` com nivel configuravel via arquivo de config. Documentar formato de log.
- [ ] O2 Adicionar metrica de duracao do download (tempo total, throughput aproximado) e enviar para o logger acima.
- [ ] O3 Criar testes unitarios para `TranscriptionHandler._effective_chunk_length` cobrindo GPUs com memorias diferentes (mocks de `torch.cuda.mem_get_info`).
- [ ] O4 Implementar teste de integracao da UI cobrindo `apply_settings` em modo headless (usar `pytest` + `pytest-customtkinter` se disponivel).

## Proximas validacoes sugeridas
Lista de checagens para garantir qualidade apos implementarmos os itens acima.

- [ ] V1 Executar smoke test manual: iniciar app sem modelos, confirmar prompt unico e transicao correta para `STATE_ERROR_MODEL` quando o usuario recusa download.
- [ ] V2 Testar troca rapida de backend/modelo pela UI verificando se `apply_settings_from_external` dispara reload apenas uma vez e se o estado final e IDLE.
- [ ] V3 Verificar `config.json` apos salvar configuracoes para garantir ausencia de chaves obsoletas (`gpu_selection_var`, `ct2_quant_var`).
- [ ] V4 Revisar logs apos um ciclo completo (abrir app -> baixar modelo -> transcrever) garantindo mensagens com contexto suficiente para suporte.

## Catalogo de Bugs (PLAN2)

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
