# Plano Operacional - Validações Críticas do Whisper Flash Transcriber (05/10/2025)

## Objetivos
- Garantir que o fluxo de download de modelos responda corretamente a interrupções, falta de espaço em disco e reintentos.
- Verificar a migração automática de diretórios ao alterar a raiz de armazenamento pela UI, preservando gravações e cache de modelos.
- Auditar dependências críticas para antecipar atualizações e riscos de incompatibilidade.

## Pré-requisitos gerais
- Ambiente virtual ativo com as dependências mínimas instaladas (`pip install -r requirements.txt`).
- Acesso ao diretório de perfil configurado pelo aplicativo (padrão `~/.cache/whisper_flash_transcriber`).
- Permissões suficientes para mover diretórios e ajustar ACLs locais.

## Checklist A — Download e resiliência de modelos
1. [ ] Remover ou renomear o diretório do modelo alvo dentro do cache atual (`<perfil>/asr/<backend>/<modelo>`).
2. [ ] Iniciar o aplicativo (`python src/main.py`) e aguardar o prompt de download.
3. [ ] Aceitar o download e monitorar o log: confirmar emissão de `status=in_progress` no `config.json`.
4. [ ] Aguardar ~30% da barra de progresso e cancelar manualmente.
5. [ ] Validar que o diretório parcial foi excluído e que `status=cancelled` + `details=<caminho>` foram gravados.
6. [ ] Reabrir o prompt, aceitar novamente e aguardar a conclusão completa do download.
7. [ ] Confirmar a presença de `install.json`, `status=success` e limpeza do diretório temporário.
8. [ ] Executar `python -m compileall src` seguido de `pytest` para garantir estabilidade pós-download.

### Contingências
- **Falha por espaço insuficiente:**
  - Liberar espaço ou apontar `models_storage_dir` para uma partição com >10% de folga adicional.
  - Registrar o erro em `plans/` com os logs relevantes antes de repetir o processo.
- **Download interrompido pelo provedor:**
  - Repetir a etapa 2 com um segundo modelo menor para diferenciar falha externa de problema local.
  - Se persistir, ativar VPN ou espelhamento alternativo e atualizar a seção de riscos.

## Checklist B — Migração de diretórios
1. [ ] Criar diretórios vazios destino (ex.: `D:\whisper_cache` no Windows ou `/mnt/data/whisper_cache` no Linux).
2. [ ] Abrir a janela de configurações e alterar apenas `storage_root_dir` para o novo caminho.
3. [ ] Aplicar as mudanças e observar os logs: procurar por `config.bootstrap.ASR cache_migrated` e `Recordings directory migrated`.
4. [ ] Validar no filesystem que `asr/` e `recordings/` foram movidos quando não havia overrides.
5. [ ] Ativar overrides manuais para `models_storage_dir` e `recordings_dir`, aplicar novamente e confirmar que os logs informam “skipping migration”.
6. [ ] Rodar `pytest -k vad` para garantir que os caminhos migrados não quebraram testes dependentes de áudio.

### Contingências
- **Falha ao preparar diretório destino:**
  - Checar permissões e reexecutar a etapa 2 com privilégios elevados.
  - Documentar o stack trace e manter os diretórios originais até resolver o bloqueio.
- **Dados duplicados após migração:**
  - Interromper gravações, validar integridade dos arquivos e remover manualmente apenas as cópias redundantes.
  - Atualizar o plano com o método de deduplicação usado.

## Checklist C — Auditoria de dependências
1. [ ] Executar `pip list --outdated` dentro do ambiente virtual.
2. [ ] Priorizar bibliotecas críticas (`torch`, `transformers`, `ctranslate2`, `faster-whisper`, `sounddevice`).
3. [ ] Para cada pacote alvo, consultar changelog oficial e riscos de regressão.
4. [ ] Registrar na tabela abaixo a decisão (atualizar agora, adiar, requer teste adicional) e o racional técnico.
5. [ ] Se decidir atualizar, criar *branch* dedicado, aplicar `pip install --upgrade <pacote>` e reexecutar `pytest` + testes manuais das checklists A/B.
6. [ ] Atualizar `requirements*.txt` somente após validar compatibilidade.

| Pacote | Versão atual | Versão alvo | Decisão | Notas |
| --- | --- | --- | --- | --- |
| `torch` | _(preencher)_ | _(preencher)_ | _( ) Atualizar _( ) Adiar_( ) Investigar | Impacto em CUDA e `torch.compile`. |
| `transformers` | _(preencher)_ | _(preencher)_ | _( ) Atualizar _( ) Adiar_( ) Investigar | Verificar breaking changes na API de `WhisperModel`. |
| `ctranslate2` | _(preencher)_ | _(preencher)_ | _( ) Atualizar _( ) Adiar_( ) Investigar | Confirmar compatibilidade do compute type. |
| `faster-whisper` | _(preencher)_ | _(preencher)_ | _( ) Atualizar _( ) Adiar_( ) Investigar | Validar parâmetros aceitos em `transcribe`. |
| `sounddevice` | _(preencher)_ | _(preencher)_ | _( ) Atualizar _( ) Adiar_( ) Investigar | Testar captura em Windows e WSL. |

### Contingências
- **Atualização crítica falhou:**
  - Reverter o pacote para a versão anterior (`pip install <pacote>==<versão antiga>`) e anexar a saída do erro aos registros.
  - Avaliar fallback temporário (ex.: usar backend alternativo) até corrigir a dependência.
- **Dependência não suporta ambiente atual:**
  - Documentar a limitação e propor plano de migração de SO/driver caso necessário.

## Registro final
- [ ] Consolidar logs relevantes (`logs/*.log`, prints do terminal) em um diretório datado.
- [ ] Atualizar `plans/` com resultados e próximas ações recomendadas.
- [ ] Comunicar stakeholders sobre bloqueios ou ajustes aplicados.
