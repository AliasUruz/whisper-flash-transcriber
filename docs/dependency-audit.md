# Dependency Audit Workflow

Este documento descreve o painel de auditoria de dependências introduzido no `UIManager` e o relatório estruturado produzido por `src/utils/dependency_audit.py`. O objetivo é acelerar a detecção de ambientes quebrados logo após o bootstrap, oferecendo comandos prontos para alinhar o ambiente aos manifestos oficiais (`requirements*.txt`).

## Quando a auditoria é executada

1. **Bootstrap do AppCore** – Logo após o carregamento da configuração, `AppCore` chama `audit_environment()` e registra o resultado em `ConfigManager.record_runtime_notice()`.
2. **Publicação de evento** – Assim que o `StateManager` é instanciado, o núcleo dispara `StateEvent.DEPENDENCY_AUDIT_READY` com um resumo textual.
3. **Entrega à UI** – Quando o `UIManager` se registra como assinante, o painel não modal é aberto automaticamente exibindo o relatório completo. O painel permanece acessível até ser fechado manualmente.

## Categorias de diagnóstico

| Categoria | Descrição | Ação sugerida |
| --- | --- | --- |
| **Dependências ausentes** | Pacote não encontrado no ambiente atual. | Execute o comando de instalação/copiar todos e reinstale conforme indicado. |
| **Versão fora da especificação** | Versão instalada não atende ao specifier definido no requirements. | Utilize o botão "Copiar comando" para aplicar `pip install --upgrade ...`. |
| **Divergência de hash** | Hash registrado em `requirements*.txt` não coincide com o hash do artefato instalado (quando disponível via `direct_url.json`). | Reinstale o pacote com `pip install --force-reinstall --no-cache-dir ...` ou execute o comando sugerido. |

> **Observação:** Se os manifestos não contiverem hashes (`--hash=sha256:...`), a seção de divergências permanecerá vazia.

## Uso do painel

1. **Resumo** – O topo do painel exibe o status geral (ex.: `Dependency audit completed — 1 missing`) e o carimbo de data/hora UTC convertidos para o timezone local.
2. **Listas por categoria** – Cada card apresenta o requisito, o arquivo/linha de origem, detalhes de versão e hashes detectados.
3. **Comandos individuais** – O botão "Copiar comando" copia um `python -m pip install ...` já normalizado (inclui extras e specifiers relevantes). Cola diretamente no terminal/PowerShell.
4. **Copiar todos os comandos** – Gera uma lista deduplicada (ordenada pela primeira ocorrência) contendo um comando por item em conflito.
5. **Abrir documentação** – Abre este arquivo usando o navegador padrão para consulta rápida.

## Troubleshooting comum

| Sintoma | Possível causa | Mitigação |
| --- | --- | --- |
| Painel informa "Dependency audit failed" | `packaging` ou `importlib.metadata` indisponíveis, ou arquivo `requirements*.txt` inacessível. | Verifique se o ambiente virtual está ativo e se os arquivos existem. Execute `python -m pip install -r requirements.txt`. |
| Seções vazias mas ambiente quebrado | Manifesto não lista a dependência problemática ou marker `; ...` filtrou o pacote para o sistema atual. | Revise o arquivo `requirements*.txt` correspondente e ajuste a política de markers se necessário. |
| Divergências de hash recorrentes | Instalação manual via wheel local ou mirror privado que gera `direct_url.json` com hash diferente. | Reavalie o hash no manifesto ou use `pip install --no-deps --force-reinstall <wheel>` com hash atualizado. |
| Botão de cópia não faz nada | Clipboard bloqueado pelo SO ou sessão Tk sem foco. | Dê foco à janela principal ou execute o aplicativo com privilégios adequados. |

## Referência programática

- **API principal:** `DependencyAuditResult` expõe `missing`, `version_mismatches` e `hash_mismatches` (listas de `DependencyIssue`).
- **Reaproveitamento em scripts:** chame `audit_environment()` diretamente (retorna objeto serializável via `to_serializable()`).
- **Registro em runtime:** consulte `ConfigManager.get_runtime_notices()` para capturar o último resumo gerado durante o bootstrap.

## Próximos passos sugeridos

1. Automatizar a execução de `audit_environment()` em jobs CI para validar pull requests que alterem requirements.
2. Persistir o último relatório em disco (`logs/`) para inspeção histórica.
3. Expandir o parser para suportar diretórios extras (`-r path/to/requirements.txt`) caso o projeto passe a organizar manifestos em subpastas.

