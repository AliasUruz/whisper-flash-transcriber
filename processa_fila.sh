#!/bin/bash
set -euo pipefail

# Script alinhado ao plano 2025-09-28-operational-optimization-blueprint.md
# Objetivo: processar PRs etiquetados com "codex" em sequência, evitando conflitos manuais.

REPO="AliasUruz/whisper-flash-transcriber"
PR_LABEL="codex"
TARGET_BRANCH="stable"

log() {
  printf '[processa_fila] %s\n' "$1"
}

log "Iniciando processamento controlado da fila (label=${PR_LABEL})."

git merge --abort >/dev/null 2>&1 || true
log "Atualizando branch base '${TARGET_BRANCH}'."
git checkout "${TARGET_BRANCH}"
git pull --ff-only origin "${TARGET_BRANCH}"

PRS=$(gh pr list --repo "${REPO}" --label "${PR_LABEL}" --state open --json number,createdAt \\
  | jq -r 'sort_by(.createdAt) | .[] | .number')

if [[ -z "${PRS}" ]]; then
  log "Nenhum PR pendente encontrado."
  exit 0
fi

log "Fila obtida: ${PRS}."

for PR in ${PRS}; do
  log "Processando PR #${PR}."
  if ! gh pr checkout "${PR}" --repo "${REPO}"; then
    log "Aviso: checkout falhou para #${PR}; seguindo para o próximo."
    continue
  fi

  BRANCH_NAME=$(git rev-parse --abbrev-ref HEAD)
  log "Branch atual: ${BRANCH_NAME}."

  if git merge --no-edit -X ours "origin/${TARGET_BRANCH}"; then
    log "Merge aplicado com sucesso."
  else
    log "Conflitos detectados; favor resolver manualmente e relançar o script."
    exit 1
  fi

  git commit --allow-empty -m "chore: trigger CI checks"
  git push origin "HEAD:${BRANCH_NAME}"
  log "PR #${PR} atualizado e enviado."

  git checkout "${TARGET_BRANCH}"
  git reset --hard "origin/${TARGET_BRANCH}"
done

log "Processamento concluído sem erros. Consulte o blueprint operacional para próximos passos."
