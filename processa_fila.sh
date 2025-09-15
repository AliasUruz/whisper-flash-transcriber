#!/bin/bash

# Encerra o script se qualquer comando falhar
set -e

# --- Configurações ---
REPO="AliasUruz/whisper-flash-transcriber"
PR_LABEL="codex"
TARGET_BRANCH="stable"

echo "--- INICIANDO PROCESSAMENTO DA FILA LOCALMENTE ---"
echo "Repositório: $REPO"
echo "Label dos PRs: $PR_LABEL"
echo "Branch Alvo: $TARGET_BRANCH"
echo "----------------------------------------------------"

# Garante que estamos na branch principal e com a versão mais recente
echo "Atualizando a branch '$TARGET_BRANCH'..."
git checkout "$TARGET_BRANCH"
git pull origin "$TARGET_BRANCH"
echo "----------------------------------------------------"

echo "Buscando PRs abertos com a label '${PR_LABEL}'..."
# --- MUDANÇA CRUCIAL AQUI ---
# Adicionado "| tr -d '\r'" para remover o caractere de quebra de linha do Windows da lista de PRs
PRS=$(gh pr list --repo "$REPO" --label "$PR_LABEL" --state open --json number,createdAt \
  | jq -r 'sort_by(.createdAt) | .[] | .number' \
  | tr -d '\r')

if [ -z "$PRS" ]; then
  echo "!!! ERRO DE DIAGNÓSTICO: Nenhum PR foi encontrado com a label '${PR_LABEL}'."
  exit 1
fi

echo "Fila de PRs encontrada (do mais antigo para o mais novo):"
echo "$PRS"
echo "----------------------------------------------------"

# Itera sobre cada PR encontrado
for PR in $PRS; do
  echo ""
  echo ">>> Processando PR #${PR}..."

  echo "Fazendo checkout do branch do PR..."
  if ! gh pr checkout "$PR" --repo "$REPO"; then
    echo "!!! AVISO: Falha ao fazer checkout do PR #${PR}. Pulando para o próximo."
    continue
  fi

  BRANCH_NAME=$(git rev-parse --abbrev-ref HEAD)
  echo "Branch atual: ${BRANCH_NAME}"

  echo "Mesclando '${TARGET_BRANCH}' com estratégia '-X ours' (favorecendo o PR)..."
  git merge -X ours "origin/${TARGET_BRANCH}" || true

  echo "Verificando se ainda há conflitos..."
  CONFLICTS=$(git ls-files -u | awk '{print $4}' | sort -u || true)
  if [ -n "$CONFLICTS" ]; then
    echo "Conflitos encontrados. Resolvendo arquivo por arquivo..."
    for FILE in $CONFLICTS; do
      echo "   - Resolvendo '${FILE}' favorecendo o PR (ours)..."
      git checkout --ours -- "$FILE"
      git add "$FILE"
    done
    git commit -m "chore(codex): auto-resolve conflicts preferring PR (local script)"
    echo "Commit de resolução de conflitos criado."
  else
    echo "Nenhum conflito restante."
  fi

  echo "Criando commit vazio para garantir o acionamento dos checks..."
  git commit --allow-empty -m "chore: trigger CI checks (local script)"

  echo "Enviando push para o branch '${BRANCH_NAME}'..."
  git push origin "HEAD:${BRANCH_NAME}"
  echo "Push concluído para o PR #${PR}!"

  echo "PR #${PR} processado. Voltando para '${TARGET_BRANCH}' para o próximo."
  git checkout "$TARGET_BRANCH"
done

echo ""
echo "--- PROCESSAMENTO DA FILA LOCAL CONCLUÍDO ---"