# Bloco único para RECRIAR o script 'processa_fila.sh' na versão correta

echo "--- CONSTRUINDO O SCRIPT FINAL (VERSÃO SEM EDITOR) ---"

# Constrói o arquivo 'processa_fila.sh' do zero com a lógica correta
echo '#!/bin/bash' > processa_fila.sh
echo 'set -e' >> processa_fila.sh
echo 'REPO="AliasUruz/whisper-flash-transcriber"' >> processa_fila.sh
echo 'PR_LABEL="codex"' >> processa_fila.sh
echo 'TARGET_BRANCH="stable"' >> processa_fila.sh
echo 'echo "--- INICIANDO PROCESSAMENTO (VERSÃO FINAL) ---"' >> processa_fila.sh
echo 'git merge --abort || echo "Nenhum merge para abortar. Repositório limpo."' >> processa_fila.sh
echo 'git checkout "$TARGET_BRANCH"' >> processa_fila.sh
echo 'git pull origin "$TARGET_BRANCH"' >> processa_fila.sh
echo 'echo "----------------------------------------------------"' >> processa_fila.sh
echo 'echo "Buscando PRs com a label '\''${PR_LABEL}'\''..."' >> processa_fila.sh
echo "PRS=\$(gh pr list --repo \"\$REPO\" --label \"\$PR_LABEL\" --state open --json number,createdAt | jq -r 'sort_by(.createdAt) | .[] | .number' | tr -d '\r')" >> processa_fila.sh
echo 'if [ -z "$PRS" ]; then echo "Nenhum PR encontrado."; exit 0; fi' >> processa_fila.sh
echo 'echo "Fila de PRs encontrada: \$PRS"' >> processa_fila.sh
echo 'echo "----------------------------------------------------"' >> processa_fila.sh
echo 'for PR in $PRS; do' >> processa_fila.sh
echo '  echo ""' >> processa_fila.sh
echo '  echo ">>> Processando PR #${PR}..."' >> processa_fila.sh
echo '  if ! gh pr checkout "$PR" --repo "$REPO"; then echo "!!! AVISO: Falha no checkout do PR #${PR}. Pulando."; continue; fi' >> processa_fila.sh
echo '  BRANCH_NAME=$(git rev-parse --abbrev-ref HEAD)' >> processa_fila.sh
echo '  echo "Branch atual: ${BRANCH_NAME}"' >> processa_fila.sh
echo '  echo "Mesclando '\''${TARGET_BRANCH}'\'' sem abrir o editor..."' >> processa_fila.sh
echo '  git merge --no-edit -X ours "origin/${TARGET_BRANCH}" || true' >> processa_fila.sh
echo '  echo "Criando commit vazio para acionar checks..."' >> processa_fila.sh
echo '  git commit --allow-empty -m "chore: trigger CI checks"' >> processa_fila.sh
echo '  echo "Enviando push para o branch '\''${BRANCH_NAME}'\''..."' >> processa_fila.sh
echo '  git push origin "HEAD:${BRANCH_NAME}"' >> processa_fila.sh
echo '  echo "Push concluído para o PR #${PR}!"' >> processa_fila.sh
echo '  echo "PR #${PR} processado. Voltando para '\''${TARGET_BRANCH}'\''."' >> processa_fila.sh
echo '  git checkout "$TARGET_BRANCH"' >> processa_fila.sh
echo 'done' >> processa_fila.sh
echo 'echo ""' >> processa_fila.sh
echo 'echo "--- PROCESSAMENTO CONCLUÍDO COM SUCESSO ---"' >> processa_fila.sh

# Torna o script executável
chmod +x processa_fila.sh

echo "✅ Script 'processa_fila.sh' recriado com sucesso!"