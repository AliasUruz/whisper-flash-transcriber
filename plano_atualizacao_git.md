# Plano de Atualização do Repositório Git

## Objetivo
Atualizar o repositório Git remoto com a versão atual do aplicativo local, garantindo que, em caso de conflitos de merge, a versão local dos arquivos seja priorizada. Será dada atenção especial à proteção de arquivos sensíveis.

## Contexto
O aplicativo `WhisperTeste` precisa ter sua versão local sincronizada com o repositório Git. O usuário Isaque solicitou que a versão local prevaleça em caso de conflitos de merge e que arquivos sensíveis sejam protegidos.

## Arquivos Potencialmente Sensíveis Identificados
*   [`hotkey_config.json`](hotkey_config.json)
*   [`gemini_api.py`](gemini_api.py)
*   [`openrouter_api.py`](openrouter_api.py)
*   [`.env.example`](.env.example) (Este é um exemplo, o `.env` real conteria as chaves e *não* deveria ser versionado)

## Plano de Execução

### Fase 1: Preparação e Verificação do Ambiente Git

**Objetivo:** Assegurar que o ambiente Git local esteja limpo e pronto para as operações, e identificar arquivos sensíveis.

**Checklist:**
- [ ] 1.1. Verificar o status atual do repositório Git.
    - Comando: `git status`
    - Expectativa: Entender se há arquivos modificados, novos ou não rastreados.
- [ ] 1.2. Identificar a branch atual.
    - Comando: `git branch --show-current`
    - Expectativa: Saber em qual branch as operações serão realizadas (assumindo `main` ou `master` se não especificado).
- [ ] 1.3. Verificar o conteúdo de [`hotkey_config.json`](hotkey_config.json), [`gemini_api.py`](gemini_api.py), [`openrouter_api.py`](openrouter_api.py) e [`.env.example`](.env.example) para confirmar a presença de informações sensíveis.
    - Ação: `read_file` para cada um.
    - Expectativa: Confirmar se contêm chaves, tokens ou credenciais.
- [ ] 1.4. Garantir que arquivos sensíveis (como `.env` real, se existir) estejam no [`.gitignore`](.gitignore).
    - Ação: `read_file` para `.gitignore`. Se `.env` não estiver, adicionar.
    - Expectativa: `.env` (e outros arquivos de credenciais reais) não devem ser rastreados pelo Git.

### Fase 2: Commit das Alterações Locais

**Objetivo:** Salvar todas as alterações locais no histórico do Git antes de tentar o merge.

**Checklist:**
- [ ] 2.1. Adicionar todas as alterações locais ao *staging area*.
    - Comando: `git add .`
    - Expectativa: Todas as modificações, adições e exclusões de arquivos locais estão prontas para o commit.
- [ ] 2.2. Realizar o commit das alterações locais.
    - Comando: `git commit -m "Atualização local do aplicativo: Sincronizando com a versão mais recente."`
    - Expectativa: Um novo commit foi criado com o estado atual do aplicativo.

### Fase 3: Sincronização com o Repositório Remoto (Pull e Resolução de Conflitos)

**Objetivo:** Puxar as últimas alterações do repositório remoto e resolver quaisquer conflitos, priorizando a versão local.

**Checklist:**
- [ ] 3.1. Puxar as últimas alterações do repositório remoto.
    - Comando: `git pull origin <branch_atual>` (substituir `<branch_atual>` pela branch identificada na Fase 1).
    - Expectativa: O Git tenta fazer o merge das alterações remotas com as locais.
- [ ] 3.2. **Se houver conflitos de merge:**
    - [ ] 3.2.1. Listar os arquivos com conflito.
        - Comando: `git status`
        - Expectativa: Ver a lista de arquivos que precisam de resolução.
    - [ ] 3.2.2. Para cada arquivo em conflito, priorizar a versão local.
        - Comando: `git checkout --ours <caminho_do_arquivo_em_conflito>`
        - Expectativa: A versão local do arquivo sobrescreve a versão remota no conflito.
    - [ ] 3.2.3. Adicionar os arquivos resolvidos ao *staging area*.
        - Comando: `git add <caminho_do_arquivo_resolvido>` (repetir para cada arquivo)
        - Expectativa: Os arquivos com conflito estão marcados como resolvidos.
    - [ ] 3.2.4. Finalizar o merge commit.
        - Comando: `git commit -m "Merge remoto resolvido: Priorizando versão local."`
        - Expectativa: O merge foi concluído e um novo commit de merge foi criado.
- [ ] 3.3. **Se não houver conflitos:**
    - Expectativa: O `git pull` completou com sucesso, e as alterações remotas foram integradas.

### Fase 4: Envio das Alterações para o Repositório Remoto (Push)

**Objetivo:** Enviar as alterações locais (incluindo o merge resolvido, se aplicável) para o repositório remoto.

**Checklist:**
- [ ] 4.1. Enviar as alterações para o repositório remoto.
    - Comando: `git push origin <branch_atual>` (substituir `<branch_atual>` pela branch identificada na Fase 1).
    - Expectativa: As alterações locais foram enviadas com sucesso para o repositório remoto.

## Considerações de Segurança para Arquivos Sensíveis
*   **Nunca versionar credenciais:** O arquivo `.env.example` é um *exemplo*. O arquivo `.env` (que conteria as credenciais reais) *nunca* deve ser adicionado ao controle de versão. Certifique-se de que ele esteja no `.gitignore`.
*   **Verificação manual:** Após o processo, é recomendável uma verificação manual do repositório remoto para garantir que nenhum arquivo sensível foi acidentalmente enviado.