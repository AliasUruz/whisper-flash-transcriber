# Plano de Correção e Melhorias do Aplicativo Whisper

Este documento detalha o plano de ação para corrigir os problemas identificados e melhorar a robustez geral do aplicativo.

## 1. Análise Geral da Arquitetura

O aplicativo é composto pelos seguintes módulos principais:
- **`main.py`**: Ponto de entrada, inicializa o `AppCore` e o `UIManager`.
- **`core.py`**: O núcleo da aplicação, orquestrando todos os outros módulos, gerenciando o estado e os hotkeys.
- **`config_manager.py`**: Gerencia o carregamento, salvamento e acesso a todas as configurações do usuário e segredos.
- **`ui_manager.py`**: Controla toda a interface gráfica, incluindo a janela de configurações e o ícone da bandeja do sistema.
- **`audio_handler.py`**: Responsável por capturar o áudio do microfone, utilizando VAD (Voice Activity Detection) se habilitado.
- **`transcription_handler.py`**: Gerencia o pipeline de transcrição com o modelo Whisper e orquestra a chamada para os serviços de correção de texto.
- **`gemini_api.py` / `openrouter_api.py`**: Clientes para as APIs de IA, responsáveis pela correção de texto e pelo modo agêntico.
- **`keyboard_hotkey_manager.py`**: Gerencia o registro e a escuta dos hotkeys globais.
- **`vad_manager.py`**: Implementa a lógica de detecção de atividade de voz (VAD).

## 2. Diagnóstico e Plano de Ação

A seguir, detalhamos cada problema reportado, seu diagnóstico e o plano de ação para corrigi-lo.

### Problema 1: Correção de Texto Resume ou Altera o Sentido da Transcrição
- **Diagnóstico:** O prompt enviado para a API Gemini (`gemini_prompt` em `src/config_manager.py`) é muito agressivo. Instruções como "Make the text MUCH MORE FLUID AND COHERENT" e "Connect related thoughts" dão à IA permissão para reestruturar e resumir o texto, em vez de apenas corrigir erros de transcrição, pontuação e fluidez.
- **Plano de Ação:** Modificar o prompt padrão para ser mais conservador, focando em correções e preservação do conteúdo original.
- **Status:** Planejado. Detalhes em `memory-bank/plans/Plano_Codex_P1_T1_PromptGemini.md`.

### Problema 2: Dificuldade em Alterar os Prompts de Correção e Modo Agêntico
- **Diagnóstico:** O fluxo de atualização de configurações parece complexo e pode ter uma falha na propagação do novo valor do prompt para a instância do `GeminiAPI`.
- **Plano de Ação:** Simplificar e robustecer o processo de reinicialização do cliente Gemini, garantindo que qualquer alteração nas configurações relevantes (incluindo os prompts) force uma recarga.
- **Status:** Planejado. Detalhes em `memory-bank/plans/Plano_Codex_P2_T1_SimplificarGeminiAPI.md`.

### Problema 3: Função "Cancel Transcription" Não Funciona Corretamente
- **Diagnóstico:** O cancelamento da transcrição não reverte o estado da aplicação de forma eficaz, levando a uma experiência de usuário inconsistente.
- **Plano de Ação:** Ajustar a lógica de cancelamento para que, ao detectar o evento, a tarefa não apenas pare de processar o resultado, mas também ativamente redefina o estado da aplicação para `IDLE`.
- **Status:** Planejado em duas partes:
    - Parte 1: Adicionar callback no `TranscriptionHandler` (`memory-bank/plans/Plano_Codex_P3_T1_AddCancelCallback_TH.md`).
    - Parte 2: Implementar o callback no `AppCore` (`memory-bank/plans/Plano_Codex_P3_T2_ImplementCancelCallback_AC.md`).

### Problema 4: Resumo excessivo na versão mais atual
- **Diagnóstico:** Este problema está diretamente ligado ao **Problema 1**. O prompt da IA é a causa raiz.
- **Plano de Ação:** A solução para o Problema 1 resolverá diretamente esta questão.
- **Status:** Resolvido pela solução do Problema 1.

## Próximos Passos:
- Executar as tarefas detalhadas nos planos de ação para o Codex.
- Após a implementação, realizar os testes de verificação e validação para cada problema.