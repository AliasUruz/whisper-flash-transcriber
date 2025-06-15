# Plano de Ação Técnico: Correção de Canal de Áudio para Transcrição em Memória

## 1. Introdução e Visão Geral do Projeto

O Whisper Recorder é uma aplicação desktop desenvolvida em Python, utilizando as bibliotecas Tkinter e CustomTkinter para a interface gráfica. Seu propósito principal é facilitar a transcrição de áudio em tempo real ou quase real, aproveitando o poder do modelo Whisper da OpenAI (via biblioteca `transformers`) para converter fala em texto. Além da transcrição, o aplicativo oferece funcionalidades como gravação de áudio via `sounddevice`, gerenciamento de hotkeys para controle de gravação, integração com APIs de correção de texto (como Google Gemini e OpenRouter) e um sistema de notificação via ícone na bandeja do sistema.

O objetivo contínuo deste projeto é aprimorar a qualidade, velocidade e eficiência das transcrições, especialmente para áudios mais longos, e reduzir a taxa de erros. Recentemente, foi implementada a funcionalidade de transcrição de áudio em memória para eliminar o gargalo de I/O causado pelo salvamento de arquivos `.wav` temporários em disco. No entanto, um problema surgiu durante os testes dessa nova funcionalidade, que é o foco deste plano.

## 2. Análise Detalhada do Problema

### 2.1. O Erro Específico

Durante a execução do aplicativo e a tentativa de transcrição de áudio, o seguinte erro foi observado no log:

```
ValueError: We expect a single channel audio input for AutomaticSpeechRecognitionPipeline
```

Este erro é emitido pela biblioteca `transformers`, especificamente pelo `AutomaticSpeechRecognitionPipeline` (o pipeline do Whisper), indicando uma incompatibilidade no formato do áudio de entrada.

### 2.2. Causa Raiz: Incompatibilidade de Canais de Áudio

O pipeline de transcrição do Whisper, conforme implementado na biblioteca `transformers`, é projetado para processar áudio de **canal único (mono)**. No entanto, o dispositivo de gravação de áudio do usuário (gerenciado pela biblioteca `sounddevice`) pode estar configurado para capturar áudio em **múltiplos canais (estéreo ou superior)**.

Quando o áudio é gravado, ele é armazenado em arrays NumPy. Se o dispositivo de gravação for estéreo, cada amostra de áudio terá dois valores (um para o canal esquerdo e outro para o direito), resultando em um array NumPy com duas dimensões: `(número_de_amostras, número_de_canais)`. O erro ocorre porque este array multi-canal está sendo passado diretamente para o pipeline do Whisper, que espera um array com apenas uma dimensão (ou uma segunda dimensão de tamanho 1 para um único canal).

### 2.3. Fluxo de Dados de Áudio e Ponto de Falha

Para entender onde a correção deve ser aplicada, é crucial rastrear o fluxo do `audio_data` dentro do `whisper_tkinter.py`:

1.  **Gravação (`_record_audio_task`):**
    *   O método `_record_audio_task` utiliza `sounddevice` para capturar o áudio do microfone.
    *   Os dados de áudio são lidos em chunks e armazenados na lista `self.recording_data` como arrays NumPy. A forma desses arrays dependerá da configuração do dispositivo de áudio (ex: `(N_samples,)` para mono, `(N_samples, 2)` para estéreo).

2.  **Parada da Gravação e Concatenação (`stop_recording`):**
    *   Quando a gravação é parada, o método `stop_recording` é chamado.
    *   Ele itera sobre `self.recording_data` e concatena todos os chunks de áudio válidos em um único array NumPy chamado `audio_data_copy` (linha 1726: `audio_data_copy = np.concatenate(valid_data, axis=0)`).
    *   **Este é o ponto crítico:** Se os chunks originais eram multi-canal, `audio_data_copy` também será multi-canal.

3.  **Processamento e Transcrição (`_process_audio_task` e `_transcribe_audio_task`):**
    *   `audio_data_copy` é então passado para `_process_audio_task` (linha 1751).
    *   Dentro de `_process_audio_task`, o `audio_data` (que é o `audio_data_copy` original) é passado diretamente para `_transcribe_audio_task` (linha 1863).
    *   Finalmente, em `_transcribe_audio_task`, o `audio_input` (que ainda é o array multi-canal) é passado para `self.pipe()` (linha 1913), resultando no `ValueError`.

### 2.4. Impacto da Incompatibilidade

A incompatibilidade de canais impede que qualquer transcrição seja realizada, tornando a funcionalidade central do aplicativo inoperante. Embora a implementação em memória tenha sido concluída, este erro fundamental precisa ser resolvido para que o aplicativo possa transcrever áudio com sucesso.

## 3. Racional da Solução Proposta

### 3.1. Conversão para Mono: A Necessidade

Dado que o pipeline do Whisper exige áudio mono, a solução mais direta e eficaz é converter o áudio capturado para um único canal antes de passá-lo para a transcrição. Isso garante que o formato de entrada esteja em conformidade com as expectativas do modelo, resolvendo o `ValueError`.

### 3.2. Método de Conversão: Média dos Canais (`np.mean`)

Existem algumas maneiras de converter áudio multi-canal para mono:
*   **Selecionar um único canal:** Descartar um dos canais (ex: `audio_data_copy[:, 0]`). Embora simples, isso descarta metade da informação de áudio, o que pode impactar a qualidade da transcrição.
*   **Média dos Canais:** Calcular a média dos valores de todos os canais para cada amostra (ex: `np.mean(audio_data_copy, axis=1)`). Esta abordagem é preferível porque **preserva a informação de todos os canais** ao combiná-los em um único canal. Isso geralmente resulta em uma qualidade de áudio mono mais rica e, consequentemente, em uma transcrição mais precisa.

A escolha de `np.mean(audio_data_copy, axis=1)` é a mais robusta e recomendada para este cenário, pois minimiza a perda de dados e maximiza a qualidade do áudio de entrada para o modelo de transcrição.

### 3.3. Ponto de Inserção da Lógica

O ponto ideal para inserir a lógica de conversão para mono é no método `stop_recording`, logo após a concatenação dos chunks de áudio em `audio_data_copy`. Neste ponto, temos o array completo de áudio gravado, e a conversão aqui garante que qualquer função subsequente que utilize `audio_data_copy` (como `_process_audio_task` e `_transcribe_audio_task`) receba o formato de canal correto.

## 4. Plano de Ação Detalhado para o Codex

Este plano detalha as modificações necessárias no arquivo `whisper_tkinter.py` para corrigir o problema do canal de áudio.

**Tarefa Principal: Converter Áudio para Mono antes da Transcrição**

**Objetivo da Tarefa:** Modificar o método `stop_recording` para garantir que o `audio_data_copy` seja sempre um array de áudio mono antes de ser passado para o pipeline de transcrição.

*   **Contexto da Modificação:**
    *   **Arquivo:** `whisper_tkinter.py`
    *   **Método Alvo:** `stop_recording(self, agent_mode=False)`
    *   **Localização:** Dentro do bloco `try` que prepara `audio_data_copy`, especificamente após a linha `audio_data_copy = np.concatenate(valid_data, axis=0)`.
    *   **Dependências:** A biblioteca `numpy` já está importada e sendo utilizada para manipulação de arrays.

*   **Passos de Implementação (Sub-tarefas e Sub-sub-tarefas):**

    **Sub-tarefa 1.1: Localizar o Ponto de Inserção Preciso**
    *   **Objetivo:** Identificar a linha exata no código onde a lógica de conversão para mono deve ser inserida.
    *   **Passos:**
        1.  Abra o arquivo `whisper_tkinter.py`.
        2.  Navegue até a definição do método `stop_recording(self, agent_mode=False)`.
        3.  Dentro deste método, localize o bloco `try` que começa na linha `1723` (ou próximo a ela, dependendo de futuras modificações).
        4.  Identifique a linha `1726`: `audio_data_copy = np.concatenate(valid_data, axis=0)`. Esta linha é crucial, pois é onde todos os chunks de áudio gravados são combinados em um único array. A lógica de conversão deve vir imediatamente após esta linha, dentro do mesmo bloco `if valid_data:`.

    **Sub-tarefa 1.2: Adicionar Lógica de Verificação de Canais**
    *   **Objetivo:** Criar uma condição para verificar se o `audio_data_copy` é um array multi-canal que precisa ser convertido.
    *   **Passos:**
        1.  Após a linha `audio_data_copy = np.concatenate(valid_data, axis=0);` (linha 1726), adicione uma nova linha em branco para clareza.
        2.  Na linha seguinte, insira um comentário para indicar a finalidade do código: `# Convert to mono if necessary`.
        3.  Na linha subsequente, adicione a estrutura condicional `if` para verificar as dimensões do array:
            *   `audio_data_copy.ndim == 2`: Verifica se o array tem duas dimensões. Um array mono terá `ndim == 1` (apenas amostras), enquanto um estéreo ou multi-canal terá `ndim == 2` (amostras e canais).
            *   `audio_data_copy.shape[1] > 1`: Se for 2D, verifica se o número de canais (a segunda dimensão) é maior que 1. Isso confirma que é um áudio multi-canal.
        4.  A linha completa do `if` será: `if audio_data_copy.ndim == 2 and audio_data_copy.shape[1] > 1:`.

    **Sub-tarefa 1.3: Implementar a Conversão para Mono**
    *   **Objetivo:** Realizar a conversão do áudio multi-canal para mono usando a média dos canais.
    *   **Passos:**
        1.  Dentro do bloco `if` criado na Sub-tarefa 1.2, adicione uma mensagem de log informativa. Esta mensagem ajudará na depuração e confirmará que a conversão está ocorrendo quando necessário. A mensagem deve ser: `logging.info(f"Converting audio from {audio_data_copy.shape[1]} channels to mono.")`.
        2.  Na linha seguinte, execute a operação de conversão: `audio_data_copy = np.mean(audio_data_copy, axis=1)`.
            *   `np.mean()`: Calcula a média dos elementos de um array.
            *   `audio_data_copy`: O array de áudio multi-canal.
            *   `axis=1`: Especifica que a média deve ser calculada ao longo do segundo eixo (o eixo dos canais). Isso significa que para cada amostra de tempo, os valores de todos os canais serão somados e divididos pelo número de canais, resultando em um único valor para aquela amostra.
        3.  Atribua o resultado dessa operação de volta à variável `audio_data_copy`. Isso sobrescreverá o array multi-canal original com a sua versão mono.

    **Sub-tarefa 1.4: Garantir o Fluxo de Controle Correto**
    *   **Objetivo:** Assegurar que a lógica existente para áudio inválido ou vazio continue funcionando corretamente.
    *   **Passos:**
        1.  Verifique se o bloco `else:` (que lida com `No valid audio data recorded to save.`) e o bloco `except Exception as e:` (que lida com erros gerais de preparação de áudio) permanecem inalterados e com a indentação correta. A nova lógica de conversão deve estar aninhada *apenas* dentro do `if valid_data:`.

*   **Bloco de Código Relevante (com explicações detalhadas):**

    ```python
    <<<<<<< SEARCH
                if valid_data:
                    audio_data_copy = np.concatenate(valid_data, axis=0)
                else:
                    logging.warning("No valid audio data recorded to save.")
                    self._set_state(STATE_IDLE)
                    self._log_status("No audio recorded.", error=True)
                    self.recording_data.clear() # Clear invalid data
                    self.start_time = None # Reset start time
                    return
            except Exception as e:
                 logging.error(f"Failed to prepare audio data: {e}", exc_info=True)
                 self._set_state(STATE_ERROR_AUDIO)
                 self._log_status("Error processing audio data.", error=True)
                 self.recording_data.clear() # Clear potentially corrupt data
                 self.start_time = None # Reset start time
                 return
    =======
                if valid_data:
                    # Concatena todos os chunks de áudio válidos em um único array NumPy.
                    # Se o áudio original for estéreo, audio_data_copy terá a forma (N_amostras, N_canais).
                    audio_data_copy = np.concatenate(valid_data, axis=0)

                    # Início da lógica para converter áudio multi-canal para mono.
                    # Verifica se o array tem duas dimensões (indicando amostras x canais)
                    # E se o número de canais (audio_data_copy.shape[1]) é maior que 1.
                    if audio_data_copy.ndim == 2 and audio_data_copy.shape[1] > 1:
                        # Registra no log que a conversão está ocorrendo, informando o número original de canais.
                        logging.info(f"Converting audio from {audio_data_copy.shape[1]} channels to mono.")
                        # Calcula a média dos valores de todos os canais para cada amostra.
                        # O 'axis=1' garante que a média seja feita ao longo do eixo dos canais,
                        # resultando em um array mono (uma única dimensão de amostras).
                        audio_data_copy = np.mean(audio_data_copy, axis=1)
                    # Fim da lógica de conversão para mono.

                else:
                    # Este bloco é executado se não houver dados de áudio válidos para processar.
                    logging.warning("No valid audio data recorded to save.")
                    self._set_state(STATE_IDLE) # Retorna o aplicativo ao estado ocioso.
                    self._log_status("No audio recorded.", error=True) # Exibe uma mensagem de status de erro.
                    self.recording_data.clear() # Limpa quaisquer dados inválidos que possam ter sido coletados.
                    self.start_time = None # Reseta o tempo de início da gravação.
                    return # Sai da função, pois não há áudio para salvar ou transcrever.
            except Exception as e:
                 # Este bloco captura quaisquer exceções que ocorram durante a preparação dos dados de áudio.
                 logging.error(f"Failed to prepare audio data: {e}", exc_info=True) # Registra o erro detalhadamente.
                 self._set_state(STATE_ERROR_AUDIO) # Define o estado do aplicativo como erro de áudio.
                 self._log_status("Error processing audio data.", error=True) # Exibe uma mensagem de status de erro.
                 self.recording_data.clear() # Limpa os dados de gravação, que podem estar corrompidos.
                 self.start_time = None # Reseta o tempo de início da gravação.
                 return # Sai da função após o tratamento do erro.
    >>>>>>> REPLACE
    ```

## 5. Critérios de Verificação e Teste

Após a implementação das mudanças, Isaque, por favor, execute os seguintes testes para garantir que a correção foi bem-sucedida e que não houve regressões em outras funcionalidades.

### 5.1. Cenário 1: Gravação e Transcrição de Áudio (Cenário Padrão)

*   **Objetivo:** Verificar se a transcrição funciona corretamente com a correção de canal, sem erros.
*   **Passos Detalhados:**
    1.  **Inicie o Aplicativo:** Execute o arquivo `whisper_tkinter.py` a partir do terminal ou do VS Code.
    2.  **Aguarde o Carregamento:** Espere até que o modelo Whisper seja totalmente carregado e o aplicativo entre no estado `IDLE` (o ícone da bandeja deve mudar para o estado normal).
    3.  **Inicie a Gravação:** Pressione a hotkey de gravação (F3 por padrão) para iniciar a gravação. O ícone da bandeja deve mudar para o estado de gravação.
    4.  **Fale Algo:** Fale algumas frases claras e distintas no microfone.
    5.  **Pare a Gravação:** Pressione a hotkey de gravação (F3) novamente para parar a gravação. O ícone da bandeja deve mudar para o estado de transcrição e, em seguida, para o estado `IDLE` novamente.
*   **Comportamento Esperado:**
    *   **Ausência de Erro:** O erro `ValueError: We expect a single channel audio input for AutomaticSpeechRecognitionPipeline` **NÃO** deve aparecer no console ou nos logs.
    *   **Transcrição Correta:** O texto transcrito deve aparecer corretamente na área de transferência do sistema (e ser colado automaticamente se a opção `auto_paste` estiver ativada nas configurações).
    *   **Log de Conversão (se aplicável):** Se o seu dispositivo de gravação for estéreo (ou multi-canal), você deverá ver uma mensagem no log similar a: `INFO - ProcessAudioThread - Converting audio from X channels to mono.` (onde X será o número de canais do seu dispositivo, geralmente 2). Se o seu dispositivo já for mono, esta mensagem não aparecerá, o que é o comportamento esperado.
*   **Como Verificar:**
    *   Observe o console do terminal para mensagens de erro.
    *   Verifique o conteúdo da área de transferência (Ctrl+V em um editor de texto) ou o local onde o texto é colado.
    *   Analise as mensagens de `INFO` no log para confirmar a conversão de canal, se aplicável.

### 5.2. Cenário 2: Funcionalidade `save_audio_for_debug` (Verificação de Arquivo Mono)

*   **Objetivo:** Confirmar que a opção de salvar áudio para depuração funciona e que os arquivos salvos são mono após a correção.
*   **Passos Detalhados:**
    1.  **Inicie o Aplicativo:** Execute o `whisper_tkinter.py`.
    2.  **Abra as Configurações:** Clique com o botão direito no ícone da bandeja do sistema e selecione "Settings" (Configurações).
    3.  **Ative "Save Audio for Debug":** Na janela de configurações, localize a seção "Debug Settings" e ative o switch "Save Audio for Debug".
    4.  **Salve e Feche:** Clique no botão "Apply Settings" (Aplicar Configurações) e feche a janela de configurações.
    5.  **Inicie a Gravação:** Pressione a hotkey de gravação (F3) para iniciar a gravação.
    6.  **Fale Algo:** Fale algumas frases.
    7.  **Pare a Gravação:** Pressione a hotkey de gravação (F3) novamente para parar.
*   **Comportamento Esperado:**
    *   **Arquivo WAV Salvo:** Um novo arquivo `.wav` deve ser criado no mesmo diretório do `whisper_tkinter.py` (ex: `recording_TIMESTAMP.wav`).
    *   **Formato Mono:** Ao inspecionar este arquivo `.wav` (usando um reprodutor de mídia como VLC, Windows Media Player, ou uma ferramenta de análise de áudio como Audacity), ele deve ter **um único canal (mono)**, mesmo que o dispositivo de gravação original seja estéreo.
    *   **Transcrição Funcional:** A transcrição ainda deve ocorrer sem erros e o texto deve ser gerado corretamente.
*   **Como Verificar:**
    *   Navegue até o diretório do projeto e procure por arquivos `.wav` recém-criados.
    *   Abra o arquivo `.wav` em um reprodutor de mídia e verifique suas propriedades (número de canais). Em muitos reprodutores, isso pode ser encontrado em "Propriedades" ou "Informações do Codec". No Audacity, você verá uma única trilha de áudio.

### 5.3. Cenário 3: Testes de Regressão (Funcionalidades Existentes)

*   **Objetivo:** Garantir que a correção do canal de áudio não introduziu problemas em outras funcionalidades já existentes do aplicativo.
*   **Passos Detalhados e Comportamento Esperado:**

    1.  **Verificar Hotkeys:**
        *   **Passos:** Teste as hotkeys de gravação (F3) e de agente (F4) várias vezes.
        *   **Esperado:** Ambas as hotkeys devem iniciar e parar a gravação/comando conforme configurado, sem atrasos ou falhas.

    2.  **Verificar Modos de Gravação:**
        *   **Passos:** Se o modo de gravação for "push-to-talk" (segurar F3 para gravar), teste-o.
        *   **Esperado:** A gravação deve iniciar apenas enquanto a hotkey é pressionada e parar ao soltar.

    3.  **Verificar Correção de Texto:**
        *   **Passos:** Certifique-se de que a correção de texto esteja ativada nas configurações. Grave e transcreva algo que possa se beneficiar da correção (ex: uma frase com gírias ou erros gramaticais intencionais).
        *   **Esperado:** O texto transcrito deve ser enviado para a API de correção e o resultado corrigido deve ser recebido e utilizado.

    4.  **Verificar Carregamento do Modelo:**
        *   **Passos:** Feche e reabra o aplicativo várias vezes.
        *   **Esperado:** O modelo Whisper deve carregar corretamente na inicialização, e o aplicativo deve transitar para o estado `IDLE` sem erros relacionados ao carregamento do modelo.

    5.  **Verificar Estados da Aplicação:**
        *   **Passos:** Observe o ícone da bandeja do sistema e os logs durante todo o ciclo de gravação e transcrição.
        *   **Esperado:** Os estados da aplicação (`IDLE`, `RECORDING`, `TRANSCRIBING`, `ERROR_TRANSCRIPTION`, etc.) devem ser atualizados corretamente e refletir o status atual do aplicativo.

    6.  **Verificar Tratamento de Erros (Opcional, se possível simular):**
        *   **Passos:** Se houver uma maneira de simular um erro (ex: desconectar a internet para falha da API de correção, ou tentar transcrever um arquivo de áudio corrompido se essa funcionalidade for exposta), observe o comportamento.
        *   **Esperado:** O aplicativo deve entrar em um estado de erro apropriado (`STATE_ERROR_...`), exibir uma mensagem de status e permitir a recuperação ou reinício.

Este plano detalhado deve fornecer ao Codex todas as informações necessárias para implementar a correção e a você, Isaque, um guia completo para verificar a funcionalidade.