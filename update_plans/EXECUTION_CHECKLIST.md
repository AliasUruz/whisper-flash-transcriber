# Checklist de Execução da Refatoração

Este documento acompanha as tarefas necessárias para aplicar as melhorias
de interface descritas em `UI_REFACTORING.md`. Cada item inclui passos
atômicos e observações para acompanhamento.

## 1. Estruturar diretório `ui`
- [x] Criar pasta `ui` e subpasta `components`
- [x] Adicionar arquivos `__init__.py`, `main_window.py` e `settings_window.py`
- [x] Definir esboço das classes `MainWindow` e `SettingsWindow`

## 2. Preparar dependências
- [x] Incluir `customtkinter==5.2.2` em `requirements.txt`
 - [x] Instalar dependência no ambiente de desenvolvimento

## 3. Migração inicial
- [ ] Mapear funções de interface em `whisper_tkinter.py`
 - [x] Transferir criação da janela principal para `MainWindow`
 - [x] Transferir lógica de configurações para `SettingsWindow`r

## 4. Componentização
- [ ] Implementar primeiros widgets em `ui/components`
- [ ] Documentar parâmetros e exemplos de uso

## 5. Temas e personalização
- [ ] Criar diretório `themes` com arquivos `default.json`
- [ ] Adicionar carregamento de tema na inicialização da UI

A cada etapa concluída, marcar a caixa correspondente e registrar possíveis
ajustes necessários.

