# nlp-toolkit-downloader 🚀✨

## Seu Assistente Inteligente para Preparar o Ambiente NLP! 🤖🔧

[![Python Version](https://img.shields.io/badge/Python-3.8+-blue.svg?style=for-the-badge&logo=python)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)
[![Status](https://img.shields.io/badge/Status-Ativo%20✅-success?style=for-the-badge)](https://github.com/chaos4455/nlp-toolkit-downloader/) <!-- Atualize o link se for um repo diferente -->
[![Autor](https://img.shields.io/badge/Criado%20por-Elias%20Andrade%20(chaos4455)-informational?style=for-the-badge)](https://github.com/chaos4455)

<!-- Badges dos Modelos/Dados -->
<p align="center">
  <img src="https://img.shields.io/badge/Modelo-BERT%20(base/large)-blue?style=flat-square&logo=google" alt="BERT Badge"/>
  <img src="https://img.shields.io/badge/Modelo-GPT--2%20(base/medium)-green?style=flat-square&logo=openai" alt="GPT-2 Badge"/>
  <img src="https://img.shields.io/badge/Modelo-Whisper%20(s/m/l--v3)-purple?style=flat-square&logo=openai" alt="Whisper Badge"/>
  <img src="https://img.shields.io/badge/Dados-NLTK%20Essentials-red?style=flat-square" alt="NLTK Badge"/>
  <img src="https://img.shields.io/badge/Embeddings-Word2Vec%20%26%20GloVe-orange?style=flat-square" alt="Embeddings Badge"/>
</p>

---

## 🎯 O que é isso?

![c20f2ade-ba12-49a7-9f1d-5b95026fad51](https://github.com/user-attachments/assets/6313c7ab-7992-4d6a-b1a9-9204c18202c9)
*(Exemplo visual do output colorido e informativo do script)*

Cansado de baixar manualmente cada modelo e pacote de dados para seus projetos de Processamento de Linguagem Natural (NLP)? 😩 Perder tempo configurando tudo repetidamente, apenas para descobrir depois que esqueceu algo ou que o cache não funcionou como esperado?

O **NLP Toolkit Downloader** é um script Python **robusto**, **configurável** e **altamente visual**, projetado para automatizar completamente o download e a configuração inicial dos modelos e conjuntos de dados NLP mais populares. Ele transforma um processo potencialmente tedioso e propenso a erros em uma execução simples, informativa e controlada, direto no seu console! Com feedback detalhado, colorido e cheio de emojis! 🎨📊✅

Criado por [Elias Andrade (chaos4455)](https://github.com/chaos4455), este script é ideal para pesquisadores, estudantes e desenvolvedores que precisam de um ambiente NLP pronto para usar rapidamente, com controle total sobre o que é baixado.

## ✨ Funcionalidades Principais

Este script não é apenas um downloader, é um assistente de configuração inteligente e flexível:

*   ⚙️ **Configuração Flexível:** Controle facilmente quais modelos e dados baixar através de um dicionário de configuração simples (`DOWNLOAD_CONFIG`) no topo do script. Basta definir `True` ou `False` para cada item!
*   ✅ **Download Automatizado:** Baixa múltiplos modelos e dados de fontes confiáveis (Hugging Face 🤗, NLTK, Gensim) com base na sua configuração.
*   🎨 **Console Rico e Colorido:** Utiliza `colorama` para fornecer feedback visual claro e agradável, com cores e ícones distintos para status (sucesso, erro, aviso, cache, pulado), etapas e informações.
*   📊 **Feedback Detalhado em Tempo Real:** Exibe mensagens passo a passo sobre o que está acontecendo: verificação de dependências, configuração de tarefas, verificação de cache, início de download, carregamento, testes e resumo final.
*   ⏳ **Barras de Progresso Integradas:** Exibe as barras de progresso fornecidas pelas bibliotecas `transformers` e `gensim` para downloads maiores, permitindo acompanhar o andamento real.
*   💾 **Verificação Inteligente de Cache:** Detecta *e confirma* se os modelos/dados já existem localmente (usando heurísticas e tempo de carregamento) para evitar downloads repetidos, economizando tempo e banda. Informa claramente se está usando o cache ou baixando.
*   🧠 **Suporte a Diversos Modelos e Dados:** Pré-configurado com uma variedade de ferramentas NLP essenciais (BERT, GPT-2, Whisper, NLTK, Word2Vec, GloVe).
*   🧪 **Testes Básicos Pós-Download:** Realiza verificações simples após o download (acesso à configuração, teste de pipeline básico quando aplicável) para aumentar a confiança de que os modelos foram carregados corretamente.
*   ❌ **Tratamento de Erros Robusto:** Captura e reporta erros de forma clara (dependências ausentes, problemas de rede, modelos não encontrados, falhas no download/carregamento). Inclui detalhes do erro e tracebacks para facilitar a depuração.
*   🔍 **Validação da Configuração:** Ao iniciar, avisa sobre possíveis inconsistências entre as tarefas definidas no código (`all_possible_tasks`) e as chaves presentes no seu `DOWNLOAD_CONFIG`, ajudando a evitar typos ou esquecimentos.
*   🚫 **Log de Tarefas Puladas:** Informa explicitamente quais tarefas estão sendo puladas com base na sua configuração `False` no `DOWNLOAD_CONFIG`.
*   ⏱️ **Resumo Final Detalhado:** Apresenta um sumário claro ao final da execução, mostrando o número de tarefas planejadas, puladas, tentadas, bem-sucedidas, falhas (listando quais falharam) e o tempo total decorrido.

## 📦 Modelos e Dados Suportados (Configuráveis!)

O script pode baixar os seguintes recursos. **Você controla quais deles serão efetivamente baixados editando o dicionário `DOWNLOAD_CONFIG` no início do script!**

| Ícone | Tipo         | Modelos/Dados Específicos (Exemplos)                           | Fonte Principal   | Chave no `DOWNLOAD_CONFIG`         |
| :---: | :----------- | :------------------------------------------------------------- | :---------------- | :--------------------------------- |
| 🧠    | **BERT**     | `bert-base-uncased`, `bert-large-uncased`                    | Hugging Face 🤗   | `Modelo BERT Base (Uncased)`, etc. |
| 🧠    | **GPT-2**    | `gpt2`, `gpt2-medium`                                          | Hugging Face 🤗   | `Modelo GPT-2 (Base)`, etc.        |
| 🎤    | **Whisper**  | `whisper-small`, `whisper-medium`, `whisper-large-v3`        | Hugging Face 🤗   | `Modelo Whisper Small`, etc.       |
| 📚    | **NLTK Data**| `punkt`, `stopwords`, `wordnet`, `averaged_perceptron_tagger`  | NLTK Project      | `Dados Essenciais NLTK`           |
| 📄    | **Embeddings**| `word2vec-google-news-300`, `glove-wiki-gigaword-100`          | Gensim Downloader | `Modelo Word2Vec Google News`, etc.|

*(Importante: Por padrão, no código fornecido, `Whisper Medium` e `Whisper Large v3` estão definidos como `False`. Verifique e edite o dicionário `DOWNLOAD_CONFIG` no script para selecionar exatamente o que você precisa!)*

*Aviso: O download desses modelos, especialmente os maiores (BERT Large, GPT-2 Medium, Whisper Large), pode consumir vários gigabytes de espaço em disco e levar um tempo considerável dependendo da sua conexão com a internet.* 🌐⏳

## ⚙️ Instalação e Pré-requisitos

Antes de executar, garanta que você tem o ambiente preparado:

1.  **Python:** Versão 3.8 ou superior é recomendada.
    [![Python Version](https://img.shields.io/badge/Python-3.8+-blue.svg?style=flat&logo=python)](https://python.org)
2.  **Pip:** O gerenciador de pacotes do Python (geralmente vem junto com o Python).

3.  **Bibliotecas Python:** Instale as dependências necessárias usando `pip`. Abra seu terminal ou prompt de comando e execute:

    ```bash
    pip install colorama transformers torch nltk gensim tqdm requests huggingface_hub
    ```
    [![Dependencies](https://img.shields.io/badge/Libs-Transformers%20%7C%20PyTorch%20%7C%20NLTK%20%7C%20Gensim%20%7C%20HF--Hub-critical?style=flat)](requirements.txt) <!-- Você pode criar um requirements.txt se quiser -->

    *   **Nota sobre PyTorch (`torch`):** A instalação do PyTorch pode ser específica para seu sistema operacional e hardware (CPU vs GPU com CUDA). O comando acima instala uma versão padrão. Se precisar de suporte a GPU ou encontrar problemas, consulte as instruções de instalação personalizadas no [site oficial do PyTorch](https://pytorch.org/).

## 🚀 Como Usar

É muito simples colocar o assistente para trabalhar:

1.  **Obtenha o Script:**
    *   **Opção A (Git):** Clone o repositório:
        ```bash
        git clone https://github.com/chaos4455/nlp-toolkit-downloader.git # Use o URL correto do seu repo
        cd nlp-toolkit-downloader
        ```
    *   **Opção B (Download Direto):** Baixe o arquivo `.py` diretamente do GitHub.

2.  **Instale as Dependências:**
    Se ainda não o fez, execute o comando `pip install ...` mostrado na seção anterior.

3.  **(IMPORTANTE) Configure o Download:**
    *   Abra o arquivo Python do script (ex: `download_models_advanced.py`) em um editor de texto ou IDE.
    *   Localize o dicionário `DOWNLOAD_CONFIG` logo no início do arquivo.
    *   Para cada item listado (ex: `"Modelo BERT Base (Uncased)"`), defina o valor como `True` se você deseja baixá-lo, ou `False` se deseja pulá-lo.
    *   **Salve o arquivo** após fazer suas escolhas.

4.  **Execute o Script:**
    Abra seu terminal ou prompt de comando, navegue até a pasta onde salvou o script e execute:
    ```bash
    python seu_nome_de_script.py
    ```
    *(Substitua `seu_nome_de_script.py` pelo nome real do arquivo)*

5.  **Acompanhe a Mágica:**
    Relaxe e veja o script trabalhar! O console mostrará cada passo: verificações, configuração, status do cache, downloads (com barras de progresso se aplicável), testes e, por fim, um resumo completo. 🎉

## 🖥️ Exemplo de Saída no Console (Conceitual Atualizado)

A saída real será colorida e mais detalhada, mas aqui está uma ideia do fluxo, incluindo a configuração e tarefas puladas:

```text
================================================================================
🚀 PIPELINE DE DOWNLOAD DE MODELOS E DADOS DE NLP 🚀
================================================================================

➡️ Verificando Dependências Essenciais
  ✅ Todas as dependências principais encontradas.

➡️ Configurando Tarefas de Download ⚙️
  ℹ️ Verificando 10 tarefas possíveis contra a configuração...
    ✅ Habilitado: Modelo BERT Base (Uncased)
    ✅ Habilitado: Modelo BERT Large (Uncased)
    ✅ Habilitado: Modelo GPT-2 (Base)
    ✅ Habilitado: Modelo GPT-2 Medium
    ✅ Habilitado: Modelo Whisper Small
    🚫 Pulando: Modelo Whisper Medium (definido como False)
    🚫 Pulando: Modelo Whisper Large v3 (definido como False)
    ✅ Habilitado: Dados Essenciais NLTK
    ✅ Habilitado: Modelo Word2Vec Google News
    ✅ Habilitado: Modelo GloVe Wikipedia (100d)

  ℹ️ 8 tarefas serão executadas.
  ℹ️ 2 tarefas foram puladas conforme configuração. 🚫

➡️ Tarefa: BERT Base: bert-base-uncased 🧠
  🔍 Verificando cache local para bert-base-uncased...
  💾 bert-base-uncased parece estar no cache (verificação heurística).
  ↪️ Carregando Tokenizer (bert-base-uncased)...
    ✅ Tokenizer carregado com sucesso.
    ⏱️ Tempo de carregamento (Tokenizer): 0.18s
    💾 Tokenizer confirmado no cache (tempo baixo).
  ↪️ Carregando Modelo Principal (bert-base-uncased)...
    ✅ Modelo principal carregado com sucesso.
    ⏱️ Tempo de carregamento (Modelo): 0.95s
    💾 Modelo principal confirmado no cache (tempo baixo).
  🧪 Realizando teste básico de carregamento para bert-base-uncased...
    [...]
    🧪 Teste básico para bert-base-uncased concluído.
✅ BERT Base 'bert-base-uncased' pronto para uso!
⏱️ Tarefa 'Modelo BERT Base (Uncased)' concluída em 1.35s.
----------------------------------------
➡️ Tarefa: NLTK Data 📚
  ℹ️ Pacotes solicitados: punkt, stopwords, wordnet, averaged_perceptron_tagger
  🔍 Verificando pacote NLTK: 'punkt'...
  💾 Pacote 'punkt' encontrado.
  🔍 Verificando pacote NLTK: 'stopwords'...
  💾 Pacote 'stopwords' encontrado.
  🔍 Verificando pacote NLTK: 'wordnet'...
  💾 Pacote 'wordnet' encontrado.
  🔍 Verificando pacote NLTK: 'averaged_perceptron_tagger'...
  💾 Pacote 'averaged_perceptron_tagger' encontrado.
  ✅ Todos os pacotes NLTK solicitados já estão presentes!
✅ Download e verificação dos pacotes NLTK concluído!
⏱️ Tarefa 'Dados Essenciais NLTK' concluída em 0.15s.
----------------------------------------
... (outras tarefas habilitadas rodam) ...

================================================================================
✨ RESUMO FINAL DO PIPELINE ✨
================================================================================
📊 Total de tarefas possíveis definidas no script: 10
🚫 Tarefas puladas conforme configuração: 2
⚙️ Tarefas que tentaram ser executadas: 8
✅ Tarefas concluídas com sucesso: 8
❌ Tarefas que falharam: 0
  Nenhuma tarefa executada falhou.
⏱️ Duração total do pipeline: 185.20 segundos (3.1 minutos)
================================================================================
✅ Pipeline concluído com sucesso! Os itens selecionados devem estar prontos.
```

## ⚙️ Customização

Quer baixar outros modelos ou remover alguns da lista? É fácil!

1.  Abra o arquivo Python do script (ex: `download_models_advanced.py`).
2.  Localize a lista `tasks` dentro da função `main()`.
3.  Cada item na lista é uma tupla definindo uma tarefa: `(função_download, [lista_de_argumentos], "Descrição")`.
4.  **Para adicionar um modelo:** Copie uma linha existente (ex: de BERT ou GPT-2), altere o `model_id` (ex: `"distilbert-base-uncased"`) e a descrição. Certifique-se de usar a `model_loader` correta (ex: `AutoModel` para DistilBERT).
5.  **Para remover um modelo:** Simplesmente comente (adicione `#` no início) ou delete a linha correspondente da lista `tasks`.
6.  **Para alterar dados NLTK:** Modifique a lista de pacotes no argumento da tarefa `download_nltk_data`.

Salve o arquivo e execute o script novamente.

##🤝 Contribuições

Contribuições são bem-vindas! Se você tiver sugestões para melhorar o script, adicionar mais funcionalidades ou encontrar bugs, sinta-se à vontade para:

1.  Abrir uma **Issue** para discutir a mudança.
2.  Fazer um **Fork** do repositório.
3.  Criar uma nova **Branch** para sua feature (`git checkout -b feature/AmazingFeature`).
4.  Fazer **Commit** das suas mudanças (`git commit -m 'Add some AmazingFeature'`).
5.  Fazer **Push** para a Branch (`git push origin feature/AmazingFeature`).
6.  Abrir um **Pull Request**.

##📜 Licença

Distribuído sob a licença MIT. Veja o arquivo `LICENSE` para mais informações.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](https://opensource.org/licenses/MIT)

##👨‍💻 Autor

*   **Elias Andrade (chaos4455)**
*   GitHub: [@chaos4455](https://github.com/chaos4455)

---

Espero que este script facilite seus projetos de NLP! 😊

