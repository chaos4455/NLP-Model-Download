# chaos4455

# nlp-model-downloader 🚀✨

## Seu Assistente Inteligente para Preparar o Ambiente NLP! 🤖🔧

[![Python Version](https://img.shields.io/badge/Python-3.8+-blue.svg?style=for-the-badge&logo=python)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)
[![Status](https://img.shields.io/badge/Status-Ativo%20✅-success?style=for-the-badge)](https://github.com/chaos4455/nlp-toolkit-downloader/)
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


Cansado de baixar manualmente cada modelo e pacote de dados para seus projetos de Processamento de Linguagem Natural (NLP)? 😩 Perder tempo configurando tudo repetidamente?

O **NLP Toolkit Downloader** é um script Python robusto e altamente visual projetado para automatizar completamente o download e a configuração inicial dos modelos e conjuntos de dados NLP mais populares. Ele transforma um processo potencialmente tedioso em uma execução simples e informativa, direto no seu console!  console colorido, detalhado e cheio de emojis! 🎨📊

Criado por [Elias Andrade (chaos4455)](https://github.com/chaos4455), este script é ideal para pesquisadores, estudantes e desenvolvedores que precisam de um ambiente NLP pronto para usar rapidamente.

## ✨ Funcionalidades Principais

Este script não é apenas um downloader, é um assistente de configuração inteligente:

*   ✅ **Download Automatizado:** Baixa múltiplos modelos e dados de fontes confiáveis (Hugging Face 🤗, NLTK, Gensim) com um único comando.
*   🎨 **Console Rico e Colorido:** Utiliza `colorama` para fornecer feedback visual claro, com cores diferentes para status (sucesso, erro, aviso, cache), etapas e informações.
*   📊 **Feedback Detalhado em Tempo Real:** Exibe mensagens passo a passo sobre o que está acontecendo, desde a verificação de dependências até o download e testes básicos.
*   ⏳ **Barras de Progresso:** Integra-se com as barras de progresso das bibliotecas subjacentes (`transformers`, `gensim`) para downloads maiores, mostrando o andamento real.
*   💾 **Verificação Inteligente de Cache:** Detecta se os modelos/dados já existem localmente para evitar downloads desnecessários, economizando tempo e banda.
*   🧠 **Suporte a Diversos Modelos e Dados:** Configurado para baixar uma variedade de ferramentas essenciais (veja a lista abaixo).
*   🧪 **Testes Básicos Pós-Download:** Realiza verificações simples após o download para garantir que os modelos foram carregados corretamente.
*   ❌ **Tratamento de Erros:** Captura e reporta erros de forma clara (ex: dependências ausentes, problemas de rede, modelos não encontrados).
*   ⏱️ **Resumo Final:** Apresenta um sumário ao final da execução, mostrando o número de tarefas bem-sucedidas, falhas e o tempo total decorrido.

## 📦 Modelos e Dados Baixados

O script está configurado por padrão para baixar os seguintes recursos:

| Ícone | Tipo        | Modelos/Dados Específicos                                     | Fonte Principal      | Badge Exemplo                                                                                                     |
| :---: | :---------- | :------------------------------------------------------------ | :------------------- | :---------------------------------------------------------------------------------------------------------------- |
| 🧠    | **BERT**    | `bert-base-uncased`, `bert-large-uncased`                   | Hugging Face 🤗      | ![](https://img.shields.io/badge/BERT-blue?style=flat&logo=google)                                                |
| 🧠    | **GPT-2**   | `gpt2`, `gpt2-medium`                                         | Hugging Face 🤗      | ![](https://img.shields.io/badge/GPT--2-green?style=flat&logo=openai)                                             |
| 🎤    | **Whisper** | `whisper-small`, `whisper-medium`, `whisper-large-v3`       | Hugging Face 🤗      | ![](https://img.shields.io/badge/Whisper-purple?style=flat&logo=openai)                                           |
| 📚    | **NLTK Data** | `punkt`, `stopwords`, `wordnet`, `averaged_perceptron_tagger` | NLTK Project         | ![](https://img.shields.io/badge/NLTK-red?style=flat)                                                             |
| 📄    | **Embeddings**| `word2vec-google-news-300`, `glove-wiki-gigaword-100`         | Gensim Downloader    | ![](https://img.shields.io/badge/Embeddings-orange?style=flat)                                                    |

*Observação: O download desses modelos, especialmente os maiores (BERT Large, GPT-2 Medium, Whisper Large), pode consumir vários gigabytes de espaço em disco e levar um tempo considerável dependendo da sua conexão com a internet.* 🌐⏳

## 🛠️ Pré-requisitos

Antes de executar o script, você precisará ter:

## ⚙️ Instalação das Dependências

Precisa instalar as bibliotecas Python necessárias. Você pode fazer isso facilmente usando `pip`.

Abra seu terminal ou prompt de comando e execute o seguinte comando:

```bash
pip install colorama transformers torch nltk gensim tqdm requests huggingface_hub
```

1.  **Python:** Versão 3.8 ou superior recomendada.
    [![Python Version](https://img.shields.io/badge/Python-3.8+-blue.svg?style=flat&logo=python)](https://python.org)
2.  **Pip:** O gerenciador de pacotes do Python (geralmente vem junto com o Python).
3.  **Bibliotecas Python:** As seguintes bibliotecas são necessárias e serão usadas pelo script:
    *   `colorama`: Para a saída colorida no console.
    *   `transformers`: Para baixar e carregar modelos BERT, GPT-2 e Whisper do Hugging Face Hub.
    *   `torch`: Backend de Deep Learning para os modelos do `transformers` (PyTorch).
        *   *Nota:* A instalação do PyTorch pode variar dependendo do seu sistema operacional e se você tem uma GPU compatível com CUDA. Consulte o [site oficial do PyTorch](https://pytorch.org/) para as instruções corretas.
    *   `nltk`: Para baixar dados linguísticos (tokenizers, stopwords, etc.).
    *   `gensim`: Para baixar modelos de Word Embeddings pré-treinados (Word2Vec, GloVe).
    *   `tqdm`: Usado internamente por `transformers` e `gensim` para exibir barras de progresso.
    *   `requests`: Dependência comum para operações de rede.
    *   `huggingface_hub`: Para interagir com o Hugging Face Hub e realizar verificações de cache mais avançadas.

    Você pode instalar todas de uma vez (exceto talvez `torch`, veja a nota acima) com:
    ```bash
    pip install colorama transformers nltk gensim tqdm requests huggingface_hub torch
    ```
    [![Dependencies](https://img.shields.io/badge/Libs-Transformers%20%7C%20PyTorch%20%7C%20NLTK%20%7C%20Gensim%20%7C%20HF--Hub-critical?style=flat)](requirements.txt)
    *(Nota: Considere criar um arquivo `requirements.txt` para facilitar a instalação)*

## 🚀 Como Usar

1.  **Clone o Repositório:**
    ```bash
    git clone https://github.com/chaos4455/nlp-toolkit-downloader.git
    cd nlp-toolkit-downloader
    ```
    Ou baixe o código ZIP diretamente do GitHub.

2.  **Instale as Dependências:**
    Certifique-se de ter todas as bibliotecas listadas na seção de Pré-requisitos instaladas. Use o comando pip fornecido acima. Lembre-se da instalação potencialmente específica do PyTorch!

3.  **Execute o Script:**
    Abra seu terminal ou prompt de comando, navegue até a pasta do projeto e execute:
    ```bash
    python download_models_advanced.py
    ```
    *(Substitua `download_models_advanced.py` pelo nome real do arquivo, se diferente)*

4.  **Acompanhe o Processo:**
    Sente-se e observe o console! O script mostrará cada etapa, verificações de cache, downloads (com barras de progresso quando aplicável), testes e, finalmente, um resumo. 🎉

## 🖥️ Exemplo de Saída no Console (Conceitual)

Você verá algo parecido com isto (mas muito mais detalhado e colorido!):

## 🖥️ Exemplo de Saída no Console (Conceitual)

Você verá algo parecido com isto (mas muito mais detalhado e colorido!):

```text
================================================================================
🚀 INICIANDO PIPELINE AVANÇADO DE DOWNLOAD DE MODELOS E DADOS 🚀
================================================================================

➡️ Verificando Dependências Essenciais
  ✅ Todas as dependências principais encontradas.

ℹ️ Processando Tarefa 1/10: Modelo BERT Base (Uncased) 📦
➡️ Download BERT Base: bert-base-uncased
  🔍 Verificando cache local para bert-base-uncased...
  💾 bert-base-uncased parece estar no cache.
  ↪️ Carregando Tokenizer...
    ✅ Tokenizer carregado com sucesso.
    ⏱️ Tempo de carregamento (Tokenizer): 0.15s
    💾 Tokenizer provavelmente carregado do cache (tempo baixo).
  ↪️ Carregando Modelo Principal...
    ✅ Modelo principal carregado com sucesso.
    ⏱️ Tempo de carregamento (Modelo): 0.80s
    💾 Modelo principal provavelmente carregado do cache (tempo baixo).
  🧪 Realizando teste básico de carregamento para bert-base-uncased...
    🧪 Teste de pipeline 'feature-extraction' OK.
    🧪 Teste básico para bert-base-uncased passou.
  ✅ BERT Base 'bert-base-uncased' pronto para uso!
⏱️ Tarefa 1 concluída com sucesso em 1.20s.
------------------------------
ℹ️ Processando Tarefa 2/10: Modelo BERT Large (Uncased) 📦
➡️ Download BERT Large: bert-large-uncased
  🔍 Verificando cache local para bert-large-uncased...
  🌐 bert-large-uncased não encontrado ou incompleto no cache. Preparando para download/verificação.
  ↪️ Carregando Tokenizer...
    ✅ Tokenizer carregado com sucesso.
    ⏱️ Tempo de carregamento (Tokenizer): 5.50s
  ↪️ Carregando Modelo Principal...
    Downloading pytorch_model.bin: 100%|██████████| 1.34G/1.34G [01:30<00:00, 15.8MB/s]
    ✅ Modelo principal carregado com sucesso.
    ⏱️ Tempo de carregamento (Modelo): 95.30s
  🧪 Realizando teste básico de carregamento para bert-large-uncased...
    ... (testes) ...
    🧪 Teste básico para bert-large-uncased passou.
  ✅ BERT Large 'bert-large-uncased' pronto para uso!
⏱️ Tarefa 2 concluída com sucesso em 102.10s.
------------------------------
... (outras tarefas) ...

================================================================================
✨ RESUMO FINAL DO PIPELINE ✨
================================================================================
📊 Tarefas totais planejadas: 10
✅ Tarefas concluídas com sucesso: 10
❌ Tarefas que falharam: 0
  Nenhuma tarefa falhou.
⏱️ Duração total do pipeline: 350.50 segundos (5.8 minutos)
================================================================================
✅ Pipeline concluído com sucesso! Todos os modelos/dados solicitados devem estar prontos.

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

