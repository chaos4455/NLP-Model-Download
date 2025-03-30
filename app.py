# -*- coding: utf-8 -*-
import os
import sys
import time
import nltk
import gensim.downloader
import colorama
from colorama import Fore, Style, Back
import logging
from tqdm import tqdm # Usaremos tqdm diretamente para NLTK se necessário

# --- Dependências Opcionais (Importações Seguras) ---
try:
    from transformers import (
        AutoModel, AutoTokenizer,
        WhisperProcessor, WhisperForConditionalGeneration,
        AutoModelForCausalLM, # Para GPT-2
        pipeline # Para um teste rápido pós-download
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE
    from huggingface_hub.file_download import repo_folder_name, hf_hub_download
    from huggingface_hub import list_models, model_info
    HUGGINGFACE_HUB_AVAILABLE = True
except ImportError:
    HUGGINGFACE_HUB_AVAILABLE = False

# --- Configuração Inicial ---
# Desabilita logs INFO excessivos que poluem o console
logging.getLogger("filelock").setLevel(logging.WARNING)
logging.getLogger("transformers.utils.hub").setLevel(logging.WARNING) # Menos verboso
logging.getLogger('nltk.downloader').setLevel(logging.WARNING)
logging.getLogger("urllib3.connectionpool").setLevel(logging.WARNING) # Download HTTP logs

# Inicializa Colorama
colorama.init(autoreset=True)

# --- Constantes de Emojis e Cores ---
ICON_START = "🚀"
ICON_STEP = "➡️"
ICON_SUBSTEP = "↪️"
ICON_DOWNLOAD = "📥"
ICON_UPLOAD = "📤" # Placeholder, não usado para download
ICON_INFO = "ℹ️"
ICON_SUCCESS = "✅"
ICON_ERROR = "❌"
ICON_WARN = "⚠️"
ICON_CACHE = "💾"
ICON_MODEL = "📦"
ICON_BRAIN = "🧠" # BERT, GPT
ICON_MIC = "🎤" # Whisper
ICON_BOOK = "📚" # NLTK
ICON_WORD = "📄" # Word2Vec
ICON_FINISH = "✨"
ICON_CHECK = "🔍"
ICON_NETWORK = "🌐"
ICON_CLOCK = "⏱️"
ICON_CONFIG = "⚙️"
ICON_TEST = "🧪"
ICON_SKIP = "🚫" # Ícone para tarefas puladas

COLOR_TITLE = Fore.MAGENTA + Style.BRIGHT
COLOR_STEP = Fore.CYAN + Style.BRIGHT
COLOR_SUBSTEP = Fore.CYAN
COLOR_INFO = Fore.BLUE
COLOR_SUCCESS = Fore.GREEN + Style.BRIGHT
COLOR_ERROR = Fore.RED + Style.BRIGHT
COLOR_WARN = Fore.YELLOW + Style.BRIGHT
COLOR_PROGRESS = Fore.YELLOW
COLOR_DETAIL = Fore.WHITE
COLOR_CACHE = Fore.LIGHTBLUE_EX
COLOR_NETWORK = Fore.LIGHTMAGENTA_EX
COLOR_TEST = Fore.LIGHTGREEN_EX
COLOR_SKIP = Fore.LIGHTBLACK_EX # Cor para tarefas puladas

# --- CONFIGURAÇÃO DE DOWNLOAD ---
# Edite os valores abaixo para True ou False para controlar o que será baixado.
# True = Baixar este item.
# False = Pular este item.
# ----------------------------------
DOWNLOAD_CONFIG = {
    # Modelos Transformers (Hugging Face)
    "Modelo BERT Base (Uncased)": True,
    "Modelo BERT Large (Uncased)": True,
    "Modelo GPT-2 (Base)": True,
    "Modelo GPT-2 Medium": True,
    "Modelo Whisper Small": True,
    "Modelo Whisper Medium": False,      # <<< Definido como False
    "Modelo Whisper Large v3": False,    # <<< Definido como False

    # Dados NLTK
    "Dados Essenciais NLTK": True,

    # Modelos Gensim (Word Embeddings)
    "Modelo Word2Vec Google News": True,
    "Modelo GloVe Wikipedia (100d)": True,

    # Adicione futuras tarefas aqui com True/False
    # Exemplo: "Meu Novo Modelo X": False,
}
# ----------------------------------

# --- Funções Auxiliares de Log ---
def print_color(text, color=COLOR_DETAIL, style="", end="\n", **kwargs):
    """Imprime texto colorido, resetando automaticamente."""
    print(f"{style}{color}{text}{Style.RESET_ALL}", end=end, **kwargs)
    sys.stdout.flush()

def log_header(message):
    print_color("=" * 80, COLOR_TITLE)
    print_color(f" {ICON_START} {message} {ICON_START} ".center(80, "="), COLOR_TITLE + Style.BRIGHT)
    print_color("=" * 80, COLOR_TITLE)
    print()

def log_step(message):
    print_color(f"\n{ICON_STEP} {message}", COLOR_STEP)
    print_color("-" * (len(message) + 4), COLOR_STEP)

def log_substep(message, indent=1):
    prefix = "  " * indent
    print_color(f"{prefix}{ICON_SUBSTEP} {message}", COLOR_SUBSTEP)

def log_info(message, indent=1, icon=ICON_INFO):
    prefix = "  " * indent
    print_color(f"{prefix}{icon} {message}", COLOR_INFO)

def log_detail(message, indent=2, icon=""):
    prefix = "  " * indent
    icon_str = f"{icon} " if icon else ""
    print_color(f"{prefix}{icon_str}{message}", COLOR_DETAIL)

def log_download_progress(message, indent=1):
    prefix = "  " * indent
    print_color(f"{prefix}{ICON_DOWNLOAD} {message}", COLOR_PROGRESS)

def log_cache_check(message, indent=1):
    prefix = "  " * indent
    print_color(f"{prefix}{ICON_CHECK} {message}", COLOR_CACHE)

def log_cache_hit(message, indent=1):
    prefix = "  " * indent
    print_color(f"{prefix}{ICON_CACHE} {message}", COLOR_CACHE + Style.BRIGHT)

def log_network_op(message, indent=1):
     prefix = "  " * indent
     print_color(f"{prefix}{ICON_NETWORK} {message}", COLOR_NETWORK)

def log_success(message, indent=1):
    prefix = "  " * indent
    print_color(f"{prefix}{ICON_SUCCESS} {message}", COLOR_SUCCESS)

def log_error(message, indent=1, error_details=""):
    prefix = "  " * indent
    print_color(f"{prefix}{ICON_ERROR} {message}", COLOR_ERROR)
    if error_details:
        print_color(f"{prefix}  Detalhes do erro: {error_details}", Fore.RED)
    sys.stderr.flush()

def log_warning(message, indent=1):
    prefix = "  " * indent
    print_color(f"{prefix}{ICON_WARN} {message}", COLOR_WARN)

def log_test(message, indent=1):
    prefix = "  " * indent
    print_color(f"{prefix}{ICON_TEST} {message}", COLOR_TEST)

def log_skip(message, indent=1):
    """Loga uma mensagem de tarefa pulada."""
    prefix = "  " * indent
    print_color(f"{prefix}{ICON_SKIP} {message}", COLOR_SKIP)

# --- Funções de Verificação e Download ---

def check_dependencies():
    """Verifica se as dependências principais estão instaladas."""
    log_step("Verificando Dependências Essenciais")
    all_ok = True
    # ... (restante da função check_dependencies permanece igual) ...
    if not TRANSFORMERS_AVAILABLE:
        log_error("Biblioteca 'transformers' não encontrada. Necessária para BERT, GPT-2, Whisper.", 1)
        log_info("Instale com: pip install transformers", 2)
        all_ok = False
    if not TORCH_AVAILABLE:
        # Transformers geralmente requer torch ou tensorflow
        log_error("Biblioteca 'torch' (PyTorch) não encontrada. Geralmente necessária para 'transformers'.", 1)
        log_info("Instale em https://pytorch.org/ ou com: pip install torch", 2)
        # Não definimos all_ok = False aqui, pois pode funcionar com TF, mas é menos comum
        log_warning("Continuando, mas modelos podem falhar se PyTorch for necessário.", 1)
    if not HUGGINGFACE_HUB_AVAILABLE:
        log_error("Biblioteca 'huggingface_hub' não encontrada. Necessária para checagem de cache avançada.", 1)
        log_info("Instale com: pip install huggingface_hub", 2)
        all_ok = False # Cache check vai falhar

    try:
        import nltk
    except ImportError:
        log_error("Biblioteca 'nltk' não encontrada.", 1)
        log_info("Instale com: pip install nltk", 2)
        all_ok = False

    try:
        import gensim
    except ImportError:
        log_error("Biblioteca 'gensim' não encontrada.", 1)
        log_info("Instale com: pip install gensim", 2)
        all_ok = False

    if not all_ok:
        log_error("Dependências faltando. Por favor, instale-as e tente novamente.", 0)
        sys.exit(1)
    else:
        log_success("Todas as dependências principais encontradas.", 1)
        time.sleep(1)


def check_transformer_cache_advanced(model_id):
    """Verifica heuristicamente se um modelo transformers parece estar no cache."""
    # ... (função check_transformer_cache_advanced permanece igual) ...
    if not HUGGINGFACE_HUB_AVAILABLE:
        log_warning("Biblioteca 'huggingface_hub' não disponível, pulando verificação de cache.", 2)
        return False
    try:
        resolved_cache_dir = HUGGINGFACE_HUB_CACHE
        model_cache_path = os.path.join(resolved_cache_dir, repo_folder_name(repo_id=model_id, repo_type="model"))
        if os.path.isdir(model_cache_path):
             has_config = any(f == 'config.json' for f in os.listdir(model_cache_path))
             has_model_file = any(f.endswith('.bin') or f.endswith('.safetensors') for f in os.listdir(model_cache_path))
             return has_config and has_model_file
        return False
    except Exception as e:
        log_warning(f"Não foi possível verificar o cache para {model_id}: {e}", 2)
        return False


def download_transformer_model(model_id, model_loader, tokenizer_loader=AutoTokenizer, processor_loader=None, model_type="Modelo Genérico"):
    """Função genérica para baixar modelos/tokenizers/processors do Hugging Face."""
    # Adicionando ícone específico do modelo
    icon = ICON_BRAIN if "BERT" in model_type or "GPT" in model_type else ICON_MIC if "Whisper" in model_type else ICON_MODEL
    log_step(f"Download {model_type}: {model_id} {icon}")

    if not TRANSFORMERS_AVAILABLE or not TORCH_AVAILABLE:
         log_error(f"Bibliotecas necessárias (transformers/torch) não disponíveis para {model_id}.", 1)
         return False

    model = None
    tokenizer_or_processor = None
    loaded_from_cache = False

    # 1. Checagem de Cache
    log_cache_check(f"Verificando cache local para {model_id}...", 1)
    is_cached = check_transformer_cache_advanced(model_id)
    if is_cached:
        log_cache_hit(f"{model_id} parece estar no cache.", 1)
        loaded_from_cache = True
    else:
        log_network_op(f"{model_id} não encontrado ou incompleto no cache. Preparando para download/verificação.", 1)
        if HUGGINGFACE_HUB_AVAILABLE:
            try:
                info = model_info(model_id)
                siblings = [f.rfilename for f in info.siblings if not f.rfilename.endswith('.gitattributes')]
                log_detail(f"Arquivos esperados no repositório: {len(siblings)}", 2, icon=ICON_CONFIG)
            except Exception as e:
                log_warning(f"Não foi possível obter informações detalhadas do Hub para {model_id}: {e}", 2)

    # 2. Download/Load Tokenizer ou Processor
    loader = processor_loader if processor_loader else tokenizer_loader
    loader_name = "Processor" if processor_loader else "Tokenizer"
    if loader:
        log_substep(f"Carregando {loader_name}...", 1)
        try:
            start_time = time.time()
            tokenizer_or_processor = loader.from_pretrained(model_id)
            duration = time.time() - start_time
            log_detail(f"{loader_name} carregado com sucesso.", 2, icon=ICON_SUCCESS)
            log_detail(f"Tempo de carregamento ({loader_name}): {duration:.2f}s", 2, icon=ICON_CLOCK)
            if not loaded_from_cache and duration < 0.5:
                 log_cache_hit(f"{loader_name} provavelmente carregado do cache (tempo baixo).", 2)
        except Exception as e:
            log_error(f"Falha ao carregar {loader_name} para {model_id}", 1, error_details=e)
            return False
        time.sleep(0.1)

    # 3. Download/Load Modelo Principal
    log_substep("Carregando Modelo Principal...", 1)
    try:
        start_time = time.time()
        # Adicionando uma mensagem antes do download real ser iniciado pela biblioteca
        if not loaded_from_cache:
             log_download_progress("Iniciando download/verificação do modelo principal (pode levar tempo)...", 2)
        model = model_loader.from_pretrained(model_id)
        duration = time.time() - start_time
        log_detail("Modelo principal carregado com sucesso.", 2, icon=ICON_SUCCESS)
        log_detail(f"Tempo de carregamento (Modelo): {duration:.2f}s", 2, icon=ICON_CLOCK)
        if not loaded_from_cache and duration < 1.0:
             log_cache_hit("Modelo principal provavelmente carregado do cache (tempo baixo).", 2)

        # 4. Teste Básico
        log_test(f"Realizando teste básico de carregamento para {model_id}...", 1)
        try:
            _ = model.config
            if hasattr(model, 'device'):
                log_detail(f"Modelo carregado no dispositivo: {model.device}", 2)
            else:
                 log_detail("Teste de configuração OK.", 2)

            # Testes específicos (simplificados para evitar dependências extras nos testes)
            if TRANSFORMERS_AVAILABLE and pipeline is not None:
                 if "BERT" in model_type and loader_name == "Tokenizer":
                     pipe = pipeline('feature-extraction', model=model, tokenizer=tokenizer_or_processor, device=-1)
                     _ = pipe("Teste.")
                     log_test("Teste de pipeline 'feature-extraction' OK.", 2)
                 elif "GPT-2" in model_type and loader_name == "Tokenizer":
                     pipe = pipeline('text-generation', model=model, tokenizer=tokenizer_or_processor, device=-1)
                     _ = pipe("Olá,", max_length=10)
                     log_test("Teste de pipeline 'text-generation' OK.", 2)
                 # Teste de Whisper exigiria um arquivo de áudio dummy ou mock
                 elif "Whisper" in model_type and loader_name == "Processor":
                      log_test("Teste de pipeline Whisper não implementado (requer áudio).", 2)


            log_test(f"Teste básico para {model_id} passou.", 1)

        except Exception as e:
            log_warning(f"Teste básico para {model_id} falhou ou não pôde ser executado: {e}", 1)

        log_success(f"{model_type} '{model_id}' pronto para uso!", 1)
        return True

    except Exception as e:
        log_error(f"Falha ao carregar o Modelo Principal para {model_id}", 1, error_details=e)
        return False


def download_nltk_data(packages):
    """Baixa pacotes de dados do NLTK com verificação e progresso manual se necessário."""
    log_step(f"Download de Dados NLTK {ICON_BOOK}")
    log_info(f"Pacotes solicitados: {', '.join(packages)}", 1)

    try:
        import nltk
    except ImportError:
        log_error("Biblioteca 'nltk' não encontrada.", 1)
        return False

    missing_packages = []
    for pkg_id in packages:
        log_cache_check(f"Verificando pacote NLTK: '{pkg_id}'...", 2)
        found = False
        # NLTK busca em múltiplos diretórios dentro de nltk_data
        # Simplificamos a checagem tentando encontrar diretamente
        try:
            nltk.data.find(pkg_id) # Tenta encontrar em qualquer lugar padrão
            log_cache_hit(f"Pacote '{pkg_id}' encontrado.", 2)
            found = True
        except LookupError:
             # Verifica explicitamente em subdirs comuns se a busca geral falhar
             for subdir in ['tokenizers', 'corpora', 'models', 'taggers', 'chunkers', 'grammars', 'misc']:
                 try:
                     nltk.data.find(f'{subdir}/{pkg_id}')
                     log_cache_hit(f"Pacote '{pkg_id}' encontrado (em {subdir}/).", 2)
                     found = True
                     break # Sai do loop de subdirs se encontrar
                 except LookupError:
                     continue # Tenta o próximo subdir
        if not found:
             log_warning(f"Pacote NLTK '{pkg_id}' não encontrado localmente.", 2)
             missing_packages.append(pkg_id)


    if not missing_packages:
        log_success("Todos os pacotes NLTK solicitados já estão presentes!", 1)
        return True

    log_network_op(f"Iniciando download para pacotes NLTK faltantes: {', '.join(missing_packages)}", 1)

    download_successful = False
    try:
        print_color("-" * 40, COLOR_NETWORK) # Separador visual
        # Vamos usar quiet=False para ver o output do NLTK
        nltk.download(missing_packages, quiet=False)
        download_successful = True
        print_color("-" * 40, COLOR_NETWORK) # Separador visual
    except Exception as e:
        log_error("Ocorreu um erro inesperado durante o download do NLTK.", 1, error_details=str(e))
        return False

    # Verifica novamente após a tentativa de download
    final_missing = []
    all_verified = True
    for pkg_id in missing_packages:
         found_after = False
         try:
             nltk.data.find(pkg_id)
             found_after = True
         except LookupError:
              for subdir in ['tokenizers', 'corpora', 'models', 'taggers', 'chunkers', 'grammars', 'misc']:
                 try:
                     nltk.data.find(f'{subdir}/{pkg_id}')
                     found_after = True
                     break
                 except LookupError:
                     continue
         if not found_after:
             final_missing.append(pkg_id)
             all_verified = False


    if all_verified:
         log_success("Download e verificação dos pacotes NLTK concluído!", 1)
         return True
    else:
         # O downloader do NLTK às vezes reporta sucesso mesmo falhando em baixar algo
         log_error(f"Falha ao baixar ou verificar alguns pacotes NLTK: {', '.join(final_missing)}. Verifique o log do NLTK acima.", 1)
         return False


def download_word2vec(model_name="word2vec-google-news-300"):
    """Baixa um modelo Word2Vec pré-treinado usando gensim.downloader."""
    log_step(f"Download do Modelo Word2Vec/GloVe: {model_name} {ICON_WORD}")

    try:
        import gensim.downloader
    except ImportError:
        log_error("Biblioteca 'gensim' não encontrada.", 1)
        return False

    log_info(f"Solicitando '{model_name}' ao Gensim Downloader.", 1)
    log_network_op("Gensim verificará o cache e iniciará o download se necessário.", 1)
    log_detail("Gensim mostrará barras de progresso abaixo para downloads ativos.", 2)

    try:
        start_time = time.time()
        print_color("-" * 40, COLOR_NETWORK) # Separador visual
        wv = gensim.downloader.load(model_name)
        print_color("-" * 40, COLOR_NETWORK) # Separador visual
        duration = time.time() - start_time

        log_success(f"Modelo '{model_name}' baixado/carregado com sucesso!", 1)
        log_detail(f"Tempo total (download/load): {duration:.2f}s", 2, icon=ICON_CLOCK)
        try:
            vocab_size = len(wv.key_to_index)
            vector_dim = wv.vector_size
            log_detail(f"Tamanho do vocabulário: {vocab_size}", 2, icon=ICON_CONFIG)
            log_detail(f"Dimensão dos vetores: {vector_dim}", 2, icon=ICON_CONFIG)

            log_test(f"Realizando teste básico com '{model_name}'...", 1)
            # Tenta encontrar uma palavra comum para teste
            test_word = 'computer' if 'computer' in wv else \
                        'house' if 'house' in wv else \
                        'king' if 'king' in wv else \
                        list(wv.key_to_index.keys())[0] # Pega a primeira palavra se outras falharem

            if test_word in wv:
                _ = wv[test_word]
                similars = wv.most_similar(test_word, topn=3)
                log_test(f"Vetor para '{test_word}' acessado.", 2)
                log_test(f"Palavras similares a '{test_word}': {[(w, f'{s:.2f}') for w, s in similars]}", 2)
                log_test("Teste básico Word2Vec/GloVe passou.", 1)
            else:
                log_warning(f"Não foi possível encontrar palavras comuns ('computer', 'house', 'king') para teste em '{model_name}'.", 1)

        except Exception as e:
             log_warning(f"Não foi possível obter detalhes ou testar o modelo '{model_name}': {e}", 1)

        return True

    except ValueError as e:
         if "not found" in str(e).lower():
              log_error(f"Modelo '{model_name}' não encontrado no catálogo do Gensim.", 1)
              try:
                  available_models = list(gensim.downloader.info()['models'].keys())
                  log_info(f"Modelos disponíveis (exemplo): {available_models[:10]}...", 2)
              except Exception:
                  log_info("Não foi possível listar modelos disponíveis do Gensim.", 2)
         else:
              log_error(f"Erro ao baixar/carregar '{model_name}'", 1, error_details=e)
         return False
    except Exception as e:
        log_error(f"Falha inesperada ao baixar/carregar '{model_name}'", 1, error_details=e)
        return False


# --- Pipeline Principal ---
def main():
    """Executa o pipeline de download avançado."""
    log_header("PIPELINE AVANÇADO DE DOWNLOAD DE MODELOS E DADOS")
    start_pipeline_time = time.time()

    check_dependencies()

    # --- Definição de Todas as Tarefas Possíveis ---
    # Formato: (função_download, [lista_de_argumentos], "Descrição da Tarefa")
    # A "Descrição da Tarefa" DEVE corresponder exatamente a uma chave em DOWNLOAD_CONFIG
    all_possible_tasks = [
        (download_transformer_model, ["bert-base-uncased", AutoModel, AutoTokenizer, None, "BERT Base"], "Modelo BERT Base (Uncased)"),
        (download_transformer_model, ["bert-large-uncased", AutoModel, AutoTokenizer, None, "BERT Large"], "Modelo BERT Large (Uncased)"),
        (download_transformer_model, ["gpt2", AutoModelForCausalLM, AutoTokenizer, None, "GPT-2"], "Modelo GPT-2 (Base)"),
        (download_transformer_model, ["gpt2-medium", AutoModelForCausalLM, AutoTokenizer, None, "GPT-2 Medium"], "Modelo GPT-2 Medium"),
        (download_transformer_model, ["openai/whisper-small", WhisperForConditionalGeneration, None, WhisperProcessor, "Whisper Small"], "Modelo Whisper Small"),
        (download_transformer_model, ["openai/whisper-medium", WhisperForConditionalGeneration, None, WhisperProcessor, "Whisper Medium"], "Modelo Whisper Medium"),
        (download_transformer_model, ["openai/whisper-large-v3", WhisperForConditionalGeneration, None, WhisperProcessor, "Whisper Large v3"], "Modelo Whisper Large v3"),
        (download_nltk_data, [["punkt", "stopwords", "wordnet", "averaged_perceptron_tagger"]], "Dados Essenciais NLTK"),
        (download_word2vec, ["word2vec-google-news-300"], "Modelo Word2Vec Google News"),
        (download_word2vec, ["glove-wiki-gigaword-100"], "Modelo GloVe Wikipedia (100d)"),
        # Adicione a definição de novas tarefas aqui
        # Exemplo: (minha_funcao_download, [arg1, arg2], "Meu Novo Modelo X"),
    ]

    # --- Filtrar Tarefas com base na Configuração ---
    tasks_to_run = []
    skipped_tasks_count = 0
    log_step(f"Configurando Tarefas {ICON_CONFIG}")
    log_info(f"Verificando {len(all_possible_tasks)} tarefas possíveis contra a configuração...", 1)
    time.sleep(0.5) # Pausa para leitura

    for task_definition in all_possible_tasks:
        func, args, desc = task_definition
        # Pega o valor da configuração, default é False se a chave não existir
        should_run = DOWNLOAD_CONFIG.get(desc, False)

        if should_run:
            tasks_to_run.append(task_definition)
            log_info(f"Habilitado: {desc}", 2, icon=ICON_SUCCESS)
        else:
            skipped_tasks_count += 1
            # Verifica se a chave existe mas está False, ou se a chave não existe
            if desc in DOWNLOAD_CONFIG:
                 log_skip(f"Pulando: {desc} (definido como False na configuração)", 2)
            else:
                 log_warning(f"Pulando: {desc} (não encontrado em DOWNLOAD_CONFIG, assumindo False)", 2)

    if not tasks_to_run:
         log_warning("\nNenhuma tarefa habilitada para execução na configuração.", 0)
         log_info("Edite o dicionário DOWNLOAD_CONFIG no topo do script para habilitar tarefas.", 0)
         # Sai mais cedo se não houver nada a fazer
         end_pipeline_time = time.time()
         pipeline_duration = end_pipeline_time - start_pipeline_time
         log_info(f"\nDuração total do pipeline: {pipeline_duration:.2f} segundos", 0, icon=ICON_CLOCK)
         print_color("=" * 80, COLOR_TITLE)
         colorama.deinit()
         return # Termina a execução

    log_info(f"\n{len(tasks_to_run)} tarefas serão executadas.", 1)
    if skipped_tasks_count > 0:
        log_info(f"{skipped_tasks_count} tarefas foram puladas conforme configuração.", 1, icon=ICON_SKIP)
    time.sleep(1.5) # Pausa antes de começar

    # --- Execução das Tarefas Selecionadas ---
    results = {"success": 0, "failed": 0, "skipped": skipped_tasks_count, "total_attempted": len(tasks_to_run), "failed_tasks": []}

    for i, (func, args, desc) in enumerate(tasks_to_run):
        # Usar i+1 sobre o total a ser rodado, não o total possível
        log_info(f"Executando Tarefa {i+1}/{results['total_attempted']}: {desc}", indent=0)
        task_start_time = time.time()
        success = False
        try:
            # Chamada da função de download real
            success = func(*args)
        except Exception as e:
             log_error(f"Erro catastrófico ao executar a tarefa '{desc}'", 0, error_details=e)
             import traceback
             log_detail(traceback.format_exc(), 1)
             success = False

        task_duration = time.time() - task_start_time

        if success:
            results["success"] += 1
            # Log sucesso já é feito dentro das funções de download, adicionamos só a duração aqui
            log_detail(f"Tarefa {i+1} ('{desc}') concluída em {task_duration:.2f}s.", 0, icon=ICON_CLOCK)
        else:
            results["failed"] += 1
            results["failed_tasks"].append(desc)
            # Log de erro já deve ter ocorrido dentro da função, adicionamos duração
            log_detail(f"Tarefa {i+1} ('{desc}') falhou após {task_duration:.2f}s.", 0, icon=ICON_CLOCK)

        if i < results["total_attempted"] - 1:
             log_detail("-" * 30, indent=0) # Separador visual
             time.sleep(1.5) # Pausa entre tarefas

    # --- Resumo Final ---
    end_pipeline_time = time.time()
    pipeline_duration = end_pipeline_time - start_pipeline_time

    print_color("\n" + "=" * 80, COLOR_TITLE)
    print_color(f" {ICON_FINISH} RESUMO FINAL DO PIPELINE {ICON_FINISH} ".center(80, "="), COLOR_TITLE + Style.BRIGHT)
    print_color("=" * 80, COLOR_TITLE)

    log_info(f"Total de tarefas possíveis definidas: {len(all_possible_tasks)}", 0, icon="📊")
    log_info(f"Tarefas puladas conforme configuração: {results['skipped']}", 0, icon=ICON_SKIP)
    log_info(f"Tarefas que tentaram ser executadas: {results['total_attempted']}", 0, icon="⚙️") # Usando ícone de config
    log_success(f"Tarefas concluídas com sucesso: {results['success']}", 0)

    if results['failed'] > 0:
        log_error(f"Tarefas que falharam: {results['failed']}", 0)
        for failed_task in results['failed_tasks']:
            log_detail(f"- {failed_task}", 1, icon=ICON_ERROR)
    else:
        if results['total_attempted'] > 0: # Só diz que não falhou se tentamos alguma
             log_detail("Nenhuma tarefa executada falhou.", 0)

    log_info(f"Duração total do pipeline: {pipeline_duration:.2f} segundos ({pipeline_duration/60:.1f} minutos)", 0, icon=ICON_CLOCK)
    print_color("=" * 80, COLOR_TITLE)

    if results["failed"] == 0 and results['total_attempted'] > 0:
        print_color(f"{ICON_SUCCESS} Pipeline concluído com sucesso! Itens selecionados devem estar prontos.", COLOR_SUCCESS)
    elif results["failed"] > 0:
        print_color(f"{ICON_WARN} Pipeline concluído com {results['failed']} falha(s). Verifique os logs de erro acima.", COLOR_WARN)
    elif results['total_attempted'] == 0:
         print_color(f"{ICON_INFO} Nenhuma tarefa foi executada. Verifique a configuração.", COLOR_INFO)
    else: # Caso de sucesso mas 0 tentados (não deveria acontecer com a lógica atual, mas por segurança)
         print_color(f"{ICON_INFO} Pipeline concluído, mas parece que nenhuma tarefa foi efetivamente executada.", COLOR_INFO)


    colorama.deinit()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print_color(f"\n\n{ICON_WARN} Operação interrompida pelo usuário (Ctrl+C).", COLOR_WARN)
        colorama.deinit()
        sys.exit(1)
    except SystemExit as e:
        # Permite saídas normais (ex: falha de dependência)
        if e.code != 0: # Só loga se for uma saída de erro
             print_color(f"\n{ICON_ERROR} Script terminado com código de saída: {e.code}", COLOR_ERROR)
        colorama.deinit()
        sys.exit(e.code)
    except Exception as e:
        print_color(f"\n\n{ICON_ERROR} Erro Inesperado no Pipeline Principal!", COLOR_ERROR + Style.BRIGHT)
        import traceback
        print_color(traceback.format_exc(), Fore.RED)
        colorama.deinit()
        sys.exit(1)
