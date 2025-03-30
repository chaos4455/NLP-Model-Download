# -*- coding: utf-8 -*-
import os
import sys
import time
import re # Importado para a correção do NLTK find
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
    from huggingface_hub.file_download import repo_folder_name # Removido hf_hub_download (não usado diretamente)
    from huggingface_hub import model_info # Removido list_models (não usado)
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
    sys.stdout.flush() # Garante visibilidade imediata

def log_header(message):
    """Loga um cabeçalho principal."""
    print_color("=" * 80, COLOR_TITLE)
    print_color(f" {ICON_START} {message} {ICON_START} ".center(80, "="), COLOR_TITLE + Style.BRIGHT)
    print_color("=" * 80, COLOR_TITLE)
    print()

def log_step(message):
    """Loga o início de um passo principal."""
    print_color(f"\n{ICON_STEP} {message}", COLOR_STEP)
    print_color("-" * (len(message) + 4), COLOR_STEP)

def log_substep(message, indent=1):
    """Loga um subpasso dentro de um passo principal."""
    prefix = "  " * indent
    print_color(f"{prefix}{ICON_SUBSTEP} {message}", COLOR_SUBSTEP)

def log_info(message, indent=1, icon=ICON_INFO):
    """Loga uma informação geral."""
    prefix = "  " * indent
    print_color(f"{prefix}{icon} {message}", COLOR_INFO)

def log_detail(message, indent=2, icon=""):
    """Loga um detalhe da operação."""
    prefix = "  " * indent
    icon_str = f"{icon} " if icon else ""
    print_color(f"{prefix}{icon_str}{message}", COLOR_DETAIL)

def log_download_progress(message, indent=1):
    """Loga uma mensagem relacionada ao download."""
    prefix = "  " * indent
    print_color(f"{prefix}{ICON_DOWNLOAD} {message}", COLOR_PROGRESS)

def log_cache_check(message, indent=1):
    """Loga mensagem de verificação de cache."""
    prefix = "  " * indent
    print_color(f"{prefix}{ICON_CHECK} {message}", COLOR_CACHE)

def log_cache_hit(message, indent=1):
    """Loga mensagem de cache hit."""
    prefix = "  " * indent
    print_color(f"{prefix}{ICON_CACHE} {message}", COLOR_CACHE + Style.BRIGHT)

def log_network_op(message, indent=1):
     """Loga mensagem de operação de rede."""
     prefix = "  " * indent
     print_color(f"{prefix}{ICON_NETWORK} {message}", COLOR_NETWORK)

def log_success(message, indent=1):
    """Loga uma mensagem de sucesso."""
    prefix = "  " * indent
    print_color(f"{prefix}{ICON_SUCCESS} {message}", COLOR_SUCCESS)

def log_error(message, indent=1, error_details=""):
    """Loga uma mensagem de erro."""
    prefix = "  " * indent
    print_color(f"{prefix}{ICON_ERROR} {message}", COLOR_ERROR)
    if error_details:
        # Tenta decodificar detalhes do erro se forem bytes (comum em exceções)
        if isinstance(error_details, bytes):
            try:
                error_details = error_details.decode('utf-8', errors='replace')
            except Exception:
                error_details = str(error_details) # Fallback
        elif not isinstance(error_details, str):
             error_details = str(error_details) # Garante que seja string

        # Limita o tamanho dos detalhes do erro para não poluir demais
        max_len = 500
        details_short = (error_details[:max_len] + '...') if len(error_details) > max_len else error_details
        print_color(f"{prefix}  Detalhes: {details_short.replace(os.linesep, ' ')}", Fore.RED) # Remove newlines para log conciso
    sys.stderr.flush()

def log_warning(message, indent=1):
    """Loga uma mensagem de aviso."""
    prefix = "  " * indent
    print_color(f"{prefix}{ICON_WARN} {message}", COLOR_WARN)

def log_test(message, indent=1):
    """Loga uma mensagem de teste/validação."""
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
    if not TRANSFORMERS_AVAILABLE:
        log_error("Biblioteca 'transformers' não encontrada.", 1)
        log_info("Instale com: pip install transformers[torch]", 2) # Sugere com torch
        all_ok = False
    if not TORCH_AVAILABLE:
        log_error("Biblioteca 'torch' (PyTorch) não encontrada.", 1)
        log_info("Instale em https://pytorch.org/ ou com: pip install torch", 2)
        log_warning("Transformers pode não funcionar sem PyTorch (ou TensorFlow).", 1)
        # Não definimos all_ok = False, pode funcionar com TF, mas é um aviso importante
    if not HUGGINGFACE_HUB_AVAILABLE:
        log_error("Biblioteca 'huggingface_hub' não encontrada.", 1)
        log_info("Instale com: pip install huggingface_hub", 2)
        all_ok = False # Verificação de cache avançada falhará

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
        log_error("Dependências essenciais faltando. Por favor, instale-as e tente novamente.", 0)
        sys.exit(1) # Termina se dependências críticas faltam
    else:
        log_success("Todas as dependências principais encontradas.", 1)
        time.sleep(1)


def check_transformer_cache_advanced(model_id):
    """Verifica heuristicamente se um modelo transformers parece estar no cache."""
    if not HUGGINGFACE_HUB_AVAILABLE:
        # log_warning("Biblioteca 'huggingface_hub' não disponível, pulando verificação de cache avançada.", 2)
        return False # Se não tem a lib, não tem como checar
    try:
        # Caminho esperado para o diretório do modelo no cache padrão do Hugging Face
        resolved_cache_dir = HUGGINGFACE_HUB_CACHE
        # Usa a função oficial para obter o nome da pasta, mais robusto
        model_cache_path = os.path.join(resolved_cache_dir, repo_folder_name(repo_id=model_id, repo_type="model"))

        # Verifica se o diretório existe
        if not os.path.isdir(model_cache_path):
             return False

        # Verifica se arquivos essenciais existem (heurística mais forte)
        # Ex: config.json E algum arquivo de peso (.bin, .safetensors)
        has_config = any(f.lower() == 'config.json' for f in os.listdir(model_cache_path))
        has_model_file = any(f.lower().endswith(('.bin', '.safetensors')) for f in os.listdir(model_cache_path))

        # Pode adicionar checagem de tokenizer.json se necessário
        # has_tokenizer = any(f.lower() == 'tokenizer.json' for f in os.listdir(model_cache_path))

        # Considera em cache se os arquivos principais estiverem presentes
        return has_config and has_model_file
    except Exception as e:
        log_warning(f"Não foi possível verificar o cache para {model_id}: {e}", 2)
        return False # Assume não estar em cache se a verificação falhar


def download_transformer_model(model_id, model_loader, tokenizer_loader=AutoTokenizer, processor_loader=None, model_type="Modelo Genérico"):
    """
    Função genérica para baixar/carregar modelos, tokenizers e processors
    do Hugging Face com verificação de cache e logging.
    """
    icon = ICON_BRAIN if "BERT" in model_type or "GPT" in model_type else ICON_MIC if "Whisper" in model_type else ICON_MODEL
    log_step(f"Tarefa: {model_type}: {model_id} {icon}")

    if not TRANSFORMERS_AVAILABLE:
         log_error(f"Biblioteca 'transformers' não disponível para {model_id}.", 1)
         return False
    # Aviso se PyTorch não estiver disponível, mas continua (pode usar TF)
    if not TORCH_AVAILABLE:
        log_warning(f"PyTorch não encontrado. Tentando carregar {model_id} (pode requerer TensorFlow).", 1)

    model = None
    tokenizer_or_processor = None
    loaded_from_cache = {"model": False, "tokenizer_processor": False} # Rastreia cache separadamente

    # 1. Checagem de Cache (Visão Geral)
    log_cache_check(f"Verificando cache local para {model_id}...", 1)
    is_cached_heuristic = check_transformer_cache_advanced(model_id)
    if is_cached_heuristic:
        log_cache_hit(f"{model_id} parece estar no cache (verificação heurística).", 1)
        # Marcamos como potencialmente em cache, será confirmado pelo tempo de carregamento
        loaded_from_cache["model"] = True
        loaded_from_cache["tokenizer_processor"] = True
    else:
        log_network_op(f"{model_id} não encontrado ou incompleto no cache. Preparando para download/verificação.", 1)
        if HUGGINGFACE_HUB_AVAILABLE:
            try:
                info = model_info(model_id)
                # siblings = [f.rfilename for f in info.siblings if not f.rfilename.endswith(('.gitattributes', '.gitignore'))] # Mais específico
                num_files = len([f for f in info.siblings if not f.rfilename.startswith('.')]) # Conta arquivos não ocultos
                log_detail(f"Arquivos esperados no repositório: ~{num_files}", 2, icon=ICON_CONFIG)
            except Exception as e:
                log_warning(f"Não foi possível obter informações do Hub para {model_id}: {e}", 2)

    # 2. Download/Load Tokenizer ou Processor
    loader = processor_loader if processor_loader else tokenizer_loader
    loader_name = "Processor" if processor_loader else "Tokenizer"
    if loader:
        log_substep(f"Carregando {loader_name} ({model_id})...", 1)
        try:
            start_time = time.time()
            tokenizer_or_processor = loader.from_pretrained(model_id) # cache_dir pode ser especificado se necessário
            duration = time.time() - start_time
            log_detail(f"{loader_name} carregado com sucesso.", 2, icon=ICON_SUCCESS)
            log_detail(f"Tempo de carregamento ({loader_name}): {duration:.2f}s", 2, icon=ICON_CLOCK)
            # Refina a checagem de cache com base no tempo de carregamento
            if duration < 0.5 and loaded_from_cache["tokenizer_processor"]: # Carregou rápido E heurística disse cache
                 log_cache_hit(f"{loader_name} confirmado no cache (tempo baixo).", 2)
            elif duration >= 0.5 and not loaded_from_cache["tokenizer_processor"]: # Carregou devagar E heurística disse não-cache (provável download)
                 log_download_progress(f"{loader_name} provavelmente baixado ou verificado.", 2)
            # Se a heurística e o tempo discordarem, o tempo é mais confiável
            elif duration < 0.5 and not loaded_from_cache["tokenizer_processor"]:
                 log_cache_hit(f"{loader_name} carregado rapidamente, provavelmente estava em cache apesar da heurística inicial.", 2)
                 loaded_from_cache["tokenizer_processor"] = True # Corrige estado do cache

        except Exception as e:
            log_error(f"Falha ao carregar {loader_name} para {model_id}", 1, error_details=e)
            return False # Falha crítica se o tokenizer/processor não carregar
        time.sleep(0.1) # Pausa para legibilidade

    # 3. Download/Load Modelo Principal
    log_substep(f"Carregando Modelo Principal ({model_id})...", 1)
    try:
        start_time = time.time()
        if not loaded_from_cache["model"]:
             log_download_progress("Iniciando download/verificação do modelo (pode levar tempo)...", 2)
        # A mágica acontece aqui: from_pretrained faz o download se necessário
        model = model_loader.from_pretrained(model_id)
        duration = time.time() - start_time
        log_detail("Modelo principal carregado com sucesso.", 2, icon=ICON_SUCCESS)
        log_detail(f"Tempo de carregamento (Modelo): {duration:.2f}s", 2, icon=ICON_CLOCK)

        # Refina checagem de cache para o modelo
        if duration < 1.0 and loaded_from_cache["model"]:
             log_cache_hit("Modelo principal confirmado no cache (tempo baixo).", 2)
        elif duration >= 1.0 and not loaded_from_cache["model"]:
             log_download_progress("Modelo principal provavelmente baixado ou verificado.", 2)
        elif duration < 1.0 and not loaded_from_cache["model"]:
             log_cache_hit("Modelo principal carregado rapidamente, provavelmente estava em cache.", 2)
             loaded_from_cache["model"] = True # Corrige estado

        # 4. Teste Básico (Opcional, mas útil)
        log_test(f"Realizando teste básico de carregamento para {model_id}...", 1)
        try:
            _ = model.config # Acessa a configuração (teste mínimo)
            if hasattr(model, 'device'):
                log_detail(f"Modelo carregado no dispositivo: {model.device}", 2)
            else:
                 log_detail("Teste de acesso à configuração OK.", 2)

            # Teste funcional com pipeline (se disponível e aplicável)
            if TRANSFORMERS_AVAILABLE and pipeline is not None and tokenizer_or_processor is not None:
                 task = None
                 if "BERT" in model_type: task = 'feature-extraction'
                 elif "GPT-2" in model_type: task = 'text-generation'
                 # Whisper precisa de 'automatic-speech-recognition' e input de áudio (pulamos teste funcional)

                 if task:
                     log_test(f"Tentando pipeline '{task}'...", 2)
                     pipe = pipeline(task, model=model, tokenizer=tokenizer_or_processor, device=-1) # Força CPU para teste
                     if task == 'feature-extraction': _ = pipe("Teste.")
                     elif task == 'text-generation': _ = pipe("Olá,", max_length=10)
                     log_test(f"Teste de pipeline '{task}' OK.", 2)
                 elif "Whisper" in model_type:
                      log_test("Teste de pipeline Whisper pulado (requer input de áudio).", 2)

            else:
                log_test("Teste de pipeline pulado (transformers.pipeline ou tokenizer/processor não disponível).", 2)

            log_test(f"Teste básico para {model_id} concluído.", 1)

        except Exception as e:
            log_warning(f"Teste básico para {model_id} falhou ou não pôde ser executado completamente: {e}", 1)
            # Não retorna False aqui, o download principal funcionou.

        log_success(f"{model_type} '{model_id}' pronto para uso!", 0) # Sucesso no nível do passo
        return True

    except Exception as e:
        log_error(f"Falha ao carregar o Modelo Principal para {model_id}", 1, error_details=e)
        return False


def download_nltk_data(packages):
    """
    Baixa pacotes de dados do NLTK com verificação de cache e logging.
    'packages' deve ser uma lista de strings (nomes dos pacotes).
    """
    log_step(f"Tarefa: Download de Dados NLTK {ICON_BOOK}")
    if not isinstance(packages, list) or not all(isinstance(p, str) for p in packages):
         log_error(f"Argumento inválido para download_nltk_data: esperado uma lista de strings, recebeu {type(packages)}.", 1)
         return False
    log_info(f"Pacotes solicitados: {', '.join(packages)}", 1)

    try:
        import nltk
        # Configura o downloader do NLTK para ser um pouco menos verboso por padrão
        # nltk.downloader.Downloader(quiet=True) # Pode esconder demais, vamos deixar o padrão por enquanto
    except ImportError:
        log_error("Biblioteca 'nltk' não encontrada.", 1)
        return False

    missing_packages = []
    for pkg_id in packages:
        if not isinstance(pkg_id, str): # Sanity check interno
             log_error(f"Item inválido na lista de pacotes NLTK: {pkg_id} (tipo {type(pkg_id)}). Pulando.", 2)
             continue

        log_cache_check(f"Verificando pacote NLTK: '{pkg_id}'...", 2)
        found = False
        try:
            # nltk.data.find agora é mais robusto e busca em vários locais por padrão
            nltk.data.find(pkg_id)
            log_cache_hit(f"Pacote '{pkg_id}' encontrado.", 2)
            found = True
        except LookupError:
            log_warning(f"Pacote NLTK '{pkg_id}' não encontrado localmente.", 2)
            missing_packages.append(pkg_id)
        except Exception as e:
            log_error(f"Erro ao verificar o pacote NLTK '{pkg_id}': {e}", 2)
            # Considera como faltando se houver erro na verificação
            if pkg_id not in missing_packages:
                missing_packages.append(pkg_id)


    if not missing_packages:
        log_success("Todos os pacotes NLTK solicitados já estão presentes!", 1)
        return True

    log_network_op(f"Iniciando download para pacotes NLTK faltantes: {', '.join(missing_packages)}", 1)

    download_successful = False
    try:
        print_color("-" * 40, COLOR_NETWORK) # Separador visual
        # quiet=False para ver o output detalhado do NLTK
        # O NLTK lida com a barra de progresso internamente se o TQDM estiver instalado
        nltk.download(missing_packages, quiet=False)
        # Assume sucesso se não lançar exceção, mas verificaremos novamente
        download_successful = True
        print_color("-" * 40, COLOR_NETWORK) # Separador visual
    except Exception as e:
        # Erros durante o download (rede, permissão, etc.)
        log_error("Ocorreu um erro durante o download do NLTK.", 1, error_details=str(e))
        # Tentamos verificar mesmo assim, talvez alguns tenham baixado
        # return False # Não retorna imediatamente, verifica o que foi baixado

    # Verifica novamente após a tentativa de download (crucial!)
    final_missing = []
    all_verified = True
    log_info("Verificando pacotes após tentativa de download...", 1)
    for pkg_id in missing_packages:
         try:
             nltk.data.find(pkg_id) # Verifica novamente
             log_detail(f"'{pkg_id}' verificado com sucesso após download.", 2, icon=ICON_SUCCESS)
         except LookupError:
             log_error(f"Pacote '{pkg_id}' ainda não encontrado após tentativa de download.", 2)
             final_missing.append(pkg_id)
             all_verified = False
         except Exception as e:
             log_error(f"Erro ao re-verificar o pacote NLTK '{pkg_id}': {e}", 2)
             final_missing.append(pkg_id)
             all_verified = False


    if all_verified and download_successful:
         log_success("Download e verificação dos pacotes NLTK concluído!", 0) # Sucesso no nível do passo
         return True
    else:
         if final_missing:
            log_error(f"Falha ao baixar ou verificar os seguintes pacotes NLTK: {', '.join(final_missing)}.", 0) # Erro no nível do passo
         else:
             log_error("Download NLTK concluído com erros (ver logs acima).", 0)
         log_info("Verifique sua conexão ou permissões de escrita no diretório de dados do NLTK.", 1)
         return False


def download_word2vec(model_name="word2vec-google-news-300"):
    """
    Baixa um modelo Word2Vec/GloVe pré-treinado usando gensim.downloader.
    """
    log_step(f"Tarefa: Download Word Embedding: {model_name} {ICON_WORD}")

    try:
        import gensim.downloader
    except ImportError:
        log_error("Biblioteca 'gensim' não encontrada.", 1)
        return False

    log_info(f"Solicitando '{model_name}' ao Gensim Downloader.", 1)
    log_network_op("Gensim verificará o cache e iniciará o download se necessário.", 1)
    log_detail("Gensim pode mostrar barras de progresso abaixo.", 2)

    wv = None # Inicializa wv
    try:
        start_time = time.time()
        print_color("-" * 40, COLOR_NETWORK) # Separador visual
        # A chamada load() faz o download e/ou carrega do cache
        wv = gensim.downloader.load(model_name)
        print_color("-" * 40, COLOR_NETWORK) # Separador visual
        duration = time.time() - start_time

        # Verifica se carregou algo
        if wv is None:
            log_error(f"Gensim retornou 'None' para '{model_name}', indicando falha no carregamento.", 1)
            return False

        log_success(f"Modelo '{model_name}' baixado/carregado com sucesso!", 1)
        log_detail(f"Tempo total (download/load): {duration:.2f}s", 2, icon=ICON_CLOCK)

        try:
            # Tenta obter detalhes e testar
            if hasattr(wv, 'key_to_index') and hasattr(wv, 'vector_size'):
                 vocab_size = len(wv.key_to_index)
                 vector_dim = wv.vector_size
                 log_detail(f"Tamanho do vocabulário: {vocab_size}", 2, icon=ICON_CONFIG)
                 log_detail(f"Dimensão dos vetores: {vector_dim}", 2, icon=ICON_CONFIG)

                 log_test(f"Realizando teste básico com '{model_name}'...", 1)
                 # Tenta encontrar palavras comuns para teste
                 test_words = ['computer', 'house', 'king', 'data']
                 word_found = None
                 for word in test_words:
                     if word in wv:
                         word_found = word
                         break
                 # Se nenhuma comum for encontrada, pega uma aleatória (menos ideal)
                 if not word_found and wv.key_to_index:
                     word_found = list(wv.key_to_index.keys())[0]

                 if word_found:
                     _ = wv[word_found] # Acessa o vetor
                     log_test(f"Vetor para '{word_found}' acessado.", 2)
                     if hasattr(wv, 'most_similar'):
                         similars = wv.most_similar(word_found, topn=3)
                         log_test(f"Palavras similares a '{word_found}': {[(w, f'{s:.2f}') for w, s in similars]}", 2)
                     log_test("Teste básico de acesso passou.", 1)
                 else:
                     log_warning(f"Não foi possível encontrar palavras comuns para teste em '{model_name}'.", 1)
            else:
                log_warning(f"O objeto carregado para '{model_name}' não parece ser um modelo KeyedVectors padrão.", 1)

        except Exception as e:
             log_warning(f"Não foi possível obter detalhes completos ou testar o modelo '{model_name}': {e}", 1)

        log_success(f"Word Embedding '{model_name}' pronto para uso!", 0) # Sucesso no nível do passo
        return True

    except ValueError as e:
         # Gensim levanta ValueError para modelo não encontrado no catálogo
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
        # Captura outros erros (rede, disco, formato inesperado, etc.)
        log_error(f"Falha inesperada ao baixar/carregar '{model_name}'", 1, error_details=e)
        return False


# --- Pipeline Principal ---
def main():
    """Executa o pipeline de download configurável."""
    log_header("PIPELINE DE DOWNLOAD DE MODELOS E DADOS DE NLP")
    start_pipeline_time = time.time()

    # 1. Verificar dependências primeiro
    check_dependencies()

    # 2. Definir todas as tarefas possíveis
    # Formato: (função_download, [lista_de_argumentos], "Descrição da Tarefa")
    # A "Descrição da Tarefa" DEVE corresponder EXATAMENTE a uma chave em DOWNLOAD_CONFIG
    all_possible_tasks = [
        # Transformers
        (download_transformer_model, ["bert-base-uncased", AutoModel, AutoTokenizer, None, "BERT Base"], "Modelo BERT Base (Uncased)"),
        (download_transformer_model, ["bert-large-uncased", AutoModel, AutoTokenizer, None, "BERT Large"], "Modelo BERT Large (Uncased)"),
        (download_transformer_model, ["gpt2", AutoModelForCausalLM, AutoTokenizer, None, "GPT-2"], "Modelo GPT-2 (Base)"),
        (download_transformer_model, ["gpt2-medium", AutoModelForCausalLM, AutoTokenizer, None, "GPT-2 Medium"], "Modelo GPT-2 Medium"),
        (download_transformer_model, ["openai/whisper-small", WhisperForConditionalGeneration, None, WhisperProcessor, "Whisper Small"], "Modelo Whisper Small"),
        (download_transformer_model, ["openai/whisper-medium", WhisperForConditionalGeneration, None, WhisperProcessor, "Whisper Medium"], "Modelo Whisper Medium"),
        (download_transformer_model, ["openai/whisper-large-v3", WhisperForConditionalGeneration, None, WhisperProcessor, "Whisper Large v3"], "Modelo Whisper Large v3"),

        # NLTK Data (Argumento CORRIGIDO para ser uma lista de strings)
        (download_nltk_data, ["punkt", "stopwords", "wordnet", "averaged_perceptron_tagger"], "Dados Essenciais NLTK"),

        # Gensim Word Embeddings
        (download_word2vec, ["word2vec-google-news-300"], "Modelo Word2Vec Google News"),
        (download_word2vec, ["glove-wiki-gigaword-100"], "Modelo GloVe Wikipedia (100d)"),

        # Adicione a definição de novas tarefas potenciais aqui
        # Exemplo: (minha_funcao_download, [arg1, arg2], "Meu Novo Modelo X"),
    ]

    # 3. Filtrar Tarefas com base na Configuração
    tasks_to_run = []
    skipped_tasks_count = 0
    log_step(f"Configurando Tarefas de Download {ICON_CONFIG}")
    log_info(f"Verificando {len(all_possible_tasks)} tarefas possíveis contra a configuração...", 1)
    time.sleep(0.5) # Pausa para leitura

    configured_task_names = set(DOWNLOAD_CONFIG.keys())
    defined_task_names = {desc for _, _, desc in all_possible_tasks}

    # Avisar sobre tarefas definidas no código mas não na configuração
    missing_in_config = defined_task_names - configured_task_names
    if missing_in_config:
        log_warning(f"As seguintes tarefas estão definidas no código mas FALTAM em DOWNLOAD_CONFIG:", 1)
        for task_name in missing_in_config:
             log_detail(f"- {task_name} (será pulada por padrão)", 2)

    # Avisar sobre tarefas na configuração mas não definidas no código (typos?)
    missing_in_code = configured_task_names - defined_task_names
    if missing_in_code:
        log_warning(f"As seguintes tarefas estão em DOWNLOAD_CONFIG mas NÃO estão definidas na lista 'all_possible_tasks':", 1)
        for task_name in missing_in_code:
             log_detail(f"- {task_name} (será ignorada)", 2)


    for task_definition in all_possible_tasks:
        func, args, desc = task_definition
        # Pega o valor da configuração; default é False se a chave não existir
        should_run = DOWNLOAD_CONFIG.get(desc, False)

        if should_run:
            tasks_to_run.append(task_definition)
            log_info(f"Habilitado: {desc}", 2, icon=ICON_SUCCESS)
        else:
            skipped_tasks_count += 1
            # Loga apenas se a tarefa estava definida no código
            if desc in defined_task_names:
                reason = "(definido como False)" if desc in DOWNLOAD_CONFIG else "(não encontrado ou False em DOWNLOAD_CONFIG)"
                log_skip(f"Pulando: {desc} {reason}", 2)

    if not tasks_to_run:
         log_warning("\nNenhuma tarefa habilitada para execução na configuração.", 0)
         log_info("Edite o dicionário DOWNLOAD_CONFIG no topo do script para habilitar tarefas.", 0)
         # Conclui o pipeline mesmo sem tarefas para executar
    else:
        log_info(f"\n{len(tasks_to_run)} tarefas serão executadas.", 1)
        if skipped_tasks_count > 0:
            log_info(f"{skipped_tasks_count} tarefas foram puladas conforme configuração.", 1, icon=ICON_SKIP)

    time.sleep(1.5) # Pausa antes de começar a execução

    # 4. Execução das Tarefas Selecionadas
    results = {
        "success": 0,
        "failed": 0,
        "skipped": skipped_tasks_count,
        "total_attempted": len(tasks_to_run),
        "failed_tasks": []
    }

    for i, (func, args, desc) in enumerate(tasks_to_run):
        # Mensagem movida para dentro das funções de download (log_step)
        # log_info(f"Executando Tarefa {i+1}/{results['total_attempted']}: {desc}", indent=0)
        task_start_time = time.time()
        success = False
        try:
            # Chama a função de download real (ex: download_transformer_model)
            success = func(*args)
        except Exception as e:
             # Captura erro inesperado na *chamada* da função (improvável, mas seguro)
             log_error(f"Erro catastrófico ao tentar iniciar a tarefa '{desc}'", 0, error_details=e)
             import traceback
             log_detail(f"Traceback: {traceback.format_exc()}", 1) # Imprime stack trace
             success = False # Garante que seja marcado como falha

        task_duration = time.time() - task_start_time

        if success:
            results["success"] += 1
            # Log de sucesso principal já é feito dentro da função de download
            # Adicionamos apenas a duração aqui no resumo da tarefa
            log_detail(f"Tarefa '{desc}' concluída em {task_duration:.2f}s.", 0, icon=ICON_CLOCK)
        else:
            results["failed"] += 1
            results["failed_tasks"].append(desc)
            # Log de erro principal já deve ter ocorrido dentro da função
            log_detail(f"Tarefa '{desc}' falhou (ver logs acima). Duração: {task_duration:.2f}s.", 0, icon=ICON_CLOCK)

        # Pausa entre tarefas para melhor legibilidade e evitar sobrecarga de rede/disco
        if i < results["total_attempted"] - 1:
             log_detail("-" * 40, indent=0) # Separador visual entre tarefas
             time.sleep(1.5)

    # 5. Resumo Final
    end_pipeline_time = time.time()
    pipeline_duration = end_pipeline_time - start_pipeline_time

    print_color("\n" + "=" * 80, COLOR_TITLE)
    print_color(f" {ICON_FINISH} RESUMO FINAL DO PIPELINE {ICON_FINISH} ".center(80, "="), COLOR_TITLE + Style.BRIGHT)
    print_color("=" * 80, COLOR_TITLE)

    log_info(f"Total de tarefas possíveis definidas no script: {len(all_possible_tasks)}", 0, icon="📊")
    log_info(f"Tarefas puladas conforme configuração: {results['skipped']}", 0, icon=ICON_SKIP)
    log_info(f"Tarefas que tentaram ser executadas: {results['total_attempted']}", 0, icon=ICON_CONFIG) # Ícone de engrenagem
    log_success(f"Tarefas concluídas com sucesso: {results['success']}", 0)

    if results['failed'] > 0:
        log_error(f"Tarefas que falharam: {results['failed']}", 0)
        for failed_task in results['failed_tasks']:
            log_detail(f"- {failed_task}", 1, icon=ICON_ERROR)
    else:
        # Apenas diz que não falhou se tentamos executar alguma tarefa
        if results['total_attempted'] > 0:
             log_detail("Nenhuma tarefa executada falhou.", 0)
        elif results['skipped'] == len(all_possible_tasks): # Se todas foram puladas
             log_info("Nenhuma tarefa foi executada (todas puladas ou desabilitadas).", 0)


    log_info(f"Duração total do pipeline: {pipeline_duration:.2f} segundos ({pipeline_duration/60:.1f} minutos)", 0, icon=ICON_CLOCK)
    print_color("=" * 80, COLOR_TITLE)

    # Mensagem final baseada nos resultados
    if results["failed"] == 0 and results['success'] > 0:
        print_color(f"{ICON_SUCCESS} Pipeline concluído com sucesso! Os itens selecionados devem estar prontos.", COLOR_SUCCESS)
    elif results["failed"] > 0:
        print_color(f"{ICON_WARN} Pipeline concluído com {results['failed']} falha(s). Verifique os logs de erro acima.", COLOR_WARN)
    elif results['total_attempted'] == 0:
         print_color(f"{ICON_INFO} Nenhuma tarefa foi executada. Verifique a configuração 'DOWNLOAD_CONFIG'.", COLOR_INFO)
    else: # Caso de 0 falhas e 0 sucessos (ex: todas tentaram mas falharam antes de começar?) - Improvável
         print_color(f"{ICON_INFO} Pipeline concluído, mas sem sucessos ou falhas registradas. Verifique os logs.", COLOR_INFO)


    colorama.deinit() # Desliga o colorama no final

# --- Ponto de Entrada ---
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print_color(f"\n\n{ICON_WARN} Operação interrompida pelo usuário (Ctrl+C).", COLOR_WARN)
        colorama.deinit() # Garante a desinicialização
        sys.exit(130) # Código de saída comum para interrupção
    except SystemExit as e:
        # Permite saídas normais (ex: falha de dependência em check_dependencies)
        if e.code != 0: # Só loga se for uma saída de erro explícita
             print_color(f"\n{ICON_ERROR} Script terminado com código de saída: {e.code}", COLOR_ERROR)
        colorama.deinit()
        sys.exit(e.code)
    except Exception as e:
        # Captura qualquer outra exceção não tratada no nível superior
        print_color(f"\n\n{ICON_ERROR} Erro Inesperado e Não Tratado no Pipeline Principal!", COLOR_ERROR + Style.BRIGHT)
        import traceback
        # Imprime o traceback completo para depuração
        print_color(traceback.format_exc(), Fore.RED)
        colorama.deinit()
        sys.exit(1) # Código de erro genérico
