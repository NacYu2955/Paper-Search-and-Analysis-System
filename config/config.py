import os

CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CONFIG_DIR)
RESOURCES_DIR = os.path.join(BASE_DIR, 'src')
LEGACY_RESOURCES_DIR = os.path.join(RESOURCES_DIR, 'resources')
DATASET_DIR = os.path.join(BASE_DIR, 'dataset')


def _pick_existing_database_path():
    candidates = [
        os.path.join(RESOURCES_DIR, 'papers.db'),
        os.path.join(LEGACY_RESOURCES_DIR, 'papers.db'),
        '/root/autodl-fs/pasa/src/resources/papers.db',
        '/autodl-fs/data/pasa/src/resources/papers.db',
    ]
    existing = [path for path in candidates if os.path.exists(path)]
    if existing:
        return max(existing, key=lambda path: os.path.getsize(path))
    return candidates[0]

# DeepSeek API configuration
DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY', 'your DeepSeek key')
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"

# Model paths remain relative to the project root unless overridden.
MODEL_PATH = os.getenv('MODEL_PATH', os.path.join(BASE_DIR, 'model', 'all-MiniLM-L6-v2'))
SELECTOR_PATH = os.getenv('SELECTOR_PATH', os.path.join(BASE_DIR, 'model', 'pasa-7b-selector'))
EMBEDDING_MODEL_NAME = os.getenv('EMBEDDING_MODEL_NAME', 'all-MiniLM-L6-v2')

# Runtime/data files.
DATABASE_PATH = os.getenv('DATABASE_PATH', _pick_existing_database_path())
TESTSET_PATH = os.getenv('TESTSET_PATH', os.path.join(DATASET_DIR, 'testset.jsonl'))
DICTIONARY_PATH = os.getenv('DICTIONARY_PATH', os.path.join(RESOURCES_DIR, 'frequency_dictionary_en_82_765.txt'))
if not os.path.exists(DICTIONARY_PATH):
    DICTIONARY_PATH = os.path.join(LEGACY_RESOURCES_DIR, 'frequency_dictionary_en_82_765.txt')
AGENT_PROMPT_PATH = os.getenv('AGENT_PROMPT_PATH', os.path.join(RESOURCES_DIR, 'agent_prompt.json'))
if not os.path.exists(AGENT_PROMPT_PATH):
    AGENT_PROMPT_PATH = os.path.join(LEGACY_RESOURCES_DIR, 'agent_prompt.json')
TEMPLATES_DIR = os.getenv('TEMPLATES_DIR', os.path.join(RESOURCES_DIR, 'templates'))
STATIC_DIR = os.getenv('STATIC_DIR', os.path.join(RESOURCES_DIR, 'static'))

# Upload configuration
UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', os.path.join(RESOURCES_DIR, 'pdfs'))
ALLOWED_EXTENSIONS = {'pdf'}

# Tencent COS configuration
COS_SECRET_ID = os.getenv('COS_SECRET_ID', 'your_COS_SECRET_ID')
COS_SECRET_KEY = os.getenv('COS_SECRET_KEY', 'your_COS_SECRET_KEY')
COS_REGION = os.getenv('COS_REGION', 'your_COS_REGION')
COS_BUCKET_NAME = os.getenv('COS_BUCKET_NAME', 'your_COS_BUCKET_NAME')
COS_FOLDER = 'pdfs'
USE_COS_STORAGE = os.getenv('USE_COS_STORAGE', 'false').lower() == 'true'

# Server configuration
HOST = '0.0.0.0'
PORT = 6006
DEBUG = False

# RAG retrieval configuration
RAG_ENABLE_MILVUS = os.getenv('RAG_ENABLE_MILVUS', 'true').lower() == 'true'
RAG_ENABLE_IN_MEMORY_FALLBACK = os.getenv('RAG_ENABLE_IN_MEMORY_FALLBACK', 'true').lower() == 'true'
RAG_MILVUS_URI = os.getenv('RAG_MILVUS_URI', os.path.join(BASE_DIR, 'output', 'rag', 'milvus.db'))
RAG_MILVUS_COLLECTION = os.getenv('RAG_MILVUS_COLLECTION', 'pasa_papers')
RAG_MAX_TEXT_CHARS = int(os.getenv('RAG_MAX_TEXT_CHARS', '8192'))
RAG_EMBED_BATCH_SIZE = int(os.getenv('RAG_EMBED_BATCH_SIZE', '32'))
RAG_RRF_K = int(os.getenv('RAG_RRF_K', '60'))
