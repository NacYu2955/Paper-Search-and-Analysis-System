import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# DeepSeek API配置
DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY', 'your_deepseek_api_key')
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"

# 模型路径配置
MODEL_PATH = os.getenv('MODEL_PATH', os.path.join(BASE_DIR, 'all-MiniLM-L6-v2'))
EMBEDDING_MODEL_NAME = os.getenv('EMBEDDING_MODEL_NAME', 'sentence-transformers/all-MiniLM-L6-v2')
SELECTOR_PATH = os.getenv('SELECTOR_PATH', os.path.join(BASE_DIR, 'checkpoints', 'pasa-7b-selector'))

# 数据库配置
DATABASE_PATH = os.getenv('DATABASE_PATH', os.path.join(BASE_DIR, 'papers.db'))

# RAG / Milvus配置
RAG_ENABLE_RERANK = os.getenv('RAG_ENABLE_RERANK', 'false').lower() == 'true'
RAG_ENABLE_MILVUS = os.getenv('RAG_ENABLE_MILVUS', 'true').lower() == 'true'
RAG_ENABLE_IN_MEMORY_FALLBACK = os.getenv('RAG_ENABLE_IN_MEMORY_FALLBACK', 'true').lower() == 'true'
RAG_MILVUS_URI = os.getenv('RAG_MILVUS_URI', os.path.join(BASE_DIR, 'data', 'milvus', 'paper_rag.db'))
RAG_MILVUS_COLLECTION = os.getenv('RAG_MILVUS_COLLECTION', 'paper_rag_chunks')
RAG_RETRIEVAL_MODE = os.getenv('RAG_RETRIEVAL_MODE', 'dense')
RAG_RETRIEVAL_TOP_K = int(os.getenv('RAG_RETRIEVAL_TOP_K', '50'))
RAG_RERANK_TOP_K = int(os.getenv('RAG_RERANK_TOP_K', '5'))
RAG_EMBED_BATCH_SIZE = int(os.getenv('RAG_EMBED_BATCH_SIZE', '32'))
RAG_MAX_TEXT_CHARS = int(os.getenv('RAG_MAX_TEXT_CHARS', '8192'))
RAG_RRF_K = int(os.getenv('RAG_RRF_K', '60'))

# 文件上传配置
UPLOAD_FOLDER = 'pdfs'

ALLOWED_EXTENSIONS = {'pdf'}

# 腾讯云COS配置
COS_SECRET_ID = os.getenv('COS_SECRET_ID', 'your_cos_secret_id')
COS_SECRET_KEY = os.getenv('COS_SECRET_KEY', 'your_cos_secret_key')
COS_REGION = os.getenv('COS_REGION', 'ap_your_location')
COS_BUCKET_NAME = os.getenv('COS_BUCKET_NAME', 'your_cos_bucket_name')
COS_FOLDER = 'pdfs'

# 是否启用腾讯云COS存储（如果为False，则使用本地存储）
USE_COS_STORAGE = os.getenv('USE_COS_STORAGE', 'true').lower() == 'true'

# 服务器配置
HOST = '0.0.0.0'
PORT = 6006
DEBUG = False
