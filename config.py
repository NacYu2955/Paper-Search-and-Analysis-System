# 配置文件
import os

# DeepSeek API配置
DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY', 'your_deepseek_api_key')
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"

# 模型路径配置
MODEL_PATH = './all-MiniLM-L6-v2'
SELECTOR_PATH = "checkpoints/pasa-7b-selector"

# 数据库配置
DATABASE_PATH = 'papers.db'

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
