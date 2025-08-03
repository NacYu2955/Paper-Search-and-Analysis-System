from flask import Flask, render_template, request, jsonify, send_file, Response
import json
from paper_search import PaperSearch
import os
import logging
import time
from flask_socketio import SocketIO, emit, join_room, leave_room
import threading
import sqlite3
import bibtexparser
from bibtexparser.bparser import BibTexParser
from bibtexparser.customization import convert_to_unicode
from werkzeug.utils import secure_filename
import uuid
import requests
from config import (
    MODEL_PATH, SELECTOR_PATH, UPLOAD_FOLDER, ALLOWED_EXTENSIONS, 
    HOST, PORT, DEBUG, COS_SECRET_ID, COS_SECRET_KEY, COS_REGION, 
    COS_BUCKET_NAME, COS_FOLDER, USE_COS_STORAGE
)
import PyPDF2
import io
from cos_utils import COSUtils

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# 配置Flask应用，确保所有错误都返回JSON
app.config['PROPAGATE_EXCEPTIONS'] = True

# PDF文件配置
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# 初始化PaperSearch
# 全局变量存储PaperSearch实例
searcher = None
user_last_active = {}  # 存储用户最后活跃时间
online_users = set()
user_lock = threading.Lock()

# 初始化腾讯云COS客户端
cos_client = None
if USE_COS_STORAGE:
    try:
        cos_client = COSUtils(COS_SECRET_ID, COS_SECRET_KEY, COS_REGION, COS_BUCKET_NAME)
        logger.info("腾讯云COS客户端初始化成功")
    except Exception as e:
        logger.error(f"腾讯云COS客户端初始化失败: {str(e)}")
        cos_client = None
else:
    logger.info("使用本地文件存储模式")



def get_db():
    conn = sqlite3.connect('papers.db')
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    # 确保上传文件夹存在（本地存储备用）
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
        
    with get_db() as db:
        db.execute('''CREATE TABLE IF NOT EXISTS papers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            authors TEXT NOT NULL,
            abstract TEXT NOT NULL,
            year INTEGER NOT NULL,
            journal TEXT NOT NULL,
            doi TEXT,
            status TEXT NOT NULL DEFAULT 'pending',
            submitter TEXT,
            review_comment TEXT,
            reviewed_by TEXT,
            submitted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            reviewed_at TIMESTAMP,
            pdf_file_path TEXT
        )''')
        
        # 检查是否存在pdf_file_path字段，如果不存在则添加
        cursor = db.cursor()
        cursor.execute("PRAGMA table_info(papers)")
        columns = [column[1] for column in cursor.fetchall()]
        
        if 'pdf_file_path' not in columns:
            db.execute('ALTER TABLE papers ADD COLUMN pdf_file_path TEXT')
            db.commit()
            

init_db()

def extract_pdf_text(pdf_path_or_cos_key, is_cos_key=False):
    """提取PDF文件的文本内容"""
    try:
        if is_cos_key:
            # 从COS获取文件内容
            if not cos_client:
                logger.error("COS客户端未初始化")
                return None
            
            file_content = cos_client.download_file(pdf_path_or_cos_key)
            if not file_content:
                logger.warning(f"无法从COS下载文件: {pdf_path_or_cos_key}")
                return None
            
            # 从内存中读取PDF
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_content))
        else:
            # 从本地文件系统读取
            if not os.path.exists(pdf_path_or_cos_key):
                logger.warning(f"PDF文件不存在: {pdf_path_or_cos_key}")
                return None
            
            # 读取文件内容到内存，避免文件关闭问题
            with open(pdf_path_or_cos_key, 'rb') as file:
                file_content = file.read()
            
            # 从内存中读取PDF
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_content))
        
        text_content = ""
        # 提取所有页面的文本
        for page_num in range(len(pdf_reader.pages)):
            try:
                page = pdf_reader.pages[page_num]
                page_text = page.extract_text()
                if page_text:
                    text_content += page_text + "\n"
            except Exception as e:
                logger.warning(f"提取第{page_num + 1}页文本失败: {str(e)}")
                continue
        
        # 清理文本内容
        if text_content:
            # 移除多余的空白字符
            text_content = ' '.join(text_content.split())
            # 限制文本长度，避免token过多
            if len(text_content) > 8000:  # 大约2000-3000个token
                text_content = text_content[:8000] + "..."
        
        logger.info(f"成功提取PDF文本，长度: {len(text_content)} 字符")
        return text_content
        
    except Exception as e:
        logger.error(f"PDF文本提取失败: {str(e)}")
        return None

def get_paper_pdf_content(paper_id):
    """获取论文的PDF内容（优先使用本地文件）"""
    try:
        db = get_db()
        cursor = db.cursor()
        cursor.execute("SELECT pdf_file_path FROM papers WHERE id = ?", (paper_id,))
        result = cursor.fetchone()
        db.close()
        
        if not result or not result['pdf_file_path']:
            logger.info(f"论文ID {paper_id} 没有PDF文件路径")
            return None
        
        # 优先检查本地文件
        local_pdf_path = os.path.join(UPLOAD_FOLDER, result['pdf_file_path'])
        logger.info(f"检查本地PDF文件: {local_pdf_path}")
        
        if os.path.exists(local_pdf_path):
            # 从本地文件读取
            logger.info(f"从本地文件读取PDF内容: {local_pdf_path}")
            return extract_pdf_text(local_pdf_path, is_cos_key=False)
        else:
            logger.warning(f"本地PDF文件不存在: {local_pdf_path}")
            
            # 如果本地文件不存在，尝试从COS获取
            if USE_COS_STORAGE and cos_client:
                cos_key = f"{COS_FOLDER}/{result['pdf_file_path']}"
                logger.info(f"从COS获取PDF内容: {cos_key}")
                return extract_pdf_text(cos_key, is_cos_key=True)
            else:
                logger.error("COS存储未启用或客户端未初始化")
                return None
        
    except Exception as e:
        logger.error(f"获取论文PDF内容失败: {str(e)}")
        return None

def init_searcher(max_retries=3):
    """初始化搜索器，支持重试机制"""
    global searcher
    if searcher is not None:
        return searcher
    
    for attempt in range(max_retries):
        try:
            logger.info(f"正在初始化模型和加载论文数据... (尝试 {attempt + 1}/{max_retries})")
            searcher = PaperSearch(MODEL_PATH, SELECTOR_PATH)
            searcher.load_papers()
            logger.info("模型初始化和论文数据加载完成")
            return searcher
        except Exception as e:
            logger.error(f"模型初始化失败 (尝试 {attempt + 1}/{max_retries}): {str(e)}")
            if attempt < max_retries - 1:
                logger.info("等待5秒后重试...")
                import time
                time.sleep(5)
            else:
                logger.error("所有初始化尝试都失败了")
                raise

# 在应用启动时初始化
def initialize_app():
    """应用初始化函数"""
    global searcher
    try:
        init_searcher()
        logger.info("应用初始化成功")
    except Exception as e:
        logger.error(f"应用启动时初始化失败: {str(e)}")
        logger.info("将在首次请求时尝试重新初始化")

# 延迟初始化，避免启动时阻塞
initialize_app()

# 全局错误处理器
@app.errorhandler(404)
def not_found(error):
    if request.path.startswith('/api/') or request.path.startswith('/get_pdf_url/') or request.path.startswith('/view_pdf/'):
        return jsonify({'error': '页面不存在'}), 404
    return '<h1>404 - 页面未找到</h1><p>抱歉，您访问的页面不存在。</p><a href="/">返回首页</a>', 404

@app.errorhandler(500)
def internal_error(error):
    if request.path.startswith('/api/') or request.path.startswith('/get_pdf_url/') or request.path.startswith('/view_pdf/'):
        return jsonify({'error': '服务器内部错误'}), 500
    return render_template('500.html'), 500

@app.errorhandler(400)
def bad_request(error):
    if request.path.startswith('/api/') or request.path.startswith('/get_pdf_url/') or request.path.startswith('/view_pdf/'):
        return jsonify({'error': '请求参数错误'}), 400
    return render_template('400.html'), 400

@app.errorhandler(403)
def forbidden(error):
    if request.path.startswith('/api/') or request.path.startswith('/get_pdf_url/') or request.path.startswith('/view_pdf/'):
        return jsonify({'error': '访问被拒绝'}), 403
    return render_template('403.html'), 403

@app.errorhandler(Exception)
def handle_exception(e):
    logger.error(f"未处理的异常: {str(e)}")
    if request.path.startswith('/api/') or request.path.startswith('/get_pdf_url/') or request.path.startswith('/view_pdf/'):
        return jsonify({'error': '服务器错误'}), 500
    return render_template('error.html'), 500

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('connect')
def handle_connect():
    user_id = request.args.get('uid') or request.remote_addr
    with user_lock:
        online_users.add(user_id)
    emit('online_users', {'online_users': len(online_users)}, broadcast=True)

@socketio.on('disconnect')
def handle_disconnect():
    user_id = request.args.get('uid') or request.remote_addr
    with user_lock:
        online_users.discard(user_id)
    emit('online_users', {'online_users': len(online_users)}, broadcast=True)

@app.route('/search', methods=['POST'])
def search():
    try:
        data = request.get_json()
        query = data.get('query', '')
        mode = data.get('mode', 'original')
        query_level = int(data.get('query_level', 2))  # 默认适中
        
        if not query:
            return jsonify({'error': '请输入查询内容'}), 400
            
        # 检查searcher是否已初始化
        if searcher is None:
            logger.error("搜索器未初始化，尝试重新初始化...")
            try:
                init_searcher()
            except Exception as e:
                logger.error(f"重新初始化搜索器失败: {str(e)}")
                return jsonify({'error': '搜索器初始化失败，请稍后重试'}), 500
        
        # 增加搜索次数统计
        try:
            with get_db() as db:
                # 检查是否存在search_stats表
                cursor = db.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='search_stats'")
                if not cursor.fetchone():
                    # 创建search_stats表
                    db.execute('''CREATE TABLE search_stats (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        total_searches INTEGER DEFAULT 0,
                        last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )''')
                    db.commit()
                
                # 获取或创建统计记录
                cursor.execute("SELECT * FROM search_stats LIMIT 1")
                stats = cursor.fetchone()
                
                if stats:
                    # 更新现有记录
                    cursor.execute("""
                        UPDATE search_stats 
                        SET total_searches = total_searches + 1, 
                            last_updated = CURRENT_TIMESTAMP 
                        WHERE id = ?
                    """, (stats['id'],))
                else:
                    # 创建新记录
                    cursor.execute("""
                        INSERT INTO search_stats (total_searches, last_updated) 
                        VALUES (1, CURRENT_TIMESTAMP)
                    """)
                
                db.commit()
        except Exception as e:
            logger.error(f"更新搜索统计失败: {str(e)}")
        
        if mode == 'original':
            # 原文查询模式直接执行，不生成分级查询
            results = searcher.search_papers(query, mode='original')
            return jsonify({
                'success': True,
                'results': {'level_0': results},
                'queries': [query],
                'selected_level': 0
            })
        else:
            # 分级查询模式才生成分级查询
            queries = searcher.generate_queries(query)
            results = searcher.search_papers(query, mode='multi-level', target_level=query_level)
            return jsonify({
                'success': True,
                'results': {f'level_{query_level}': results},
                'queries': queries,
                'selected_level': query_level
            })
    except Exception as e:
        logger.error(f"搜索出错: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/analyze_paper', methods=['POST'])
def analyze_paper():
    try:
        data = request.get_json()
        query = data.get('query', '')
        paper = data.get('paper', {})
        if not paper:
            return jsonify({'error': '缺少必要参数'}), 400
        
        # 检查searcher是否已初始化
        if searcher is None:
            logger.error("搜索器未初始化，尝试重新初始化...")
            try:
                init_searcher()
            except Exception as e:
                logger.error(f"重新初始化搜索器失败: {str(e)}")
                return jsonify({'error': '搜索器初始化失败，请稍后重试'}), 500
        
        # 分析论文
        analysis_result = searcher.analyze_paper(query, paper)
        return jsonify({
            'success': True,
            'citation_analysis': analysis_result['citation_analysis']
        })
    except Exception as e:
        logger.error(f"论文分析出错: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/selector_infer', methods=['POST'])
def selector_infer():
    try:
        data = request.get_json()
        query = data.get('query', '')
        paper = data.get('paper', {})
        query_level = data.get('query_level', 0)
        queries = data.get('queries', [])
        if not query or not paper:
            return jsonify({'error': '缺少必要参数'}), 400
        
        # 检查searcher是否已初始化
        if searcher is None:
            logger.error("搜索器未初始化，尝试重新初始化...")
            try:
                init_searcher()
            except Exception as e:
                logger.error(f"重新初始化搜索器失败: {str(e)}")
                return jsonify({'error': '搜索器初始化失败，请稍后重试'}), 500
        
        # 使用对应查询级别的查询
        level_query = queries[query_level] if queries and query_level < len(queries) else query
        # 构造prompt
        prompt = searcher.prompts["get_selected"].format(
            title=paper['title'],
            abstract=paper.get('abstract', ''),
            user_query=level_query
        )
        # 推理
        analysis = searcher.selector.infer(prompt)
        if isinstance(analysis, list):
            analysis = analysis[0]
        return jsonify({'success': True, 'selector_infer': str(analysis)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/search_papers', methods=['POST'])
def search_papers():
    try:
        data = request.get_json()
        query = data.get('query', '')
        level = data.get('level', None)  # 新增level参数
        
        if not query:
            return jsonify({'error': '查询内容不能为空'}), 400
            
        # 初始化搜索器
        searcher = PaperSearch(MODEL_PATH, SELECTOR_PATH)
        searcher.load_papers()
        
        # 根据level参数决定搜索模式
        if level is not None:
            # 分级搜索模式
            results = searcher.search_papers(query, mode='multi-level', target_level=level)
        else:
            # 原始搜索模式
            results = searcher.search_papers(query, mode='original')
            
        return jsonify(results)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/upload')
def upload():
    return render_template('upload.html')

@app.route('/admin')
def admin():
    return render_template('admin.html')

@app.route('/admin/review')
def admin_review():
    return render_template('admin_review.html')



@app.route('/check_searcher_status')
def check_searcher_status():
    """检查搜索器状态的调试接口"""
    try:
        if searcher is None:
            return jsonify({
                'status': 'not_initialized',
                'message': '搜索器未初始化'
            })
        else:
            # 尝试获取一些基本信息
            try:
                paper_count = len(searcher.papers) if hasattr(searcher, 'papers') else 'unknown'
                return jsonify({
                    'status': 'initialized',
                    'message': '搜索器已初始化',
                    'paper_count': paper_count,
                    'has_model': hasattr(searcher, 'model'),
                    'has_selector': hasattr(searcher, 'selector')
                })
            except Exception as e:
                return jsonify({
                    'status': 'error',
                    'message': f'搜索器状态检查失败: {str(e)}'
                })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'检查失败: {str(e)}'
        })

# 存储上传任务状态
upload_tasks = {}

def cleanup_expired_tasks():
    """清理过期的上传任务（超过1小时的任务）"""
    import time
    current_time = time.time()
    expired_tasks = []
    
    for task_id, task in upload_tasks.items():
        # 如果任务创建时间超过1小时，标记为过期
        if 'created_time' not in task:
            task['created_time'] = current_time
        elif current_time - task['created_time'] > 3600:  # 1小时 = 3600秒
            expired_tasks.append(task_id)
    
    # 删除过期的任务和对应的PDF文件
    for task_id in expired_tasks:
        task = upload_tasks[task_id]
        if task['status'] == 'completed' and task.get('pdf_path'):
            pdf_path = task['pdf_path']
            full_path = os.path.join(app.config['UPLOAD_FOLDER'], pdf_path)
            try:
                if os.path.exists(full_path):
                    os.remove(full_path)
                    logger.info(f"清理过期任务，删除PDF文件: {full_path}")
            except Exception as e:
                logger.error(f"删除过期PDF文件失败: {str(e)}")
        
        del upload_tasks[task_id]
        logger.info(f"清理过期任务: {task_id}")
    
    if expired_tasks:
        logger.info(f"清理了 {len(expired_tasks)} 个过期任务")

# 启动定时清理任务
def start_cleanup_scheduler():
    """启动定时清理调度器"""
    def cleanup_scheduler():
        while True:
            try:
                cleanup_expired_tasks()
                time.sleep(1800)  # 每30分钟清理一次
            except Exception as e:
                logger.error(f"定时清理任务出错: {str(e)}")
                time.sleep(1800)
    
    cleanup_thread = threading.Thread(target=cleanup_scheduler)
    cleanup_thread.daemon = True
    cleanup_thread.start()
    logger.info("定时清理任务已启动")

@app.route('/upload_pdf_async', methods=['POST'])
def upload_pdf_async():
    """异步上传PDF文件"""
    try:
        if 'pdf_file' not in request.files:
            return jsonify({'error': '没有选择文件'}), 400
        
        file = request.files['pdf_file']
        if file.filename == '':
            return jsonify({'error': '没有选择文件'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': '只允许上传PDF文件'}), 400
        
        # 生成任务ID
        task_id = str(uuid.uuid4())
        
        # 读取文件内容到内存
        file_content = file.read()
        original_filename = file.filename
        
        # 初始化任务状态
        import time
        upload_tasks[task_id] = {
            'status': 'uploading',
            'progress': 0,
            'filename': original_filename,
            'pdf_path': None,
            'error': None,
            'created_time': time.time()
        }
        
        def upload_task():
            try:
                # 生成唯一文件名
                filename = f"{uuid.uuid4()}.pdf"
                
                # 更新进度到30%
                upload_tasks[task_id]['progress'] = 30
                
                # 上传到腾讯云COS
                if not cos_client:
                    raise Exception("COS客户端未初始化")
                logger.info(f"异步上传PDF文件到COS: {filename}")
                upload_result = cos_client.upload_file(file_content, filename, COS_FOLDER)
                
                if upload_result['success']:
                    # 更新进度到70%
                    upload_tasks[task_id]['progress'] = 70
                    
                    # 从COS下载到本地pdfs文件夹
                    cos_key = f"{COS_FOLDER}/{filename}"
                    logger.info(f"从COS下载PDF文件到本地: {cos_key}")
                    
                    # 确保本地pdfs文件夹存在
                    if not os.path.exists(UPLOAD_FOLDER):
                        os.makedirs(UPLOAD_FOLDER)
                    
                                            # 使用预签名URL从COS下载文件
                        logger.info(f"生成预签名URL下载文件: {cos_key}")
                        presigned_url = cos_client.get_file_url(cos_key, expires=3600)
                        if presigned_url:
                            logger.info(f"预签名URL生成成功: {presigned_url[:50]}...")
                            
                            # 使用requests下载文件
                            response = requests.get(presigned_url, timeout=30)
                        if response.status_code == 200:
                            downloaded_content = response.content
                            logger.info(f"通过预签名URL下载成功，文件大小: {len(downloaded_content)} bytes")
                            
                            # 保存到本地
                            local_pdf_path = os.path.join(UPLOAD_FOLDER, filename)
                            with open(local_pdf_path, 'wb') as f:
                                f.write(downloaded_content)
                            logger.info(f"PDF文件已从COS下载到本地: {local_pdf_path}")
                            
                            # 更新任务状态
                            upload_tasks[task_id].update({
                                'status': 'completed',
                                'progress': 100,
                                'pdf_path': filename
                            })
                            
                            logger.info(f"PDF文件异步上传和下载完成: {filename}")
                        else:
                            logger.error(f"通过预签名URL下载失败，状态码: {response.status_code}")
                            logger.error(f"响应内容: {response.text[:200]}")
                            raise Exception("从云存储下载文件失败")
                    else:
                        logger.error(f"生成预签名URL失败: {cos_key}")
                        raise Exception("生成下载链接失败")
                else:
                    raise Exception(f"上传到COS失败: {upload_result['error']}")
            except Exception as e:
                logger.error(f"PDF文件异步上传失败: {str(e)}")
                upload_tasks[task_id].update({
                    'status': 'failed',
                    'error': str(e)
                })
        
        # 在后台线程中执行上传
        thread = threading.Thread(target=upload_task)
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'success': True,
            'task_id': task_id,
            'message': 'PDF文件上传已开始'
        })
        
    except Exception as e:
        logger.error(f"PDF异步上传失败: {str(e)}")
        return jsonify({'error': f'上传失败: {str(e)}'}), 500

@app.route('/upload_status/<task_id>')
def upload_status(task_id):
    """获取上传任务状态"""
    if task_id not in upload_tasks:
        return jsonify({'error': '任务不存在'}), 404
    
    task = upload_tasks[task_id]
    return jsonify({
        'task_id': task_id,
        'status': task['status'],
        'progress': task['progress'],
        'filename': task['filename'],
        'pdf_path': task.get('pdf_path'),
        'error': task.get('error')
    })

@app.route('/cleanup_unused_pdf', methods=['POST'])
def cleanup_unused_pdf():
    """清理未使用的PDF文件"""
    try:
        data = request.get_json()
        task_id = data.get('task_id')
        
        if not task_id:
            return jsonify({'error': '缺少任务ID'}), 400
        
        logger.info(f"清理未使用的PDF文件，任务ID: {task_id}")
        
        if task_id in upload_tasks:
            task = upload_tasks[task_id]
            
                        # 如果任务已完成，删除文件
            if task['status'] == 'completed':
                # 清理COS中的文件
                if task.get('pdf_path') and USE_COS_STORAGE and cos_client:
                    try:
                        cos_key = f"{COS_FOLDER}/{task['pdf_path']}"
                        if cos_client.delete_file(cos_key):
                            logger.info(f"已删除COS中未使用的PDF文件: {cos_key}")
                        else:
                            logger.warning(f"删除COS中PDF文件失败: {cos_key}")
                    except Exception as e:
                        logger.error(f"删除COS中PDF文件失败: {str(e)}")
                
                # 清理本地文件
                if task.get('pdf_path'):
                    pdf_path = task['pdf_path']
                    full_path = os.path.join(app.config['UPLOAD_FOLDER'], pdf_path)
                    
                    try:
                        if os.path.exists(full_path):
                            os.remove(full_path)
                            logger.info(f"已删除本地未使用的PDF文件: {full_path}")
                        else:
                            logger.warning(f"本地PDF文件不存在: {full_path}")
                    except Exception as e:
                        logger.error(f"删除本地PDF文件失败: {str(e)}")
            
            # 删除任务记录
            del upload_tasks[task_id]
            logger.info(f"已删除任务记录: {task_id}")
            
            return jsonify({'success': True, 'message': 'PDF文件清理成功'})
        else:
            logger.warning(f"任务不存在: {task_id}")
            return jsonify({'success': True, 'message': '任务不存在，无需清理'})
            
    except Exception as e:
        logger.error(f"清理PDF文件失败: {str(e)}")
        return jsonify({'error': f'清理失败: {str(e)}'}), 500

@app.route('/submit_paper', methods=['POST'])
def submit_paper():
    try:
        logger.info("开始处理论文提交请求")
        
        # 判断请求类型
        if request.content_type and request.content_type.startswith('multipart/form-data'):
            bibtex_text = request.form.get('bibtex')
            abstract = request.form.get('abstract')
            task_id = request.form.get('task_id')  # 获取上传任务ID
            logger.info(f"接收到multipart/form-data请求: bibtex长度={len(bibtex_text) if bibtex_text else 0}, abstract长度={len(abstract) if abstract else 0}, task_id={task_id}")
        else:
            data = request.get_json()
            bibtex_text = data.get('bibtex')
            abstract = data.get('abstract')
            task_id = data.get('task_id')
            logger.info(f"接收到JSON请求: bibtex长度={len(bibtex_text) if bibtex_text else 0}, abstract长度={len(abstract) if abstract else 0}, task_id={task_id}")

        if not bibtex_text or not abstract:
            logger.error("缺少必要字段: bibtex或abstract")
            return jsonify({'error': '请提供BibTeX格式和摘要'}), 400

        # 检查PDF上传任务状态
        pdf_path = None
        if task_id:
            if task_id not in upload_tasks:
                return jsonify({'error': 'PDF上传任务不存在或已过期'}), 400
            
            task = upload_tasks[task_id]
            if task['status'] != 'completed':
                return jsonify({'error': f'PDF文件上传未完成，当前状态: {task["status"]}'}), 400
            
            # 获取PDF文件路径
            pdf_path = task.get('pdf_path')
            
            if pdf_path:
                logger.info(f"使用已上传的PDF文件: {pdf_path}")
            else:
                logger.warning("任务完成但没有找到PDF文件路径")

        # 检查PDF路径（前端直传）
        pdf_path = request.form.get('pdf_path') or (data.get('pdf_path') if not request.content_type.startswith('multipart/form-data') else None)
        if not pdf_path:
            logger.error("缺少PDF文件路径（pdf_path）")
            return jsonify({'error': '请先上传PDF文件'}), 400

        # 解析BibTeX
        logger.info("开始解析BibTeX")
        parser = BibTexParser()
        parser.customization = convert_to_unicode
        bib_database = bibtexparser.loads(bibtex_text, parser=parser)

        if not bib_database.entries:
            logger.error("BibTeX解析失败，没有找到条目")
            return jsonify({'error': '无法解析BibTeX格式'}), 400

        entry = bib_database.entries[0]  # 获取第一个条目
        logger.info(f"成功解析BibTeX条目: {entry.get('title', '未知标题')}")

        # 提取必要字段
        title = entry.get('title', '').strip('{}')
        authors = entry.get('author', '').strip('{}')
        year = entry.get('year', '')
        journal = entry.get('journal', '')
        if not journal:
            journal = entry.get('booktitle', '')  # 如果是会议论文，使用booktitle

        # 验证必要字段
        if not all([title, authors, year]):
            logger.error(f"BibTeX缺少必要字段: title={bool(title)}, authors={bool(authors)}, year={bool(year)}")
            return jsonify({'error': 'BibTeX格式缺少必要字段（标题、作者、年份）'}), 400

        # 数据库操作
        try:
            logger.info("开始插入数据库")
            db = get_db()
            cursor = db.cursor()
            cursor.execute('''
                INSERT INTO papers (
                    title, authors, abstract, year, journal, status, pdf_file_path
                ) VALUES (?, ?, ?, ?, ?, 'pending', ?)
            ''', (
                title,
                authors,
                abstract,
                int(year),
                journal,
                pdf_path
            ))
            db.commit()
            db.close()
            logger.info("论文提交成功")
            
            # 从COS下载PDF到本地（使用预签名URL）
            if pdf_path and USE_COS_STORAGE and cos_client:
                try:
                    cos_key = f"{COS_FOLDER}/{pdf_path}"
                    logger.info(f"从COS下载PDF文件到本地: {cos_key}")
                    logger.info(f"COS配置: bucket={cos_client.bucket_name}, region={cos_client.region}")
                    
                    # 检查文件是否存在于COS中
                    if cos_client.file_exists(cos_key):
                        logger.info(f"文件存在于COS中: {cos_key}")
                    else:
                        logger.warning(f"文件不存在于COS中: {cos_key}")
                        return jsonify({'error': 'PDF文件在COS中不存在'}), 400
                    
                    # 确保本地pdfs文件夹存在
                    if not os.path.exists(UPLOAD_FOLDER):
                        os.makedirs(UPLOAD_FOLDER)
                        logger.info(f"创建本地文件夹: {UPLOAD_FOLDER}")
                    
                    # 使用预签名URL下载文件
                    logger.info(f"生成预签名URL下载文件: {cos_key}")
                    presigned_url = cos_client.get_file_url(cos_key, expires=3600)
                    if presigned_url:
                        logger.info(f"预签名URL生成成功: {presigned_url[:50]}...")
                        
                        # 使用requests下载文件
                        response = requests.get(presigned_url, timeout=30)
                        if response.status_code == 200:
                            downloaded_content = response.content
                            logger.info(f"通过预签名URL下载成功，文件大小: {len(downloaded_content)} bytes")
                            
                            # 检查文件内容的前几个字节，确认是PDF
                            if len(downloaded_content) > 4 and downloaded_content[:4] == b'%PDF':
                                logger.info("文件内容确认是PDF格式")
                            else:
                                logger.warning(f"文件内容可能不是PDF格式，前20字节: {downloaded_content[:20]}")
                            
                            # 保存到本地
                            local_pdf_path = os.path.join(UPLOAD_FOLDER, pdf_path)
                            with open(local_pdf_path, 'wb') as f:
                                f.write(downloaded_content)
                            logger.info(f"PDF文件已从COS下载到本地: {local_pdf_path}")
                        else:
                            logger.error(f"通过预签名URL下载失败，状态码: {response.status_code}")
                            logger.error(f"响应内容: {response.text[:200]}")
                    else:
                        logger.error(f"生成预签名URL失败: {cos_key}")
                except Exception as e:
                    logger.error(f"下载PDF到本地失败: {str(e)}")
                    logger.error(f"异常详情: ", exc_info=True)
            
            # 清理上传任务
            if task_id and task_id in upload_tasks:
                del upload_tasks[task_id]
            
        except Exception as e:
            logger.error(f"数据库操作失败: {str(e)}")
            # 如果数据库操作失败，删除已保存的文件
            if pdf_path and USE_COS_STORAGE and cos_client:
                try:
                    cos_key = f"{COS_FOLDER}/{pdf_path}"
                    if cos_client.delete_file(cos_key):
                        logger.info(f"已删除COS中的PDF文件: {cos_key}")
                except Exception as del_e:
                    logger.error(f"删除COS中PDF文件失败: {str(del_e)}")
            
            if pdf_path:
                try:
                    pdf_full_path = os.path.join(app.config['UPLOAD_FOLDER'], pdf_path)
                    if os.path.exists(pdf_full_path):
                        os.remove(pdf_full_path)
                        logger.info(f"已删除本地PDF文件: {pdf_full_path}")
                except Exception as del_e:
                    logger.error(f"删除本地PDF文件失败: {str(del_e)}")
            return jsonify({'error': f'数据库操作失败：{str(e)}'}), 500

        return jsonify({'message': '论文提交成功！'}), 200

    except Exception as e:
        logger.error(f"论文提交失败: {str(e)}", exc_info=True)
        return jsonify({'error': f'提交失败：{str(e)}'}), 500

@app.route('/my_papers', methods=['GET'])
def my_papers():
    uid = request.args.get('uid', '匿名')
    db = get_db()
    papers = db.execute('SELECT * FROM papers WHERE submitter=? ORDER BY submitted_at DESC', (uid,)).fetchall()
    return jsonify([dict(row) for row in papers])

@app.route('/admin/pending_papers')
def admin_pending_papers():
    db = get_db()
    papers = db.execute('SELECT * FROM papers WHERE status=? ORDER BY submitted_at DESC', ('pending',)).fetchall()
    return jsonify([dict(row) for row in papers])

@app.route('/admin/review_paper', methods=['POST'])
def admin_review_paper():
    data = request.get_json()
    db = get_db()
    db.execute('''
        UPDATE papers SET status=?, review_comment=?, reviewed_by=?, reviewed_at=CURRENT_TIMESTAMP
        WHERE id=?
    ''', (data['action'], data.get('comment', ''), data.get('reviewed_by', 'admin'), data['paper_id']))
    db.commit()
    # 审核后重新加载论文数据
    global searcher
    if searcher is not None:
        searcher.load_papers()
    return jsonify({'message': '审核完成'})

@app.route('/admin/reviewed_papers')
def admin_reviewed_papers():
    db = get_db()
    papers = db.execute("SELECT * FROM papers WHERE status IN ('approved', 'approve', 'rejected', 'reject') ORDER BY reviewed_at DESC").fetchall()
    return jsonify([dict(row) for row in papers])

@app.route('/admin/delete_paper', methods=['POST'])
def admin_delete_paper():
    try:
        data = request.get_json()
        paper_id = data.get('paper_id')
        
        if not paper_id:
            return jsonify({'error': '缺少论文ID'}), 400
        
        db = get_db()
        cursor = db.cursor()
        
        # 先查询论文信息，获取PDF文件路径
        cursor.execute('SELECT pdf_file_path FROM papers WHERE id=?', (paper_id,))
        result = cursor.fetchone()
        
        if result and result['pdf_file_path']:
            pdf_path = result['pdf_file_path']
            
            # 删除本地PDF文件
            local_pdf_path = os.path.join(UPLOAD_FOLDER, pdf_path)
            if os.path.exists(local_pdf_path):
                try:
                    os.remove(local_pdf_path)
                    logger.info(f"已删除本地PDF文件: {local_pdf_path}")
                except Exception as e:
                    logger.error(f"删除本地PDF文件失败: {str(e)}")
            else:
                logger.info(f"本地PDF文件不存在: {local_pdf_path}")
            
            # 删除COS中的文件（无论USE_COS_STORAGE配置如何都尝试）
            if cos_client:
                try:
                    cos_key = f"{COS_FOLDER}/{pdf_path}"
                    if cos_client.delete_file(cos_key):
                        logger.info(f"已删除COS中的PDF文件: {cos_key}")
                    else:
                        logger.warning(f"删除COS中PDF文件失败: {cos_key}")
                except Exception as e:
                    logger.error(f"删除COS中PDF文件失败: {str(e)}")
        
        # 删除数据库记录
        cursor.execute('DELETE FROM papers WHERE id=?', (paper_id,))
        db.commit()
        db.close()
        
        logger.info(f"成功删除论文ID: {paper_id}")
        return jsonify({'message': '删除成功'})
        
    except Exception as e:
        logger.error(f"删除论文失败: {str(e)}")
        return jsonify({'error': f'删除失败：{str(e)}'}), 500

@app.route('/admin/update_paper', methods=['POST'])
def admin_update_paper():
    data = request.get_json()
    db = get_db()
    db.execute('''
        UPDATE papers SET title=?, authors=?, abstract=?, year=?, journal=?, doi=?
        WHERE id=?
    ''', (
        data['title'], data['authors'], data['abstract'], data['year'],
        data['journal'], data['doi'], data['paper_id']
    ))
    db.commit()
    return jsonify({'message': '修改成功'})

@app.route('/check_spelling', methods=['POST'])
def check_spelling():
    try:
        data = request.get_json()
        query = data.get('query', '')
        if not query:
            return jsonify({'error': '请输入查询内容'}), 400
        
        # 检查searcher是否已初始化
        if searcher is None:
            logger.error("搜索器未初始化，尝试重新初始化...")
            try:
                init_searcher()
            except Exception as e:
                logger.error(f"重新初始化搜索器失败: {str(e)}")
                return jsonify({'error': '搜索器初始化失败，请稍后重试'}), 500
            
        # 使用PaperSearch实例的拼写检查功能
        corrected_query = searcher.correct_spelling(query)
        
        return jsonify({
            'success': True,
            'original_query': query,
            'corrected_query': corrected_query if corrected_query != query else None
        })
    except Exception as e:
        logger.error(f"拼写检查出错: {str(e)}")
        return jsonify({'error': str(e)}), 500



@app.route('/admin/check_duplicate', methods=['POST'])
def admin_check_duplicate():
    data = request.get_json()
    title = data.get('title', '').strip().lower()
    db = get_db()
    # 只查已通过的论文
    result1 = db.execute('SELECT COUNT(*) FROM papers WHERE TRIM(LOWER(title)) = ? AND status IN ("approved", "approve")', (title,)).fetchone()
    # 查 paper_submissions 表（如果有）
    try:
        result2 = db.execute('SELECT COUNT(*) FROM paper_submissions WHERE TRIM(LOWER(title)) = ?', (title,)).fetchone()
    except Exception:
        result2 = [0]
    exists = result1[0] > 0 or result2[0] > 0
    return jsonify({'exists': exists})

@app.route('/admin/change_paper_status', methods=['POST'])
def admin_change_paper_status():
    try:
        data = request.get_json()
        paper_id = data.get('paper_id')
        status = data.get('status')
        comment = data.get('comment', '')
        if not paper_id or not status:
            return jsonify({'error': '参数不完整'}), 400
        db = get_db()
        db.execute('''
            UPDATE papers SET status=?, review_comment=?, reviewed_by=?, reviewed_at=CURRENT_TIMESTAMP
            WHERE id=?
        ''', (status, comment, 'admin', paper_id))
        db.commit()
        # 审核后重新加载论文数据
        global searcher
        if searcher is not None:
            searcher.load_papers()
        return jsonify({'message': '状态修改成功'})
    except Exception as e:
        return jsonify({'error': f'修改状态失败: {str(e)}'}), 500

@app.route('/upload_pdf', methods=['POST'])
def upload_pdf():
    """上传PDF文件（上传到COS后下载到本地）"""
    try:
        if 'pdf_file' not in request.files:
            return jsonify({'error': '没有选择文件'}), 400
        file = request.files['pdf_file']
        paper_id = request.form.get('paper_id')
        if file.filename == '':
            return jsonify({'error': '没有选择文件'}), 400
        if not allowed_file(file.filename):
            return jsonify({'error': '只允许上传PDF文件'}), 400
        if not paper_id:
            return jsonify({'error': '缺少论文ID'}), 400
        
        # 生成唯一文件名
        filename = f"{uuid.uuid4()}.pdf"
        # 读取文件内容
        file_content = file.read()
        
        # 上传到腾讯云COS
        if not cos_client:
            return jsonify({'error': 'COS客户端未初始化'}), 500
        
        logger.info(f"上传PDF文件到COS: {filename}")
        upload_result = cos_client.upload_file(file_content, filename, COS_FOLDER)
        
        if upload_result['success']:
            # 从COS下载到本地pdfs文件夹（使用预签名URL）
            cos_key = f"{COS_FOLDER}/{filename}"
            logger.info(f"从COS下载PDF文件到本地: {cos_key}")
            
            # 确保本地pdfs文件夹存在
            if not os.path.exists(UPLOAD_FOLDER):
                os.makedirs(UPLOAD_FOLDER)
            
                            # 使用预签名URL下载文件
                logger.info(f"生成预签名URL下载文件: {cos_key}")
                presigned_url = cos_client.get_file_url(cos_key, expires=3600)
                if presigned_url:
                    logger.info(f"预签名URL生成成功: {presigned_url[:50]}...")
                    
                    # 使用requests下载文件
                    response = requests.get(presigned_url, timeout=30)
                if response.status_code == 200:
                    downloaded_content = response.content
                    logger.info(f"通过预签名URL下载成功，文件大小: {len(downloaded_content)} bytes")
                    
                    # 保存到本地
                    local_pdf_path = os.path.join(UPLOAD_FOLDER, filename)
                    with open(local_pdf_path, 'wb') as f:
                        f.write(downloaded_content)
                    logger.info(f"PDF文件已从COS下载到本地: {local_pdf_path}")
                else:
                    logger.error(f"通过预签名URL下载失败，状态码: {response.status_code}")
                    logger.error(f"响应内容: {response.text[:200]}")
                    return jsonify({'error': '从云存储下载文件失败'}), 500
            else:
                logger.error(f"生成预签名URL失败: {cos_key}")
                return jsonify({'error': '生成下载链接失败'}), 500
                
                # 更新数据库
                db = get_db()
                cursor = db.cursor()
                cursor.execute("UPDATE papers SET pdf_file_path = ? WHERE id = ?", (filename, paper_id))
                db.commit()
                db.close()
                
                logger.info(f"PDF文件上传和下载成功: {filename}")
                return jsonify({
                    'success': True, 
                    'message': 'PDF文件上传成功', 
                    'file_path': filename
                })
        else:
            logger.error(f"PDF文件上传到COS失败: {upload_result['error']}")
            return jsonify({'error': f"上传到COS失败: {upload_result['error']}"}), 500
    except Exception as e:
        logger.error(f"PDF上传失败: {str(e)}")
        return jsonify({'error': f'上传失败: {str(e)}'}), 500

@app.route('/view_pdf/<int:paper_id>')
def view_pdf(paper_id):
    """查看PDF文件（从本地pdfs文件夹）"""
    try:
        logger.info(f"尝试查看PDF，论文ID: {paper_id}")
        
        db = get_db()
        cursor = db.cursor()
        cursor.execute("SELECT pdf_file_path FROM papers WHERE id = ?", (paper_id,))
        result = cursor.fetchone()
        db.close()
        
        logger.info(f"查询结果: {result}")
        
        if not result:
            logger.error(f"论文ID {paper_id} 不存在")
            return jsonify({'error': '论文不存在'}), 404
        
        if not result['pdf_file_path']:
            logger.error(f"论文ID {paper_id} 没有PDF文件")
            return jsonify({'error': 'PDF文件不存在'}), 404
        
        # 从本地pdfs文件夹查看PDF
        local_pdf_path = os.path.join(UPLOAD_FOLDER, result['pdf_file_path'])
        logger.info(f"从本地路径获取PDF文件: {local_pdf_path}")
        
        # 检查本地文件是否存在
        if os.path.exists(local_pdf_path):
            logger.info(f"本地PDF文件存在，直接返回文件")
            return send_file(local_pdf_path, mimetype='application/pdf')
        else:
            logger.warning(f"本地PDF文件不存在: {local_pdf_path}")
            
            # 如果本地文件不存在，尝试从COS下载
            if USE_COS_STORAGE and cos_client:
                cos_key = f"{COS_FOLDER}/{result['pdf_file_path']}"
                logger.info(f"尝试从COS下载PDF文件到本地: {cos_key}")
                
                if cos_client.file_exists(cos_key):
                    # 使用预签名URL从COS下载文件到本地
                    logger.info(f"生成预签名URL下载文件: {cos_key}")
                    presigned_url = cos_client.get_file_url(cos_key, expires=3600)
                    if presigned_url:
                        logger.info(f"预签名URL生成成功: {presigned_url[:50]}...")
                        
                        # 使用requests下载文件
                        response = requests.get(presigned_url, timeout=30)
                        if response.status_code == 200:
                            file_content = response.content
                            logger.info(f"通过预签名URL下载成功，文件大小: {len(file_content)} bytes")
                            
                            # 保存到本地
                            with open(local_pdf_path, 'wb') as f:
                                f.write(file_content)
                            logger.info(f"PDF文件已从COS下载到本地: {local_pdf_path}")
                            return send_file(local_pdf_path, mimetype='application/pdf')
                        else:
                            logger.error(f"通过预签名URL下载失败，状态码: {response.status_code}")
                            logger.error(f"响应内容: {response.text[:200]}")
                            return jsonify({'error': '从云存储下载文件失败'}), 500
                    else:
                        logger.error(f"生成预签名URL失败: {cos_key}")
                        return jsonify({'error': '生成下载链接失败'}), 500
                else:
                    logger.error(f"PDF文件在COS中不存在: {cos_key}")
                    return jsonify({'error': 'PDF文件在云存储中不存在'}), 404
            else:
                logger.error("COS存储未启用或客户端未初始化")
                return jsonify({'error': '云存储服务不可用'}), 500
        
    except Exception as e:
        logger.error(f"查看PDF失败: {str(e)}")
        return jsonify({'error': f'查看失败: {str(e)}'}), 500

@app.route('/get_pdf_info/<int:paper_id>')
def get_pdf_info(paper_id):
    """获取PDF文件信息（优先检查本地文件）"""
    try:
        logger.info(f"获取PDF信息，论文ID: {paper_id}")
        
        db = get_db()
        cursor = db.cursor()
        cursor.execute("SELECT pdf_file_path FROM papers WHERE id = ?", (paper_id,))
        result = cursor.fetchone()
        db.close()
        
        if not result:
            logger.error(f"论文ID {paper_id} 不存在")
            return jsonify({'error': '论文不存在'}), 404
        
        if not result['pdf_file_path']:
            logger.info(f"论文ID {paper_id} 没有PDF文件路径")
            return jsonify({'has_pdf': False, 'error': 'PDF文件不存在'})
        
        # 优先检查本地文件
        local_pdf_path = os.path.join(UPLOAD_FOLDER, result['pdf_file_path'])
        logger.info(f"检查本地PDF文件: {local_pdf_path}")
        
        if os.path.exists(local_pdf_path):
            # 获取本地文件信息
            file_size = os.path.getsize(local_pdf_path)
            logger.info(f"本地PDF文件存在，大小: {file_size} bytes")
            
            return jsonify({
                'has_pdf': True,
                'storage_type': 'local',
                'file_path': result['pdf_file_path'],
                'file_size': file_size,
                'content_type': 'application/pdf'
            })
        else:
            logger.warning(f"本地PDF文件不存在: {local_pdf_path}")
            
            # 如果本地文件不存在，检查COS
            if USE_COS_STORAGE and cos_client:
                cos_key = f"{COS_FOLDER}/{result['pdf_file_path']}"
                logger.info(f"检查COS中的PDF文件: {cos_key}")
                
                if cos_client.file_exists(cos_key):
                    # 获取COS文件信息
                    file_info = cos_client.get_file_info(cos_key)
                    if file_info:
                        logger.info(f"PDF文件在COS中，大小: {file_info['size']} bytes")
                        
                        return jsonify({
                            'has_pdf': True,
                            'storage_type': 'cos',
                            'file_path': result['pdf_file_path'],
                            'file_size': file_info['size'],
                            'content_type': file_info['content_type']
                        })
                    else:
                        logger.error(f"无法获取COS文件信息: {cos_key}")
                        return jsonify({'error': '无法获取文件信息'}), 500
                else:
                    logger.info(f"PDF文件在COS中也不存在: {cos_key}")
                    return jsonify({'has_pdf': False, 'error': 'PDF文件不存在'})
            else:
                logger.error("COS存储未启用或客户端未初始化")
                return jsonify({'error': '云存储服务不可用'}), 500
        
    except Exception as e:
        logger.error(f"获取PDF信息失败: {str(e)}")
        return jsonify({'error': f'获取信息失败: {str(e)}'}), 500



@app.route('/paper_chat', methods=['POST'])
def paper_chat():
    """处理针对特定论文的对话"""
    try:
        data = request.get_json()
        paper_index = data.get('paper_index')
        paper = data.get('paper')
        message = data.get('message')
        conversation_history = data.get('conversation_history', [])
        user_id = request.remote_addr
        
        if not paper or not message:
            return jsonify({'error': '缺少必要参数'}), 400
        
        # 尝试获取PDF内容
        pdf_content = None
        if paper.get('id'):
            pdf_content = get_paper_pdf_content(paper['id'])
            logger.info(f"论文ID {paper['id']} 的PDF内容获取结果: {'成功' if pdf_content else '失败或无PDF'}")
        
        # 构建针对论文的对话上下文
        paper_context = f"""
论文信息：
标题：{paper.get('title', '未知')}
作者：{', '.join(paper.get('authors', ['未知']))}
年份：{paper.get('year', '未知')}
摘要：{paper.get('abstract', '无摘要')}
        """.strip()
        
        # 如果有PDF内容，添加到上下文中
        if pdf_content:
            paper_context += f"""

论文PDF内容：
{pdf_content}

注意：以上是论文PDF文件的文本内容，您可以基于这些具体内容来回答用户的问题。"""
        
        # 构建系统提示
        system_prompt = f"""你是一个专业的学术论文分析助手。用户正在询问关于以下论文的问题：

{paper_context}

请根据论文信息回答用户的问题。回答要专业、准确、有帮助。如果用户的问题超出了论文内容范围，请说明并建议用户查看其他相关文献。

请用中文回答，保持友好和专业的语调。"""
        
        # 构建消息历史
        messages = [{"role": "system", "content": system_prompt}]
        
        # 添加对话历史（限制长度以避免token过多）
        for msg in conversation_history[-10:]:  # 只保留最近10条消息
            if msg['role'] in ['user', 'assistant']:
                messages.append({"role": msg['role'], "content": msg['content']})
        
        # 添加当前用户消息
        messages.append({"role": "user", "content": message})
        
        # 调用DeepSeek API
        try:
            response = searcher.deepseek_client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
                stream=False,
                max_tokens=800,  # 减少token数量
                temperature=0.6  # 降低随机性
            )
            
            assistant_message = response.choices[0].message.content
            
            return jsonify({
                'success': True,
                'response': assistant_message,
                'paper_title': paper.get('title', '未知'),
                'has_pdf_content': pdf_content is not None
            })
            
        except Exception as e:
            logger.error(f"DeepSeek API调用失败: {str(e)}")
            error_message = f"抱歉，我暂时无法回答您的问题。错误信息：{str(e)}"
            
            return jsonify({
                'success': False,
                'response': error_message
            })
            
    except Exception as e:
        logger.error(f"论文对话处理失败: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/citation_chat', methods=['POST'])
def citation_chat():
    """处理引用分析多轮对话"""
    try:
        data = request.get_json()
        query = data.get('query')  # 新论文描述
        paper = data.get('paper')  # 被引用论文
        message = data.get('message')
        conversation_history = data.get('conversation_history', [])
        if not query or not paper or not message:
            return jsonify({'error': '缺少必要参数'}), 400

        # 尝试获取PDF内容
        pdf_content = None
        if paper.get('id'):
            pdf_content = get_paper_pdf_content(paper['id'])
            logger.info(f"论文ID {paper['id']} 的PDF内容获取结果: {'成功' if pdf_content else '失败或无PDF'}")

        # 构建引用分析上下文
        citation_context = f"""
当前新论文：{query}

待引用论文：
标题：{paper.get('title', '未知')}
作者：{', '.join(paper.get('authors', ['未知']))}
年份：{paper.get('year', '未知')}
摘要：{paper.get('abstract', '无摘要')}
        """.strip()

        # 如果有PDF内容，添加到上下文中
        if pdf_content:
            citation_context += f"""

论文PDF内容：
{pdf_content}

注意：以上是论文PDF文件的文本内容，您可以基于这些具体内容来回答用户的问题。"""

        # 系统提示 - 优化为更简洁的提示
        system_prompt = f"""你是学术引用分析助手。分析用户问题并给出简洁、实用的建议。

{citation_context}

回答要求：
1. 先简要分析问题要点
2. 给出具体建议
3. 使用中文回答
4. 如果提供了PDF内容，请基于PDF内容进行更深入的分析"""

        # 构建消息历史
        messages = [{"role": "system", "content": system_prompt}]
        for msg in conversation_history[-6:]:  # 减少历史消息数量
            if msg['role'] in ['user', 'assistant']:
                messages.append({"role": msg['role'], "content": msg['content']})
        messages.append({"role": "user", "content": message})

        # 调用 DeepSeek
        try:
            response = searcher.deepseek_client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
                stream=False,
                temperature=0.5
            )
            assistant_message = response.choices[0].message.content
            return jsonify({
                'success': True,
                'response': assistant_message,
                'has_pdf_content': pdf_content is not None
            })
        except Exception as e:
            logger.error(f"DeepSeek API调用失败: {str(e)}")
            error_message = f"抱歉，我暂时无法回答您的问题。错误信息：{str(e)}"
            return jsonify({
                'success': False,
                'response': error_message
            })
    except Exception as e:
        logger.error(f"引用分析对话处理失败: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/citation_chat_stream', methods=['POST'])
def citation_chat_stream():
    """处理引用分析多轮对话 - 流式输出"""
    try:
        data = request.get_json()
        query = data.get('query')  # 新论文描述
        paper = data.get('paper')  # 被引用论文
        message = data.get('message')
        conversation_history = data.get('conversation_history', [])
        if not query or not paper or not message:
            return jsonify({'error': '缺少必要参数'}), 400

        # 尝试获取PDF内容
        pdf_content = None
        if paper.get('id'):
            pdf_content = get_paper_pdf_content(paper['id'])
            logger.info(f"论文ID {paper['id']} 的PDF内容获取结果: {'成功' if pdf_content else '失败或无PDF'}")

        # 构建引用分析上下文
        citation_context = f"""
当前新论文：{query}

待引用论文：
标题：{paper.get('title', '未知')}
作者：{', '.join(paper.get('authors', ['未知']))}
年份：{paper.get('year', '未知')}
摘要：{paper.get('abstract', '无摘要')}
        """.strip()

        # 如果有PDF内容，添加到上下文中
        if pdf_content:
            citation_context += f"""

论文PDF内容：
{pdf_content}

注意：以上是论文PDF文件的文本内容，您可以基于这些具体内容来回答用户的问题。"""

        # 系统提示 - 优化为更简洁的提示
        system_prompt = f"""你是学术引用分析助手。分析用户问题并给出简洁、实用的建议。

{citation_context}

回答要求：
1. 先简要分析问题要点
2. 给出具体建议
3. 使用中文回答
4. 如果提供了PDF内容，请基于PDF内容进行更深入的分析"""

        # 构建消息历史
        messages = [{"role": "system", "content": system_prompt}]
        for msg in conversation_history[-6:]:  # 减少历史消息数量
            if msg['role'] in ['user', 'assistant']:
                messages.append({"role": msg['role'], "content": msg['content']})
        messages.append({"role": "user", "content": message})

        def generate():
            try:
                # 调用 DeepSeek 流式API
                response = searcher.deepseek_client.chat.completions.create(
                    model="deepseek-chat",
                    messages=messages,
                    stream=True,
                    temperature=0.5
                )
                
                for chunk in response:
                    if chunk.choices[0].delta.content is not None:
                        yield f"data: {json.dumps({'content': chunk.choices[0].delta.content})}\n\n"
                
                yield f"data: {json.dumps({'done': True, 'has_pdf_content': pdf_content is not None})}\n\n"
                
            except Exception as e:
                logger.error(f"DeepSeek API调用失败: {str(e)}")
                yield f"data: {json.dumps({'error': f'抱歉，我暂时无法回答您的问题。错误信息：{str(e)}'})}\n\n"

        return Response(generate(), mimetype='text/plain')
        
    except Exception as e:
        logger.error(f"引用分析对话处理失败: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/search_realtime', methods=['POST'])
def search_realtime():
    """实时搜索论文，支持批次推送"""
    try:
        data = request.get_json()
        query = data.get('query', '')
        mode = data.get('mode', 'original')
        query_level = int(data.get('query_level', 2))
        session_id = data.get('session_id', 'default')
        
        if not query:
            return jsonify({'error': '请输入查询内容'}), 400
        
        # 增加搜索次数统计
        try:
            with get_db() as db:
                # 检查是否存在search_stats表
                cursor = db.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='search_stats'")
                if not cursor.fetchone():
                    # 创建search_stats表
                    db.execute('''CREATE TABLE search_stats (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        total_searches INTEGER DEFAULT 0,
                        last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )''')
                    db.commit()
                
                # 获取或创建统计记录
                cursor.execute("SELECT * FROM search_stats LIMIT 1")
                stats = cursor.fetchone()
                
                if stats:
                    # 更新现有记录
                    cursor.execute("""
                        UPDATE search_stats 
                        SET total_searches = total_searches + 1, 
                            last_updated = CURRENT_TIMESTAMP 
                        WHERE id = ?
                    """, (stats['id'],))
                else:
                    # 创建新记录
                    cursor.execute("""
                        INSERT INTO search_stats (total_searches, last_updated) 
                        VALUES (1, CURRENT_TIMESTAMP)
                    """)
                
                db.commit()
        except Exception as e:
            logger.error(f"更新搜索统计失败: {str(e)}")
        
        # 存储当前会话的搜索结果
        if not hasattr(app, 'realtime_results'):
            app.realtime_results = {}
        
        app.realtime_results[session_id] = {
            'papers': [],
            'candidates': [],
            'original_query': query,
            'corrected_query': None,
            'completed': False
        }
        
        def callback(high_score_papers, batch_num):
            """回调函数，用于推送高评分论文"""
            try:
                # 将高评分论文添加到会话结果中
                app.realtime_results[session_id]['papers'].extend(high_score_papers)
                
                # 通过WebSocket推送结果
                socketio.emit('realtime_papers', {
                    'session_id': session_id,
                    'papers': high_score_papers,
                    'batch_num': batch_num,
                    'total_papers': len(app.realtime_results[session_id]['papers'])
                }, room=session_id)
                
                logger.info(f"批次 {batch_num} 推送了 {len(high_score_papers)} 篇高评分论文")
                
            except Exception as e:
                logger.error(f"推送高评分论文失败: {str(e)}")
        
        # 修改后的回调函数，支持翻译信息传递
        def callback_with_translation(high_score_papers, batch_num, corrected_query=None):
            """回调函数，用于推送高评分论文，支持翻译信息"""
            try:
                # 为每篇论文添加翻译信息
                for paper in high_score_papers:
                    if corrected_query:
                        paper['corrected_query'] = corrected_query
                
                # 将高评分论文添加到会话结果中
                app.realtime_results[session_id]['papers'].extend(high_score_papers)
                
                # 通过WebSocket推送结果
                socketio.emit('realtime_papers', {
                    'session_id': session_id,
                    'papers': high_score_papers,
                    'batch_num': batch_num,
                    'total_papers': len(app.realtime_results[session_id]['papers'])
                }, room=session_id)
                
                logger.info(f"批次 {batch_num} 推送了 {len(high_score_papers)} 篇高评分论文")
                
            except Exception as e:
                logger.error(f"推送高评分论文失败: {str(e)}")
        
        # 在后台线程中执行搜索
        def search_thread():
            try:
                # 检查searcher是否已初始化
                if searcher is None:
                    logger.error("搜索器未初始化，尝试重新初始化...")
                    try:
                        init_searcher()
                    except Exception as e:
                        logger.error(f"重新初始化搜索器失败: {str(e)}")
                        socketio.emit('search_error', {
                            'session_id': session_id,
                            'error': '搜索器初始化失败，请稍后重试'
                        }, room=session_id)
                        return
                
                # 检查字数，如果超过400字则转换为宽泛查询
                original_query = query
                final_query = query
                logger.info(f"原始查询长度: {len(query)} 字符")
                
                if len(query) > 400:
                    logger.info(f"检测到长查询，开始转换为宽泛查询...")
                    try:
                        # 使用searcher的generate_queries功能生成宽泛查询
                        queries = searcher.generate_queries(query)
                        if queries and len(queries) > 1:
                            # 使用第一个查询（最宽泛的）
                            final_query = queries[1]  # Level 1 (Broadest)
                            logger.info(f"转换后的宽泛查询: {final_query}")
                        else:
                            logger.warning(f"生成查询失败，使用原始查询")
                    except Exception as e:
                        logger.error(f"转换宽泛查询失败: {str(e)}")
                        final_query = query
                else:
                    logger.info(f"查询长度未超过50字，使用原文查询")
                
                # 创建带翻译信息的回调函数
                def callback_with_translation_info(high_score_papers, batch_num):
                    corrected_query = None
                    if hasattr(searcher, 'last_corrected_query'):
                        corrected_query = searcher.last_corrected_query
                    callback_with_translation(high_score_papers, batch_num, corrected_query)
                
                if mode == 'original':
                    results = searcher.search_papers_realtime(final_query, mode='original', callback=callback_with_translation_info)
                else:
                    results = searcher.search_papers_realtime(final_query, mode='multi-level', target_level=query_level, callback=callback_with_translation_info)
                
                # 更新会话结果
                app.realtime_results[session_id].update({
                    'candidates': results.get('candidates', []),
                    'original_query': original_query,
                    'final_query': final_query,
                    'corrected_query': results.get('corrected_query'),
                    'completed': True
                })
                
                # 发送完成信号
                socketio.emit('search_completed', {
                    'session_id': session_id,
                    'total_candidates': len(results.get('candidates', [])),
                    'total_papers': len(app.realtime_results[session_id]['papers'])
                }, room=session_id)
                
                logger.info(f"搜索完成，会话 {session_id} 共找到 {len(app.realtime_results[session_id]['papers'])} 篇论文")
                
            except Exception as e:
                logger.error(f"搜索线程出错: {str(e)}")
                socketio.emit('search_error', {
                    'session_id': session_id,
                    'error': str(e)
                }, room=session_id)
        
        # 启动后台搜索线程
        thread = threading.Thread(target=search_thread)
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'message': '搜索已开始，将通过WebSocket实时推送结果'
        })
        
    except Exception as e:
        logger.error(f"实时搜索出错: {str(e)}")
        return jsonify({'error': str(e)}), 500

@socketio.on('join_session')
def handle_join_session(data):
    """加入搜索会话"""
    session_id = data.get('session_id')
    if session_id:
        join_room(session_id)
        logger.info(f"用户加入会话: {session_id}")

@socketio.on('leave_session')
def handle_leave_session(data):
    """离开搜索会话"""
    session_id = data.get('session_id')
    if session_id:
        leave_room(session_id)
        logger.info(f"用户离开会话: {session_id}")

@app.route('/get_realtime_results/<session_id>')
def get_realtime_results(session_id):
    """获取实时搜索结果"""
    try:
        if not hasattr(app, 'realtime_results') or session_id not in app.realtime_results:
            return jsonify({'error': '会话不存在'}), 404
        
        results = app.realtime_results[session_id]
        return jsonify({
            'success': True,
            'results': results
        })
        
    except Exception as e:
        logger.error(f"获取实时结果失败: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/get_pdf_url/<int:paper_id>')
def get_pdf_url(paper_id):
    """获取PDF文件的直接访问URL（优先使用本地文件）"""
    try:
        logger.info(f"获取PDF URL，论文ID: {paper_id}")
        
        # 验证paper_id参数
        if not isinstance(paper_id, int) or paper_id <= 0:
            logger.error(f"无效的论文ID: {paper_id}")
            return jsonify({'error': '无效的论文ID'}), 400
        
        db = get_db()
        cursor = db.cursor()
        cursor.execute("SELECT pdf_file_path FROM papers WHERE id = ?", (paper_id,))
        result = cursor.fetchone()
        db.close()
        
        if not result:
            logger.error(f"论文ID {paper_id} 不存在")
            return jsonify({'error': '论文不存在'}), 404
        
        if not result['pdf_file_path']:
            logger.error(f"论文ID {paper_id} 没有PDF文件")
            return jsonify({'error': 'PDF文件不存在'}), 404
        
        # 优先检查本地文件
        local_pdf_path = os.path.join(UPLOAD_FOLDER, result['pdf_file_path'])
        logger.info(f"检查本地PDF文件: {local_pdf_path}")
        
        if os.path.exists(local_pdf_path):
            # 返回本地文件的URL
            local_url = f"/view_pdf/{paper_id}"
            logger.info(f"返回本地PDF文件URL: {local_url}")
            return jsonify({
                'success': True,
                'url': local_url,
                'storage_type': 'local',
                'expires_in': None,
                'preview_mode': 'inline'
            })
        else:
            logger.warning(f"本地PDF文件不存在: {local_pdf_path}")
            
            # 如果本地文件不存在，尝试从COS获取
            if USE_COS_STORAGE and cos_client:
                cos_key = f"{COS_FOLDER}/{result['pdf_file_path']}"
                logger.info(f"从COS获取PDF URL: {cos_key}")
                
                # 检查文件是否存在
                try:
                    if not cos_client.file_exists(cos_key):
                        logger.error(f"PDF文件在COS中不存在: {cos_key}")
                        return jsonify({'error': 'PDF文件在云存储中不存在'}), 404
                except Exception as e:
                    logger.error(f"检查COS文件存在性失败: {str(e)}")
                    return jsonify({'error': '检查云存储文件失败'}), 500
                
                # 获取COS预签名URL（1小时有效，启用在线预览）
                presigned_url = cos_client.get_file_url(cos_key, expires=3600, inline=True)
                if not presigned_url:
                    logger.error(f"生成预签名URL失败: {cos_key}")
                    return jsonify({'error': '生成预签名URL失败'}), 500
                logger.info(f"生成COS预签名URL（在线预览）: {presigned_url}")
                return jsonify({
                    'success': True,
                    'url': presigned_url,
                    'storage_type': 'cos',
                    'expires_in': 3600,
                    'preview_mode': 'inline'
                })
            else:
                logger.error("COS存储未启用或客户端未初始化")
                return jsonify({'error': '云存储服务不可用'}), 500
        
    except Exception as e:
        logger.error(f"获取PDF URL失败: {str(e)}")
        return jsonify({'error': f'获取URL失败: {str(e)}'}), 500

@app.route('/download_pdf/<int:paper_id>')
def download_pdf(paper_id):
    """下载PDF文件（优先使用本地文件）"""
    try:
        logger.info(f"尝试下载PDF，论文ID: {paper_id}")
        
        db = get_db()
        cursor = db.cursor()
        cursor.execute("SELECT pdf_file_path FROM papers WHERE id = ?", (paper_id,))
        result = cursor.fetchone()
        db.close()
        
        if not result:
            logger.error(f"论文ID {paper_id} 不存在")
            return jsonify({'error': '论文不存在'}), 404
        
        if not result['pdf_file_path']:
            logger.error(f"论文ID {paper_id} 没有PDF文件")
            return jsonify({'error': 'PDF文件不存在'}), 404
        
        # 优先检查本地文件
        local_pdf_path = os.path.join(UPLOAD_FOLDER, result['pdf_file_path'])
        logger.info(f"检查本地PDF文件: {local_pdf_path}")
        
        if os.path.exists(local_pdf_path):
            # 从本地文件下载
            logger.info(f"从本地文件下载PDF: {local_pdf_path}")
            return send_file(
                local_pdf_path, 
                mimetype='application/pdf',
                as_attachment=True,
                download_name=f"paper_{paper_id}.pdf"
            )
        else:
            logger.warning(f"本地PDF文件不存在: {local_pdf_path}")
            
            # 如果本地文件不存在，尝试从COS下载
            if USE_COS_STORAGE and cos_client:
                cos_key = f"{COS_FOLDER}/{result['pdf_file_path']}"
                logger.info(f"从COS下载PDF文件: {cos_key}")
                
                # 检查文件是否存在
                if cos_client.file_exists(cos_key):
                    # 获取带下载参数的预签名URL
                    presigned_url = cos_client.get_file_url(cos_key, expires=3600, inline=False)
                    if presigned_url:
                        logger.info(f"重定向到COS预签名URL（下载）: {presigned_url}")
                        # 重定向到腾讯云COS的预签名URL
                        return Response(
                            f'<html><head><meta http-equiv="refresh" content="0;url={presigned_url}"></head><body>正在下载PDF文件...</body></html>',
                            mimetype='text/html'
                        )
                    else:
                        logger.error(f"生成预签名URL失败: {cos_key}")
                        return jsonify({'error': '生成预签名URL失败'}), 500
                else:
                    logger.error(f"PDF文件在COS中不存在: {cos_key}")
                    return jsonify({'error': 'PDF文件在云存储中不存在'}), 404
            else:
                logger.error("COS存储未启用或客户端未初始化")
                return jsonify({'error': '云存储服务不可用'}), 500
        
    except Exception as e:
        logger.error(f"下载PDF失败: {str(e)}")
        return jsonify({'error': f'下载失败: {str(e)}'}), 500

@app.route('/cos_sts_token', methods=['GET'])
def cos_sts_token():
    """获取前端直传COS的临时密钥（STS）"""
    try:
        allow_prefix = f"{COS_FOLDER}/*"
        token = cos_client.get_upload_sts_token(allow_prefix=allow_prefix, duration_seconds=1800)
        if token:
            return jsonify({'success': True, 'credentials': token['credentials'], 'startTime': token.get('startTime'), 'expiredTime': token.get('expiredTime')})
        else:
            return jsonify({'success': False, 'error': '获取COS临时密钥失败'}), 500
    except Exception as e:
        logger.error(f"获取COS临时密钥失败: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/admin/search_stats')
def admin_search_stats():
    """获取搜索统计信息"""
    try:
        with get_db() as db:
            # 检查是否存在search_stats表
            cursor = db.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='search_stats'")
            if not cursor.fetchone():
                # 创建search_stats表
                db.execute('''CREATE TABLE search_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    total_searches INTEGER DEFAULT 0,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )''')
                db.commit()
            
            # 获取统计记录
            cursor.execute("SELECT * FROM search_stats LIMIT 1")
            stats = cursor.fetchone()
            
            if stats:
                return jsonify({
                    'success': True,
                    'total_searches': stats['total_searches'],
                    'last_updated': stats['last_updated']
                })
            else:
                # 如果没有记录，创建一条初始记录
                cursor.execute("""
                    INSERT INTO search_stats (total_searches, last_updated) 
                    VALUES (0, CURRENT_TIMESTAMP)
                """)
                db.commit()
                
                return jsonify({
                    'success': True,
                    'total_searches': 0,
                    'last_updated': 'CURRENT_TIMESTAMP'
                })
    except Exception as e:
        logger.error(f"获取搜索统计失败: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    try:
        logger.info("正在启动服务器...")
        # 启动定时清理任务
        start_cleanup_scheduler()
        socketio.run(app, host=HOST, port=PORT, debug=DEBUG, allow_unsafe_werkzeug=True)
    except Exception as e:
        logger.error(f"服务器启动失败: {str(e)}")
        raise 
