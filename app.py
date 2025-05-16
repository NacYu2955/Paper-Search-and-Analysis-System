from flask import Flask, render_template, request, jsonify
from paper_search import PaperSearch
import os
import logging
import time
from flask_socketio import SocketIO, emit
import threading
import sqlite3
import bibtexparser
from bibtexparser.bparser import BibTexParser
from bibtexparser.customization import convert_to_unicode

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# 初始化PaperSearch
model_path = './all-MiniLM-L6-v2'
selector_path = "checkpoints/pasa-7b-selector"

# 全局变量存储PaperSearch实例
searcher = None
user_last_active = {}  # 存储用户最后活跃时间
online_users = set()
user_lock = threading.Lock()

def get_db():
    conn = sqlite3.connect('papers.db')
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
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
            reviewed_at TIMESTAMP
        )''')
init_db()

def init_searcher():
    global searcher
    if searcher is None:
        logger.info("正在初始化模型和加载论文数据...")
        try:
            searcher = PaperSearch(model_path, selector_path)
            searcher.load_papers()
            logger.info("模型初始化和论文数据加载完成")
        except Exception as e:
            logger.error(f"模型初始化失败: {str(e)}")
            raise

# 在应用启动时初始化
try:
    init_searcher()
except Exception as e:
    logger.error(f"应用启动失败: {str(e)}")
    raise

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
        model_path = './all-MiniLM-L6-v2'
        selector_path = "/root/autodl-tmp/pasa/checkpoints/pasa-7b-selector"
        searcher = PaperSearch(model_path, selector_path)
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

@app.route('/submit_paper', methods=['POST'])
def submit_paper():
    try:
        data = request.get_json()
        bibtex_text = data.get('bibtex')
        abstract = data.get('abstract')

        if not bibtex_text or not abstract:
            return jsonify({'error': '请提供BibTeX格式和摘要'}), 400

        # 解析BibTeX
        parser = BibTexParser()
        parser.customization = convert_to_unicode
        bib_database = bibtexparser.loads(bibtex_text, parser=parser)

        if not bib_database.entries:
            return jsonify({'error': '无法解析BibTeX格式'}), 400

        entry = bib_database.entries[0]  # 获取第一个条目

        # 提取必要字段
        title = entry.get('title', '').strip('{}')
        authors = entry.get('author', '').strip('{}')
        year = entry.get('year', '')
        journal = entry.get('journal', '')
        if not journal:
            journal = entry.get('booktitle', '')  # 如果是会议论文，使用booktitle

        # 可选字段
        booktitle = entry.get('booktitle', '')
        organization = entry.get('organization', '')
        volume = entry.get('volume', '')
        number = entry.get('number', '')
        pages = entry.get('pages', '')
        publisher = entry.get('publisher', '')
        citation_key = entry.get('ID', '')  # BibTeX解析库通常用'ID'字段存key

        # 验证必要字段
        if not all([title, authors, year]):
            return jsonify({'error': 'BibTeX格式缺少必要字段（标题、作者、年份）'}), 400

        db = get_db()
        db.execute('''
            INSERT INTO papers (
                title, authors, abstract, year, journal, status,
                booktitle, organization, volume, number, pages, publisher, citation_key
            ) VALUES (?, ?, ?, ?, ?, 'pending', ?, ?, ?, ?, ?, ?, ?)
        ''', (
            title,
            authors,
            abstract,
            int(year),
            journal,
            booktitle,
            organization,
            volume,
            number,
            pages,
            publisher,
            citation_key
        ))
        db.commit()

        return jsonify({'message': '论文提交成功！'}), 200

    except Exception as e:
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
    data = request.get_json()
    db = get_db()
    db.execute('DELETE FROM papers WHERE id=?', (data['paper_id'],))
    db.commit()
    return jsonify({'message': '删除成功'})

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
    # 查 papers 表
    result1 = db.execute('SELECT COUNT(*) FROM papers WHERE TRIM(LOWER(title)) = ?', (title,)).fetchone()
    # 查 paper_submissions 表（如果有）
    try:
        result2 = db.execute('SELECT COUNT(*) FROM paper_submissions WHERE TRIM(LOWER(title)) = ?', (title,)).fetchone()
    except Exception:
        result2 = [0]
    exists = result1[0] > 0 or result2[0] > 0
    return jsonify({'exists': exists})

if __name__ == '__main__':
    try:
        logger.info("正在启动服务器...")
        socketio.run(app, host='0.0.0.0', port=6006, debug=False)
    except Exception as e:
        logger.error(f"服务器启动失败: {str(e)}")
        raise 