from flask import Flask, render_template, request, jsonify
from paper_search import PaperSearch
import os
import logging
import time

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# 初始化PaperSearch
model_path = './all-MiniLM-L6-v2'
selector_path = "checkpoints/pasa-7b-selector"

# 全局变量存储PaperSearch实例
searcher = None
user_last_active = {}  # 存储用户最后活跃时间

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

@app.route('/search', methods=['POST'])
def search():
    try:
        data = request.get_json()
        query = data.get('query', '')
        mode = data.get('mode', 'original')
        query_level = int(data.get('query_level', 2))  # 默认适中
        
        if not query:
            return jsonify({'error': '请输入查询内容'}), 400
            
        # 生成分级查询
        queries = searcher.generate_queries(query)
        
        if mode == 'original':
            # 只返回原文查询结果
            results = searcher.search_papers(query, mode='original')
            return jsonify({
                'success': True,
                'results': {'level_0': results},
                'queries': [query],
                'selected_level': 0
            })
        else:
            # 返回指定level的结果
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

@app.route('/heartbeat', methods=['POST'])
def heartbeat():
    data = request.get_json(silent=True) or {}
    user_id = data.get('uid') or request.remote_addr
    user_last_active[user_id] = time.time()
    return jsonify({'status': 'ok'})

@app.route('/online_users')
def online_users():
    now = time.time()
    count = sum(1 for t in user_last_active.values() if now - t <= 10)
    return jsonify({'online_users': count})

if __name__ == '__main__':
    try:
        logger.info("正在启动服务器...")
        app.run(host='0.0.0.0', port=6006, debug=False)
    except Exception as e:
        logger.error(f"服务器启动失败: {str(e)}")
        raise 