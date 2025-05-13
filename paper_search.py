import json
import numpy as np
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import torch
from models import Agent
from openai import OpenAI

# 设置环境变量以优化内存使用
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

class PaperSearch:
    def __init__(self, model_path, selector_path):
        """
        初始化模型和数据
        :param model_path: all-MiniLM-L6-v2模型路径
        :param selector_path: pasa-7b-selector模型路径
        """
        try:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"模型路径不存在: {model_path}")
                
            print(f"正在从本地加载模型: {model_path}")
            self.model = SentenceTransformer(model_path, device='cuda' if torch.cuda.is_available() else 'cpu')
            print("模型加载完成")
            
            # 初始化selector模型
            if selector_path:
                print(f"正在加载selector模型: {selector_path}")
                self.selector = Agent(selector_path)
                print("selector模型加载完成")
                # 加载提示模板
                self.prompts = json.load(open("agent_prompt.json"))
            else:
                self.selector = None
            
            # 初始化DeepSeek客户端
            self.deepseek_client = OpenAI(
                api_key="sk-0ab105e607314279b67232d2de420d55",
                base_url="https://api.deepseek.com/v1"
            )
            
        except Exception as e:
            print(f"初始化失败: {str(e)}")
            raise

        self.papers = []
        self.embeddings = None
        
    def load_papers(self):
        """加载papers.jsonl数据库"""
        try:
            if not os.path.exists("papers.jsonl"):
                raise FileNotFoundError("找不到papers.jsonl文件")
                
            with open("papers.jsonl", "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        paper = json.loads(line)
                        self.papers.append(paper)
                    except json.JSONDecodeError:
                        print(f"警告: 跳过无效的JSON行")
                        continue
                    
            if not self.papers:
                raise ValueError("papers.jsonl文件为空")
                
            # 生成论文的文本表示
            texts = [f"{paper['title']} {paper['abstract']}" for paper in self.papers]
            
            # 计算embeddings
            print("正在计算论文向量...")
            self.embeddings = self.model.encode(texts, show_progress_bar=True,
                                              batch_size=32)  # 直接获取numpy数组
            print(f"完成向量计算,共 {len(self.papers)} 篇论文")
            
        except Exception as e:
            print(f"加载论文数据失败: {str(e)}")
            raise

    def analyze_citation(self, query_paper, selected_paper):
        """
        使用DeepSeek API分析论文引用关系并给出引用建议
        """
        try:
            prompt = f"""请分析以下两篇论文的关系，并给出如何在新论文中引用该文献的具体建议。

请从以下几个方面简要分析（中英文各2-3句话）：
1. 这篇论文与当前研究的相关性
2. 这篇论文的创新点或主要贡献
3. 在新论文中如何引用和对比（例如：可以在哪个部分引用，引用时需要强调哪些异同点）

当前新论文：{query_paper}

待引用论文：
标题：{selected_paper['title']}
摘要：{selected_paper.get('abstract', '无摘要')}
作者：{', '.join(selected_paper.get('authors', ['未知']))}
年份：{selected_paper.get('year', '未知')}

请按照以下格式输出：

中文引用建议：
[此处是中文分析，包含相关性、创新点和具体引用建议]

Citation Suggestion:
[Here is the English analysis, including relevance, innovation points, and specific citation suggestions]"""

            response = self.deepseek_client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "你是一个专业的双语学术论文分析助手，擅长分析论文间的关系并提供具体的引用建议。请确保建议简洁、专业且实用。"},
                    {"role": "user", "content": prompt}
                ],
                stream=False
            )
            
            return response.choices[0].message.content
        except Exception as e:
            return f"引用分析生成失败: {str(e)}"

    def clear_gpu_memory(self):
        """清理GPU内存"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def batch_process_selector(self, results, user_query, batch_size=5):
        """分批处理selector评分"""
        scores = []
        for i in range(0, len(results), batch_size):
            batch = results[i:i + batch_size]
            select_prompts = []
            for result in batch:
                paper = result['paper']
                prompt = self.prompts["get_selected"].format(
                    title=paper['title'],
                    abstract=paper.get('abstract', ''),
                    user_query=user_query
                )
                select_prompts.append(prompt)
            try:
                batch_scores = self.selector.infer_score(select_prompts)
                scores.extend(batch_scores)
                self.clear_gpu_memory()
            except Exception as e:
                print(f"处理批次 {i//batch_size + 1} 时出错: {str(e)}")
                scores.extend([0.0] * len(batch))
        return scores

    def generate_bibtex(self, paper):
        """
        生成论文的BibTeX格式
        """
        try:
            # 获取引用key
            if 'citations' in paper and paper['citations'] and 'key' in paper['citations'][0]:
                entry_key = paper['citations'][0]['key']
            elif 'key' in paper:
                entry_key = paper['key']
            else:
                # 如果没有key字段，则使用标题作为备选
                entry_key = paper['title'][:30].replace(' ', '_').lower()

            # 获取论文类型
            if 'citations' in paper and paper['citations'] and 'type' in paper['citations'][0]:
                entry_type = paper['citations'][0]['type']
            elif 'type' in paper:
                entry_type = paper['type']
            else:
                entry_type = 'article'

            # 处理作者
            if 'citations' in paper and paper['citations'] and 'author' in paper['citations'][0]:
                author_str = paper['citations'][0]['author']
            elif 'author' in paper:
                author_str = paper['author']
            else:
                author_str = 'Unknown'

            # 构建非空字段列表
            fields = []
            fields.append(f"    author = {{{author_str}}}")
            fields.append(f"    title = {{{paper['title']}}}")

            # 处理期刊/会议信息
            if entry_type == 'article':
                if 'citations' in paper and paper['citations'] and 'journal' in paper['citations'][0]:
                    fields.append(f"    journal = {{{paper['citations'][0]['journal']}}}")
                elif 'journal' in paper:
                    fields.append(f"    journal = {{{paper['journal']}}}")
            elif entry_type == 'inproceedings':
                if 'citations' in paper and paper['citations'] and 'booktitle' in paper['citations'][0]:
                    fields.append(f"    booktitle = {{{paper['citations'][0]['booktitle']}}}")
                elif 'booktitle' in paper:
                    fields.append(f"    booktitle = {{{paper['booktitle']}}}")
                if 'citations' in paper and paper['citations'] and 'organization' in paper['citations'][0]:
                    fields.append(f"    organization = {{{paper['citations'][0]['organization']}}}")
                elif 'organization' in paper:
                    fields.append(f"    organization = {{{paper['organization']}}}")

            # 添加其他字段
            for field in ['volume', 'number', 'pages', 'year', 'publisher']:
                if 'citations' in paper and paper['citations'] and field in paper['citations'][0]:
                    fields.append(f"    {field} = {{{paper['citations'][0][field]}}}")
                elif field in paper:
                    fields.append(f"    {field} = {{{paper[field]}}}")
            
            # 生成格式化的BibTeX
            bibtex = "@{}{{{},\n".format(entry_type, entry_key)
            bibtex += ",\n".join(fields)
            bibtex += "\n}"
            
            return bibtex
        except Exception as e:
            print(f"BibTeX生成失败: {str(e)}")
            return ""

    def generate_queries(self, text):
        """
        使用DeepSeek API生成3个不同宽泛程度的查询，并添加原始查询
        :param text: 输入文本
        :return: 包含4个查询的列表，从原始查询到最具体
        """
        try:
            # 首先添加原始查询，直接使用输入文本
            queries = [text]
            
            prompt = f"""请将以下文本改写成3个不同宽泛程度的"show me research on xxx"格式查询，要求：
1. 第一个查询最宽泛，关注整体领域和主题
2. 第二个查询适中，关注具体研究问题
3. 第三个查询最具体，关注具体方法和技术
4. 每个查询不超过50字
5. 使用专业学术语言


示例1：
输入：机器学习在医疗诊断中的应用
输出：
Level 1 (Broadest): show me research on artificial intelligence in healthcare
Level 2 (Moderate): show me research on deep learning for medical image analysis
Level 3 (Specific): show me research on convolutional neural network based tumor detection in magnetic resonance imaging scans



文本内容：
{text}

请按以下格式输出：
Level 1 (Broadest): show me research on [最宽泛查询]
Level 2 (Moderate): show me research on [适中查询]
Level 3 (Specific): show me research on [最具体查询]"""

            response = self.deepseek_client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "你是一个专业的学术助手，擅长将文本改写成不同广泛程度的查询。请确保使用完整的术语，不要使用任何缩写。"},
                    {"role": "user", "content": prompt}
                ],
                stream=False
            )
            
            result = response.choices[0].message.content.strip()
            
            # 解析返回的结果
            for line in result.split('\n'):
                if 'Level 1' in line:
                    queries.append(line.split(':', 1)[1].strip())
                elif 'Level 2' in line:
                    queries.append(line.split(':', 1)[1].strip())
                elif 'Level 3' in line:
                    queries.append(line.split(':', 1)[1].strip())
            
            return queries
        except Exception as e:
            print(f"查询生成失败: {str(e)}")
            return ["", "", "", ""]

    def search_papers(self, query_paper, top_k=5, user_query=None, mode='multi-level', target_level=None):
        """
        支持两种模式：
        - mode='original'：只返回原文查询结果（不分级）
        - mode='multi-level'：返回指定level的结果
        :param target_level: 指定要搜索的level（1-3），如果为None则使用原文查询
        """
        if isinstance(query_paper, str):
            query_text = query_paper
        else:
            query_text = f"{query_paper['title']} {query_paper.get('abstract', '')}"

        # 原文查询模式直接使用向量检索
        if mode == 'original' or target_level is None:
            print("\n使用原文查询模式...")
            query_embedding = self.model.encode([query_text])[0]
            similarities = cosine_similarity([query_embedding], self.embeddings)[0]
            indices = np.argsort(similarities)[::-1][:50]
            candidate_results = []
            for idx in indices:
                paper = self.papers[idx]
                bibtex = self.generate_bibtex(paper)
                candidate_results.append({
                    "paper": paper,
                    "similarity": similarities[idx],
                    "bibtex": bibtex
                })
            print(f"\n找到 {len(candidate_results)} 篇候选论文")
            
            # 格式化候选结果
            formatted_candidates = []
            for result in candidate_results:
                paper = result['paper']
                formatted_result = {
                    'title': paper['title'],
                    'authors': paper['author'].split(' and ') if paper.get('author') else ['Unknown'],
                    'year': paper.get('year', 'Unknown'),
                    'abstract': paper.get('abstract', 'No abstract available'),
                    'similarity': float(result['similarity']),
                    'bibtex': result['bibtex']
                }
                formatted_candidates.append(formatted_result)
            
            # 使用 selector 过滤结果
            if self.selector:
                try:
                    print("\n使用selector进行过滤...")
                    select_scores = self.batch_process_selector(candidate_results, query_text)
                    for i, result in enumerate(candidate_results):
                        result['select_score'] = select_scores[i]
                    candidate_results.sort(key=lambda x: (x['select_score'], x['similarity']), reverse=True)
                    seen_titles = set()
                    unique_results = []
                    for result in candidate_results:
                        title = result['paper']['title'].lower().strip()
                        if title not in seen_titles:
                            seen_titles.add(title)
                            unique_results.append(result)
                    candidate_results = unique_results
                    filtered_results = [r for r in candidate_results if r['select_score'] > 0.3]
                    if not filtered_results:
                        print("没有论文评分大于0.3，返回评分最高的1篇论文")
                        candidate_results = candidate_results[:1]
                    else:
                        candidate_results = filtered_results
                    print(f"\n筛选后保留 {len(candidate_results)} 篇论文")
                except Exception as e:
                    print(f"Selector过滤失败: {str(e)}")
                    print("将使用相似度排序结果")
                    candidate_results = candidate_results[:top_k]
            else:
                candidate_results = candidate_results[:top_k]
            
            # 格式化最终结果
            formatted_results = []
            for result in candidate_results:
                paper = result['paper']
                formatted_result = {
                    'title': paper['title'],
                    'authors': paper['author'].split(' and ') if paper.get('author') else ['Unknown'],
                    'year': paper.get('year', 'Unknown'),
                    'abstract': paper.get('abstract', 'No abstract available'),
                    'similarity': float(result['similarity']),
                    'select_score': float(result.get('select_score', 0.0)),
                    'citation_analysis': result.get('citation_analysis', '分析失败'),
                    'bibtex': result['bibtex']
                }
                formatted_results.append(formatted_result)
            
            return {
                'candidates': formatted_candidates,
                'filtered': formatted_results
            }
            
        # 分级查询模式
        print(f"\n正在生成第{target_level}级查询...")
        queries = self.generate_queries(query_text)
        query = queries[target_level]
        print(f"生成的查询: {query}")
        
        query_embedding = self.model.encode([query])[0]
        similarities = cosine_similarity([query_embedding], self.embeddings)[0]
        indices = np.argsort(similarities)[::-1][:50]
        candidate_results = []
        for idx in indices:
            paper = self.papers[idx]
            bibtex = self.generate_bibtex(paper)
            candidate_results.append({
                "paper": paper,
                "similarity": similarities[idx],
                "bibtex": bibtex
            })
        print(f"\n找到 {len(candidate_results)} 篇候选论文")
        
        formatted_candidates = []
        for result in candidate_results:
            paper = result['paper']
            formatted_result = {
                'title': paper['title'],
                'authors': paper['author'].split(' and ') if paper.get('author') else ['Unknown'],
                'year': paper.get('year', 'Unknown'),
                'abstract': paper.get('abstract', 'No abstract available'),
                'similarity': float(result['similarity']),
                'bibtex': result['bibtex']
            }
            formatted_candidates.append(formatted_result)
            
        if self.selector:
            try:
                print(f"\n使用selector进行过滤: {query}")
                select_scores = self.batch_process_selector(candidate_results, query)
                for i, result in enumerate(candidate_results):
                    result['select_score'] = select_scores[i]
                candidate_results.sort(key=lambda x: (x['select_score'], x['similarity']), reverse=True)
                seen_titles = set()
                unique_results = []
                for result in candidate_results:
                    title = result['paper']['title'].lower().strip()
                    if title not in seen_titles:
                        seen_titles.add(title)
                        unique_results.append(result)
                candidate_results = unique_results
                filtered_results = [r for r in candidate_results if r['select_score'] > 0.3]
                if not filtered_results:
                    print(f"没有论文评分大于0.3，返回评分最高的1篇论文")
                    candidate_results = candidate_results[:1]
                else:
                    candidate_results = filtered_results
                print(f"\n筛选后保留 {len(candidate_results)} 篇论文")
            except Exception as e:
                print(f"Selector过滤失败: {str(e)}")
                print("将使用相似度排序结果")
                candidate_results = candidate_results[:top_k]
        else:
            candidate_results = candidate_results[:top_k]
            
        formatted_results = []
        for result in candidate_results:
            paper = result['paper']
            formatted_result = {
                'title': paper['title'],
                'authors': paper['author'].split(' and ') if paper.get('author') else ['Unknown'],
                'year': paper.get('year', 'Unknown'),
                'abstract': paper.get('abstract', 'No abstract available'),
                'similarity': float(result['similarity']),
                'select_score': float(result.get('select_score', 0.0)),
                'citation_analysis': result.get('citation_analysis', '分析失败'),
                'bibtex': result['bibtex']
            }
            formatted_results.append(formatted_result)
            
        return {
            'candidates': formatted_candidates,
            'filtered': formatted_results
        }

    def analyze_paper(self, query_paper, paper):
        """
        分析单篇论文并生成BibTeX
        """
        try:
            # 进行引用分析
            citation_analysis = self.analyze_citation(query_paper, paper)
            
            # 生成BibTeX格式
            if 'citations' in paper and paper['citations']:
                citation = paper['citations'][0]
                # 使用citation中的所有字段
                entry_type = citation.get('type', 'article')
                entry_key = citation.get('key', '')
                journal = citation.get('journal', '')
                volume = citation.get('volume', '')
                number = citation.get('number', '')
                pages = citation.get('pages', '')
                year = citation.get('year', '')
                publisher = citation.get('publisher', '')
            else:
                # 如果没有citation信息，使用默认值
                entry_type = 'article'
                entry_key = paper['title'][:30].replace(' ', '_')
                journal = paper.get('secondary_title', paper.get('journal', 'Unknown'))
                volume = paper.get('volume', '')
                number = paper.get('number', '')
                pages = paper.get('pages', '')
                year = paper.get('year', 'n.d.')
                publisher = paper.get('publisher', '')
            
            # 构建非空字段列表
            fields = []
            fields.append(f"    title = {{{paper['title']}}}")
            fields.append(f"    author = {{{' and '.join(paper.get('authors', ['Unknown']))}}}")
            
            if year:
                fields.append(f"    year = {{{year}}}")
            if journal:
                fields.append(f"    journal = {{{journal}}}")
            if volume:
                fields.append(f"    volume = {{{volume}}}")
            if number:
                fields.append(f"    number = {{{number}}}")
            if pages:
                fields.append(f"    pages = {{{pages}}}")
            if publisher:
                fields.append(f"    publisher = {{{publisher}}}")
            
            # 生成格式化的BibTeX
            bibtex = "@{}{{{}，\n".format(entry_type, entry_key)
            bibtex += ",\n".join(fields)
            bibtex += "\n}"
            
            return {
                'citation_analysis': citation_analysis,
                'bibtex': bibtex
            }
        except Exception as e:
            print(f"论文分析失败: {str(e)}")
            return {
                'citation_analysis': f"分析失败: {str(e)}",
                'bibtex': ''
            }

def main():
    """主函数"""
    try:
        # 指定本地模型路径
        model_path = './all-MiniLM-L6-v2'
        selector_path = "/root/autodl-tmp/pasa/checkpoints/pasa-7b-selector"
        
        if not os.path.exists(model_path):
            print(f"错误: 模型文件夹不存在: {model_path}")
            print("请确保您已经下载了模型并放置在正确的位置")
            return
            
        searcher = PaperSearch(model_path, selector_path)
        
        # 加载论文数据
        searcher.load_papers()
        
        # 获取用户输入的论文标题
        print("\n请输入要搜索的内容:")
        query_title = input().strip()
        
        if not query_title:
            raise ValueError("论文标题不能为空")
            
        # 使用论文标题作为查询要求
        user_query = query_title
        
        # 搜索相似论文
        print("\n正在搜索相似论文...")
        similar_papers = searcher.search_papers(query_title, user_query=user_query)
        
        # 输出相似论文的基本信息
        print(f"\n输入论文标题: {query_title}")
        print(f"使用selector进行过滤，查询要求: {user_query}")
        print("\n找到的相似论文:")
        for level, results in similar_papers.items():
            level_name = "原始查询" if level == "level_0" else f"第{level.split('_')[1]}级查询"
            print(f"\n{level_name}结果:")
            for i, result in enumerate(results['filtered'], 1):
                paper = result['paper']
                similarity = result['similarity']
                select_score = result['select_score']
                citation_analysis = result['citation_analysis']
                bibtex = result['bibtex']
                print(f"\n{i}. 论文标题: {paper['title']}")
                print(f"   相似度得分: {similarity:.3f}")
                print(f"   selector得分: {select_score:.3f}")
                print(f"   引用分析: {citation_analysis}")
                print(f"   BibTeX: {bibtex}")
            
    except Exception as e:
        print(f"程序运行出错: {str(e)}")
        raise

if __name__ == "__main__":
    main() 