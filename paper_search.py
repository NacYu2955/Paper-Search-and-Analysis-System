import json
import numpy as np
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import torch
from models import Agent
from openai import OpenAI

# Set environment variable to optimize memory usage
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

class PaperSearch:
    def __init__(self, model_path, selector_path):
        """
        Initialize models and data
        :param model_path: path to all-MiniLM-L6-v2 model
        :param selector_path: path to pasa-7b-selector model
        """
        try:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"模型路径不存在: {model_path}")
                
            print(f"正在从本地加载模型: {model_path}")
            self.model = SentenceTransformer(model_path, device='cuda' if torch.cuda.is_available() else 'cpu')
            print("模型加载完成")
            
            # Initialize selector model
            if selector_path:
                print(f"正在加载selector模型: {selector_path}")
                self.selector = Agent(selector_path)
                print("selector模型加载完成")
                # Load prompt templates
                self.prompts = json.load(open("agent_prompt.json"))
            else:
                self.selector = None
            
            # Initialize DeepSeek client
            self.deepseek_client = OpenAI(
                api_key="your deepseek key",
                base_url="https://api.deepseek.com/v1"
            )
            
        except Exception as e:
            print(f"初始化失败: {str(e)}")
            raise

        self.papers = []
        self.embeddings = None
        
    def load_papers(self):
        # Load papers.jsonl database
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
                
            # Generate text representations of papers
            texts = [f"{paper['title']} {paper['abstract']}" for paper in self.papers]
            
            # Compute embeddings
            print("正在计算论文向量...")
            self.embeddings = self.model.encode(texts, show_progress_bar=True, batch_size=32)  
            print(f"完成向量计算,共 {len(self.papers)} 篇论文")
            
        except Exception as e:
            print(f"加载论文数据失败: {str(e)}")
            raise

    def analyze_citation(self, query_paper, selected_paper):
        """
        Use DeepSeek API to analyze citation relationship and provide suggestions
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

    def generate_keywords(self, text):
        """
        使用DeepSeek API从文本中提取关键词
        """
        try:
            prompt = f"""请从以下文本中提取1-3个最相关的关键词，用逗号分隔：

{text}

请只返回关键词，不要包含其他文字。关键词应该是专业术语或重要概念。"""

            response = self.deepseek_client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "你是一个专业的学术助手，擅长从文本中提取关键概念和术语。"},
                    {"role": "user", "content": prompt}
                ],
                stream=False
            )
            
            keywords = response.choices[0].message.content.strip()
            return [k.strip() for k in keywords.split(',')]
        except Exception as e:
            print(f"关键词生成失败: {str(e)}")
            return []

    def clear_gpu_memory(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def batch_process_selector(self, results, user_query, batch_size=5):
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
        Batch process selector scores
        """
        try:
            if 'citations' in paper and paper['citations'] and 'key' in paper['citations'][0]:
                entry_key = paper['citations'][0]['key']
            elif 'key' in paper:
                entry_key = paper['key']
            else:
                entry_key = paper['title'][:30].replace(' ', '_').lower()

            if 'citations' in paper and paper['citations'] and 'type' in paper['citations'][0]:
                entry_type = paper['citations'][0]['type']
            elif 'type' in paper:
                entry_type = paper['type']
            else:
                entry_type = 'article'

            if 'citations' in paper and paper['citations'] and 'author' in paper['citations'][0]:
                author_str = paper['citations'][0]['author']
            elif 'author' in paper:
                author_str = paper['author']
            else:
                author_str = 'Unknown'

            fields = []
            fields.append(f"    author = {{{author_str}}}")
            fields.append(f"    title = {{{paper['title']}}}")

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

            for field in ['volume', 'number', 'pages', 'year', 'publisher']:
                if 'citations' in paper and paper['citations'] and field in paper['citations'][0]:
                    fields.append(f"    {field} = {{{paper['citations'][0][field]}}}")
                elif field in paper:
                    fields.append(f"    {field} = {{{paper[field]}}}")
            
            bibtex = "@{}{{{},\n".format(entry_type, entry_key)
            bibtex += ",\n".join(fields)
            bibtex += "\n}"
            
            return bibtex
        except Exception as e:
            print(f"BibTeX生成失败: {str(e)}")
            return ""

    def search_papers(self, query_paper, top_k=5, user_query=None):
        """
        Search for similar papers using vector similarity and filter them with the selector model
        :param query_paper: Input the paper information, which can be a string (title) or a dictionary {"title": str, "abstract": str}
        :param top_k: Returns the top k most similar papers
        :param user_query: User query, used for selector filtering
        :return: [(paper, similarity_score, select_score)]
        """
        # Processing input, supporting direct input of title strings
        if isinstance(query_paper, str):
            query_text = query_paper
        else:
            query_text = f"{query_paper['title']} {query_paper.get('abstract', '')}"
        
        print("\n正在生成关键词...")
        keywords = self.generate_keywords(query_text)
        print(f"生成的关键词: {', '.join(keywords)}")
        
        # Use keywords to enhance queries
        enhanced_query = f"{query_text} {' '.join(keywords)}"
        
        # Generate the embedding of the query paper
        query_embedding = self.model.encode([enhanced_query])[0]
        
        similarities = cosine_similarity([query_embedding], self.embeddings)[0]
        
        # Obtain the index of the most similar papers and retain 50 for selector filtering
        top_indices = np.argsort(similarities)[::-1][:100]
        
        # Return the paper and the similarity score
        results = []
        for idx in top_indices:
            paper = self.papers[idx]
            bibtex = self.generate_bibtex(paper)
            results.append({
                "paper": paper,
                "similarity": similarities[idx],
                "bibtex": bibtex
            })
            
        print(f"\n通过相似度初筛找到 {len(results)} 篇候选论文")
        
        if self.selector and user_query:
            try:
                print(f"\n使用selector进行过滤...")
                
                scores = self.batch_process_selector(results, user_query)
                print(f"Selector评分结果: {scores}")
                
                for i, result in enumerate(results):
                    result['select_score'] = scores[i]
                
                results.sort(key=lambda x: (x['select_score'], x['similarity']), reverse=True)
                
                # Deduplication: Remove papers with the same title
                seen_titles = set()
                unique_results = []
                for result in results:
                    title = result['paper']['title'].lower().strip()
                    if title not in seen_titles:
                        seen_titles.add(title)
                        unique_results.append(result)
                results = unique_results
                
                # Only retain the results with a selector score greater than 0.3
                filtered_results = [r for r in results if r['select_score'] > 0.3]
                
                # If the filtered result is empty, the top 3 papers with the highest ratings will be returned
                if not filtered_results:
                    print("没有论文评分大于0.3，返回评分最高的3篇论文")
                    results = results[:3]
                else:
                    results = filtered_results
                
                print(f"\n经过selector筛选后保留 {len(results)} 篇论文")
                
            except Exception as e:
                print(f"Selector过滤失败: {str(e)}")
                print("将使用相似度排序结果")
                results = results[:top_k]
        else:
            results = results[:top_k]
            
        return results

    def analyze_paper(self, query_paper, paper):
        """
        Analyze individual papers and generate BibTeX
        """
        try:
            citation_analysis = self.analyze_citation(query_paper, paper)
            
            if 'citations' in paper and paper['citations']:
                citation = paper['citations'][0]
                entry_type = citation.get('type', 'article')
                entry_key = citation.get('key', '')
                journal = citation.get('journal', '')
                volume = citation.get('volume', '')
                number = citation.get('number', '')
                pages = citation.get('pages', '')
                year = citation.get('year', '')
                publisher = citation.get('publisher', '')
            else:
                entry_type = 'article'
                entry_key = paper['title'][:30].replace(' ', '_')
                journal = paper.get('secondary_title', paper.get('journal', 'Unknown'))
                volume = paper.get('volume', '')
                number = paper.get('number', '')
                pages = paper.get('pages', '')
                year = paper.get('year', 'n.d.')
                publisher = paper.get('publisher', '')
            
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
        model_path = './all-MiniLM-L6-v2'
        selector_path = "checkpoints/pasa-7b-selector"
        
        if not os.path.exists(model_path):
            print(f"错误: 模型文件夹不存在: {model_path}")
            print("请确保您已经下载了模型并放置在正确的位置")
            return
            
        searcher = PaperSearch(model_path, selector_path)
        
        searcher.load_papers()
        
        print("\n请输入要搜索的内容:")
        query_title = input().strip()
        
        if not query_title:
            raise ValueError("论文标题不能为空")
            
        user_query = query_title
        
        print("\n正在搜索相似论文...")
        similar_papers = searcher.search_papers(query_title, user_query=user_query)
        
        print(f"\n输入论文标题: {query_title}")
        print(f"使用selector进行过滤，查询要求: {user_query}")
        print("\n找到的相似论文:")
        for i, result in enumerate(similar_papers, 1):
            paper = result["paper"]
            similarity = result["similarity"]
            select_score = result.get('select_score', 0.0)
            citation_analysis = result.get('citation_analysis', '')
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
