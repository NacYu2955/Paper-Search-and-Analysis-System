# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Please note that:
1. You need to first apply for a Google Search API key at https://serpapi.com/,
   and replace the 'your google keys' below before you can use it.
2. The service for searching arxiv and obtaining paper contents is relatively simple. 
   If there are any bugs or improvement suggestions, you can submit pull requests.
   We would greatly appreciate and look forward to your contributions!!
"""
import re
import bs4
import json
import arxiv
import urllib
import zipfile
import warnings
import requests
import time
import random
from datetime import datetime
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm import tqdm
warnings.simplefilter("always")


DEEPSEEK_API_KEY = 'your Deepseek keys'
arxiv_client = arxiv.Client(delay_seconds = 0.05)
id2paper     = json.load(open("data/paper_database/id2paper.json"))
paper_db     = zipfile.ZipFile("data/paper_database/cs_paper_2nd.zip", "r")
arxiv_client = arxiv.Client()
# 配置请求头
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
    'Connection': 'keep-alive',
}

# 配置重试策略
RETRY_STRATEGY = Retry(
    total=3,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504],
)

# 创建带有重试机制的session
session = requests.Session()
adapter = HTTPAdapter(max_retries=RETRY_STRATEGY)
session.mount("http://", adapter)
session.mount("https://", adapter)

# 访问频率限制
RATE_LIMIT = {
    'arxiv': 3,  # 每3秒一次
    'google': 1,  # 每1秒一次
    'last_request': {}
}

def rate_limit(service):
    """实现访问频率限制"""
    current_time = time.time()
    if service in RATE_LIMIT['last_request']:
        time_since_last = current_time - RATE_LIMIT['last_request'][service]
        if time_since_last < RATE_LIMIT[service]:
            time.sleep(RATE_LIMIT[service] - time_since_last)
    RATE_LIMIT['last_request'][service] = time.time()

def google_search_arxiv_id(query, num=10, end_date=None):
    print(f"\n🔍 正在通过Google搜索论文: {query}")
    rate_limit('google')
    url = "https://google.serper.dev/search"

    search_query = f"{query} site:arxiv.org"
    if end_date:
        try:
            end_date = datetime.strptime(end_date, '%Y%m%d').strftime('%Y-%m-%d')
            search_query = f"{query} before:{end_date} site:arxiv.org"
        except:
            search_query = f"{query} site:arxiv.org"
    
    payload = json.dumps({
        "q": search_query, 
        "num": num, 
        "page": 1, 
    })

    headers = {
        'X-API-KEY': GOOGLE_KEY,
        'Content-Type': 'application/json',
        **HEADERS
    }
    assert headers['X-API-KEY'] != 'your google keys', "add your google search key!!!"

    for attempt in range(3):
        try:
            print(f"  尝试 {attempt + 1}/3...")
            response = session.post(url, headers=headers, data=payload)
            if response.status_code == 200:
                results = json.loads(response.text)
                arxiv_id_list = []
                for paper in results['organic']:
                    if re.search(r'arxiv\.org/(?:abs|pdf|html)/(\d{4}\.\d+)', paper["link"]):
                        arxiv_id = re.search(r'arxiv\.org/(?:abs|pdf|html)/(\d{4}\.\d+)', paper["link"]).group(1)
                        arxiv_id_list.append(arxiv_id)
                arxiv_id_list = list(set(arxiv_id_list))
                print(f"✅ 找到 {len(arxiv_id_list)} 篇相关论文")
                return arxiv_id_list
        except Exception as e:
            warnings.warn(f"Google search failed (attempt {attempt + 1}/3), query: {query}, error: {str(e)}")
            time.sleep(2 ** attempt)
            continue
    print("❌ 搜索失败")
    return []

def parse_metadata(metas):
    """
    Parse concatenated metadata string into authors, title, and journal.
    """
    # Get and clean metas
    metas = [item.replace('\n', ' ') for item in metas]
    meta_string = ' '.join(metas)
    
    authors, title, journal = "", "", ""
        
    if len(metas) == 3: # author / title / journal
        authors, title, journal = metas
    else:
        # Remove the year suffix (e.g., 2022a) from the metadata string
        meta_string = re.sub(r'\.\s\d{4}[a-z]?\.', '.', meta_string)
        # Regular expression to match the pattern
        regex = r"^(.*?\.\s)(.*?)(\.\s.*|$)"
        match = re.match(regex, meta_string, re.DOTALL)
        if match:
            authors = match.group(1).strip() if match.group(1) else ""
            title = match.group(2).strip() if match.group(2) else ""
            journal = match.group(3).strip() if match.group(3) else ""

            if journal.startswith('. '):
                journal = journal[2:]

    return {
        "meta_list": metas, 
        "meta_string": meta_string, 
        "authors": authors,
        "title": title,
        "journal": journal
    }

def create_dict_for_citation(ul_element):
    citation_dict, futures, id_attrs = {}, [], []
    for li in ul_element.find_all("li", recursive=False):
        id_attr = li['id']
        metas = [x.text.strip() for x in li.find_all('span', class_='ltx_bibblock')]
        id_attrs.append(id_attr)
        futures.append(parse_metadata(metas))
    results = list(zip(id_attrs, futures))
    citation_dict = dict(results)
    return citation_dict

def generate_full_toc(soup):
    toc = []
    stack = [(0, toc)]
    
    # Mapping of heading tags to their levels
    heading_tags = {'h1': 1, 'h2': 2, 'h3': 3, 'h4': 4, 'h5': 5}
    
    for tag in soup.find_all(heading_tags.keys()):
        level = heading_tags[tag.name]
        title = tag.get_text()
        
        # Ensure the stack has the correct level
        while stack and stack[-1][0] >= level:
            stack.pop()
        
        current_level = stack[-1][1]

        # Find the nearest enclosing section with an id
        section = tag.find_parent('section', id=True)
        section_id = section.get('id') if section else None
        
        # Create the new entry
        new_entry = {'title': title, 'id': section_id, 'subsections': []}
        
        current_level.append(new_entry)
        stack.append((level, new_entry['subsections']))
    
    return toc

def parse_text(local_text, tag):
    ignore_tags = ['a', 'figure', 'center', 'caption', 'td', 'h1', 'h2', 'h3', 'h4']
    # latexmlc
    ignore_tags += ['sup']
    max_math_length = 300000

    for child in tag.children:
        child_type = type(child)
        if child_type == bs4.element.NavigableString:
                txt = child.get_text()
                local_text.append(txt)

        elif child_type == bs4.element.Comment:
            continue
        elif child_type == bs4.element.Tag:

                if child.name in ignore_tags or (child.has_attr('class') and child['class'][0] == 'navigation'):
                    continue
                elif child.name == 'cite':
                    # add hrefs
                    hrefs = [a.get('href').strip('#') for a in child.find_all('a', class_='ltx_ref')]
                    local_text.append('~\cite{' + ', '.join(hrefs) + '}')
                elif child.name == 'img' and child.has_attr('alt'):
                    math_txt = child.get('alt')
                    if len(math_txt) < max_math_length:
                        local_text.append(math_txt)

                elif child.has_attr('class') and (child['class'][0] == 'ltx_Math' or child['class'][0] == 'ltx_equation'):
                    math_txt = child.get_text()
                    if len(math_txt) < max_math_length:
                        local_text.append(math_txt)

                elif child.name == 'section':
                    return
                else:
                    parse_text(local_text, child)
        else:
            raise RuntimeError('Unhandled type')

def clean_text(text):
    delete_items = ['=-1', '\t', u'\xa0', '[]', '()', 'mathbb', 'mathcal', 'bm', 'mathrm', 'mathit', 'mathbf', 'mathbfcal', 'textbf', 'textsc', 'langle', 'rangle', 'mathbin']
    for item in delete_items:
        text = text.replace(item, '')
    text = re.sub(' +', ' ', text)
    text = re.sub(r'[[,]+]', '', text)
    text = re.sub(r'\.(?!\d)', '. ', text)
    text = re.sub('bib. bib', 'bib.bib', text)
    return text

def remove_stop_word_sections_and_extract_text(toc, soup, stop_words=['references', 'acknowledgments', 'about this document', 'apopendix']):
    def has_stop_word(title, stop_words):
        return any(stop_word.lower() in title.lower() for stop_word in stop_words)
    
    def extract_text(entry, soup):
        section_id = entry['id']
        if section_id: # section_id
            section = soup.find(id=section_id)
            if section is not None:
                local_text = []
                parse_text(local_text, section)
                if local_text:
                    processed_text = clean_text(''.join(local_text))
                    entry['text'] = processed_text
        return 0 
    
    def filter_and_update_toc(entries):
        filtered_entries = []
        for entry in entries:
            if not has_stop_word(entry['title'], stop_words):
                # Get clean text
                extract_text(entry, soup)                
                entry['subsections'] = filter_and_update_toc(entry['subsections'])
                filtered_entries.append(entry)
        return filtered_entries
    
    return filter_and_update_toc(toc)

def parse_html(html_file):
    soup = bs4.BeautifulSoup(html_file, "lxml")
    # parse title
    title = soup.head.title.get_text().replace("\n", " ")
    # parse abstract
    abstract = soup.find(class_='ltx_abstract').get_text()
    # parse citation
    citation = soup.find(class_='ltx_biblist')
    citation_dict = create_dict_for_citation(citation)
    # generate the full toc without text
    sections = generate_full_toc(soup)
    # remove the sections need to skip and extract the text of the rest sections
    sections = remove_stop_word_sections_and_extract_text(sections, soup)
    document = {
        "title": title, 
        "abstract": abstract, 
        "sections": sections, 
        "references": citation_dict,
    }
    return document 

def extract_title_with_deepseek(reference_string):
    """
    使用 DeepSeek API 从引用字符串中提取论文标题
    """
    try:
        # 构建提示词
        prompt = f"""请从以下引用字符串中提取论文标题。只返回标题，不要包含任何其他信息。
引用字符串: {reference_string}"""
        
        # 调用 DeepSeek API
        response = requests.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "deepseek-chat",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1
            }
        )
        
        if response.status_code == 200:
            result = response.json()
            title = result["choices"][0]["message"]["content"].strip()
            return title
        else:
            warnings.warn(f"DeepSeek API 调用失败: {response.status_code}")
            return None
            
    except Exception as e:
        warnings.warn(f"提取标题时发生错误: {str(e)}")
        return None

def search_section_by_arxiv_id(entry_id, cite):
    rate_limit('arxiv')
    print(f"\n🔍 正在处理论文 {entry_id}")
    #warnings.warn("Using search_section_by_arxiv_id function may return wrong title because ar5iv parsing citation error. To solve this, You can prompt any LLM to extract the paper title from the reference string")
    assert re.match(r'^\d+\.\d+$', entry_id)
    url = f'https://ar5iv.labs.arxiv.org/html/{entry_id}'
    
    for attempt in range(3):
        try:
            print(f"  ⏳ 正在获取HTML内容 (尝试 {attempt + 1}/3)...")
            response = session.get(url, headers=HEADERS)
            if response.status_code == 200:
                html_content = response.text
                if not 'https://ar5iv.labs.arxiv.org/html' in html_content:
                    print("  ❌ 无效的ar5iv HTML文档")
                    warnings.warn(f'Invalid ar5iv HTML document: {url}')
                    return None
                else:
                    try:
                        print("  📄 正在解析HTML文档...")
                        document = parse_html(html_content)
                    except Exception as e:
                        print(f"  ❌ HTML解析失败: {str(e)}")
                        warnings.warn(f'Wrong format HTML document: {url}, error: {str(e)}')
                        return None
                    try:
                        print("  📑 正在提取章节内容...")
                        sections = get_2nd_section(document["sections"][0]["subsections"])
                    except Exception as e:
                        print(f"  ❌ 章节提取失败: {str(e)}")
                        warnings.warn(f'Get subsections error: {str(e)}')
                        return None
                    
                    print("  🔎 正在处理引用...")
                    sections2title = {}
                    total_sections = len(sections.items())
                    for idx, (k, v) in enumerate(sections.items(), 1):
                        print(f"    处理章节 {idx}/{total_sections}: {k[:50]}...")
                        k = " ".join(k.split("\n"))
                        sections2title[k] = set()
                        bibs = re.findall(cite, v, re.DOTALL)
                        
                        if bibs:
                            print(f"      找到 {len(bibs)} 个引用")
                            for bib_idx, bib in enumerate(bibs, 1):
                                bib = bib.split(",")
                                for b in bib:
                                    if b not in document["references"]:
                                        continue
                                    # 使用 DeepSeek API 提取标题
                                    reference_string = document["references"][b]["meta_string"]
                                    title = extract_title_with_deepseek(reference_string)
                                    if title:
                                        sections2title[k].add(title)
                                    else:
                                        # 如果 DeepSeek API 失败，使用原始标题
                                        sections2title[k].add(document["references"][b]["title"])
                        
                        if len(sections2title[k]) == 0:
                            print("      ⚠️ 本章节未找到有效引用")
                            del sections2title[k]
                        else:
                            sections2title[k] = list(sections2title[k])
                            print(f"      ✅ 成功提取 {len(sections2title[k])} 个引用标题")
                    
                    print(f"\n✅ 处理完成! 共处理 {len(sections2title)} 个章节")
                    return sections2title
        except Exception as e:
            print(f"  ❌ 请求失败 (尝试 {attempt + 1}/3): {str(e)}")
            warnings.warn(f"Request failed (attempt {attempt + 1}/3): {str(e)}")
            time.sleep(2 ** attempt)  # 指数退避
            continue
    
    print("\n❌ 处理失败，已达到最大重试次数")
    return None

def keep_letters(s):
    letters = [c for c in s if c.isalpha()]
    result = ''.join(letters)
    return result.lower()

def search_paper_by_arxiv_id(arxiv_id):
    """
    Search paper by arxiv id.
    :param arxiv_id: arxiv id of the paper
    :return: paper list
    """
    print(f"\n🔍 正在搜索论文 ID: {arxiv_id}")
    
    # 首先尝试从本地数据库搜索
    if arxiv_id in id2paper:
        print("  正在从本地数据库搜索...")
        title_key = keep_letters(id2paper[arxiv_id])
        if title_key in paper_db.namelist():
            print("✅ 在本地数据库中找到论文")
            with paper_db.open(title_key) as f:
                data = json.loads(f.read().decode("utf-8"))
            return {
                "arxiv_id": arxiv_id,
                "title": data["title"].replace("\n", " "),
                "abstract": data["abstract"],
                "sections": data["sections"],
                "source": 'SearchFrom:local_paper_db',
            }
        else:
            print("❌ 本地数据库未找到论文")

    # 如果本地没有，尝试从arxiv搜索
    print("  正在从arXiv搜索...")
    search = arxiv.Search(
        query = "",
        id_list = [arxiv_id],
        max_results = 10,
        sort_by = arxiv.SortCriterion.Relevance,
        sort_order = arxiv.SortOrder.Descending,
    )

    try:
        results = list(arxiv_client.results(search))
    except Exception as e:
        warnings.warn(f"Failed to search arxiv id: {arxiv_id}, error: {str(e)}")
        print("❌ arXiv搜索失败")
        return None

    res = None
    for arxiv_id_result in results:
        entry_id = arxiv_id_result.entry_id.split("/")[-1]
        entry_id = entry_id.split('v')[0]
        if entry_id == arxiv_id:
            print("✅ 在arXiv中找到论文")
            res = {
                "arxiv_id": arxiv_id,
                "title": arxiv_id_result.title.replace("\n", " "),
                "abstract": arxiv_id_result.summary.replace("\n", " "),
                "sections": "",
                "source": 'SearchFrom:arxiv',
            }
            break
    
    if res is None:
        print("❌ 未找到论文")
    return res
    
def search_arxiv_id_by_title(title):
    print(f"\n🔍 正在通过标题搜索论文: {title}")
    rate_limit('arxiv')
    
    for attempt in range(3):
        try:
            print(f"  尝试 {attempt + 1}/3...")
            
            search = arxiv.Search(query=f'ti:"{title}"', max_results=200)
            results = list(arxiv_client.results(search))
            
            if results:
                print(f"✅ 找到 {len(results)} 个匹配结果")
                for result in results:
                    title_find = result.title.lower().strip('.').replace(' ', '').replace('\n', '')
                    title_search = title.lower().strip('.').replace(' ', '').replace('\n', '')
                    if title_find == title_search:
                        print(f"✅ 找到完全匹配的论文 ID: {result.get_short_id()}")
                        return result.get_short_id()
                print("❌ 未找到完全匹配的论文")
                return None
            
            print("❌ 未找到任何结果")
            return None
            
        except Exception as e:
            warnings.warn(f"API请求失败 (attempt {attempt + 1}/3): {str(e)}")
            time.sleep(2 ** attempt)
            continue
            
    print("❌ 搜索失败")
    return None

def search_paper_by_title(title):
    """
    Search paper by title.
    :param title: title of the paper
    :return: paper list
    """
    print(f"\n🔍 正在搜索论文: {title}")
    title_id = search_arxiv_id_by_title(title)
    if title_id is None:
        return None
    title_id = title_id.split('v')[0]
    return search_paper_by_arxiv_id(title_id)

def get_subsection(sections):
    res = {}
    for section in sections:
        if "text" in section and section["text"].strip() != "":
            res[section["title"].strip()] = section["text"].strip()
        subsections = get_subsection(section["subsections"])
        for k, v in subsections.items():
            res[k] = v
    return res

def get_1st_section(sections):
    res = {}
    for section in sections:
        subsections = get_subsection(section["subsections"])
        if "text" in section and section["text"].strip() != "" or len(subsections) > 0:
            if "text" in section and section["text"].strip() != "":
                res[section["title"].strip()] = section["text"].strip()
            else:
                res[section["title"].strip()] = ""
            for k, v in subsections.items():
                res[section["title"].strip()] += v.strip()
    res_new = {}
    for k, v in res.items():
        if "appendix" not in k.lower():
            res_new[" ".join(k.split("\n")).strip()] = v
    return res_new

def get_2nd_section(sections):
    res = {}
    for section in sections:
        subsections = get_1st_section(section["subsections"])
        if "text" in section and section["text"].strip() != "":
            if "text" in section and section["text"].strip() != "":
                res[section["title"].strip()] = section["text"].strip()
        for k, v in subsections.items():
            res[section["title"].strip() + " " + k.strip()] = v.strip()
    res_new = {}
    for k, v in res.items():
        if "appendix" not in k.lower():
            res_new[" ".join(k.split("\n")).strip()] = v
    return res_new

def cal_micro(pred_set, label_set):
    if len(label_set) == 0:
        return 0, 0, 0

    if len(pred_set) == 0:
        return 0, 0, len(label_set)

    tp = len(pred_set & label_set)
    fp = len(pred_set - label_set)
    fn = len(label_set - pred_set)

    assert tp + fn == len(label_set)
    assert len(label_set) != 0
    return tp, fp, fn

if __name__ == "__main__":
    print(search_section_by_arxiv_id("2307.00235", r"~\\cite\{(.*?)\}"))
    # print(search_paper_by_arxiv_id("2307.00235"))
    # print(search_paper_by_title("A hybrid approach to CMB lensing reconstruction on all-sky intensity maps"))
