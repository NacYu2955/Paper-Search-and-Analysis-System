# PASA 论文检索系统

一个基于深度学习的智能论文检索系统，支持实时搜索、多轮对话、PDF管理和引用分析等功能。

## 🚀 系统特性

- **智能搜索**: 基于语义向量相似度和大模型理解的混合检索
- **实时推送**: WebSocket实时推送搜索结果
- **多语言支持**: 支持中英文查询和自动翻译
- **PDF管理**: 支持PDF文件上传、存储和查看
- **引用分析**: 智能分析论文引用关系和对话
- **拼写检查**: 自动修正查询中的拼写错误
- **宽泛查询**: 自动将长查询转换为宽泛查询

## 🏗️ 系统架构

### 核心组件

1. **前端**: HTML5 + JavaScript + Bootstrap + Socket.IO
2. **后端**: Flask + SQLite + WebSocket
3. **AI模型**: 
   - SentenceTransformer (all-MiniLM-L6-v2)
   - PASA-7B-Selector (论文筛选模型)
   - DeepSeek API (查询生成和对话)
4. **存储**: 腾讯云COS + 本地SQLite
5. **网络**: 花生壳内网穿透

## 🔧 外部接口和平台

### AI模型和API

1. **DeepSeek API**
   - 用途: 查询生成、多轮对话、引用分析
   - 配置: `DEEPSEEK_API_KEY`, `DEEPSEEK_BASE_URL`
   - 开销: 按token数量计费 (输入¥0.5-2/百万tokens，输出¥8/百万tokens)

2. **SentenceTransformer (all-MiniLM-L6-v2)**
   - 用途: 文本向量化
   - 部署: 本地加载
   - 开销: 免费开源模型

3. **PASA-7B-Selector**
   - 用途: 论文相关性评分
   - 部署: 本地加载 (7B参数)
   - 开销: 免费开源模型

### 云服务

1. **腾讯云COS (对象存储)**
   - 用途: PDF文件存储
   - 配置: `COS_SECRET_ID`, `COS_SECRET_KEY`, `COS_REGION`, `COS_BUCKET_NAME`
   - 开销: 对象存储资源包¥9.77/年 + 流量费¥3.5/GB

2. **花生壳内网穿透**
   - 用途: 公网访问
   - 部署: 本地服务
   - 域名: dicalab-paper.com
   - 开销: 专业版 ¥398/年

### 硬件要求

- **GPU**: NVIDIA RTX 3090 (24GB显存)
- **内存**: 建议32GB+ RAM
- **存储**: 建议100GB+ SSD
- **网络**: 稳定的互联网连接

## 💰 系统开销

### 硬件开销

1. **GPU显存使用**
   - PASA-7B-Selector: ~14GB
   - SentenceTransformer: ~2GB
   - 总计: ~16GB (3090 24GB足够)

2. **内存使用**
   - 模型加载: ~8GB
   - 论文向量: ~2GB
   - 系统运行: ~4GB
   - 总计: ~14GB

3. **存储开销**
   - 模型文件: ~15GB
   - 论文数据库: ~100MB
   - PDF文件: 按需增长
   - 总计: ~15GB基础

### 云服务开销

1. **DeepSeek API**
   - 输入 (缓存命中): ¥0.5/百万tokens
   - 输入 (缓存未命中): ¥2/百万tokens
   - 输出: ¥8/百万tokens

2. **腾讯云COS**
   - 对象存储资源包: ¥9.77/年
   - 流量费用: ¥3.5/GB (按需)

3. **花生壳**
   - 专业版: ¥398/年
   - 域名服务: ¥99/年 (dicalab-paper.com)
   - HTTP/HTTPS映射服务: ¥10/年

### 已购买服务费用

- **花生壳专业版**: ¥398/年
- **域名服务 (dicalab-paper.com)**: ¥99/年
- **HTTP/HTTPS映射服务**: ¥10/年
- **腾讯云COS对象存储资源包**: ¥9.77/年
- **总计固定年费**: ¥516.77/年

### 按需付费服务

- **DeepSeek API**: 按token使用量计费
- **腾讯云COS流量费**: 按实际流量计费
- **优化建议**: 缓存常用查询结果，提高DeepSeek缓存命中率

## 🔄 系统维护

### 服务器重启

- **频率**: 每30天自动重启
- **原因**: 释放内存、更新系统、清理缓存
- **影响**: 服务中断约2-3分钟

## 📦 安装部署

### 安装步骤

1. **克隆项目**
```bash
git clone <repository-url>
cd pasa
```

2. **安装Python依赖**
```bash
pip install -r requirements.txt
```

3. **下载模型文件**
```bash
# 下载SentenceTransformer模型
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# 下载PASA-7B-Selector模型 (需要手动下载到checkpoints目录)
```

4. **配置环境变量**
```bash
export DEEPSEEK_API_KEY="your-api-key"
export COS_SECRET_ID="your-cos-secret-id"
export COS_SECRET_KEY="your-cos-secret-key"
```

5. **初始化数据库**
```bash
python -c "from app import init_db; init_db()"
```

6. **启动服务**
```bash
python start.py
```

## 🚀 快速开始

1. **访问系统**: http://dicalab-paper.com
2. **输入查询**: 支持中英文自然语言查询
3. **查看结果**: 实时推送高相关性论文
4. **深度分析**: 点击论文查看详情和引用分析

## 📊 性能指标

- **搜索响应时间**: 2-5秒
- **实时推送延迟**: <1秒
- **并发用户数**: 支持10-20个并发用户
- **论文库规模**: 支持10万+论文
- **GPU利用率**: 平均60-80%

## 🔧 配置说明

### 主要配置文件

- `config.py`: 系统配置
- `agent_prompt.json`: AI提示模板
- `start.py`: 启动脚本

### 环境变量

```bash
# DeepSeek API
DEEPSEEK_API_KEY=sk-xxx
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1

# 腾讯云COS
COS_SECRET_ID=AKIDxxx
COS_SECRET_KEY=xxx
COS_REGION=ap-location
COS_BUCKET_NAME=bucket_name-xxx

# 系统配置
USE_COS_STORAGE=true
HOST=0.0.0.0
PORT=6006
```

## 🤝 贡献

欢迎提交Issue和Pull Request来改进项目。

## 📞 联系方式

如有问题或建议，请通过以下方式联系：
- 邮箱: [your-email]
- GitHub: [your-github]

---

**注意**: 请确保在使用前正确配置所有API密钥和云服务配置。 
