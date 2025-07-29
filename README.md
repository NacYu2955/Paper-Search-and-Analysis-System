# Paper Search and Analysis System

An intelligent paper retrieval system based on deep learning, supporting real-time search, multi-turn dialogue, PDF management, and citation analysis.

## 🌐 Online Access

**Project Website**: [http://dicalab-paper.com](http://dicalab-paper.com)

---

## 🚀 System Features

- **Intelligent Search**: Hybrid retrieval based on semantic vector similarity and large model understanding
- **Real-time Push**: WebSocket real-time push of search results
- **Multi-language Support**: Support for Chinese and English queries with automatic translation
- **PDF Management**: Support for PDF file upload, storage, and viewing
- **Citation Analysis**: Intelligent analysis of paper citation relationships and dialogue
- **Spell Check**: Automatic correction of spelling errors in queries
- **Broad Query**: Automatic conversion of long queries to broad queries

## 🏗️ System Architecture

### Core Components

1. **Frontend**: HTML5 + JavaScript + Bootstrap + Socket.IO
2. **Backend**: Flask + SQLite + WebSocket
3. **AI Models**: 
   - SentenceTransformer (all-MiniLM-L6-v2)
   - PASA-7B-Selector (Paper Selection Model)
   - DeepSeek API (Query Generation and Dialogue)
4. **Storage**: Tencent Cloud COS + Local SQLite
5. **Network**: Peanut Shell Intranet Penetration

### Database Structure

The system uses SQLite database to store paper information, with the following main fields:

| Field Name | Type | Description |
|------------|------|-------------|
| id | INTEGER | Paper unique identifier |
| title | TEXT | Paper title |
| authors | TEXT | Author information |
| abstract | TEXT | Paper abstract |
| year | INTEGER | Publication year |
| journal | TEXT | Journal name |
| doi | TEXT | DOI identifier |
| status | TEXT | Paper status |
| submitter | TEXT | Submitter |
| review_comment | TEXT | Review comments |
| reviewed_by | TEXT | Reviewer |
| submitted_at | DATETIME | Submission time |
| reviewed_at | DATETIME | Review time |
| type | TEXT | Paper type |
| citation_key | TEXT | Citation key |
| booktitle | TEXT | Book title/Conference name |
| organization | TEXT | Organization/Institution |
| volume | TEXT | Volume number |
| number | TEXT | Issue number |
| pages | TEXT | Page numbers |
| publisher | TEXT | Publisher |
| citations | TEXT | Citation information |
| pdf_file_path | TEXT | PDF file path |

## 🔧 External Interfaces and Platforms

### AI Models and APIs

1. **DeepSeek API**
   - Purpose: Query generation, multi-turn dialogue, citation analysis
   - Configuration: `DEEPSEEK_API_KEY`, `DEEPSEEK_BASE_URL`
   - Cost: Billed by token usage (Input ¥0.5-2/million tokens, Output ¥8/million tokens)

2. **SentenceTransformer (all-MiniLM-L6-v2)**
   - Purpose: Text vectorization
   - Deployment: Local loading
   - Cost: Free open-source model

3. **PASA-7B-Selector**
   - Purpose: Paper relevance scoring
   - Deployment: Local loading (7B parameters)
   - Cost: Free open-source model

### Cloud Services

1. **Tencent Cloud COS (Object Storage)**
   - Purpose: PDF file storage
   - Configuration: `COS_SECRET_ID`, `COS_SECRET_KEY`, `COS_REGION`, `COS_BUCKET_NAME`
   - Cost: Object storage resource package ¥9.77/year + traffic fee ¥3.5/GB

2. **Peanut Shell Intranet Penetration**
   - Purpose: Public network access
   - Deployment: Local service
   - Domain: dicalab-paper.com
   - Cost: Professional version ¥398/year

### Hardware Requirements

- **GPU**: NVIDIA RTX 3090 (24GB VRAM)
- **Memory**: Recommended 32GB+ RAM
- **Storage**: Recommended 100GB+ SSD
- **Network**: Stable internet connection

## 💰 System Costs

### Hardware Costs

1. **GPU Memory Usage**
   - PASA-7B-Selector: ~14GB
   - SentenceTransformer: ~2GB
   - Total: ~16GB (3090 24GB sufficient)

2. **Memory Usage**
   - Model loading: ~8GB
   - Paper vectors: ~2GB
   - System operation: ~4GB
   - Total: ~14GB

3. **Storage Costs**
   - Model files: ~15GB
   - Paper database: ~100MB
   - PDF files: Grows as needed
   - Total: ~15GB base

### Cloud Service Costs

1. **DeepSeek API**
   - Input (cache hit): ¥0.5/million tokens
   - Input (cache miss): ¥2/million tokens
   - Output: ¥8/million tokens

2. **Tencent Cloud COS**
   - Object storage resource package: ¥9.77/year
   - Traffic fee: ¥3.5/GB (as needed)

3. **Peanut Shell**
   - Professional version: ¥398/year (dicalab-paper.com domain)
   - Domain service: ¥99/year (dicalab-paper.com)
   - HTTP/HTTPS mapping service: ¥10/year

### Purchased Service Costs

- **Peanut Shell Professional**: ¥398/year
- **Domain Service (dicalab-paper.com)**: ¥99/year
- **HTTP/HTTPS Mapping Service**: ¥10/year
- **Tencent Cloud COS Object Storage Resource Package**: ¥9.77/year
- **Total Fixed Annual Fee**: ¥516.77/year

### Pay-as-you-go Services

- **DeepSeek API**: Billed by token usage
- **Tencent Cloud COS Traffic Fee**: Billed by actual traffic
- **Optimization Suggestion**: Cache common query results, improve DeepSeek cache hit rate

## 🔄 System Maintenance

### Server Restart

- **Frequency**: Automatic restart every 30 days
- **Reason**: Release memory, update system, clean cache
- **Impact**: Service interruption about 2-3 minutes

## 📦 Installation and Deployment

### Data Preparation

**Important**: This project does not include pre-built database files. Users need to:
- Prepare paper data themselves
- Or use the system's upload function to add papers
- Database file `papers.db` will be automatically created on first run

### Installation Steps

1. **Clone Project**
```bash
git clone https://github.com/NacYu2955/Paper-Search-and-Analysis-System.git
```

2. **Install Python Dependencies**
```bash
pip install -r requirements.txt
```

3. **Download Model Files**
```bash
# Download SentenceTransformer model
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Download PASA-7B-Selector model (need to manually download to checkpoints directory)
```

4. **Configure Environment Variables**
```bash
export DEEPSEEK_API_KEY="your-api-key"
export COS_SECRET_ID="your-cos-secret-id"
export COS_SECRET_KEY="your-cos-secret-key"
```

5. **Initialize Database**
```bash
python -c "from app import init_db; init_db()"
```

6. **Start Service**
```bash
python start.py
```

**Note**: This project does not include pre-built database files. Users need to:
- Prepare paper data themselves
- Or use the system's upload function to add papers
- Database file `papers.db` will be automatically created on first run

## 🚀 Quick Start

1. **Access System**: [http://dicalab-paper.com](http://dicalab-paper.com)
2. **Input Query**: Support Chinese and English natural language queries
3. **View Results**: Real-time push of high-relevance papers
4. **Deep Analysis**: Click papers to view details and citation analysis

## 📊 Performance Metrics

- **Search Response Time**: 2-5 seconds
- **Real-time Push Delay**: <1 second
- **Concurrent Users**: Support 10-20 concurrent users
- **Paper Library Scale**: Support 100,000+ papers
- **GPU Utilization**: Average 60-80%

## 🔧 Configuration

### Main Configuration Files

- `config.py`: System configuration
- `agent_prompt.json`: AI prompt templates
- `start.py`: Startup script

### Environment Variables

```bash
# DeepSeek API
DEEPSEEK_API_KEY=your_deepseek_api_key
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1

# Tencent Cloud COS
COS_SECRET_ID=your_cos_secret_id
COS_SECRET_KEY=your_cos_secret_key
COS_REGION=ap-location
COS_BUCKET_NAME=your_bucket_name

# System Configuration
USE_COS_STORAGE=true
HOST=0.0.0.0
PORT=6006
```

## 🤝 Contributing

Welcome to submit Issues and Pull Requests to improve the project.

## 📞 Contact

For questions or suggestions, please contact via:
- Email: [xinyu9026@gmail.com]

---

**Note**: Please ensure all API keys and cloud service configurations are correctly set up before use. 
