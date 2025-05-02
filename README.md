# Paper Search and Analysis System

This project implements a web-based academic paper search and analysis system based on **pasa-7b-selector** and **DeepSeek API**.  
It first utilizes `pasa-7b-selector` to **score and filter candidate papers** from a local database,  
and then leverages the **DeepSeek API** to perform **relevance analysis** between the input query and the filtered papers.

---


## Model Preparation
Download models [`pasa-7b-selector`](https://huggingface.cobytedance-research/pasa-7b-selector) and [`all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) and save them in the folder.
```
.
└── checkpoints/           
   └── pasa-7b-selector
└── all-MiniLM-L6-v2      
```

---

## How to Use

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Server

```bash
git clone git@github.com:hyc2026/transformers.git
cd transformers
pip install -e .
cd ..
pip install -r requirements.txt

```
You need to first apply for a Deepseek API key at serper.dev, and replace 'your Deepseek keys' in utils.py.
```bash
python app.py
```
### 3. Access the Web Interface

Open your browser and go to:

```
http://localhost:6006
```





