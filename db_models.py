from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import re

db = SQLAlchemy()

class PaperSubmission(db.Model):
    """论文提交模型
    
    用于存储用户提交的论文信息，包括基本信息、提交状态和审核信息。
    
    Attributes:
        id (int): 主键ID
        title (str): 论文标题
        authors (str): 作者列表，以逗号分隔
        abstract (str): 论文摘要
        year (int): 发表年份
        journal (str): 期刊/会议名称
        doi (str): DOI标识符
        keywords (str): 关键词，以逗号分隔
        url (str): 论文链接
        status (str): 审核状态 (pending/approved/rejected)
        submitted_at (datetime): 提交时间
        reviewed_at (datetime): 审核时间
        reviewer_notes (str): 审核意见
        submitted_by (str): 提交者邮箱
        reviewer (str): 审核人
        version (int): 提交版本号
        last_modified (datetime): 最后修改时间
        type (str): BibTeX类型
        citation_key (str): BibTeX引用key
        booktitle (str): 会议名称
        organization (str): 组织机构
        volume (str): 卷号
        number (str): 期号
        pages (str): 页码
        publisher (str): 出版社
        citations (str): 原始citations字段，JSON字符串
    """
    
    __tablename__ = 'paper_submissions'
    
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(500), nullable=False, index=True)
    authors = db.Column(db.String(1000), nullable=False)
    abstract = db.Column(db.Text)
    year = db.Column(db.Integer, index=True)
    journal = db.Column(db.String(500), index=True)
    doi = db.Column(db.String(100), unique=True, index=True)
    keywords = db.Column(db.String(500))
    url = db.Column(db.String(500))
    status = db.Column(db.String(20), default='pending', index=True)  # pending, approved, rejected
    submitted_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    reviewed_at = db.Column(db.DateTime)
    reviewer_notes = db.Column(db.Text)
    submitted_by = db.Column(db.String(100), index=True)  # 提交者邮箱
    reviewer = db.Column(db.String(100))  # 审核人
    version = db.Column(db.Integer, default=1)  # 提交版本号
    last_modified = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    type = db.Column(db.String(50))              # BibTeX类型
    citation_key = db.Column(db.String(100))     # BibTeX引用key
    booktitle = db.Column(db.String(500))        # 会议名称
    organization = db.Column(db.String(200))     # 组织机构
    volume = db.Column(db.String(50))
    number = db.Column(db.String(50))
    pages = db.Column(db.String(50))
    publisher = db.Column(db.String(200))
    citations = db.Column(db.Text)               # 原始citations字段，JSON字符串
    
    def __init__(self, **kwargs):
        super(PaperSubmission, self).__init__(**kwargs)
        self.validate()
    
    def validate(self):
        """验证提交的论文信息
        
        Raises:
            ValueError: 当验证失败时抛出异常
        """
        if not self.title or len(self.title.strip()) < 5:
            raise ValueError("论文标题不能为空且长度必须大于5个字符")
            
        if not self.authors or len(self.authors.strip()) < 2:
            raise ValueError("作者信息不能为空且长度必须大于2个字符")
            
        if not self.abstract or len(self.abstract.strip()) < 50:
            raise ValueError("摘要不能为空且长度必须大于50个字符")
            
        if self.year and (self.year < 1900 or self.year > datetime.now().year):
            raise ValueError("发表年份无效")
            
        if self.doi and not self._validate_doi(self.doi):
            raise ValueError("DOI格式无效")
            
        if self.submitted_by and not self._validate_email(self.submitted_by):
            raise ValueError("提交者邮箱格式无效")
    
    def _validate_doi(self, doi):
        """验证DOI格式
        
        Args:
            doi (str): 要验证的DOI
            
        Returns:
            bool: 是否为有效的DOI格式
        """
        doi_pattern = r'^10\.\d{4,9}/[-._;()/:A-Z0-9]+$'
        return bool(re.match(doi_pattern, doi, re.IGNORECASE))
    
    def _validate_email(self, email):
        """验证邮箱格式
        
        Args:
            email (str): 要验证的邮箱
            
        Returns:
            bool: 是否为有效的邮箱格式
        """
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(email_pattern, email))
    
    def approve(self, reviewer, notes=None):
        """通过论文审核
        
        Args:
            reviewer (str): 审核人
            notes (str, optional): 审核意见
        """
        self.status = 'approved'
        self.reviewer = reviewer
        self.reviewer_notes = notes
        self.reviewed_at = datetime.utcnow()
    
    def reject(self, reviewer, notes):
        """拒绝论文审核
        
        Args:
            reviewer (str): 审核人
            notes (str): 拒绝原因
        """
        self.status = 'rejected'
        self.reviewer = reviewer
        self.reviewer_notes = notes
        self.reviewed_at = datetime.utcnow()
    
    def update(self, **kwargs):
        """更新论文信息
        
        Args:
            **kwargs: 要更新的字段和值
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self.version += 1
        self.last_modified = datetime.utcnow()
        self.validate()
    
    def to_dict(self):
        """将模型转换为字典
        
        Returns:
            dict: 包含所有字段的字典
        """
        return {
            'id': self.id,
            'title': self.title,
            'authors': self.authors,
            'abstract': self.abstract,
            'year': self.year,
            'journal': self.journal,
            'doi': self.doi,
            'keywords': self.keywords,
            'url': self.url,
            'status': self.status,
            'submitted_at': self.submitted_at.isoformat() if self.submitted_at else None,
            'reviewed_at': self.reviewed_at.isoformat() if self.reviewed_at else None,
            'reviewer_notes': self.reviewer_notes,
            'submitted_by': self.submitted_by,
            'reviewer': self.reviewer,
            'version': self.version,
            'last_modified': self.last_modified.isoformat() if self.last_modified else None,
            'type': self.type,
            'citation_key': self.citation_key,
            'booktitle': self.booktitle,
            'organization': self.organization,
            'volume': self.volume,
            'number': self.number,
            'pages': self.pages,
            'publisher': self.publisher,
            'citations': self.citations
        }
    
    def __repr__(self):
        return f'<PaperSubmission {self.id}: {self.title}>'

class PendingPaper(db.Model):
    """待审核的论文"""
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(500), nullable=False)
    abstract = db.Column(db.Text)
    authors = db.Column(db.String(500))
    year = db.Column(db.Integer)
    journal = db.Column(db.String(200))
    doi = db.Column(db.String(100))
    submitted_by = db.Column(db.String(100))  # 提交者邮箱
    submitted_at = db.Column(db.DateTime, default=datetime.utcnow)
    status = db.Column(db.String(20), default='pending')  # pending, approved, rejected
    reviewed_by = db.Column(db.String(100))  # 审核者邮箱
    reviewed_at = db.Column(db.DateTime)
    review_comment = db.Column(db.Text)

class User(db.Model):
    """用户模型"""
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    is_admin = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Paper(db.Model):
    """已审核通过的论文"""
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(500), nullable=False)
    abstract = db.Column(db.Text)
    authors = db.Column(db.String(500))
    year = db.Column(db.Integer)
    journal = db.Column(db.String(200))
    doi = db.Column(db.String(100))
    added_at = db.Column(db.DateTime, default=datetime.utcnow)
    added_by = db.Column(db.String(100))  # 添加者邮箱
    embedding = db.Column(db.LargeBinary)  # 存储论文的向量表示
    type = db.Column(db.String(50))              # BibTeX类型
    citation_key = db.Column(db.String(100))     # BibTeX引用key
    booktitle = db.Column(db.String(500))        # 会议名称
    organization = db.Column(db.String(200))     # 组织机构
    volume = db.Column(db.String(50))
    number = db.Column(db.String(50))
    pages = db.Column(db.String(50))
    publisher = db.Column(db.String(200))
    citations = db.Column(db.Text)               # 原始citations字段，JSON字符串