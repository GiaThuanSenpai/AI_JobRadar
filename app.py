import os
import logging
import time
import re
import json
import traceback
import numpy as np
import pandas as pd
import torch
from datetime import datetime, timedelta
from functools import wraps, lru_cache
from threading import Thread, Lock
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from flask import Flask, request, jsonify
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
import psutil
import PyPDF2
import docx2txt
from dotenv import load_dotenv
import google.generativeai as genai
import redis
import hashlib
import pickle
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from cryptography.fernet import Fernet
import base64
# Thiết lập mã hóa
os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Tải biến môi trường
load_dotenv()

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Khởi tạo Flask
app = Flask(__name__)
# Cấu hình CORS
CORS(app, resources={
    r"/*": {
        "origins": ["http://localhost:3000"],
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

# Cấu hình file paths
SEARCH_HISTORY_FILEPATH = os.environ.get("SEARCH_HISTORY_FILEPATH", os.path.join(os.path.dirname(os.path.dirname(__file__)), "search.csv"))
JOBS_FILEPATH = os.environ.get("JOBS_FILEPATH", os.path.join(os.path.dirname(os.path.dirname(__file__)), "job_post.csv"))
# Cấu hình Gemini API
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    logger.warning("GEMINI_API_KEY không được thiết lập.")
MODEL_NAME = "gemini-2.0-flash"

# Khởi tạo mô hình SentenceTransformer
embedding_model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
embedding_model.to(DEVICE)
embedding_model.eval()
logger.info(f"Mô hình SentenceTransformer được tải trên {DEVICE}")

# Khởi tạo Redis cache
REDIS_URL = os.environ.get("REDIS_URL")
redis_client = None
try:
    if REDIS_URL:
        redis_client = redis.from_url(REDIS_URL)
        redis_client.ping()
        logger.info("Kết nối Redis thành công")
    else:
        logger.warning("REDIS_URL không được thiết lập, sử dụng bộ nhớ cache trong ứng dụng")
except Exception as e:
    logger.error(f"Không thể kết nối Redis: {e}")
    redis_client = None

# Cache trong bộ nhớ nếu không có Redis
in_memory_cache = {}

# Biến cấu hình tối ưu hóa
MAX_WORKERS = 8
PRECOMPUTE_EMBEDDINGS = True
EMBEDDINGS_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "embeddings_cache.pkl")
EMBEDDING_REFRESH_HOURS = 24
SEMANTIC_SEARCH_TIMEOUT = 30
USE_JOB_INDEXING = True
ENABLE_PREFILTERING = True
MAX_JOBS_FOR_GEMINI = 50
EMBEDDING_CACHE_SIZE = 1000
CACHE_VERSION = "1.0"  # Thêm version để quản lý cache

# Biến toàn cục
job_index = {}
job_embeddings = {}
last_embedding_update = None
JOB_VECTOR_CACHE = {}
jobs = []
search_history = []
csv_write_lock = Lock()
csv_read_lock = Lock()
model_lock = Lock()

# Danh sách stop words tiếng Việt
VIETNAMESE_STOP_WORDS = [
    "và", "của", "là", "các", "cho", "trong", "tại", "được", "với", "một",
    "những", "để", "từ", "có", "không", "người", "này", "đã", "ra", "trên",
    "bằng", "vào", "hay", "thì", "đó", "nào", "ở", "lại", "còn", "như",
    "thành", "phố", "hồ", "chí", "minh", "title", "tuyển", "dụng", "việc",
    "làm", "công", "ty", "tại", "lương", "cao", "hấp", "dẫn", "dịch", "vụ",
    "quản", "lý", "hệ", "thống"
]

# Từ khóa ngành nghề
DRIVER_KEYWORDS = [
    'tài xế', 'lái xe', 'lái xe tải', 'tài xế xe tải', 'lái xe container',
    'giao hàng', 'vận chuyển'
]
TECH_KEYWORDS = [
    'công nghệ ô tô', 'sửa chữa ô tô', 'kiểm tra ghế lái', 'kiểm tra phanh xe',
    'kỹ sư ô tô', 'thiết kế ô tô', 'bảo dưỡng ô tô', 'truyền động',
    'kỹ thuật viên ô tô', 'sửa chữa truyền động', 'xe điện', 'phụ tùng ô tô'
]
ECOMMERCE_KEYWORDS = [
    'thương mại điện tử', 'mua sắm trực tuyến', 'sàn thương mại',
    'thanh toán điện tử', 'marketplace', 'e-commerce'
]
MARKETING_KEYWORDS = [
    'quảng cáo', 'tiếp thị số', 'truyền thông xã hội', 'seo', 'sem', 'thương hiệu'
]
IT_HARDWARE_KEYWORDS = [
    'phần cứng', 'sửa chữa máy tính', 'hạ tầng mạng', 'server', 'linh kiện điện tử'
]
IT_SOFTWARE_KEYWORDS = [
    'lập trình', 'phát triển phần mềm', 'kỹ sư phần mềm', 'trí tuệ nhân tạo',
    'ứng dụng di động', 'java', 'python', 'javascript', 'devops', 'database',
    'cloud', 'web developer', 'mobile developer', 'software engineer', 'api',
    'microservices', 'blockchain', 'machine learning', 'fullstack', 'backend',
    'frontend', 'phần mềm'
]
HOSPITALITY_KEYWORDS = [
    'nhà hàng', 'khách sạn', 'đầu bếp', 'phục vụ', 'lễ tân', 'du lịch'
]
DESIGN_KEYWORDS = [
    'thiết kế đồ họa', 'ui/ux', 'minh họa', 'thiết kế bao bì', 'nhận diện thương hiệu'
]
MECHANICAL_KEYWORDS = [
    'cơ khí', 'bảo trì máy móc', 'thiết kế cơ khí', 'tự động hóa', 'cad/cam'
]
BUSINESS_KEYWORDS = [
    'bán hàng', 'phát triển thị trường', 'chiến lược kinh doanh', 'đàm phán hợp đồng'
]
EDUCATION_KEYWORDS = [
    'giảng dạy', 'giáo viên', 'đào tạo', 'giáo dục trực tuyến', 'phát triển chương trình'
]
CONSTRUCTION_KEYWORDS = [
    'kiến trúc', 'xây dựng', 'thiết kế kiến trúc', 'giám sát công trình', 'quy hoạch đô thị'
]
FINANCE_KEYWORDS = [
    'tài chính', 'ngân hàng', 'kế toán tài chính', 'phân tích tài chính', 'đầu tư'
]
TELECOM_KEYWORDS = [
    'viễn thông', 'mạng di động', 'cáp quang', '5g', 'kỹ thuật viễn thông', "mạng"
]
HEALTHCARE_KEYWORDS = [
    'bác sĩ', 'điều dưỡng', 'dược sĩ', 'chăm sóc bệnh nhân', 'y tế công cộng'
]
LOGISTICS_KEYWORDS = [
    'vận tải', 'chuỗi cung ứng', 'kho bãi', 'giao nhận hàng hóa', 'xuất nhập khẩu', 'hải quan', 'logistics'
]
ACCOUNTING_KEYWORDS = [
    'kế toán', 'kiểm toán', 'báo cáo tài chính', 'thuế', 'quản lý ngân sách'
]
MANUFACTURING_KEYWORDS = [
    'sản xuất', 'vận hành máy móc', 'kiểm soát chất lượng', 'sản xuất'
]
LEGAL_KEYWORDS = [
    'luật sư', 'tư vấn pháp lý', 'hợp đồng', 'pháp chế', 'sở hữu trí tuệ'
]
TRANSLATION_KEYWORDS = [
    'phiên dịch', 'dịch thuật', 'thông dịch', 'đa ngôn ngữ', 'hiệu đính'
]
EMBEDDED_IOT_KEYWORDS = [
    'hệ thống nhúng', 'iot', 'cảm biến', 'thiết bị thông minh', 'firmware'
]
RELATED_INTENTS = {
    'it_hardware': ['it_software', 'mechanical', 'embedded_iot'],
    'it_software': ['it_hardware', 'embedded_iot'],
    'mechanical': ['it_hardware', 'manufacturing'],
    'embedded_iot': ['it_hardware', 'it_software'],
}

# Hàm tiện ích
def log_resource_usage():
    process = psutil.Process()
    mem = process.memory_info().rss / 1024 / 1024
    cpu = psutil.cpu_percent(interval=1)
    logger.info(f"Sử dụng bộ nhớ: {mem:.2f} MB | CPU: {cpu:.2f}%")

def normalize_keyword(keyword):
    keyword = keyword.lower().strip()
    no_space = re.sub(r'\s+', '', keyword)
    return keyword, no_space

def get_cache_key(query, filters=None):
    if filters is None:
        filters = {}
    serialized = json.dumps({"query": query, "filters": filters}, sort_keys=True)
    return hashlib.md5(serialized.encode()).hexdigest()

def set_cache(key, value, ttl=3600):
    try:
        if redis_client:
            redis_client.setex(key, ttl, json.dumps(value))
            return True
        else:
            in_memory_cache[key] = {
                "data": value,
                "timestamp": time.time(),
                "ttl": ttl
            }
            return True
    except Exception as e:
        logger.error(f"Lỗi khi lưu cache: {e}")
        return False

def get_cache(key):
    try:
        if redis_client:
            data = redis_client.get(key)
            if data:
                return json.loads(data)
            return None
        else:
            if key in in_memory_cache:
                cache_item = in_memory_cache[key]
                now = time.time()
                if now - cache_item["timestamp"] < cache_item["ttl"]:
                    return cache_item["data"]
                else:
                    del in_memory_cache[key]
            return None
    except Exception as e:
        logger.error(f"Lỗi khi truy xuất cache: {e}")
        return None

def clean_expired_cache():
    if not redis_client:
        now = time.time()
        keys_to_delete = []
        for key, cache_item in in_memory_cache.items():
            if now - cache_item["timestamp"] >= cache_item["ttl"]:
                keys_to_delete.append(key)
        for key in keys_to_delete:
            del in_memory_cache[key]

@lru_cache(maxsize=EMBEDDING_CACHE_SIZE)
def get_text_embedding(text):
    try:
        return embedding_model.encode(text, device=DEVICE, normalize_embeddings=True)
    except Exception as e:
        logger.error(f"Lỗi khi tạo embedding: {e}")
        return None

def get_job_embedding(job):
    try:
        job_id = job.get('id') or job.get('jobId') or job.get('postId')
        if not job_id:
            job_content = f"{job.get('title', '')}{job.get('description', '')}{job.get('requirements', '')}"
            job_id = hashlib.md5(job_content.encode()).hexdigest()
        
        if job_id in job_embeddings:
            return job_embeddings[job_id]
        
        job_text = f"{job.get('title', '')} {job.get('description', '')} {job.get('requirements', '')}"
        embedding = embedding_model.encode(job_text, device=DEVICE, normalize_embeddings=True)
        job_embeddings[job_id] = embedding
        return embedding
    except Exception as e:
        logger.error(f"Lỗi khi tạo job embedding: {e}")
        return None

def compute_similarity(embedding1, embedding2):
    try:
        similarity = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
        return float(similarity)
    except Exception as e:
        logger.error(f"Lỗi khi tính similarity: {e}")
        return 0.0

def preprocess_text(text, bypass_stop_words=False, normalize_for_keywords=False):
    if not isinstance(text, str):
        logger.debug(f"Văn bản không phải chuỗi: {text}")
        return ""
    text = text.lower()
    text = re.sub(r'[^\w\s_àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệđìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵ]', ' ', text, flags=re.UNICODE)
    text = re.sub(r'\s+', ' ', text).strip()
    if not text:
        return ""
    words = text.split()
    if normalize_for_keywords:
        words = [word for word in words if len(word) >= 2]
    elif not bypass_stop_words:
        words = [word for word in words if word not in VIETNAMESE_STOP_WORDS and len(word) >= 2]
    else:
        words = [word for word in words if len(word) >= 2]
    if not words:
        return text
    
    boosted_words = []
    for word in words:
        boosted_words.append(word)
        if any(kw in word for kw in (
            DRIVER_KEYWORDS + TECH_KEYWORDS + ECOMMERCE_KEYWORDS + MARKETING_KEYWORDS +
            IT_HARDWARE_KEYWORDS + IT_SOFTWARE_KEYWORDS + HOSPITALITY_KEYWORDS + DESIGN_KEYWORDS +
            MECHANICAL_KEYWORDS + BUSINESS_KEYWORDS + EDUCATION_KEYWORDS + CONSTRUCTION_KEYWORDS +
            FINANCE_KEYWORDS + TELECOM_KEYWORDS + HEALTHCARE_KEYWORDS + LOGISTICS_KEYWORDS +
            ACCOUNTING_KEYWORDS + MANUFACTURING_KEYWORDS + LEGAL_KEYWORDS + TRANSLATION_KEYWORDS +
            EMBEDDED_IOT_KEYWORDS
        )):
            boosted_words.append(word)
    return ' '.join(boosted_words)

class ImprovedCommentFilter:
    def __init__(self):
        self.patterns = self._init_patterns()
        self.model = None
        try:
            self.model = setup_gemini(os.getenv('GEMINI_API_KEY'))
        except Exception as e:
            logger.error(f"Lỗi khi khởi tạo Gemini model: {str(e)}")

    def _init_patterns(self):
        return {
            'vietnamese': [
                r'\b(địt|đụ|đéo|cặc|lồn|đĩ|điếm|chó|ngu|ngu ngốc|ngu si|đần|đần độn|khốn nạn|đồ khốn|đồ ngu|đồ chó|đồ điếm|đồ đĩ|đồ khốn nạn|đồ vô dụng|đồ vô tích sự|đồ bỏ đi|đồ rác rưởi|đồ hèn|đồ hèn nhát|đồ hèn mạt|đồ hèn hạ|đồ hèn kém|đồ hèn mọn|đồ hèn nhược|đồ hèn yếu)\b',
            ],
            'english': [
                r'\b(fuck|shit|asshole|bitch|cunt|dick|pussy|bastard|motherfucker|retard|idiot|moron|stupid|dumb|fool|loser|jerk|scum|trash|garbage|worthless|useless)\b',
            ]
        }

    def _check_patterns(self, text):
        for lang, patterns in self.patterns.items():
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    return True
        return False

    def _get_model_score(self, text):
        try:
            if not self.model:
                return 0.0
                
            prompt = f"""Analyze the following text for toxicity and inappropriate content. 
            Return a score between 0 and 1, where:
            - 0 means completely safe and appropriate
            - 1 means highly toxic or inappropriate
            
            Text: {text}
            
            Score:"""
            
            response = self.model.generate_content(prompt)
            score = float(response.text.strip())
            return min(max(score, 0.0), 1.0)
        except Exception as e:
            logger.error(f"Lỗi khi lấy điểm từ model: {str(e)}")
            return 0.0

    def check_with_gemini(self, text, is_vietnamese=False):
        try:
            if not self.model:
                return False, 0.0

            prompt = f"""Analyze the following text for toxicity and inappropriate content.
            The text is in {'Vietnamese' if is_vietnamese else 'English'}.
            
            Consider:
            1. Profanity and offensive language
            2. Hate speech or discriminatory content
            3. Threats or violent content
            4. Sexual content
            5. Personal attacks or harassment
            
            Text: {text}
            
            Return a JSON response with:
            - is_toxic: boolean
            - score: float between 0 and 1
            - reason: brief explanation
            
            Response:"""

            response = self.model.generate_content(prompt)
            result = json.loads(response.text)
            
            return result.get('is_toxic', False), result.get('score', 0.0)
            
        except Exception as e:
            logger.error(f"Lỗi khi kiểm tra với Gemini: {str(e)}")
            return False, 0.0

    def is_vietnamese(self, text):
        # Kiểm tra xem text có phải tiếng Việt không
        return bool(re.search(r'[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]', text))

    def is_toxic(self, text, threshold=0.5):
        # Kiểm tra patterns trước
        if self._check_patterns(text):
            return True, 1.0
            
        # Nếu không match patterns, kiểm tra với model
        is_vietnamese = self.is_vietnamese(text)
        is_toxic, score = self.check_with_gemini(text, is_vietnamese)
        
        return is_toxic, score

    def filter_comment(self, text, threshold=0.5):
        """
        Kiểm tra và lọc comment không phù hợp
        Returns: (is_toxic: bool, score: float)
        """
        try:
            # Kiểm tra patterns trước
            if self._check_patterns(text):
                return True, 1.0
                
            # Nếu không match patterns, kiểm tra với model
            is_vietnamese = self.is_vietnamese(text)
            is_toxic, score = self.check_with_gemini(text, is_vietnamese)
            
            if is_toxic and score >= threshold:
                return True, score
                
            return False, score
            
        except Exception as e:
            logger.error(f"Lỗi khi lọc comment: {str(e)}")
            return False, 0.0

# Khởi tạo CommentFilter
comment_filter = ImprovedCommentFilter()

# Hàm xử lý CV
def setup_gemini(api_key):
    genai.configure(api_key=api_key)

def extract_text_from_pdf(pdf_file):
    text = ""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        for page in pdf_reader.pages:
            page_text = page.extract_text() or ""
            text += page_text
    except Exception as e:
        logger.error(f"Lỗi khi trích xuất PDF: {e}")
    return text

def extract_text_from_docx(docx_file):
    try:
        return docx2txt.process(docx_file)
    except Exception as e:
        logger.error(f"Lỗi khi trích xuất DOCX: {e}")
        return ""

def create_analysis_prompt(cv_text, job_data):
    """Tạo prompt để gửi đến Gemini"""
    # Trích xuất thông tin từ job_data
    job_description = job_data.get('description', '')
    job_requirements = job_data.get('requirement', '')
    job_benefits = job_data.get('benefit', '')
    job_experience = job_data.get('experience', '')
    job_skills = job_data.get('skills', [])
    job_position = job_data.get('position', '')
    job_nice_to_haves = job_data.get('niceToHaves', '')
    
    # Tạo một mô tả công việc đầy đủ
    full_job_description = f"""
    # Tiêu đề: {job_data.get('title', '')}
    # Vị trí: {job_position}
    
    ## Mô tả công việc:
    {job_description}
    
    ## Yêu cầu công việc:
    {job_requirements}
    
    ## Trách nhiệm công việc:
    {job_nice_to_haves}
    
    ## Quyền lợi:
    {job_benefits}
    
    ## Yêu cầu kinh nghiệm:
    {job_experience}
    
    ## Kỹ năng yêu cầu:
    {', '.join([skill.get('skillName', '') for skill in job_skills]) if isinstance(job_skills, list) else ''}
    """
    
    prompt = f"""
    Bạn là một chuyên gia phân tích CV và đánh giá mức độ phù hợp với vị trí công việc. 
    
    Hãy phân tích CV và mô tả công việc dưới đây để đánh giá mức độ phù hợp giữa ứng viên và vị trí.
    
    # CV của ứng viên:
    ```
    {cv_text}
    ```
    
    # Mô tả công việc đầy đủ:
    ```
    {full_job_description}
    ```
    
    Hãy thực hiện phân tích chi tiết gồm:
    1. Trích xuất tất cả kỹ năng từ CV
    2. Trích xuất tất cả kỹ năng yêu cầu từ mô tả công việc
    3. So sánh kỹ năng để xác định kỹ năng phù hợp và còn thiếu
    4. Phân tích học vấn trong CV và so sánh với yêu cầu
    5. Phân tích kinh nghiệm trong CV và so sánh với yêu cầu
    6. Đánh giá độ tương đồng tổng thể giữa CV và mô tả công việc
    
    Hãy trả về một chuỗi JSON hợp lệ với các trường sau:
    ```json
    {{
        "matching_score": {{
            "totalScore": [điểm số từ 0-100],
            "matchedSkills": ["skill1", "skill2", ...],
            "missingSkills": ["skill1", "skill2", ...],
            "extraSkills": ["skill1", "skill2", ...],
            "detailedScores": {{
                "skills_match": [điểm số từ 0-100],
                "education_match": [điểm số từ 0-100], 
                "experience_match": [điểm số từ 0-100],
                "overall_similarity": [điểm số từ 0-100],
                "context_score": [điểm số từ 0-100]
            }},
            "suitabilityLevel": ["Extremely Well Suited", "Well Suited", "Moderately Suited", "Somewhat Suited", "Not Well Suited"],
            "recommendations": ["đề xuất 1", "đề xuất 2", ...],
            "cvImprovementSuggestions": ["gợi ý cải thiện 1", "gợi ý cải thiện 2", ...]
        }},
        "detailedAnalysis": {{
            "skills": {{
                "score": [điểm số từ 0-100],
                "matched_skills": ["skill1", "skill2", ...],
                "missing_skills": ["skill1", "skill2", ...],
                "reason": "Lý do đánh giá"
            }},
            "education": {{
                "score": [điểm số từ 0-100],
                "cv_level": "Trình độ học vấn được phát hiện trong CV",
                "job_level": "Trình độ học vấn yêu cầu bởi công việc",
                "cv_majors": ["chuyên ngành 1", ...],
                "job_majors": ["chuyên ngành yêu cầu 1", ...],
                "reason": "Lý do đánh giá"
            }},
            "experience": {{
                "score": [điểm số từ 0-100],
                "cv_years": [số năm kinh nghiệm từ CV],
                "job_years": [số năm kinh nghiệm yêu cầu],
                "reason": "Lý do đánh giá"
            }},
            "weights": {{
                "skills": [trọng số từ 0-1],
                "education": [trọng số từ 0-1],
                "experience": [trọng số từ 0-1],
                "overall_similarity": [trọng số từ 0-1],
                "context": [trọng số từ 0-1]
            }}
        }}
    }}
    ```

    Chỉ trả về đối tượng JSON đúng định dạng, không bao gồm markdown hoặc bất kỳ văn bản nào khác. 
    Đảm bảo rằng tất cả trường là dữ liệu đúng kiểu: điểm số là số, danh sách là mảng, và văn bản là chuỗi.
    """
    return prompt

def analyze_cv_with_gemini(cv_text, job_data, api_key):
    """Sử dụng Gemini để phân tích CV và mô tả công việc"""
    # Thiết lập API
    setup_gemini(api_key)
    
    # Tạo prompt
    prompt = create_analysis_prompt(cv_text, job_data)
    
    try:
        # Gọi Gemini API
        model = genai.GenerativeModel(MODEL_NAME)
        response = model.generate_content(prompt)
        
        # Xử lý phản hồi
        if response:
            try:
                # Gemini có thể trả về text có thể bao gồm markdown hoặc backticks
                # Cần làm sạch để chỉ lấy phần JSON
                response_text = response.text
                
                # Loại bỏ backticks và json markers nếu có
                cleaned_text = response_text
                if "```json" in response_text:
                    cleaned_text = response_text.split("```json")[1].split("```")[0].strip()
                elif "```" in response_text:
                    cleaned_text = response_text.split("```")[1].split("```")[0].strip()
                
                # Parse JSON
                result = json.loads(cleaned_text)
                return result
            except Exception as e:
                logger.error(f"Lỗi xử lý phản hồi từ Gemini: {e}")
                logger.error(f"Phản hồi gốc: {response.text}")
                raise Exception(f"Không thể phân tích phản hồi từ Gemini: {e}")
        else:
            raise Exception("Không nhận được phản hồi từ Gemini")
    except Exception as e:
        logger.error(f"Lỗi khi gọi Gemini API: {e}")
        raise e
class CVSecurity:
    def __init__(self):
        # Tạo key mã hóa
        self.key = Fernet.generate_key()
        self.cipher_suite = Fernet(self.key)
    def encrypt_cv_data(self, cv_text):
        """Mã hóa nội dung CV"""
        try:
            # Mã hóa dữ liệu
            encrypted_data = self.cipher_suite.encrypt(cv_text.encode())
            return base64.b64encode(encrypted_data).decode()
        except Exception as e:
            logger.error(f"Lỗi khi mã hóa CV: {e}")
            return None

    def decrypt_cv_data(self, encrypted_data):
        """Giải mã nội dung CV"""
        try:
            # Giải mã dữ liệu
            decrypted_data = self.cipher_suite.decrypt(base64.b64decode(encrypted_data))
            return decrypted_data.decode()
        except Exception as e:
            logger.error(f"Lỗi khi giải mã CV: {e}")
            return None
class SensitiveDataProcessor:
    def __init__(self):
        # Các pattern để nhận diện thông tin nhạy cảm
        self.patterns = {
            'email': r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
            'phone': r'(?:\+84|0)[0-9]{9,10}',
            'id_card': r'[0-9]{9}|[0-9]{12}',
            'address': r'(?:đường|phố|quận|huyện|tỉnh|thành phố)[\s\w]+',
        }
    
    def mask_sensitive_data(self, cv_text):
        """Che giấu thông tin nhạy cảm trong CV"""
        masked_text = cv_text
        
        for data_type, pattern in self.patterns.items():
            if data_type == 'email':
                masked_text = re.sub(pattern, '[EMAIL]', masked_text)
            elif data_type == 'phone':
                masked_text = re.sub(pattern, '[PHONE]', masked_text)
            elif data_type == 'id_card':
                masked_text = re.sub(pattern, '[ID_CARD]', masked_text)
            elif data_type == 'address':
                masked_text = re.sub(pattern, '[ADDRESS]', masked_text)
                
        return masked_text

# Khởi tạo các đối tượng bảo mật
cv_security = CVSecurity()
sensitive_processor = SensitiveDataProcessor()

def preprocess_cv(cv_text, max_length=4000):
    """
    Tiền xử lý CV để giảm kích thước và tối ưu hóa nội dung trước khi phân tích
    
    Args:
        cv_text (str): Nội dung CV gốc
        max_length (int): Độ dài tối đa cho phép của CV sau khi xử lý
        
    Returns:
        str: CV đã được tiền xử lý
    """
    try:
        # 1. Loại bỏ thông tin cá nhân nhạy cảm
        patterns = [
            r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',  # email
            r'(?:\+84|0)[0-9]{9,10}',  # phone
            r'[0-9]{9}|[0-9]{12}',  # id card
            r'(?:đường|phố|quận|huyện|tỉnh|thành phố)[\s\w]+',  # address (VN)
            r'(?:address|city|district|province|country)[\s\w]+',  # address (EN)
            r'(?:ngày sinh|date of birth|dob)[\s\w:/.]+',  # date of birth
            r'(?:họ tên|full name|name)[\s\w:]+',  # name
        ]
        for pattern in patterns:
            cv_text = re.sub(pattern, '', cv_text, flags=re.IGNORECASE)

        # 2. Chỉ giữ lại các section quan trọng
        important_sections = [
            r'(?:education|học vấn|trình độ học vấn)(.*?)(?:\n\n|\Z|experience|kinh nghiệm|skills|kỹ năng|projects|dự án|certifications|chứng chỉ|objective|mục tiêu nghề nghiệp)',
            r'(?:experience|kinh nghiệm)(.*?)(?:\n\n|\Z|education|học vấn|skills|kỹ năng|projects|dự án|certifications|chứng chỉ|objective|mục tiêu nghề nghiệp)',
            r'(?:skills|kỹ năng)(.*?)(?:\n\n|\Z|education|học vấn|experience|kinh nghiệm|projects|dự án|certifications|chứng chỉ|objective|mục tiêu nghề nghiệp)',
            r'(?:projects|dự án)(.*?)(?:\n\n|\Z|education|học vấn|experience|kinh nghiệm|skills|kỹ năng|certifications|chứng chỉ|objective|mục tiêu nghề nghiệp)',
            r'(?:certifications|chứng chỉ)(.*?)(?:\n\n|\Z|education|học vấn|experience|kinh nghiệm|skills|kỹ năng|projects|dự án|objective|mục tiêu nghề nghiệp)',
            r'(?:objective|mục tiêu nghề nghiệp)(.*?)(?:\n\n|\Z|education|học vấn|experience|kinh nghiệm|skills|kỹ năng|projects|dự án|certifications|chứng chỉ)',
        ]
        
        extracted = ""
        for section in important_sections:
            matches = re.findall(section, cv_text, flags=re.IGNORECASE | re.DOTALL)
            for match in matches:
                extracted += match.strip() + "\n\n"

        # 3. Nếu không tìm thấy section nào, sử dụng nội dung đã loại bỏ thông tin cá nhân
        if not extracted.strip():
            extracted = cv_text.strip()

        # 4. Loại bỏ khoảng trắng thừa và ký tự đặc biệt
        extracted = re.sub(r'\s+', ' ', extracted)
        extracted = re.sub(r'[^\w\s_àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệđìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵ]', ' ', extracted, flags=re.UNICODE)
        
        # 5. Giới hạn độ dài
        if len(extracted) > max_length:
            # Cắt theo câu hoàn chỉnh
            sentences = re.split(r'[.!?]+', extracted)
            truncated = ""
            for sentence in sentences:
                if len(truncated + sentence) <= max_length:
                    truncated += sentence + "."
                else:
                    break
            extracted = truncated.strip()

        return extracted

    except Exception as e:
        logger.error(f"Lỗi khi tiền xử lý CV: {e}")
        return cv_text[:max_length]  # Fallback: cắt bớt nếu có lỗi


# Hàm xử lý công việc và gợi ý
def load_jobs_from_csv(filepath=JOBS_FILEPATH, max_jobs=1400):
    global jobs
    try:
        logger.info(f"Đang tải công việc từ {filepath}...")
        df = pd.read_csv(filepath, encoding='utf-8', low_memory=False)
        required_columns = ['postId', 'title', 'companyId']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.error(f"Thiếu cột bắt buộc trong job_post.csv: {missing_columns}")
            return []
        df = df.drop_duplicates(subset=['postId'], keep='first')
        df['createDate'] = pd.to_datetime(df['createDate'], errors='coerce')
        df['expireDate'] = pd.to_datetime(df['expireDate'], errors='coerce')
        current_date = datetime.now()
        df_active = df[df['expireDate'].notna() & (df['expireDate'] > current_date)].copy()
        df_sorted = df_active.sort_values(by='createDate', ascending=False).head(max_jobs)
        df_sorted = df_sorted.fillna({
            'title': 'Không có tiêu đề',
            'description': '',
            'location': '',
            'salary': 0,
            'experience': '',
            'typeOfWork': '',
            'companyName': '',
            'cityName': '',
            'logo': '',
            'industryNames': ''
        })
        df_sorted['createDate'] = df_sorted['createDate'].dt.strftime('%Y-%m-%dT%H:%M:%S')
        df_sorted['expireDate'] = df_sorted['expireDate'].dt.strftime('%Y-%m-%dT%H:%M:%S')
        jobs = df_sorted.to_dict(orient='records')
        logger.info(f"Đã tải {len(jobs)} công việc.")
        return jobs
    except Exception as e:
        logger.error(f"Lỗi khi tải công việc từ CSV: {e}")
        return []

last_search_csv_modified = 0

def load_search_history_from_csv(filepath):
    global search_history, last_search_csv_modified
    try:
        if not os.path.exists(filepath):
            logger.warning(f"Tệp {filepath} không tồn tại.")
            search_history = []
            return search_history
        current_modified = os.path.getmtime(filepath)
        if current_modified <= last_search_csv_modified:
            logger.info(f"Tệp {filepath} chưa thay đổi.")
            return search_history
        logger.info(f"Đang tải lịch sử tìm kiếm từ {filepath}...")
        df = pd.read_csv(filepath, encoding='utf-8', low_memory=False)
        if 'SeekerID' not in df or 'Search Query' not in df:
            logger.warning("Thiếu cột bắt buộc trong lịch sử tìm kiếm.")
            search_history = []
            return search_history
        if 'Search Date' in df.columns:
            df['Search Date'] = pd.to_datetime(df['Search Date'], errors='coerce')
            df_sorted = df.sort_values(by='Search Date', ascending=False).copy()
        else:
            df_sorted = df.copy()
        df_sorted['Search Query'] = df_sorted['Search Query'].astype(str).apply(
            lambda x: re.sub(r'(CityName:|IndustryNames:|MaxSalary:|TypesOfWork:|Title:\s*\d+\s*|\s*\|\s*)', '', x).strip()
        )
        search_history = df_sorted.fillna('').to_dict(orient='records')
        last_search_csv_modified = current_modified
        logger.info(f"Đã tải {len(search_history)} mục lịch sử tìm kiếm.")
        return search_history
    except Exception as e:
        logger.error(f"Lỗi khi tải lịch sử tìm kiếm từ CSV: {e}")
        search_history = []
        return search_history

def filter_jobs_by_category(jobs, user_queries, top_n=2000):
    if not user_queries:
        return jobs[:top_n]
    
    keywords = set()
    cities = set()
    query_intents = set()
    for query in user_queries:
        cleaned_query = query.get('Search Query', '').replace('Title:', '').strip()
        processed_query = preprocess_text(cleaned_query, normalize_for_keywords=True)
        no_space_query = re.sub(r'\s+', '', processed_query)
        words = processed_query.split()
        keywords.update([word for word in words if word not in VIETNAMESE_STOP_WORDS])
        if "CityName:" in query.get('Search Query', ''):
            city = re.search(r'CityName:([^|]+)', query.get('Search Query', ''))
            if city:
                cities.add(city.group(1).strip().lower())
        
        for kw, kw_no_space in [normalize_keyword(k) for k in DRIVER_KEYWORDS]:
            if kw in processed_query or kw_no_space in no_space_query:
                query_intents.add('driver')
        for kw, kw_no_space in [normalize_keyword(k) for k in IT_SOFTWARE_KEYWORDS]:
            if kw in processed_query or kw_no_space in no_space_query:
                query_intents.add('it_software')
        # Thêm các ngành khác tương tự...

    seen_job_ids = set()
    filtered_jobs = []
    for job in jobs:
        job_id = job.get('postId')
        if job_id in seen_job_ids:
            continue
        seen_job_ids.add(job_id)
        job_text = " ".join(str(job.get(field, '')) for field in ['title', 'description', 'typeOfWork', 'industryNames'])
        if not job_text.strip() or len(job_text.strip().split()) < 3:
            continue
        job_text = preprocess_text(job_text, normalize_for_keywords=False)
        job_city = str(job.get('city', '')).lower()
        job_title = job.get('title', '').lower()
        if cities and job_city not in cities:
            continue
        if not query_intents:
            filtered_jobs.append(job)
            continue
        intent_match = False
        no_space_job_title = re.sub(r'\s+', '', job_title)
        for intent in query_intents:
            keywords = {
                'driver': [normalize_keyword(k) for k in DRIVER_KEYWORDS],
                'it_software': [normalize_keyword(k) for k in IT_SOFTWARE_KEYWORDS],
                # Thêm các ngành khác...
            }.get(intent, [])
            for kw, kw_no_space in keywords:
                if kw in job_title or kw_no_space in no_space_job_title:
                    intent_match = True
                    break
        if intent_match:
            filtered_jobs.append(job)
    
    return filtered_jobs[:top_n]

# API Endpoints
@app.route('/', methods=['GET'])
def index():
    return jsonify({
        "message": "JobRadar API is running",
        "endpoints": {
            "/analyze": "Phân tích CV với mô tả công việc",
            "/semantic-search": "Tìm kiếm ngữ nghĩa",
            "/recommend-jobs/phobert": "Gợi ý việc làm",
            "/save-search": "Lưu tìm kiếm",
            "/health": "Kiểm tra sức khỏe hệ thống",
            "/check-comment": "Kiểm tra nội dung bình luận"
        },
        "status": "ok"
    })

@app.route('/analyze', methods=['POST'])
def analyze_cv():
    try:
        if 'cv' not in request.files:
            return jsonify({"error": "Thiếu file CV"}), 400
        job_data = {}
        if 'job_data' in request.form:
            job_data = json.loads(request.form['job_data'])
        elif 'job_description' in request.form:
            job_data = {'description': request.form['job_description']}
        else:
            return jsonify({"error": "Thiếu mô tả công việc"}), 400
        
        cv_file = request.files['cv']
        if not GEMINI_API_KEY:
            return jsonify({"error": "GEMINI_API_KEY không được thiết lập"}), 500
        
        cv_text = ""
        if cv_file.filename.endswith('.pdf'):
            cv_text = extract_text_from_pdf(cv_file)
        elif cv_file.filename.endswith('.docx'):
            cv_text = extract_text_from_docx(cv_file)
        else:
            return jsonify({"error": "Định dạng file không được hỗ trợ"}), 400
        
        if not cv_text:
            return jsonify({"error": "Không thể trích xuất văn bản từ CV"}), 400
        
        result = analyze_cv_with_gemini(cv_text, job_data, GEMINI_API_KEY)
        return jsonify(result)
    except Exception as e:
        logger.exception(f"Lỗi server: {str(e)}")
        return jsonify({"error": f"Lỗi server: {str(e)}"}), 500

@app.route('/semantic-search', methods=['POST'])
def semantic_search():
    try:
        data = request.json
        if not data:
            return jsonify({"error": "Không có dữ liệu được gửi"}), 400
        query = data.get('query', '')
        jobs_to_search = data.get('jobs', jobs)
        filters = data.get('filters', {})
        if not query or not jobs_to_search:
            return jsonify({"error": "Thiếu truy vấn hoặc danh sách công việc"}), 400
        
        cache_key = get_cache_key(query, filters)
        cached_result = get_cache(cache_key)
        if cached_result:
            return jsonify(cached_result)
        
        query_embedding = get_text_embedding(query)
        if PRECOMPUTE_EMBEDDINGS:
            if not last_embedding_update or (datetime.now() - last_embedding_update) > timedelta(hours=EMBEDDING_REFRESH_HOURS):
                precompute_embeddings(jobs_to_search)
            if USE_JOB_INDEXING:
                create_job_index(jobs_to_search)
        
        filtered_jobs = jobs_to_search
        if ENABLE_PREFILTERING:
            keywords = query.lower().split()
            if USE_JOB_INDEXING and job_index:
                candidate_job_ids = set()
                for keyword in keywords:
                    if keyword in job_index:
                        candidate_job_ids.update(job_index[keyword])
                filtered_jobs = [job for job in jobs_to_search if (job.get('id') or job.get('jobId') or job.get('postId')) in candidate_job_ids]
            else:
                filtered_jobs = filter_jobs_by_keywords(jobs_to_search, keywords)
        
        vector_results = fast_vector_search(query_embedding, filtered_jobs)
        final_results = []
        for job, score in vector_results:
            job_with_score = job.copy()
            job_with_score['similarity_score'] = score
            final_results.append(job_with_score)
        
        MIN_SIMILARITY_SCORE = 0.3
        filtered_final_results = [job for job in final_results if job.get('similarity_score', 0) >= MIN_SIMILARITY_SCORE]
        set_cache(cache_key, filtered_final_results, ttl=7200)
        return jsonify(filtered_final_results)
    except Exception as e:
        logger.exception(f"Lỗi trong semantic search: {str(e)}")
        return jsonify({"error": str(e)}), 500

def filter_jobs_by_keywords(jobs, keywords, threshold=1):
    if not keywords:
        return jobs
    filtered_jobs = []
    keywords_lower = [k.lower() for k in keywords]
    for job in jobs:
        job_text = f"{job.get('title', '')} {job.get('description', '')} {job.get('requirements', '')}".lower()
        matches = sum(1 for keyword in keywords_lower if keyword in job_text)
        if matches >= threshold:
            filtered_jobs.append(job)
    return filtered_jobs

def create_job_index(jobs):
    global job_index
    job_index = {}
    for job in jobs:
        job_id = job.get('id') or job.get('jobId') or job.get('postId')
        if not job_id:
            job_content = f"{job.get('title', '')}{job.get('description', '')}{job.get('requirements', '')}"
            job_id = hashlib.md5(job_content.encode()).hexdigest()
        job_text = f"{job.get('title', '')} {job.get('description', '')} {job.get('requirements', '')}".lower()
        words = set(re.findall(r'\b\w+\b', job_text))
        vietnamese_stopwords = {'và', 'là', 'của', 'có', 'trong', 'cho', 'với', 'để', 'các'}
        keywords = [w for w in words if len(w) > 2 and w not in vietnamese_stopwords]
        for keyword in keywords:
            if keyword not in job_index:
                job_index[keyword] = []
            job_index[keyword].append(job_id)

def load_embeddings_from_file():
    global job_embeddings, last_embedding_update
    if os.path.exists(EMBEDDINGS_FILE):
        try:
            with open(EMBEDDINGS_FILE, 'rb') as f:
                data = pickle.load(f)
                # Kiểm tra version cache
                if data.get('version') == CACHE_VERSION:
                    job_embeddings = data.get('embeddings', {})
                    last_embedding_update = data.get('timestamp', None)
                    logger.info(f"Đã tải {len(job_embeddings)} embeddings từ cache")
                    return True
                else:
                    logger.info("Cache không hợp lệ do version không khớp")
                    return False
        except Exception as e:
            logger.error(f"Lỗi khi tải embeddings từ file: {e}")
            return False
    return False

def save_embeddings_to_file():
    try:
        with open(EMBEDDINGS_FILE, 'wb') as f:
            pickle.dump({
                'embeddings': job_embeddings,
                'timestamp': datetime.now(),
                'version': CACHE_VERSION
            }, f)
        logger.info(f"Đã lưu {len(job_embeddings)} embeddings vào cache")
        return True
    except Exception as e:
        logger.error(f"Lỗi khi lưu embeddings: {e}")
        return False

def precompute_embeddings(jobs):
    global job_embeddings, last_embedding_update
    
    # Kiểm tra cache trước
    if load_embeddings_from_file():
        # Kiểm tra xem cache có còn hợp lệ không
        if last_embedding_update and (datetime.now() - last_embedding_update) < timedelta(hours=EMBEDDING_REFRESH_HOURS):
            logger.info("Sử dụng embeddings từ cache")
            return
    
    logger.info("Tạo mới embeddings cho công việc")
    job_ids = []
    for job in jobs:
        job_id = job.get('id') or job.get('jobId') or job.get('postId')
        if not job_id:
            job_content = f"{job.get('title', '')}{job.get('description', '')}{job.get('requirements', '')}"
            job_id = hashlib.md5(job_content.encode()).hexdigest()
        job_ids.append(job_id)
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        job_texts = [f"{job.get('title', '')} {job.get('description', '')} {job.get('requirements', '')}" for job in jobs]
        BATCH_SIZE = 100
        all_embeddings = []
        for i in range(0, len(job_texts), BATCH_SIZE):
            batch_texts = job_texts[i:i+BATCH_SIZE]
            batch_embeddings = list(executor.map(lambda text: embedding_model.encode(text, device=DEVICE, normalize_embeddings=True), batch_texts))
            all_embeddings.extend(batch_embeddings)
    
    for i, job_id in enumerate(job_ids):
        if i < len(all_embeddings):
            job_embeddings[job_id] = all_embeddings[i]
    
    # Lưu embeddings vào file
    save_embeddings_to_file()
    last_embedding_update = datetime.now()

def fast_vector_search(query_embedding, jobs, top_k=100):
    results = []
    seen_job_ids = set()

    # Sử dụng job_embeddings để tính độ tương đồng
    for job in jobs:
        job_id = job.get('id') or job.get('jobId') or job.get('postId')
        if not job_id or job_id in seen_job_ids:
            continue
        seen_job_ids.add(job_id)

        # Lấy embedding từ job_embeddings nếu có
        job_embedding = job_embeddings.get(job_id)
        if job_embedding is None:
            job_embedding = get_job_embedding(job)  # Tạo mới nếu chưa có
            if job_embedding is None:
                continue
        
        similarity = compute_similarity(query_embedding, job_embedding)
        results.append((job, similarity))
    
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]

@app.route('/recommend-jobs/phobert', methods=['POST'])
def recommend_jobs_mbert():
    try:
        user_id = request.headers.get('X-User-Id')
        if not user_id:
            return jsonify({'error': 'Yêu cầu header X-User-Id'}), 400
        
        with csv_read_lock:
            load_search_history_from_csv(SEARCH_HISTORY_FILEPATH)
        
        user_search_history = [h for h in search_history if str(h.get('SeekerID', '')) == str(user_id)]
        user_search_history.sort(key=lambda x: pd.Timestamp(x.get('Search Date', '1970-01-01')), reverse=True)
        last_five_searches = user_search_history[:5]
        
        if not last_five_searches:
            return jsonify([]), 200
        
        recent_search = last_five_searches[0]
        query_text = recent_search.get('Search Query', '').replace('Title:', '').strip()
        
        # Kiểm tra cache
        cache_key = get_cache_key(query_text, {'user_id': user_id})
        cached_result = get_cache(cache_key)
        if cached_result:
            logger.info(f"Sử dụng kết quả từ cache cho truy vấn: {query_text}")
            return jsonify(cached_result)
        
        query_embedding = get_text_embedding(query_text)
        if query_embedding is None:
            return jsonify({'error': 'Không thể tạo embedding cho truy vấn'}), 500
        
        recent_jobs = filter_jobs_by_category(jobs, [recent_search])
        vector_results = fast_vector_search(query_embedding, recent_jobs, top_k=8)
        
        recommended_jobs_dict = {}
        for job, sim_score in vector_results:
            job_id = job.get('postId')
            if job_id and job_id not in recommended_jobs_dict:
                reason = f"Tương đồng: {sim_score:.4f} | Dựa trên tìm kiếm: '{query_text}'"
                job_with_reason = job.copy()
                job_with_reason['recommendation_reason'] = reason
                job_with_reason['similarity_score'] = float(sim_score)
                recommended_jobs_dict[job_id] = job_with_reason
        
        top_recommended_jobs_list = list(recommended_jobs_dict.values())
        top_recommended_jobs_list.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        # Lưu vào cache
        set_cache(cache_key, top_recommended_jobs_list, ttl=7200)
        
        return jsonify(top_recommended_jobs_list), 200, {
            'Content-Type': 'application/json; charset=utf-8',
            'Cache-Control': 'no-store, no-cache, must-revalidate, proxy-revalidate'
        }
    except Exception as e:
        logger.exception(f"Lỗi trong recommend_jobs_mbert: {str(e)}")
        return jsonify({'error': 'Lỗi server nội bộ'}), 500

@app.route('/save-search', methods=['POST'])
def save_search():
    try:
        user_id = request.headers.get('X-User-Id')
        if not user_id:
            return jsonify({'error': 'Yêu cầu header X-User-Id', 'query': None}), 400
        data = request.get_json()
        search_query = data.get('query')
        if not search_query:
            return jsonify({'error': 'Yêu cầu truy vấn tìm kiếm', 'query': None}), 400
        cleaned_query = re.sub(r'(CityName:|IndustryNames:|MaxSalary:|TypesOfWork:|Title:\s*\d+\s*|\s*\|\s*)', '', str(search_query)).strip()
        new_entry = {
            'SeekerID': user_id,
            'Search Query': cleaned_query,
            'Search Date': datetime.now().isoformat()
        }
        with csv_write_lock:
            search_history.append(new_entry)
            df = pd.DataFrame([new_entry])
            df.to_csv(SEARCH_HISTORY_FILEPATH, mode='a', header=not os.path.exists(SEARCH_HISTORY_FILEPATH), index=False)
            load_search_history_from_csv(SEARCH_HISTORY_FILEPATH)
        logger.info(f"Đã lưu tìm kiếm mới cho người dùng {user_id}: '{cleaned_query}'")
        return jsonify({'message': 'Đã lưu tìm kiếm', 'query': cleaned_query}), 200
    except Exception as e:
        logger.error(f"Lỗi khi lưu tìm kiếm: {str(e)}")
        return jsonify({'error': 'Lỗi server nội bộ', 'query': None}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "jobs_count": len(jobs),
        "search_history_count": len(search_history),
        "model_loaded": True
    }), 200

@app.route('/check-comment', methods=['POST'])
def check_comment():
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        if not text:
            return jsonify({
                'is_toxic': False,
                'message': 'No text provided'
            }), 400

        # Khởi tạo bộ lọc comment
        comment_filter = ImprovedCommentFilter()
        
        # Kiểm tra nội dung
        is_toxic, score = comment_filter.filter_comment(text)
        
        # Tạo message dựa trên kết quả
        message = 'Nội dung không phù hợp' if is_toxic else 'Nội dung phù hợp'
        
        return jsonify({
            'is_toxic': is_toxic,
            'score': score,
            'message': message
        })

    except Exception as e:
        logger.error(f"Lỗi khi kiểm tra comment: {str(e)}")
        return jsonify({
            'is_toxic': False,
            'message': f'Lỗi server: {str(e)}'
        }), 500

if __name__ == '__main__':
    logger.info("Starting JobRadar service on port 5000")
    jobs = load_jobs_from_csv(JOBS_FILEPATH)
    search_history = load_search_history_from_csv(SEARCH_HISTORY_FILEPATH)
    
    if PRECOMPUTE_EMBEDDINGS:
        # Chỉ tính toán lại embeddings nếu cache không hợp lệ
        if not load_embeddings_from_file() or \
           not job_embeddings or \
           len(job_embeddings) != len(jobs) or \
           (last_embedding_update and (datetime.now() - last_embedding_update) > timedelta(hours=EMBEDDING_REFRESH_HOURS)):
            logger.info("Tạo mới embeddings vì cache không hợp lệ hoặc đã hết hạn")
            precompute_embeddings(jobs)
    
    if USE_JOB_INDEXING:
        create_job_index(jobs)
    
    app.run(host='0.0.0.0', port=5000, debug=True)