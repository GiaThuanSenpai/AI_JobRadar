import os
import logging
import time
import re
import traceback
import numpy as np
import pandas as pd
import torch
from datetime import datetime, timedelta
from functools import wraps, lru_cache
from threading import Thread, Lock
import concurrent.futures
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from flask import Flask, request, jsonify
from flask_cors import CORS
from sentence_transformers import SentenceTransformer, __version__ as sentence_transformers_version
import psutil
import PyPDF2
import docx2txt
import json
import google.generativeai as genai
from dotenv import load_dotenv
import redis
import hashlib
from concurrent.futures import ThreadPoolExecutor
import pickle
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Thiết lập encoding
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

# Flask app setup
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*", "allow_headers": ["Authorization", "X-User-Id"]}})

# Cấu hình file paths
BASE_DIR = os.getenv('BASE_DIR', os.path.dirname(os.path.abspath(__file__)))
JOBS_FILEPATH = os.path.join(BASE_DIR, 'data', 'job_post.csv')
SEARCH_HISTORY_FILEPATH = os.path.join(BASE_DIR, 'data', 'search.csv')
EMBEDDINGS_FILE = os.path.join(BASE_DIR, 'cache', 'job_embeddings.pkl')
JOB_VECTOR_CACHE_FILE = os.path.join(BASE_DIR, 'cache', 'job_vector_cache.pkl')
MODEL_CACHE_DIR = os.path.join(BASE_DIR, 'model_cache')

# Tạo thư mục nếu chưa tồn tại
os.makedirs(os.path.join(BASE_DIR, 'data'), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, 'cache'), exist_ok=True)
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

# Cấu hình Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = "gemini-2.0-flash"
if not GEMINI_API_KEY:
    logger.warning("GEMINI_API_KEY không được thiết lập.")

# Cấu hình Redis
REDIS_URL = os.getenv("REDIS_URL")
redis_client = None
try:
    if REDIS_URL:
        redis_client = redis.from_url(REDIS_URL)
        redis_client.ping()
        logger.info("Kết nối Redis thành công")
except Exception as e:
    logger.error(f"Không thể kết nối Redis: {e}")
    redis_client = None

# Cache trong bộ nhớ
in_memory_cache = {}
EMBEDDING_CACHE_SIZE = 1000
MAX_WORKERS = 8
PRECOMPUTE_EMBEDDINGS = True
EMBEDDING_REFRESH_HOURS = 24
SEMANTIC_SEARCH_TIMEOUT = 30
USE_JOB_INDEXING = True
ENABLE_PREFILTERING = True
MAX_JOBS_FOR_GEMINI = 50

# Biến toàn cục cho job recommendation
MODEL_RECOMMEND = None
DEVICE = None
TFIDF_VECTORIZER = None
JOB_VECTOR_CACHE = {}
jobs = []
search_history = []
csv_write_lock = Lock()
csv_read_lock = Lock()
last_search_csv_modified = 0

# Biến toàn cục cho CV analysis
MODEL_ANALYZE = None
job_embeddings = {}
last_embedding_update = None
job_index = {}

# Danh sách stop words tiếng Việt
VIETNAMESE_STOP_WORDS = [
    "và", "của", "là", "các", "cho", "trong", "tại", "được", "với", "một",
    "những", "để", "từ", "có", "không", "người", "này", "đã", "ra", "trên",
    "bằng", "vào", "hay", "thì", "đó", "nào", "ở", "lại", "còn", "như",
    "thành", "phố", "hồ", "chí", "minh", "title", "tuyển", "dụng", "việc",
    "làm", "công", "ty", "tại", "lương", "cao", "hấp", "dẫn", "dịch", "vụ",
    "quản", "lý", "hệ", "thống"
]

# Từ khóa theo ngành nghề (giữ nguyên như trong File 1)
DRIVER_KEYWORDS = ['tài xế', 'lái xe', 'lái xe tải', 'tài xế xe tải', 'lái xe container', 'giao hàng', 'vận chuyển']
TECH_KEYWORDS = ['công nghệ ô tô', 'sửa chữa ô tô', 'kiểm tra ghế lái', 'kiểm tra phanh xe', 'kỹ sư ô tô', 'thiết kế ô tô', 'bảo dưỡng ô tô', 'truyền động', 'kỹ thuật viên ô tô', 'sửa chữa truyền động', 'xe điện', 'phụ tùng ô tô']
ECOMMERCE_KEYWORDS = ['thương mại điện tử', 'mua sắm trực tuyến', 'sàn thương mại', 'thanh toán điện tử', 'marketplace', 'e-commerce']
MARKETING_KEYWORDS = ['quảng cáo', 'tiếp thị số', 'truyền thông xã hội', 'seo', 'sem', 'thương hiệu']
IT_HARDWARE_KEYWORDS = ['phần cứng', 'sửa chữa máy tính', 'hạ tầng mạng', 'server', 'linh kiện được tử']
IT_SOFTWARE_KEYWORDS = ['lập trình', 'phát triển phần mềm', 'kỹ sư phần mềm', 'trí tuệ nhân tạo', 'ứng dụng di động', 'java', 'python', 'javascript', 'devops', 'database', 'cloud', 'web developer', 'mobile developer', 'software engineer', 'api', 'microservices', 'blockchain', 'machine learning', 'fullstack', 'backend', 'frontend', 'phần mềm']
HOSPITALITY_KEYWORDS = ['nhà hàng', 'khách sạn', 'đầu bếp', 'phục vụ', 'lễ tân', 'du lịch']
DESIGN_KEYWORDS = ['thiết kế đồ họa', 'ui/x', 'minh họa', 'thiết kế bao bì', 'nhận diện thương hiệu']
MECHANICAL_KEYWORDS = ['cơ khí', 'bảo trì máy móc', 'thiết kế cơ khí', 'tự động hóa', 'cad/cam']
BUSINESS_KEYWORDS = ['bán hàng', 'phát triển thị trường', 'chiến lược kinh doanh', 'đàm phán hợp đồng']
EDUCATION_KEYWORDS = ['giảng dạy', 'giáo viên', 'đào tạo', 'giáo dục trực tuyến', 'phát triển chương trình']
CONSTRUCTION_KEYWORDS = ['kiến trúc', 'xây dựng', 'thiết kế kiến trúc', 'giám sát công trình', 'quy hoạch đô thị']
FINANCE_KEYWORDS = ['tài liệu', 'chính', 'ngân hàng', 'kế toán tài chính', 'phân tích tài chính', 'đầu tư']
TELECOM_KEYWORDS = ['viễn thông', 'mạng di động', 'cáp quang', '5g', 'kỹ thuật viễn thông', 'mạng']
HEALTHCARE_KEYWORDS = ['bác sĩ', 'điều dưỡng', 'dược sĩ', 'chăm sóc bệnh nhân', 'y tế công cộng']
LOGISTICS_KEYWORDS = ['vận tải', 'chuỗi cung ứng', 'kho bãi', 'giao nhận hàng hóa', 'xuất nhập khẩu', 'hải quan', 'logistics']
ACCOUNTING_KEYWORDS = ['kế toán', 'kiểm toán', 'báo cáo tài chính', 'thuế', 'quản lý ngân sách']
MANUFACTURING_KEYWORDS = ['sản xuất', 'vận hành máy móc', 'kiểm soát chất lượng', 'sản xuất']
LEGAL_KEYWORDS = ['luật sư', 'tư vấn pháp lý', 'hợp đồng', 'pháp chế', 'sở hữu trí tuệ']
TRANSLATION_KEYWORDS = ['phiên dịch', 'dịch thuật', 'thông dịch', 'đa ngôn ngữ', 'hiệu đính']
EMBEDDED_IOT_KEYWORDS = ['hệ thống nhúng', 'iot', 'cảm biến', 'hệ thống bị thông minh', 'firmware']
RELATED_INTENTS = {
    'it_hardware': ['it_software', 'mechanical', 'embedded_iot'],
    'it_software': ['it_hardware', 'embedded_iot'],
    'mechanical': ['it_hardware', 'manufacturing'],
    'embedded_iot': ['it_hardware', 'it_software'],
}

# Utility functions
def normalize_keyword(keyword):
    keyword = keyword.lower().strip()
    no_space = re.sub(r'\s+', '', keyword)
    return keyword, no_space

def timeout(seconds):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kw):
            result = [None]
            error = [None]

            def target():
                try:
                    result[0] = request(*args, **kw)
                except Exception as e:
                    error[0] = e

                thread = Thread(target=target)
                thread.daemon = True
                thread.start()
                thread.join(seconds)

                if thread.is_alive():
                    raise TimeoutError(f"Function {func.__name__} timed out after {seconds} seconds")

                if error[0]:
                    raise error[0]
                return result[0]
            return wrapper
        return decorator

def log_resource_usage():
    process = requestutil.Process()
    mem = process.memory_info().rss / 1024 / 1024
    cpu = process.cpu_percent(interval=1)
    logger.info(f"Sở hữu dụng bộ nhớ: {mem:.2f} MB | CPU: {cpu:.2f}%")

# Load mô hình recommendation
def load_recommend_mbert():
    global MODEL_RECOMMEND, _DEVICE, TFIDF_MODELIZER
    try:
        logger.info("Kiểm tra phiên bản thư viện...")
        logger.info(f"Sentence Transformers: {sentence_transformers_version}")
        logger.info(f"PyTorch: {torch.__version__}")
        if os.path.exists(MODEL_CACHE_DIR) and os.path.isdir(MODEL_CACHE_DIR):
            logger.info(f"Tải mô hình từ cục bộ: {MODEL_CACHE_DIR}")
            MODEL_RECOMMEND = SentenceTransformer(MODEL_CACHE_DIR)
        else:
            logger.info("Tải mô hình từ Hugging Face...")
            MODEL_RECOMMEND = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
            MODEL_RECOMMEND.save(MODEL_CACHE_DIR)
        logger.info("Mô hình tải thành công!")

        if torch.cuda.is_available():
            logger.info("CUDA khả dụng. Sử dụng GPU.")
            DEVICE = torch.device("cuda")
        else:
            logger.info("CUDA không khả dụng. Sử dụng CPU.")
            DEVICE = torch.device("cpu")

        MODEL_RECOMMEND.to(DEVICE)
        MODEL_RECOMMEND.eval()
        
        TFIDF_VECTORIZER = TfidfVectorizer(
            stop_words=VIETNAMESE_STOP_WORDS,
            max_df=0.8,
            min_df=2,
            max_features=5000
        )
        logger.info("TF-IDF vectorizer khởi tạo thành công!")

        log_resource_usage()
    except Exception as e:
        logger.error(f"Lỗi khi tải mô hình recommendation: {e}", exc_info=True)
        MODEL_RECOMMEND = None
        DEVICE = None
        TFIDF_MODELIZER = None
        raise

# Load mô hình phân tích CV
def load_analyze_model():
    global MODEL_ANALYZE
    try:
        MODEL_ANALYZE = SentenceTransformer('paraphrase-multilingual-MiniLM-L-12-v2')
        logger.info("Mô hình phân tích CV tải thành công!")
    except Exception as e:
        logger.error(f"Lỗi khi tải mô hình phân tích CV: {e}")
        MODEL_ANALAYZE = None
        raise

# Class ImprovedCommentFilter (giữ nguyên)
class ImprovedCommentFilter:
    def __init__(self):
        try:
            self.en_model = AutoModelForSequenceClassification.from_pretrained("unitary/toxic-bert")
            self._en_tokenizer = AutoTokenizer.from_pretrained("unitary/toxic-bert")
            self.en_model.eval()
            self.patterns = self._init_patterns()
            self.feedback_data = []
            logger.info("Khởi tạo ImprovedCommentFilter thành công")
        except Exception as e:
            logger.error(f"Lỗi khởi tạo CommentFilter: {e}")
            raise

    def _init_patterns(self):
        vn_patterns = [
            r'\b[đdĐ][ịiỉĩíì]+[tT]*\w*',
            r'\b[cC][ặăắẳ]+[cC]*\w*',
            r'\b[lL][ồôỗổ]+[nN]*\w*',
            r'\b[đĐ][éoèẻẽẹêếềểễệ]+[oO]*\w*',
            r'\b[đdĐ][ũuụủứừ][mM][aàáảãạ]*\w*',
            r'\b[cC][hH][óoôốồổỗộơớ]+\w*',
            r'\b[sS][úuứừửữự]+[cC]\w*',
            r'\b[nN][gG][uU]\w*',
            r'\b[kK][hH][óoô]\[nN]\w*'
        ]
        en_patterns = [
            r'\bf+u[cC][kK]+\w*',
            r'\bs+h+[i1]+t+\w*',
            r'\bb+i[t|cC][hH]+\w*',
            r'\ba+s+[hH][oO][lL][eE]+\w*',
            r'\bd+u[m]+b+\w*',
            r'\bi+d+i+[oO][tT]+\w*',
            r'\bs+t+u+p[i][dD]+\w*'
        ]
        return re.compile('|'.join(vn_patterns | en_patterns), re.IGNORECASE | re.UNICODE)

    def _check_patterns(self, text):
        return bool(self.patterns.search(text))

    def _get_model_score(self, text):
        try:
            inputs = self.en_tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
            with torch.no_grad():
                outputs = self.en_model(**inputs)
                probs = torch.softmax(outputs.logits, dim=1)
                toxic_score = probs[0][1].item()
            return toxic_score
        except Exception as e:
            logger.error(f"Lỗi khi lấy điểm từ model: {e}")
            return 0.5

    def check_with_gemini(self, text, is_vietnamese=False):
        try:
            prompt = f"""
            Hãy phân tích nội dung bình luận sau và đánh giá mức độ phù hợp:
            "{text}"

            Hãy phân tích các khía cạnh sau:
            1. Chửi thề, văng tục
            2. Xúc phạm, công kích cá nhân hoặc tập thể
            3. Phân biệt đối xử (giới tính, chủng tộc, tôn giáo...)
            4. Đe dọa, quấy rối
            5. Spam hoặc nội dung rác
            6. Ngôn từ tiêu cực, gây hấn
            7. Nội dung khiêu dâm, không phù hợp thuần phong mỹ tục

            {"Lưu ý phân tích thêm về ngữ cảnh và cách dùng từ trong tiếng Việt." if is_vietnamese else ""}

            Chỉ trả về kết quả dạng JSON với cấu trúc chính xác như sau:
            {{
                "is_toxic": true/false,
                "confidence": 0.95,
                "categories": ["loại vi phạm 1", "loại vi phạm 2"],
                "explanation": "giải thích ngắn gọn",
                "severity": "low/medium/high",
                "context_analysis": "nhận xét về ngữ cảnh"
            }}
            """
            genai.configure(api_key=GEMINI_API_KEY)
            model = genai.GenerativeModel(MODEL_NAME)
            response = model.generate_content(prompt)
            if not response or not response.text:
                raise Exception("Không nhận được phản hồi từ Gemini")
            response_text = response.text.strip()
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()
            result = json.loads(response_text)
            required_fields = ['is_toxic', 'confidence', 'categories', 'explanation', 'severity']
            for field in required_fields:
                if field not in result:
                    result[field] = None if field in ['explanation', 'severity'] else ([] if field == 'categories' else False if field == 'is_toxic' else 0.5)
            return result
        except Exception as e:
            logger.error(f"Lỗi khi gọi Gemini: {e}")
            return {
                "is_toxic": False,
                "confidence": 0.5,
                "categories": [],
                "explanation": f"Lỗi khi phân tích: {str(e)}",
                "severity": "low",
                "context_analysis": ""
            }

    def is_vietnamese(self, text):
        vietnamese_chars = "áàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđ"
        count = sum(1 for char in text.lower() if char in vietnamese_chars)
        return count > 0

    def is_toxic(self, text, threshold=0.5):
        if self._check_patterns(text):
            return True, 1.0
        is_vn = self.is_vietnamese(text)
        if is_vn:
            gemini_result = self.check_with_gemini(text, is_vietnamese=True)
            return gemini_result["is_toxic"], gemini_result["confidence"]
        else:
            model_score = self._get_model_score(text)
            if 0.4 <= model_score <= 0.6:
                gemini_result = self.check_with_gemini(text, is_vietnamese=False)
                if gemini_result["is_toxic"]:
                    return True, max(model_score, gemini_result["confidence"])
            return model_score > threshold, model_score

    def filter_comment(self, text, threshold=0.5):
        is_toxic, score = self.is_toxic(text, threshold)
        is_vn = self.is_vietnamese(text)
        if is_toxic or (0.4 <= score <= 0.6):
            gemini_result = self.check_with_gemini(text, is_vietnamese=is_vn)
            return {
                'is_toxic': is_toxic,
                'score': score,
                'message': 'Bình luận của bạn chứa nội dung không phù hợp.' if is_toxic else 'Bình luận phù hợp.',
                'details': {
                    'categories': gemini_result.get('categories', []),
                    'explanation': gemini_result.get('explanation', ''),
                    'severity': gemini_result.get('severity', 'low'),
                    'context_analysis': gemini_result.get('context_analysis', '')
                }
            }
        return {
            'is_toxic': False,
            'score': score,
            'message': 'Bình luận phù hợp.'
        }

comment_filter = ImprovedCommentFilter()

# Job recommendation functions (from File 1)
def load_jobs_from_csv(filepath, max_jobs=1400):
    global TFIDF_VECTORIZER, JOB_VECTOR_CACHE, jobs
    try:
        logger.info(f"Đang tải công việc từ {filepath}...")
        start_time = time.time()
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
        if TFIDF_VECTORIZER and jobs:
            job_texts = [" ".join(str(job.get(field, '')) for field in ['title', 'description', 'typeOfWork', 'companyName']) for job in jobs]
            try:
                TFIDF_VECTORIZER.fit(job_texts)
                logger.info("TF-IDF vectorizer đã được fit với dữ liệu công việc.")
            except Exception as e:
                logger.error(f"Lỗi khi fit TF-IDF vectorizer: {e}")
                TFIDF_VECTORIZER = None
        if os.path.exists(JOB_VECTOR_CACHE_FILE):
            try:
                JOB_VECTOR_CACHE.update(pd.read_pickle(JOB_VECTOR_CACHE_FILE))
                invalid_ids = [job_id for job_id, vec in JOB_VECTOR_CACHE.items() if not np.any(vec) or np.linalg.norm(vec) < 1e-6]
                for job_id in invalid_ids:
                    logger.warning(f"Xóa vector không hợp lệ trong cache cho Job ID {job_id}")
                    del JOB_VECTOR_CACHE[job_id]
                pd.to_pickle(JOB_VECTOR_CACHE, JOB_VECTOR_CACHE_FILE)
            except Exception as e:
                logger.error(f"Lỗi khi tải cache: {e}")
                JOB_VECTOR_CACHE.clear()
        if MODEL_RECOMMEND:
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                for job in jobs:
                    job_id = job.get('postId')
                    if job_id and job_id not in JOB_VECTOR_CACHE:
                        job_text = " ".join(str(job.get(field, '')) for field in ['title', 'description', 'typeOfWork', 'companyName'])
                        if not job_text.strip() or len(job_text.strip().split()) < 3:
                            JOB_VECTOR_CACHE[job_id] = np.zeros(768)
                            continue
                        vector = executor.submit(get_bert_vector, job_text, MODEL_RECOMMEND, None, DEVICE, TFIDF_VECTORIZER).result()
                        if np.any(vector) and np.linalg.norm(vector) >= 1e-6:
                            JOB_VECTOR_CACHE[job_id] = vector
                        else:
                            JOB_VECTOR_CACHE[job_id] = np.zeros(768)
            try:
                pd.to_pickle(JOB_VECTOR_CACHE, JOB_VECTOR_CACHE_FILE)
            except Exception as e:
                logger.error(f"Lỗi khi lưu cache: {e}")
        load_time = time.time() - start_time
        logger.info(f"Đã tải {len(jobs)} công việc trong {load_time:.2f} giây.")
        log_resource_usage()
        return jobs
    except Exception as e:
        logger.error(f"Lỗi khi tải công việc từ CSV: {e}")
        return []

def load_search_history_from_csv(filepath):
    global search_history, last_search_csv_modified
    try:
        if not os.path.exists(filepath):
            search_history = []
            return search_history
        current_modified = os.path.getmtime(filepath)
        if current_modified <= last_search_csv_modified:
            return search_history
        logger.info(f"Đang tải lịch sử tìm kiếm từ {filepath}...")
        start_time = time.time()
        df = pd.read_csv(filepath, encoding='utf-8', low_memory=False)
        if 'SeekerID' not in df or 'Search Query' not in df:
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
        load_time = time.time() - start_time
        logger.info(f"Đã tải {len(search_history)} mục lịch sử tìm kiếm trong {load_time:.2f} giây.")
        return search_history
    except Exception as e:
        logger.error(f"Lỗi khi tải lịch sử tìm kiếm từ CSV: {e}")
        search_history = []
        return search_history

def preprocess_text(text, bypass_stop_words=False, normalize_for_keywords=False):
    if not isinstance(text, str):
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
        for intent, kw_list in [
            ('driver', DRIVER_KEYWORDS), ('tech', TECH_KEYWORDS), ('ecommerce', ECOMMERCE_KEYWORDS),
            ('marketing', MARKETING_KEYWORDS), ('it_hardware', IT_HARDWARE_KEYWORDS),
            ('it_software', IT_SOFTWARE_KEYWORDS), ('hospitality', HOSPITALITY_KEYWORDS),
            ('design', DESIGN_KEYWORDS), ('mechanical', MECHANICAL_KEYWORDS),
            ('business', BUSINESS_KEYWORDS), ('education', EDUCATION_KEYWORDS),
            ('construction', CONSTRUCTION_KEYWORDS), ('finance', FINANCE_KEYWORDS),
            ('telecom', TELECOM_KEYWORDS), ('healthcare', HEALTHCARE_KEYWORDS),
            ('logistics', LOGISTICS_KEYWORDS), ('accounting', ACCOUNTING_KEYWORDS),
            ('manufacturing', MANUFACTURING_KEYWORDS), ('legal', LEGAL_KEYWORDS),
            ('translation', TRANSLATION_KEYWORDS), ('embedded_iot', EMBEDDED_IOT_KEYWORDS)
        ]:
            for kw, kw_no_space in [normalize_keyword(k) for k in kw_list]:
                if kw in processed_query or kw_no_space in no_space_query:
                    query_intents.add(intent)
                    if intent in RELATED_INTENTS:
                        query_intents.update(RELATED_INTENTS[intent])
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
        if not query_intents or any(intent in query_intents for intent in ['other']):
            filtered_jobs.append(job)
            continue
        intent_match = False
        no_space_job_title = re.sub(r'\s+', '', job_title)
        for intent in query_intents:
            keywords = {
                'driver': [normalize_keyword(k) for k in DRIVER_KEYWORDS],
                'tech': [normalize_keyword(k) for k in TECH_KEYWORDS],
                'ecommerce': [normalize_keyword(k) for k in ECOMMERCE_KEYWORDS],
                'marketing': [normalize_keyword(k) for k in MARKETING_KEYWORDS],
                'it_hardware': [normalize_keyword(k) for k in IT_HARDWARE_KEYWORDS],
                'it_software': [normalize_keyword(k) for k in IT_SOFTWARE_KEYWORDS],
                'hospitality': [normalize_keyword(k) for k in HOSPITALITY_KEYWORDS],
                'design': [normalize_keyword(k) for k in DESIGN_KEYWORDS],
                'mechanical': [normalize_keyword(k) for k in MECHANICAL_KEYWORDS],
                'business': [normalize_keyword(k) for k in BUSINESS_KEYWORDS],
                'education': [normalize_keyword(k) for k in EDUCATION_KEYWORDS],
                'construction': [normalize_keyword(k) for k in CONSTRUCTION_KEYWORDS],
                'finance': [normalize_keyword(k) for k in FINANCE_KEYWORDS],
                'telecom': [normalize_keyword(k) for k in TELECOM_KEYWORDS],
                'healthcare': [normalize_keyword(k) for k in HEALTHCARE_KEYWORDS],
                'logistics': [normalize_keyword(k) for k in LOGISTICS_KEYWORDS],
                'accounting': [normalize_keyword(k) for k in ACCOUNTING_KEYWORDS],
                'manufacturing': [normalize_keyword(k) for k in MANUFACTURING_KEYWORDS],
                'legal': [normalize_keyword(k) for k in LEGAL_KEYWORDS],
                'translation': [normalize_keyword(k) for k in TRANSLATION_KEYWORDS],
                'embedded_iot': [normalize_keyword(k) for k in EMBEDDED_IOT_KEYWORDS]
            }.get(intent, [])
            for kw, kw_no_space in keywords:
                if kw in job_title or kw_no_space in no_space_job_title:
                    intent_match = True
                    break
        if intent_match:
            filtered_jobs.append(job)
    return filtered_jobs[:top_n]

@timeout(30)
def get_bert_vector(text, model, tokenizer, device, tfidf_vectorizer, bypass_tfidf=False):
    if model is None:
        return np.zeros(768)
    if not isinstance(text, str) or not text.strip():
        return np.zeros(768)
    original_text = text
    text = preprocess_text(text, bypass_stop_words=False, normalize_for_keywords=False)
    if not text:
        text = preprocess_text(original_text, bypass_stop_words=True, normalize_for_keywords=False)
        if not text:
            return np.zeros(768)
    if len(text) > 1400:
        text = text[:1400]
    try:
        processed_text = text
        if tfidf_vectorizer and hasattr(tfidf_vectorizer, 'vocabulary_') and not bypass_tfidf:
            tfidf_vec = tfidf_vectorizer.transform([text]).toarray()[0]
            important_words = [word for word, idx in tfidf_vectorizer.vocabulary_.items() if tfidf_vec[idx] > 0.2]
            if important_words:
                processed_text = " ".join(important_words)
        vector = model.encode(processed_text, device=device, normalize_embeddings=True, show_progress_bar=False)
        if np.allclose(vector, np.zeros(768), atol=1e-6):
            if not bypass_tfidf:
                return get_bert_vector(original_text, model, tokenizer, device, tfidf_vectorizer, bypass_tfidf=True)
        return vector
    except Exception as e:
        logger.error(f"Lỗi khi tạo vector: {e}")
        return np.zeros(768)

@timeout(60)
def get_bert_recommendations(user_history, all_jobs, model, tokenizer, device, tfidf_vectorizer, top_n=8):
    start_time = time.time()
    if not user_history or not all_jobs or model is None:
        return [], []
    recent_search = user_history[0]
    query_text = recent_search.get('Search Query', '').replace('Title:', '').strip()
    processed_query = preprocess_text(query_text, normalize_for_keywords=True)
    no_space_query = re.sub(r'\s+', '', processed_query)
    query_intents = set()
    for intent, kw_list in [
        ('driver', DRIVER_KEYWORDS), ('tech', TECH_KEYWORDS), ('ecommerce', ECOMMERCE_KEYWORDS),
        ('marketing', MARKETING_KEYWORDS), ('it_hardware', IT_HARDWARE_KEYWORDS),
        ('it_software', IT_SOFTWARE_KEYWORDS), ('hospitality', HOSPITALITY_KEYWORDS),
        ('design', DESIGN_KEYWORDS), ('mechanical', MECHANICAL_KEYWORDS),
        ('business', BUSINESS_KEYWORDS), ('education', EDUCATION_KEYWORDS),
        ('construction', CONSTRUCTION_KEYWORDS), ('finance', FINANCE_KEYWORDS),
        ('telecom', TELECOM_KEYWORDS), ('healthcare', HEALTHCARE_KEYWORDS),
        ('logistics', LOGISTICS_KEYWORDS), ('accounting', ACCOUNTING_KEYWORDS),
        ('manufacturing', MANUFACTURING_KEYWORDS), ('legal', LEGAL_KEYWORDS),
        ('translation', TRANSLATION_KEYWORDS), ('embedded_iot', EMBEDDED_IOT_KEYWORDS)
    ]:
        for kw, kw_no_space in [normalize_keyword(k) for k in kw_list]:
            if kw in processed_query or kw_no_space in no_space_query:
                query_intents.add(intent)
                if intent in RELATED_INTENTS:
                    query_intents.update(RELATED_INTENTS[intent])
    user_query_vector = get_bert_vector(query_text, model, tokenizer, device, tfidf_vectorizer)
    if not np.any(user_query_vector) or np.linalg.norm(user_query_vector) < 1e-6:
        return [], []
    user_query_vector /= np.linalg.norm(user_query_vector)
    recent_jobs = filter_jobs_by_category(all_jobs, [recent_search], top_n=2000)
    job_similarities = []
    processed_job_ids = set()
    for job in recent_jobs:
        job_id = job.get('postId')
        if job_id in processed_job_ids:
            continue
        processed_job_ids.add(job_id)
        if not job_id:
            continue
        job_text = " ".join(str(job.get(field, '')) for field in ['title', 'description', 'typeOfWork', 'companyName'])
        if not job_text.strip() or len(job_text.strip().split()) < 3:
            continue
        if job_id not in JOB_VECTOR_CACHE:
            try:
                vector = get_bert_vector(job_text, model, tokenizer, device, tfidf_vectorizer)
                if np.any(vector) and np.linalg.norm(vector) >= 1e-6:
                    JOB_VECTOR_CACHE[job_id] = vector
                    pd.to_pickle(JOB_VECTOR_CACHE, JOB_VECTOR_CACHE_FILE)
                else:
                    continue
            except Exception as e:
                continue
        else:
            job_vector = JOB_VECTOR_CACHE[job_id]
            if not np.any(job_vector) or np.linalg.norm(job_vector) < 1e-6:
                try:
                    vector = get_bert_vector(job_text, model, tokenizer, device, tfidf_vectorizer, bypass_tfidf=True)
                    if np.any(vector) and np.linalg.norm(vector) >= 1e-6:
                        JOB_VECTOR_CACHE[job_id] = vector
                        pd.to_pickle(JOB_VECTOR_CACHE, JOB_VECTOR_CACHE_FILE)
                    else:
                        continue
                except Exception as e:
                    continue
            else:
                job_vector = JOB_VECTOR_CACHE[job_id]
        similarity = cosine_similarity(user_query_vector.reshape(1, -1), job_vector.reshape(1, -1))[0][0]
        if similarity > 0.3:
            job_title = job.get('title', '').lower()
            no_space_job_title = re.sub(r'\s+', '', job_title)
            intent_boost = 1.0
            for intent in query_intents:
                kw_list = {
                    'driver': DRIVER_KEYWORDS, 'tech': TECH_KEYWORDS, 'ecommerce': ECOMMERCE_KEYWORDS,
                    'marketing': MARKETING_KEYWORDS, 'it_hardware': IT_HARDWARE_KEYWORDS,
                    'it_software': IT_SOFTWARE_KEYWORDS, 'hospitality': HOSPITALITY_KEYWORDS,
                    'design': DESIGN_KEYWORDS, 'mechanical': MECHANICAL_KEYWORDS,
                    'business': BUSINESS_KEYWORDS, 'education': EDUCATION_KEYWORDS,
                    'construction': CONSTRUCTION_KEYWORDS, 'finance': FINANCE_KEYWORDS,
                    'telecom': TELECOM_KEYWORDS, 'healthcare': HEALTHCARE_KEYWORDS,
                    'logistics': LOGISTICS_KEYWORDS, 'accounting': ACCOUNTING_KEYWORDS,
                    'manufacturing': MANUFACTURING_KEYWORDS, 'legal': LEGAL_KEYWORDS,
                    'translation': TRANSLATION_KEYWORDS, 'embedded_iot': EMBEDDED_IOT_KEYWORDS
                }.get(intent, [])
                if any(kw in job_title or kw_no_space in no_space_job_title for kw, kw_no_space in [normalize_keyword(k) for k in kw_list]):
                    intent_boost = 1.0
                    break
            boosted_similarity = similarity * intent_boost
            job_similarities.append({'job': job, 'similarity': boosted_similarity})
    if not job_similarities:
        return [], []
    sorted_jobs = sorted(job_similarities, key=lambda x: x['similarity'], reverse=True)
    top_results = sorted_jobs[:top_n]
    recommended_job_dicts = [item['job'] for item in top_results]
    similarity_scores = [item['similarity'] for item in top_results]
    total_time = time.time() - start_time
    logger.info(f"Đã chọn top {len(recommended_job_dicts)} gợi ý trong {total_time:.2f} giây.")
    log_resource_usage()
    return recommended_job_dicts, similarity_scores

# CV analysis functions (from File 2)
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
    job_description = job_data.get('description', '')
    job_requirements = job_data.get('requirements', '')
    job_benefits = job_data.get('benefit', '')
    job_experience = job_data.get('experience', '')
    job_skills = job_data.get('skills', [])
    job_position = job_data.get('position', '')
    job_nice_to_haves = job_data.get('niceTo', '')
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
            "totalScore": 0,
            "matchedSkills": [],
            "missingSkills": [],
            "extraSkills": [],
            "detailedScores": {{
                "skills_match": 0,
                "education_match": 0,
                "experience_match": 0,
                "overall_similarity": 0,
                "context_score": 0
            }},
            "suitabilityLevel": "Not Well Suited",
            "recommendations": [],
            "cvImprovementSuggestions": []
        }},
        "detailedAnalysis": {{
            "skills": {{
                "score": 0,
                "matched_skills": [],
                "missing_skills": [],
                "reason": ""
            }},
            "education": {{
            "score": 0,
            "matched_education": [],
            "missing_education": [],
            "reason": ""
            }},
            "experience": {{
                "score": 0,
                "matched_experience": [],
                "missing_experience": [],
                "reason": ""
            }},
            "overall_similarity": {{
                "score": 0,
                "reason": ""
            }}
        }}
    }}
    ```
    Lưu ý:
    - CV và mô tả công việc có thể bằng tiếng Việt hoặc tiếng Anh. Hãy phân tích đúng ngữ cảnh ngôn ngữ.
    - Nếu CV hoặc mô tả công việc thiếu thông tin, hãy đưa ra giả định hợp lý và ghi rõ trong "reason".
    - Điểm số (score) từ 0 đến 100. Điểm totalScore là trung bình có trọng số của các điểm thành phần (skills: 40%, education: 20%, experience: 30%, overall_similarity: 10%).
    - suitabilityLevel có thể là: "Not Well Suited", "Somewhat Suited", "Well Suited", "Highly Suited".
    - Đưa ra ít nhất 2 khuyến nghị (recommendations) và 2 gợi ý cải thiện CV (cvImprovementSuggestions).
    """
    return prompt

def get_cache_key(data):
    if isinstance(data, dict):
        data_str = json.dumps(data, sort_keys=True)
    else:
        data_str = str(data)
    return hashlib.md5(data_str.encode('utf-8')).hexdigest()

def cache_result(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        cache_key = get_cache_key((args, kwargs))
        if redis_client:
            try:
                cached = redis_client.get(cache_key)
                if cached:
                    return pickle.loads(cached)
            except Exception as e:
                logger.error(f"Lỗi khi đọc từ Redis: {e}")
        elif cache_key in in_memory_cache:
            return in_memory_cache[cache_key]
        result = func(*args, **kwargs)
        if redis_client:
            try:
                redis_client.setex(cache_key, 3600, pickle.dumps(result))
            except Exception as e:
                logger.error(f"Lỗi khi lưu vào Redis: {e}")
        else:
            in_memory_cache[cache_key] = result
            if len(in_memory_cache) > EMBEDDING_CACHE_SIZE:
                in_memory_cache.pop(next(iter(in_memory_cache)))
        return result
    return wrapper

# Job recommendation endpoints
@app.route('/recommend-jobs/phobert', methods=['POST'])
def recommend_jobs_phobert():
    try:
        data = request.get_json()
        seeker_id = data.get('seekerId')
        if not seeker_id:
            return jsonify({"error": "Thiếu seekerId"}), 400
        load_search_history_from_csv(SEARCH_HISTORY_FILEPATH)
        user_history = [h for h in search_history if str(h.get('SeekerID')) == str(seeker_id)]
        if not user_history:
            return jsonify({"error": "Không tìm thấy lịch sử tìm kiếm cho seekerId này"}), 404
        global jobs
        if not jobs:
            jobs = load_jobs_from_csv(JOBS_FILEPATH)
        if not jobs:
            return jsonify({"error": "Không có công việc nào được tải"}), 500
        recommended_jobs, similarity_scores = get_bert_recommendations(
            user_history, jobs, MODEL_RECOMMEND, None, DEVICE, TFIDF_VECTORIZER, top_n=8
        )
        response = []
        for job, score in zip(recommended_jobs, similarity_scores):
            job_data = {
                'postId': job.get('postId'),
                'title': job.get('title'),
                'companyId': job.get('companyId'),
                'companyName': job.get('companyName'),
                'salary': job.get('salary'),
                'location': job.get('location'),
                'createDate': job.get('createDate'),
                'expireDate': job.get('expireDate'),
                'description': job.get('description'),
                'typeOfWork': job.get('typeOfWork'),
                'cityName': job.get('cityName'),
                'logo': job.get('logo'),
                'industryNames': job.get('industryNames'),
                'similarity_score': float(score)
            }
            response.append(job_data)
        return jsonify({"recommended_jobs": response}), 200
    except Exception as e:
        logger.error(f"Lỗi khi gợi ý công việc: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/save-search', methods=['POST'])
def save_search():
    try:
        data = request.get_json()
        seeker_id = data.get('seekerId')
        search_query = data.get('searchQuery')
        if not seeker_id or not search_query:
            return jsonify({"error": "Thiếu seekerId hoặc searchQuery"}), 400
        search_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        search_entry = {
            'SeekerID': seeker_id,
            'Search Query': search_query,
            'Search Date': search_date
        }
        global search_history
        with csv_write_lock:
            search_history.append(search_entry)
            try:
                df = pd.DataFrame([search_entry])
                if os.path.exists(SEARCH_HISTORY_FILEPATH):
                    df.to_csv(SEARCH_HISTORY_FILEPATH, mode='a', header=False, index=False, encoding='utf-8')
                else:
                    df.to_csv(SEARCH_HISTORY_FILEPATH, mode='w', header=True, index=False, encoding='utf-8')
                global last_search_csv_modified
                last_search_csv_modified = time.time()
            except Exception as e:
                logger.error(f"Lỗi khi lưu lịch sử tìm kiếm vào CSV: {e}")
                return jsonify({"error": "Không thể lưu lịch sử tìm kiếm"}), 500
        return jsonify({"message": "Lịch sử tìm kiếm đã được lưu"}), 200
    except Exception as e:
        logger.error(f"Lỗi khi lưu tìm kiếm: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    try:
        health_status = {
            "status": "healthy",
            "model_recommend_loaded": MODEL_RECOMMEND is not None,
            "model_analyze_loaded": MODEL_ANALYZE is not None,
            "device": str(DEVICE) if DEVICE else "not set",
            "jobs_loaded": len(jobs) if jobs else 0,
            "search_history_loaded": len(search_history) if search_history else 0,
            "redis_connected": redis_client.ping() if redis_client else False,
            "memory_usage_mb": psutil.Process().memory_info().rss / 1024 / 1024,
            "cpu_usage_percent": psutil.cpu_percent(interval=1)
        }
        return jsonify(health_status), 200
    except Exception as e:
        logger.error(f"Lỗi khi kiểm tra sức khỏe: {e}", exc_info=True)
        return jsonify({"status": "unhealthy", "error": str(e)}), 500

# CV analysis endpoints
@app.route('/analyze', methods=['POST'])
@cache_result
def analyze_cv():
    try:
        if 'cv' not in request.files or not request.form.get('job_data'):
            return jsonify({"error": "Thiếu file CV hoặc dữ liệu công việc"}), 400
        cv_file = request.files['cv']
        job_data = json.loads(request.form.get('job_data'))
        if not cv_file or not job_data:
            return jsonify({"error": "Dữ liệu không hợp lệ"}), 400
        file_extension = cv_file.filename.rsplit('.', 1)[-1].lower()
        if file_extension == 'pdf':
            cv_text = extract_text_from_pdf(cv_file)
        elif file_extension == 'docx':
            cv_text = extract_text_from_docx(cv_file)
        else:
            return jsonify({"error": "Định dạng file không được hỗ trợ. Chỉ hỗ trợ PDF hoặc DOCX."}), 400
        if not cv_text.strip():
            return jsonify({"error": "Không thể trích xuất nội dung từ CV"}), 400
        setup_gemini(GEMINI_API_KEY)
        prompt = create_analysis_prompt(cv_text, job_data)
        model = genai.GenerativeModel(MODEL_NAME)
        response = model.generate_content(prompt)
        if not response or not response.text:
            return jsonify({"error": "Không nhận được phản hồi từ Gemini"}), 500
        response_text = response.text.strip()
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()
        result = json.loads(response_text)
        return jsonify(result), 200
    except Exception as e:
        logger.error(f"Lỗi khi phân tích CV: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/analyze/document', methods=['POST'])
def analyze_document():
    try:
        if 'document' not in request.files:
            return jsonify({'error': 'Không có file được tải lên'}), 400
        document = request.files['document']
        file_extension = document.filename.rsplit('.', 1)[-1].lower()
        if file_extension == 'pdf':
            text = extract_text_from_pdf(document)
        elif file_extension == 'docx':
            text = extract_text_from_docx(document)
        else:
            return jsonify({'error': 'Định dạng file không được hỗ trợ. Chỉ hỗ trợ PDF hoặc DOCX.'}), 400
        if not text.strip():
            return jsonify({'error': 'Không thể trích xuất nội dung từ file'}), 400
        prompt = f"""
        Phân tích nội dung tài liệu sau và trích xuất thông tin quan trọng:
        - Kỹ năng
        - Trình độ học vấn
        - Kinh nghiệm làm việc
        - Thành tựu nổi bật
        Tài liệu:
        ```
        {text}
        ```
        Trả về kết quả dạng JSON:
        ```json
        {{
            "skills": [],
            "education": [],
            "experience": [],
            "achievements": []
        }}
        ```
        """
        setup_gemini(GEMINI_API_KEY)
        model = genai.GenerativeModel(MODEL_NAME)
        response = model.generate_content(prompt)
        if not response.text:
            return jsonify({'error': 'Không nhận được phản hồi từ Gemini API'}), 500
        response_text = response.text.strip()
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        result = json.loads(response_text)
        return jsonify(result), 200
    except Exception as e:
        logger.error(f"Lỗi khi phân tích tài liệu: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/analyze/job', methods=['POST'])
def analyze_job():
    try:
        data = request.get_json()
        job_data = data.get('job_data')
        if not job_data:
            return jsonify({'error': "Thiếu dữ liệu công việc"}), 400  # Sửa lỗi: thêm dấu nháy đơn
        
        prompt = f"""
        Phân tích mô tả công việc sau và trích xuất các yêu cầu chính:
        - Kỹ năng bắt buộc
        - Trình độ học vấn tối thiểu
        - Kinh nghiệm yêu cầu
        - Trách nhiệm công việc
        - Kỹ năng ưu tiên (nếu có)
        Mô tả công việc:
        ```
        {json.dumps(job_data, ensure_ascii=False)}
        ```
        Trả về kết quả dạng JSON:
        ```json
        {{
            "required_skills": [],
            "education": "",
            "experience": [],
            "responsibilities": [],
            "preferred_skills": []
        }}
        ```
        """  # Sửa lỗi: thêm dấu nháy ba đóng và dấu ngoặc nhọn
        
        setup_gemini(GEMINI_API_KEY)
        model = genai.GenerativeModel(MODEL_NAME)
        response = model.generate_content(prompt)
        
        if not response.text:
            return jsonify({'error': 'Không nhận được phản hồi từ Gemini API'}), 500
        
        response_text = response.text.strip()
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        
        result = json.loads(response_text)
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"Lỗi khi phân tích mô tả công việc: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/semantic-search', methods=['POST'])
def semantic_search():
    try:
        data = request.get_json()
        query = data.get('query', '')
        top_n = data.get('top_n', 10)
        
        if not query:
            return jsonify({'error': "Thiếu câu truy vấn"}), 400  # Sửa lỗi: thêm dấu nháy đơn
        
        if not isinstance(top_n, int) or top_n <= 0:
            top_n = 10

        query_embedding = MODEL_ANALYZE.encode([query], normalize_embeddings=True)[0]
        results = []
        
        for doc_id, embedding in job_embeddings.items():
            if embedding is not None:
                similarity = cosine_similarity([query_embedding], [embedding])[0][0]
                results.append({'doc_id': doc_id, 'score': similarity})

        results = sorted(results, key=lambda x: x['score'], reverse=True)[:top_n]
        response = []
        
        for result in results:
            doc_id = result['doc_id']
            if doc_id in job_index:
                response.append({
                    'id': doc_id,
                    'job_data': job_index[doc_id],  # Sửa lỗi: bỏ dấu phẩy và {}, thay bằng ]
                    'score': result['score']
                })

        return jsonify({'results': response}), 200

    except Exception as e:
        logger.error(f"Lỗi khi thực hiện tìm kiếm ngữ nghĩa: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/check-comment', methods=['POST'])
def check_comment():
    try:
        data = request.get_json()
        comment = data.get('comment', '')
        threshold = data.get('threshold', 0.5)  # Sửa lỗi: bỏ float(), dùng get() thông thường
        
        if not comment:
            return jsonify({'error': "Thiếu bình luận để kiểm tra"}), 400  # Sửa lỗi: thêm dấu nháy đơn
        
        result = comment_filter.filter_comment(comment, threshold=threshold)
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"Lỗi khi kiểm tra bình luận: {e}")
        return jsonify({'error': str(e)}), 500

# Initialize models and data
if __name__ == '__main__':
    try:
        load_recommend_mbert()
        load_analyze_model()
        jobs = load_jobs_from_csv(JOBS_FILEPATH)
        load_search_history_from_csv(SEARCH_HISTORY_FILEPATH)
        
        # Precompute embeddings for semantic search
        if PRECOMPUTE_EMBEDDINGS and MODEL_ANALYZE and jobs:
            logger.info("Precomputing job embeddings for semantic search...")
            for job in jobs:
                job_id = job.get('postId')
                job_text = " ".join(str(job.get(field, '')) for field in ['title', 'description', 'typeOfWork', 'companyName'])
                if job_id and job_text.strip():
                    job_embeddings[str(job_id)] = MODEL_ANALYZE.encode(job_text, normalize_embeddings=True)
                    job_index[str(job_id)] = job
            logger.info(f"Precomputed embeddings for {len(job_embeddings)} jobs.")

        app.run(host='0.0.0.0', port=int(os.getenv('PORT', 8000)), debug=False)
        
    except Exception as e:
        logger.error(f"Lỗi khi khởi động ứng dụng: {e}", exc_info=True)
        raise