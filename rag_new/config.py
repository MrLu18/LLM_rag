# --- Configuration ---
DOCUMENTS_DIR = "./documents"  # Modify to your document directory
PERSIST_DIR = "./chroma_db"     # Vector database storage directory 向量数据库 存储数据的（个人理解）
EMBEDDING_MODEL_PATH = "model/bge-m3" # 嵌入模型路径 将文本转换为向量  
EMBEDDING_DEVICE = "cuda:1" # Or 'cpu' 嵌入模型设备
# VLLM Server details (using OpenAI compatible endpoint)
VLLM_BASE_URL = "http://localhost:7861/v1"  # 使用正确的端口 7861
#VLLM_BASE_URL = "http://172.16.20.193:8000/v1"  
VLLM_API_KEY = "dummy-key" # Required by ChatOpenAI, but VLLM server doesn't usually check it 
VLLM_MODEL_NAME = "/mnt/jrwbxx/LLM/model/qwen3-1.7b"  # 修正模型路径

# 检索参数 检索的配置 视情况改
CHUNK_SIZE = 1000 # Adjusted for bge-m3, which can handle more context  文本块大小
CHUNK_OVERLAP = 100  # Adjusted overlap (approx 20% of CHUNK_SIZE)  文本块重叠大小 这个的目的我个人觉得是确保每个块之间有联系
SEARCH_K = 5# Retrieve more chunks to increase chances of finding specific sentences  检索到的结果的数量
SIMILARYTY_MODEL = "paraphrase-MiniLM-L6-v2"
# --- End Configuration --- 