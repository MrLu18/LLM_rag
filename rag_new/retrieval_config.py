"""
检索算法配置文件
包含不同的检索策略配置，可以根据具体需求选择
"""

# 基础检索配置
BASIC_RETRIEVAL = {
    "chunk_size": 512,
    "chunk_overlap": 100,
    "search_k": 10,
    "search_type": "similarity"
}

# 高精度检索配置（适合精确问答）
HIGH_PRECISION_RETRIEVAL = {
    "chunk_size": 256,  # 更小的块，更精确
    "chunk_overlap": 50,
    "search_k": 5,  # 更少的文档，但更相关
    "search_type": "similarity_score_threshold",
    "score_threshold": 0.8,  # 高阈值
    "fetch_k": 15,
    "rerank_top_k": 3
}

# 高召回率检索配置（适合全面搜索）
HIGH_RECALL_RETRIEVAL = {
    "chunk_size": 1024,  # 更大的块，包含更多上下文
    "chunk_overlap": 200,
    "search_k": 20,  # 更多文档
    "search_type": "mmr",
    "fetch_k": 30,
    "lambda_mult": 0.5,  # 平衡相关性和多样性
    "rerank_top_k": 10
}

# 混合检索配置（关键词+语义）
HYBRID_RETRIEVAL = {
    "chunk_size": 512,
    "chunk_overlap": 100,
    "search_k": 15,
    "search_type": "similarity",
    "use_hybrid_search": True,
    "keyword_weight": 0.3,
    "semantic_weight": 0.7,
    "fetch_k": 25,
    "rerank_top_k": 8
}

# 煤矿领域专用配置
MINING_DOMAIN_RETRIEVAL = {
    "chunk_size": 768,  # 适合技术文档
    "chunk_overlap": 150,
    "search_k": 12,
    "search_type": "mmr",
    "lambda_mult": 0.4,
    "use_query_expansion": True,
    "expansion_terms": 5,
    "fetch_k": 20,
    "rerank_top_k": 6,
    # 煤矿领域特定同义词
    "domain_synonyms": {
        "煤矿": ["矿井", "矿山", "采煤", "煤矿企业"],
        "安全": ["安全生产", "安全管理", "安全措施", "安全规程"],
        "设备": ["机械", "装置", "设施", "采掘设备"],
        "人员": ["员工", "工人", "职工", "矿工"],
        "培训": ["教育", "学习", "训练", "安全培训"],
        "通风": ["通风系统", "通风管理", "通风设备"],
        "瓦斯": ["瓦斯检测", "瓦斯管理", "瓦斯防治"],
        "顶板": ["顶板管理", "顶板支护", "顶板安全"]
    }
}

# 快速检索配置（适合实时问答）
FAST_RETRIEVAL = {
    "chunk_size": 1024,  # 大块减少处理时间
    "chunk_overlap": 100,
    "search_k": 5,  # 最少文档
    "search_type": "similarity",
    "fetch_k": 8,
    "rerank_top_k": 3,
    "use_query_expansion": False,
    "use_hybrid_search": False
}

# 配置选择函数
def get_retrieval_config(config_name="basic"):
    """根据配置名称返回检索配置"""
    configs = {
        "basic": BASIC_RETRIEVAL,
        "precision": HIGH_PRECISION_RETRIEVAL,
        "recall": HIGH_RECALL_RETRIEVAL,
        "hybrid": HYBRID_RETRIEVAL,
        "mining": MINING_DOMAIN_RETRIEVAL,
        "fast": FAST_RETRIEVAL
    }
    
    return configs.get(config_name, BASIC_RETRIEVAL)

def apply_retrieval_config(config):
    """将配置应用到全局变量"""
    global CHUNK_SIZE, CHUNK_OVERLAP, SEARCH_K, SEARCH_TYPE
    global SIMILARITY_THRESHOLD, MMR_DIVERSITY_FACTOR, FETCH_K, RERANK_TOP_K
    global USE_HYBRID_SEARCH, KEYWORD_WEIGHT, SEMANTIC_WEIGHT
    global USE_QUERY_EXPANSION, EXPANSION_TERMS
    
    CHUNK_SIZE = config.get("chunk_size", 512)
    CHUNK_OVERLAP = config.get("chunk_overlap", 100)
    SEARCH_K = config.get("search_k", 10)
    SEARCH_TYPE = config.get("search_type", "similarity")
    SIMILARITY_THRESHOLD = config.get("score_threshold", 0.7)
    MMR_DIVERSITY_FACTOR = config.get("lambda_mult", 0.3)
    FETCH_K = config.get("fetch_k", 20)
    RERANK_TOP_K = config.get("rerank_top_k", 5)
    USE_HYBRID_SEARCH = config.get("use_hybrid_search", False)
    KEYWORD_WEIGHT = config.get("keyword_weight", 0.3)
    SEMANTIC_WEIGHT = config.get("semantic_weight", 0.7)
    USE_QUERY_EXPANSION = config.get("use_query_expansion", False)
    EXPANSION_TERMS = config.get("expansion_terms", 3)
    
    print(f"已应用检索配置: {config.get('name', 'unknown')}")
    print(f"块大小: {CHUNK_SIZE}, 检索数量: {SEARCH_K}, 检索类型: {SEARCH_TYPE}")

# 性能测试查询
TEST_QUERIES = [
    ("煤矿安全生产规程有哪些？", ["安全", "规程", "生产"]),
    ("瓦斯检测设备的使用方法", ["瓦斯", "检测", "设备", "使用"]),
    ("矿工安全培训内容包括什么？", ["培训", "安全", "内容", "矿工"]),
    ("通风系统如何维护？", ["通风", "系统", "维护"]),
    ("顶板支护技术有哪些？", ["顶板", "支护", "技术"])
]

if __name__ == "__main__":
    # 测试配置加载
    config = get_retrieval_config("mining")
    print("煤矿领域检索配置:")
    for key, value in config.items():
        print(f"  {key}: {value}") 