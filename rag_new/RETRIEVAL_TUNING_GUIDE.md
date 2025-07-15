# RAG 检索算法调优指南

## 概述

本指南介绍如何在 `rag_client.py` 中调优检索算法，而不是更换大模型或嵌入模型路径。检索算法的调优对RAG系统的性能有重要影响。

## 检索算法调优参数

### 1. 基础参数

```python
CHUNK_SIZE = 512        # 文本块大小
CHUNK_OVERLAP = 100     # 文本块重叠
SEARCH_K = 10          # 检索结果数量
```

### 2. 高级检索参数

```python
SEARCH_TYPE = "similarity"              # 检索类型
SIMILARITY_THRESHOLD = 0.7             # 相似度阈值
MMR_DIVERSITY_FACTOR = 0.3             # MMR多样性因子
FETCH_K = 20                          # 初始获取文档数量
RERANK_TOP_K = 5                      # 重排序后保留数量
```

### 3. 混合检索参数

```python
USE_HYBRID_SEARCH = False             # 启用混合检索
KEYWORD_WEIGHT = 0.3                  # 关键词权重
SEMANTIC_WEIGHT = 0.7                 # 语义权重
```

### 4. 查询扩展参数

```python
USE_QUERY_EXPANSION = False           # 启用查询扩展
EXPANSION_TERMS = 3                   # 扩展术语数量
```

## 检索策略配置

系统提供了多种预配置的检索策略：

### 1. 基础检索 (basic)
- 适合一般用途
- 平衡精度和召回率

### 2. 高精度检索 (precision)
- 适合精确问答
- 更小的文本块，更高的相似度阈值
- 检索更少但更相关的文档

### 3. 高召回率检索 (recall)
- 适合全面搜索
- 更大的文本块，更多文档
- 使用MMR算法平衡相关性和多样性

### 4. 混合检索 (hybrid)
- 结合关键词和语义检索
- 提高检索的鲁棒性

### 5. 煤矿领域检索 (mining)
- 专门针对煤矿领域优化
- 包含领域特定同义词扩展
- 适合技术文档

### 6. 快速检索 (fast)
- 适合实时问答
- 大块减少处理时间
- 最少文档数量

## 如何调优

### 1. 通过配置文件调优

编辑 `retrieval_config.py` 中的配置：

```python
# 示例：自定义高精度配置
CUSTOM_PRECISION = {
    "chunk_size": 256,
    "chunk_overlap": 50,
    "search_k": 5,
    "search_type": "similarity_score_threshold",
    "score_threshold": 0.85,  # 更高阈值
    "fetch_k": 15,
    "rerank_top_k": 3
}
```

### 2. 通过界面调优

在Gradio界面中：
1. 切换到"上传与管理文档"标签页
2. 选择检索策略配置
3. 点击"应用配置"
4. 使用"测试检索性能"评估效果

### 3. 参数调优建议

#### 文本块大小 (CHUNK_SIZE)
- **小块 (256-512)**: 适合精确问答，减少噪声
- **中块 (512-768)**: 平衡精度和上下文
- **大块 (768-1024)**: 适合需要更多上下文的复杂问题

#### 检索数量 (SEARCH_K)
- **少量 (3-5)**: 高精度，快速响应
- **中等 (8-12)**: 平衡精度和召回率
- **大量 (15-20)**: 高召回率，适合全面搜索

#### 相似度阈值 (SIMILARITY_THRESHOLD)
- **高阈值 (0.8-0.9)**: 只返回高度相关的文档
- **中等阈值 (0.6-0.8)**: 平衡相关性和数量
- **低阈值 (0.4-0.6)**: 返回更多文档，可能包含噪声

#### MMR多样性因子 (MMR_DIVERSITY_FACTOR)
- **低值 (0.1-0.3)**: 注重相关性
- **高值 (0.5-0.7)**: 注重多样性

## 性能评估

系统提供了检索性能评估功能：

1. 使用预定义的测试查询
2. 比较不同检索策略的效果
3. 计算平均相关性分数
4. 帮助选择最佳配置

## 常见调优场景

### 场景1：精确问答
```python
# 配置建议
CHUNK_SIZE = 256
SEARCH_K = 5
SEARCH_TYPE = "similarity_score_threshold"
SIMILARITY_THRESHOLD = 0.8
```

### 场景2：全面搜索
```python
# 配置建议
CHUNK_SIZE = 1024
SEARCH_K = 20
SEARCH_TYPE = "mmr"
MMR_DIVERSITY_FACTOR = 0.5
```

### 场景3：实时问答
```python
# 配置建议
CHUNK_SIZE = 1024
SEARCH_K = 5
USE_QUERY_EXPANSION = False
USE_HYBRID_SEARCH = False
```

### 场景4：技术文档
```python
# 配置建议
CHUNK_SIZE = 768
SEARCH_K = 12
USE_QUERY_EXPANSION = True
EXPANSION_TERMS = 5
```

## 注意事项

1. **向量数据库重建**: 更改 `CHUNK_SIZE` 或 `CHUNK_OVERLAP` 需要重建向量数据库
2. **嵌入模型兼容性**: 确保嵌入模型支持选择的检索策略
3. **性能平衡**: 在精度、召回率和速度之间找到平衡
4. **领域适配**: 根据具体应用领域调整参数

## 故障排除

### 问题1：检索结果不相关
- 降低 `SIMILARITY_THRESHOLD`
- 增加 `CHUNK_SIZE`
- 启用查询扩展

### 问题2：检索速度慢
- 减少 `SEARCH_K`
- 增加 `CHUNK_SIZE`
- 禁用查询扩展和混合检索

### 问题3：缺少重要信息
- 增加 `SEARCH_K`
- 降低 `SIMILARITY_THRESHOLD`
- 使用MMR检索类型

通过合理的参数调优，可以显著提升RAG系统的检索效果，而无需更换大模型或嵌入模型。 