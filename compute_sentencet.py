from sentence_transformers import SentenceTransformer, util
import re

# 加载embedding模型（一次加载即可）
model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

def rewrite_question_if_needed(current_question: str, previous_question: str, similarity_threshold=0.65):
    """
    判断当前问题是否需要重写，如果需要，则使用大模型重写一个更合理的问题，否则返回原始问题。
    
    注意：此独立文件版本使用简单重写方式。要使用大模型重写功能，请使用 app.py 或 rag_client.py 中的版本。
    """
    # 1. 先做Embedding
    current_embedding = model.encode(current_question, convert_to_tensor=True)
    previous_embedding = model.encode(previous_question, convert_to_tensor=True)

    # 2. 计算余弦相似度
    cosine_sim = util.pytorch_cos_sim(current_embedding, previous_embedding).item()
    print(f"当前的问题是:{current_question},上一个问题是:{previous_question},他们的余弦相似度是：{cosine_sim}")

    # 3. 判断是否需要重写
    need_rewrite = cosine_sim >= similarity_threshold 

    if need_rewrite:
        # 注意：此独立文件版本使用简单重写方式
        # 要使用大模型重写功能，请使用 app.py 或 rag_client.py 中的版本
        rewritten = f"关于\"{previous_question}\"，{current_question}"
        print(f"问题已重写: {rewritten}")
        return rewritten
    else:
        # 不需要改写
        return current_question

# 示例：
if __name__ == "__main__":
    prev_q = "瓦斯爆炸的预防措施有哪些？"
    curr_q = "这种情况要怎么处理？"
    result = rewrite_question_if_needed(curr_q, prev_q)
    print("改写结果:", result)
