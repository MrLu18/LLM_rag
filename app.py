import os
from sentence_transformers import SentenceTransformer, util
import rag_new.rag_core
from collections import deque

chat_history = deque(maxlen=3) #全局变量定义这样就可以和fastapi_app同步 

from langchain.schema import Document  # Import Document for chat history storage
from rag_new.config import (
    DOCUMENTS_DIR,
    PERSIST_DIR,
    EMBEDDING_MODEL_PATH,
    SIMILARYTY_MODEL,
    EMBEDDING_DEVICE,
)

model = SentenceTransformer(SIMILARYTY_MODEL,device=EMBEDDING_DEVICE)
"""
上传和管理PDF/DOCX文档知识库
基于文档内容提出自然语言问题
系统自动检索相关文档内容并生成答案
使用Qwen 3-8B大语言模型进行回答生成
支持流式输出和ChatGPT风格界面
"""
#工作流程 文档加载 → 文本分割 → 向量嵌入 → 向量数据库存储 → 问题检索 → RAG回答生成
def rewrite_question_if_needed(current_question: str, previous_question: str, similarity_threshold=0.65):
    """
    判断当前问题是否需要重写，如果需要，则使用大模型重写一个更合理的问题，否则返回原始问题。
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
        #使用大模型重写问题
        rewrite_prompt = f"""请根据上下文重写以下问题，使其更加清晰和完整。

前一个问题：{previous_question}
当前问题：{current_question}

请重写当前问题，使其：
1. 结合前一个问题，使问题更加明确和具体
2. 重写后的问题不需要结合其他问题也知道在问什么
3. 只需要改写问题，不需要解释说明
4. 如果当前问题已经明确，不需要改写，则直接返回当前问题
重写后的问题："""
        
        try:
            # 使用RAG核心中的LLM来重写问题
            rewritten_response = ""
            for chunk in rag_new.rag_core.llm.stream(rewrite_prompt):
                rewritten_response += chunk.content
            
            # 清理响应，只保留重写的问题部分
            rewritten_question = rewritten_response.strip()

            print("这是大模型重写的结果",rewritten_question)
            
            # 如果响应太长，可能包含了额外的解释，尝试提取问题部分
            if len(rewritten_question) > len(current_question) * 3:
                # 尝试找到最后一个问号或句号作为问题的结束
                for i in range(len(rewritten_question) - 1, -1, -1):
                    if rewritten_question[i] in ['？', '?', '。', '.']:
                        rewritten_question = rewritten_question[:i+1]
                        break
            
            print(f"问题已由大模型重写: {rewritten_question}")
            return rewritten_question
            
        except Exception as e:
            print(f"大模型重写问题失败: {e}")
            # 如果大模型重写失败，回退到简单的重写方式
            rewritten = f"关于\"{previous_question}\"，{current_question}"
            return rewritten

    else:
    # 不需要改写
         return current_question

# Updated function to handle query submission for gr.Chatbot 管理聊天历史  显示思考中状态 更新问答 清空输入框
def handle_submit_with_thinking(query_text): #chat_history组成是问题答案为一元素，由多个元素组成 这个chat_history和memory没关系 这里的参数只能用于问题重写 和传递回答 不会在上下文中出现
    # 先判断是否需要重写问题
    global chat_history
    if chat_history:
        previous_question = chat_history[-1][0]
    else:
        previous_question = ""
    rewritten_query = rewrite_question_if_needed(query_text, previous_question)
    #测试使用 后期可以删掉
    if rewritten_query != query_text:
        print(f"问题已改写: {rewritten_query}")
    else:
        print(f"问题没有改写:{rewritten_query}")
    query_to_use = rewritten_query 

    # Call the new function to handle memory and prepare the query
    full_query, updated_user_facts = rag_new.rag_core.handle_memory_and_query_prep(query_to_use, rag_new.rag_core.user_facts) #这个full_query是由一个个需要记住的事实和当前的提问组成

    # Update global user_facts in app.py after processing
    rag_new.rag_core.user_facts[:] = updated_user_facts # 更新记忆 通过user_facts[:] 可以保持所有引用同步更新 比如之前有用user_fact赋值的变量，他们也会变化

    if full_query == "": # 表示本次用户没有进行提问，或者是提问中包含记住这个词  感觉这个不太实用 
        if "记住" in query_text and chat_history and chat_history[-1][1] != "思考中...": #每个and都是单独判断的内容
            # If it was a 'remember' command, update the chat history with the confirmation message
            chat_history[-1] = (query_text, f"好的，我已记住：{rag_new.rag_core.user_facts[-1]}") #这个没什么用 除非提问要求记住什么内容
        # yield "", chat_history
        return

    if not query_text or query_text.strip() == "":
        # yield "", chat_history
        return

    chat_history.append((query_text, "思考中..."))
    # yield "", chat_history

    final_response_from_rag = "思考中..."

    for stream_chunk in rag_new.rag_core.process_query(full_query): #这个full_query理论上包含前面的所有提问以及要求记住的内容，也有本次的提问 
        final_response_from_rag = stream_chunk #这个for循环是每次process_query函数返回一个值，就循环一次 所以没必要for循环
        chat_history[-1] = (query_text, final_response_from_rag)#将回答和问题组成历史记忆
        # yield "", chat_history

    if rag_new.rag_core.vector_db is not None and query_text.strip() and final_response_from_rag.strip(): #strip去掉开头和结尾的空白字符
        # 组合成一个片段
        dialogue_text = f"用户: {query_text}\nAI: {final_response_from_rag}"
        # 创建 Document 对象
        doc = Document(page_content=dialogue_text, metadata={"type": "chat_history"}) #打个标签 表示这个是对话历史 也就是将本次的历史对话加入进去 虽然chat_history不会加入向量库，但是对话内容会加入进去
        try: 
            rag_new.rag_core.vector_db.add_documents([doc]) #相当于把历史文档加载进去 
            # 可选：print("对话已存入向量库") 
        except Exception as e:
            print(f"存储对话到向量库失败: {e}")

# 10. 主函数 (Modified for Gradio Blocks, Upload, Doc List, Streaming, and Usage Guide)
#
print(f"IMPORTANT: Current embedding model is {EMBEDDING_MODEL_PATH}.")
print(f"If you recently changed the embedding model and encounter dimension mismatch errors,")
print(f"you MUST manually delete the ChromaDB directory: '{PERSIST_DIR}' and restart.")

# Ensure documents directory exists at start
if not os.path.exists(DOCUMENTS_DIR):
    os.makedirs(DOCUMENTS_DIR)
    print(f"创建文档目录: {DOCUMENTS_DIR}")
    print("请将您的 PDF 和 DOCX 文件添加到此目录或使用上传功能。")

# 初始化并同步rag_core的全局变量
rag_new.rag_core.embeddings = rag_new.rag_core.initialize_embeddings()
rag_new.rag_core.llm = rag_new.rag_core.initialize_openai_client()

print("执行初始索引构建...")
initial_status = rag_new.rag_core.rebuild_index_and_chain()
print(initial_status)

if rag_new.rag_core.rag_chain is None:
    print("警告：RAG 链未初始化，尝试重新创建...")
    if rag_new.rag_core.vector_db is not None:
        rag_new.rag_core.rag_chain = rag_new.rag_core.create_rag_chain_with_memory(
            rag_new.rag_core.vector_db, rag_new.rag_core.llm, rag_new.rag_core.memory)
        print("RAG 链重新创建成功。")
    else:
        print("警告：向量数据库未初始化，RAG 链可能不可用。")
else:
    print("RAG 链初始化成功。")

