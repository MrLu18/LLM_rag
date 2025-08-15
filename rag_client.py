import os
import re
import shutil # Import shutil for file operations
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
#加入记忆功能
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import Document
from sentence_transformers import SentenceTransformer, util
import torch
from langchain.embeddings.base import Embeddings
from transformers import AutoTokenizer, AutoModel
import numpy as np
from typing import List
from chromadb import Client
from chromadb.config import Settings
import json
import hashlib
from datetime import datetime
from langchain.retrievers import EnsembleRetriever
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
"""
上传和管理PDF/DOCX文档知识库
基于文档内容提出自然语言问题
系统自动检索相关文档内容并生成答案
使用Qwen 3-8B大语言模型进行回答生成
支持流式输出和ChatGPT风格界面
"""
#工作流程 文档加载 → 文本分割 → 向量嵌入 → 向量数据库存储 → 问题检索 → RAG回答生成
# --- Configuration ---
DOCUMENTS_DIR = "./documents"  # Modify to your document directory
PERSIST_DIR = "./chroma_db"     # Vector database storage directory 向量数据库 存储数据的（个人理解）
CHAT_HISTORY_PERSIST_DIR = "./chat_history_db"  # 对话历史向量数据库存储目录
EMBEDDING_MODEL_PATH = "model/bge-m3" # 嵌入模型路径 将文本转换为向量    注意这个模型出来的向量都是归一化的
EMBEDDING_DEVICE = "cuda:1" # Or 'cpu' 嵌入模型设备
# VLLM Server details (using OpenAI compatible endpoint)
VLLM_BASE_URL = "http://localhost:7861/v1"  # 使用正确的端口 7861
#VLLM_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"  #调用外部API
VLLM_API_KEY = "dummy-key" # Required by ChatOpenAI, but VLLM server doesn't usually check it 
#VLLM_API_KEY = "sk-dcc523ef2e27471895aef2bdd7a2efc4" # Required by ChatOpenAI, but VLLM server doesn't usually check it 
VLLM_MODEL_NAME = "/mnt/jrwbxx/LLM/model/qwen3-1.7b"  # 修正模型路径
#VLLM_MODEL_NAME = "qwen3-1.7b"  #使用外部的API

# 检索参数 检索的配置 视情况改
CHUNK_SIZE = 512 # Adjusted for bge-m3, which can handle more context  文本块大小
CHUNK_OVERLAP = 50  # Adjusted overlap (approx 20% of CHUNK_SIZE)  文本块重叠大小 这个的目的我个人觉得是确保每个块之间有联系
SEARCH_K = 10 # Retrieve more chunks to increase chances of finding specific sentences  检索到的结果的数量
CHAT_HISTORY_SEARCH_K = 2  # 对话历史检索数量
# --- End Configuration ---

# Global variables
rag_chain = None
vector_db = None
chat_history_vector_db = None  # 新增：对话历史向量数据库
embeddings = None
llm = None
# 移除内存存储，改为使用向量库存储对话历史
user_facts = []

# 文档索引跟踪
DOCUMENT_INDEX_FILE = "./document_index.json"  # 存储已处理文档信息的文件
document_index = {}  # 内存中的文档索引 {file_path: {"hash": file_hash, "mtime": mtime, "size": size}}

def load_document_index():
    """加载文档索引文件"""
    global document_index
    try:
        if os.path.exists(DOCUMENT_INDEX_FILE):
            with open(DOCUMENT_INDEX_FILE, 'r', encoding='utf-8') as f:
                document_index = json.load(f) #将json文件读进来，变成python对象  读取f中全部内容 必须json格式 将json字符串解析成python对象  
            print(f"已加载文档索引，包含 {len(document_index)} 个文件记录")
        else:
            document_index = {}
            print("文档索引文件不存在，将创建新的索引")
    except Exception as e:
        print(f"加载文档索引时出错: {e}")
        document_index = {}

def save_document_index():
    """保存文档索引到文件"""
    try:
        with open(DOCUMENT_INDEX_FILE, 'w', encoding='utf-8') as f:
            json.dump(document_index, f, ensure_ascii=False, indent=2)
        print(f"文档索引已保存，包含 {len(document_index)} 个文件记录")
    except Exception as e:
        print(f"保存文档索引时出错: {e}")

def get_file_info(file_path):
    """获取文件信息：哈希值、修改时间、文件大小"""
    try:
        stat = os.stat(file_path)
        mtime = stat.st_mtime
        size = stat.st_size
        
        # 计算文件哈希值（使用文件的前1MB和后1MB来快速计算）  md5是常见的hash计算方法  能检测文档内容变化  对于2mb以下文件 读取前1mb的内容 对于2mb以上文件 读取前1mb和后1mb的内容
        hash_md5 = hashlib.md5()
        with open(file_path, 'rb') as f:
            # 读取前1MB
            data = f.read(1024 * 1024)
            hash_md5.update(data)
            
            # 如果文件大于2MB，读取后1MB
            if size > 2 * 1024 * 1024:
                f.seek(-1024 * 1024, 2)  # 从文件末尾向前1MB
                data = f.read(1024 * 1024)
                hash_md5.update(data)
        
        return {
            "hash": hash_md5.hexdigest(),
            "mtime": mtime,
            "size": size
        }
    except Exception as e:
        print(f"获取文件信息时出错 {file_path}: {e}")
        return None

def is_file_unchanged(file_path):
    """检查文件是否未发生变化"""
    if file_path not in document_index:
        return False
    
    current_info = get_file_info(file_path)
    if current_info is None:
        return False
    
    stored_info = document_index[file_path]
    return (current_info["hash"] == stored_info["hash"] and    #分别查看hash 修改时间 文件大小 是否一致 一致的情况就返回True 
            current_info["mtime"] == stored_info["mtime"] and 
            current_info["size"] == stored_info["size"])

def update_document_index(file_path):
    """更新文档索引中的文件信息"""
    file_info = get_file_info(file_path)
    if file_info:
        document_index[file_path] = file_info
        print(f"已更新文档索引: {os.path.basename(file_path)}")

def rewrite_question_if_needed(current_question: str, previous_question: str):
    """
    大模型判断当前问题是否需要重写，如果需要，则使用大模型重写一个更合理的问题，否则返回原始问题。
    """

    rewrite_prompt = f"""请根据上下文重写以下问题，使其更加清晰和完整。

前一个问题：{previous_question}
当前问题：{current_question}

请重写当前问题，使其：
1. 如果当前问题不明确，不是一个完整的问题，就结合前一个问题，使问题更加明确和具体
2. 重写后的问题不需要结合其他问题也知道在问什么
3. 只需要改写问题，不需要解释说明
4. 如果当前问题和前一个问题没有直接关系，则返回当前问题
5. 如果当前问题已经是一个完整的问题，那么不需要改写，直接返回当前问题
重写后的问题："""
        
    try:
        # 使用RAG核心中的LLM来重写问题
        rewritten_response = ""
        for chunk in llm.invoke(rewrite_prompt):
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

    
# 1. 定义文档加载函数，支持PDF和Word 以及返回文档内容列表
def load_documents(directory_path, incremental=True):
    """
    加载文档，支持增量更新
    
    参数:
        directory_path: 文档目录路径
        incremental: 是否启用增量更新模式
    """
    documents = []
    new_files = []
    skipped_files = []

    for file in os.listdir(directory_path):
        file_path = os.path.join(directory_path, file)
        
        # 检查文件是否已处理且未修改
        if incremental and is_file_unchanged(file_path):
            skipped_files.append(file)
            continue
            
        try:
            if file.endswith('.pdf'): #将pdf文件解析为多个片段，然后将这些片段放入大的document中
                loader = PyPDFLoader(file_path) #这是一个加载器，读取pdf文件，转化为文档片段（document chunks）
                file_documents = loader.load() #loader.load()返回一个列表 都是document对象
                documents.extend(file_documents) #然后通过extend将所有的对象放在document中
                new_files.append(file)
                # 更新文档索引
                if incremental:
                    update_document_index(file_path)
            elif file.endswith('.docx') or file.endswith('.doc'):
                loader = Docx2txtLoader(file_path)
                file_documents = loader.load()
                documents.extend(file_documents)
                new_files.append(file)
                # 更新文档索引
                if incremental:
                    update_document_index(file_path)
        except Exception as e:
            print(f"警告：无法加载文件 {file}，跳过此文件。错误：{e}")
            continue

    if incremental:
        print(f"增量更新模式：")
        print(f"  - 新处理文件: {len(new_files)} 个")
        print(f"  - 跳过未修改文件: {len(skipped_files)} 个")
        if new_files:
            print(f"  - 新文件列表: {', '.join(new_files)}")
        # if skipped_files:
        #     print(f"  - 跳过文件列表: {', '.join(skipped_files)}")
    else:
        print(f"全量更新模式：处理了 {len(documents)} 个文档")

    return documents

def split_documents(documents: list) -> list:
    """
    只传入 documents 列表，对每个文档先按章节分，再按块分
    内部使用默认规则（分章节规则 + chunk_size + chunk_overlap）
    返回所有拆分后的 Document 对象
    """

    # 固定参数
    section_pattern = r"\n(?=\d{1,2}\s)"  # 如 1 范围、2 引用文件 章节划分的格式可以更加完善一些 目前的正则匹配是 换行+数字+空格 (?=)是正向预查

    # 拆分器配置
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", "。", ". ", " ", ""], #感觉问号 感叹号什么的还是放置一起好 
        is_separator_regex=False
    )

    all_chunks = []
    seen_contents = set()  # 用于去重的集合

    for doc in documents:
        text = doc.page_content
        metadata = doc.metadata or {}

        # 先按章节正则分
        sections = re.split(section_pattern, text)

        for section in sections:
            cleaned = section.strip()
            if not cleaned:
                continue

            # 短章节直接保留
            if len(cleaned) <= CHUNK_SIZE:
                # 检查是否已经存在相同内容
                if cleaned not in seen_contents:
                    all_chunks.append(Document(page_content=cleaned, metadata=metadata))
                    seen_contents.add(cleaned)
            else:
                # 长章节再拆块
                sub_chunks = splitter.split_text(cleaned)
                for chunk in sub_chunks:
                    chunk_cleaned = chunk.strip()
                    if chunk_cleaned and chunk_cleaned not in seen_contents:
                        all_chunks.append(Document(page_content=chunk_cleaned, metadata=metadata))
                        seen_contents.add(chunk_cleaned)

    print(f"文档分割完成：原始chunks数量 {len(all_chunks) + len(seen_contents) - len(set(seen_contents))}，去重后 {len(all_chunks)} 个chunks")
    return all_chunks

# 3. 初始化HuggingFace嵌入模型 配置gpu加速  返回文本向量化器 
def initialize_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_PATH,
        model_kwargs={'device': EMBEDDING_DEVICE},
    )

# 4. 创建或加载向量数据库 (Modified) 检测数据库状态  处理模型变更导致的维度问题 支持增量更新文档
def get_vector_db(chunks, embeddings, persist_directory):
    """Creates a new vector DB or loads an existing one."""
    if os.path.exists(persist_directory) and os.listdir(persist_directory): #有现成数据库就加在，没有就看有没有chunk，有就创建 
        print(f"加载现有向量数据库： {persist_directory}...")
        try:
            # When loading, ChromaDB will check for dimension compatibility.
            # If EMBEDDING_MODEL_PATH changed leading to a dimension mismatch, this will fail.
            return Chroma(persist_directory=persist_directory, embedding_function=embeddings) #chroma会从目录中找数据文件  并且启用embedding_function去配置
        except Exception as e: #确保embedding模型对的
            print(f"无法加载现有向量数据库： {e}.")
            # If loading fails, proceed as if it doesn't exist, but only create if chunks are given later.
            return None # Indicate loading failed or DB doesn't exist in a usable state
    else:
        # Directory doesn't exist or is empty
        if chunks:
            print(f"构建新的向量数据库： {persist_directory}...")
            print(f"构建Chroma DB 包含 {len(chunks)} 个chunks...")
            try:
                vector_db = Chroma.from_documents( #通过from_document把每个chunk转换为embedding并存到数据库
                    documents=chunks,
                    embedding=embeddings,
                    persist_directory=persist_directory #这是保存位置
                )
                return vector_db
            except Exception as e:
                print(f"创建新的向量数据库时出错: {e}")
                raise  # Re-raise the exception if creation fails
        else:
            # No chunks provided and DB doesn't exist/is empty - cannot create.
            print(f"向量数据库目录 {persist_directory} 不存在或为空, 且没有提供chunks来创建新的向量数据库。")
            return None # Indicate DB doesn't exist and cannot be created yet

def initialize_chat_history_vector_db():
    """初始化对话历史向量数据库"""
    global chat_history_vector_db, embeddings
    
    if embeddings is None:
        print("错误：Embeddings 未初始化，无法创建对话历史向量数据库")
        return None
    
    try:
        # 确保目录存在
        if not os.path.exists(CHAT_HISTORY_PERSIST_DIR):
            os.makedirs(CHAT_HISTORY_PERSIST_DIR)
            print(f"创建对话历史向量数据库目录: {CHAT_HISTORY_PERSIST_DIR}")
        
        # 尝试加载现有的对话历史向量数据库
        if os.path.exists(CHAT_HISTORY_PERSIST_DIR) and os.listdir(CHAT_HISTORY_PERSIST_DIR):
            print(f"加载现有对话历史向量数据库: {CHAT_HISTORY_PERSIST_DIR}")
            chat_history_vector_db = Chroma(persist_directory=CHAT_HISTORY_PERSIST_DIR, embedding_function=embeddings)
        else:
            print(f"创建新的对话历史向量数据库: {CHAT_HISTORY_PERSIST_DIR}")
            # 创建一个空的向量数据库
            chat_history_vector_db = Chroma(
                embedding_function=embeddings,
                persist_directory=CHAT_HISTORY_PERSIST_DIR
            )
        
        print("对话历史向量数据库初始化成功")
        return chat_history_vector_db
        
    except Exception as e:
        print(f"初始化对话历史向量数据库时出错: {e}")
        return None

# 5. 初始化连接到VLLM服务器的ChatOpenAI客户端 (Replaces initialize_llm) 连接VLLM推理服务器 配置模型 返回兼容接口
def initialize_openai_client():
    """Initializes ChatOpenAI client pointing to the VLLM server."""
    print(f"初始化ChatOpenAI client 指向VLLM服务器 {VLLM_BASE_URL}...")
    return ChatOpenAI(
        openai_api_base=VLLM_BASE_URL,
        openai_api_key=VLLM_API_KEY,
        model_name=VLLM_MODEL_NAME,
    )


#     return rag_chain
#目前设计了一个新的检索方式，这个带有记忆功能的已经被丢弃，但是为了防止新的检索不可用，就保留他 现在已经不传入memory参数
def create_rag_chain_with_memory(vector_db, llm, memory): #创建带有记忆的问答链  这部分的提示词相当于都是封装的 用别人的库 可以尝试自己手写  
    retriever = vector_db.as_retriever(search_kwargs={"k": SEARCH_K}) #将向量数据库变为一个检索器，能输入问题返回最相关的k个文本
    # 直接用 ConversationalRetrievalChain，自动管理上下文
    return ConversationalRetrievalChain.from_llm( #封装的类， 可以把检索器返回的文本，当前对话 历史对话拼接成一个prompt   （检索向量库得到最相关的k个，获取memory 拼接prompt LLM调用返回结果，分别对应下面的参数）
        llm=llm, #这个指用哪个模型回答
        retriever=retriever, #这是指用哪个检索器 
        memory=memory,
        return_source_documents=False  # 如果不需要输出检索到的源文档，可设为 False 
    )

def create_dual_retrieval_rag_chain(document_vector_db, chat_history_vector_db, llm):
    """创建支持双向量库检索的RAG链"""
    # 创建文档检索器
    document_retriever = document_vector_db.as_retriever(search_kwargs={"k": SEARCH_K})
    
    # 创建对话历史检索器
    chat_history_retriever = chat_history_vector_db.as_retriever(search_kwargs={"k": CHAT_HISTORY_SEARCH_K})
    
    # 创建自定义提示模板，分别标注文档内容和对话历史
    template = """基于以下上下文信息回答问题。如果上下文中没有相关信息，请说明无法从提供的信息中找到答案。

【文档知识库内容】：
{document_context}

【历史对话内容】：
{chat_context}

问题：{question}

请提供准确、详细的回答："""

    prompt = PromptTemplate(
        template=template,
        input_variables=["document_context", "chat_context", "question"]
    )
    
    # 创建LLM链
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    
    # 返回两个检索器和LLM链，而不是集成检索器
    return document_retriever, chat_history_retriever, llm_chain

# 7. Function to process query using the RAG chain (Modified for Streaming)
def process_query(query):
    """Processes a user query using the RAG chain and streams the answer."""
    global rag_chain, vector_db, chat_history_vector_db # Add vector_db to globals accessed here for debugging  这里是表明 这两个是全局变量 可以让函数内部修改函数外部定义的变量（如果没定义这个，函数内是不能修改全局变量） 也就是之前定义过的，如果不声明 会出错  
    if rag_chain is None:
        yield "错误：RAG 链未初始化。"
        return
    print("DEBUG rag_chain:", rag_chain)
    print("DEBUG rag_chain.llm:", getattr(rag_chain, "llm", None))
    print("DEBUG rag_chain.retriever:", getattr(rag_chain, "retriever", None))

    # --- For Debugging Retrieval --- 
    # Uncomment the block below to see what documents are retrieved by the vector DB
    # if vector_db:
    #     try:
    #         retrieved_docs = vector_db.similarity_search(query, k=SEARCH_K)
    #         print(f"\n--- 检索到的文档内容: '{query}' ---")
    #         for i, doc in enumerate(retrieved_docs):
    #             # Attempt to get score if retriever supports it (Chroma's similarity_search_with_score)
    #             # For basic similarity_search, score might not be directly in metadata.
    #             # If using retriever.get_relevant_documents(), score might be present.
    #             score = doc.metadata.get('score', 'N/A') # 先从metadata中取score 没有就返回NA
    #             if hasattr(doc, 'score'): #  有些document是带score属性  这个相当于是一种兼容性 如果前者没找到 就找这个
    #                 score = doc.score
                
    #             print(f"文档 {i+1} (Score: {score}):")
    #             print(f"Content: {doc.page_content[:500]}...") # Print first 500 chars
    #             print(f"Metadata: {doc.metadata}")
    #         print("--- 结束文档检索内容 ---\n")
    #     except Exception as e:
    #         print(f"调试文档检索时出错: {e}")
    # else:
    #     print("文档向量数据库未初始化, 跳过调试检索。")
    
    # # 调试对话历史检索
    # if chat_history_vector_db:
    #     try:
    #         retrieved_chat_docs = chat_history_vector_db.similarity_search(query, k=CHAT_HISTORY_SEARCH_K)
    #         print(f"\n--- 检索到的对话历史: '{query}' ---")
    #         for i, doc in enumerate(retrieved_chat_docs):
    #             score = doc.metadata.get('score', 'N/A')
    #             if hasattr(doc, 'score'):
    #                 score = doc.score
                
    #             print(f"对话历史 {i+1} (Score: {score}):")
    #             print(f"Content: {doc.page_content[:500]}...")
    #             print(f"Metadata: {doc.metadata}")
    #         print("--- 结束对话历史检索内容 ---\n")
    #     except Exception as e:
    #         print(f"调试对话历史检索时出错: {e}")
    # else:
    #     print("对话历史向量数据库未初始化, 跳过调试检索。")
    # --- End Debugging Retrieval ---

    # 使用新的双向量库检索系统 这个是之前被弃用的 现在重新启用


    try:
        # 使用新的双向量库检索系统
        if hasattr(rag_chain, '__iter__') and len(rag_chain) == 3: #判断rag_chain是否有可迭代对象hasattr(rag_chain, '__iter__') 这个写法可以记住
            # 新的双检索系统：document_retriever, chat_history_retriever, llm_chain
            document_retriever, chat_history_retriever, llm_chain = rag_chain
            
            # 分别从两个检索器获取内容
            document_docs = document_retriever.get_relevant_documents(query)
            chat_docs = chat_history_retriever.get_relevant_documents(query)
            
            # 打印检索结果统计
            print(f"文档检索器返回了 {len(document_docs)} 个文档片段")
            print(f"对话历史检索器返回了 {len(chat_docs)} 个对话片段")
            
            # 检查文档检索结果的多样性
            unique_doc_contents = set()
            for i, doc in enumerate(document_docs):
                content_preview = doc.page_content[:100].strip()
                unique_doc_contents.add(content_preview)
                print(f"文档片段 {i+1}: {content_preview}...")
            
            print(f"文档检索结果多样性：{len(unique_doc_contents)}/{len(document_docs)} 个不同内容")
            
            # 分别合并文档内容和对话历史内容
            document_context = "\n\n".join([doc.page_content for doc in document_docs]) if document_docs else "未找到相关文档内容"
            chat_context = "\n\n".join([doc.page_content for doc in chat_docs]) if chat_docs else "未找到相关历史对话"
            
            print(f"文档上下文: {document_context}")
            print(f"对话历史上下文: {chat_context}")
            # 打印上下文长度信息
            print(f"文档上下文长度: {len(document_context)} 字符")
            print(f"对话历史上下文长度: {len(chat_context)} 字符")
            
            # 生成回答
            result = llm_chain.invoke({
                "document_context": document_context, 
                "chat_context": chat_context, 
                "question": query
            })
            
            if isinstance(result, dict) and "text" in result:
                yield result["text"]
            else:
                yield str(result)
        else:
            # 兼容旧的RAG链
            result = rag_chain.invoke({ #rag_chain就是之前创建的包括检索的数据 历史回答 拼接得到的结果 stream可以保证回复是一边生成一边返回   invoke和predict都是一次性返回整段内容
                    "question": query,
                    #"enable_thinking": False
                                                })

           # result 可能是 dict，也可能直接是字符串，视 chain 配置而定
            if isinstance(result, dict) and "answer" in result:
                yield result["answer"]
            else:
                yield str(result)

    except Exception as e: # 先简要打印错误信息 然后通过traceback 输出完整的错误栈追踪
        print(f"处理查询时发生错误: {e}")
        yield f"处理查询时发生错误: {e}"


def safe_add_documents(vector_db, chunks, max_batch_size=5000): 
    """
    安全分批添加文档到向量数据库
    
    参数:
        vector_db: 已初始化的向量数据库对象
        chunks: 待添加的文档块列表
        max_batch_size: 单次批量上限
    """
    
    # 检查向量数据库中是否已存在相同内容
    existing_contents = set()
    try:
        # 获取现有文档的内容（这里可能需要根据ChromaDB的API调整）
        existing_docs = vector_db.similarity_search("", k=50000)  # 获取所有文档
        for doc in existing_docs:
            existing_contents.add(doc.page_content.strip())
        print(f"向量数据库中已有 {len(existing_contents)} 个不同的文档内容")
    except Exception as e:
        print(f"获取现有文档内容时出错: {e}")
    
    # 过滤掉已存在的内容
    new_chunks = []
    for chunk in chunks:
        content = chunk.page_content.strip()
        if content not in existing_contents:
            new_chunks.append(chunk)
    
    print(f"过滤重复内容：原始 {len(chunks)} 个chunks，去重后 {len(new_chunks)} 个chunks")
    
    if not new_chunks:
        print("没有新的文档需要添加")
        return
    
    for batch_start in range(0, len(new_chunks), max_batch_size):
        batch = new_chunks[batch_start : batch_start + max_batch_size]
        batch_num = (batch_start // max_batch_size) + 1
        
        try:
            print(f"🔄 正在添加第 {batch_num} 批（{len(batch)} 个chunk）...")
            vector_db.add_documents(batch)
            print(f"✅ 第 {batch_num} 批添加成功")
        except Exception as e:
           
            print(f"❌ 第 {batch_num} 批添加失败（最终尝试）：{str(e)}")
            raise  # 抛出异常终止程序

# 8. Function to rebuild the index and RAG chain (Modified to add documents)
def rebuild_index_and_chain(incremental=True): #全流程索引重建  文档加载-分割-嵌入-存储   
    """
    加载文档，创建/更新向量数据库，并重建RAG链
    
    参数:
        incremental: 是否启用增量更新模式
    """
    global vector_db, rag_chain, embeddings, llm, chat_history_vector_db

    if embeddings is None or llm is None:
        return "错误：Embeddings 或 LLM 未初始化。"

    # 加载文档索引
    load_document_index()

    # 初始化对话历史向量数据库
    if chat_history_vector_db is None:
        chat_history_vector_db = initialize_chat_history_vector_db()

    # Ensure documents directory exists
    if not os.path.exists(DOCUMENTS_DIR):
        os.makedirs(DOCUMENTS_DIR)
        print(f"创建文档目录: {DOCUMENTS_DIR}")

    # Step 1: Load documents
    print("加载文档...")
    documents = load_documents(DOCUMENTS_DIR, incremental=incremental)
    if not documents:
        print(f"在 {DOCUMENTS_DIR} 中未找到文档。")
        # Try to load existing DB even if no new documents are found
        print("尝试加载现有向量数据库...")
        # Pass None for chunks as we are just trying to load
        vector_db = get_vector_db(None, embeddings, PERSIST_DIR)
        if vector_db:
            print("没有新文档加载，将使用现有的向量数据库。重新创建 RAG 链...")
            # 使用新的双向量库RAG链
            if chat_history_vector_db:
                rag_chain = create_dual_retrieval_rag_chain(vector_db, chat_history_vector_db, llm)
            else:
                # 如果对话历史向量库不可用，使用旧的RAG链
                rag_chain = create_rag_chain_with_memory(vector_db, llm, None)

            return "没有找到新文档，已使用现有数据重新加载 RAG 链。"
        else:
            # No documents AND no existing DB
            return "错误：没有文档可加载，且没有现有的向量数据库。"

    # Step 2: Split text
    print("分割文本...")
    chunks = split_documents(documents)
    # 过滤和预处理：只保留非空字符串内容的chunk 防止输入的格式有问题 导致后续的向量库添加失败 TextEncodeInput must be Union[TextInputSequence, Tuple[InputSequence, InputSequence]]
    filtered_chunks = []
    for c in chunks:
        if hasattr(c, 'page_content') and isinstance(c.page_content, str):
            content = c.page_content.strip()
            if content:
                c.page_content = content  # 去除首尾空白
                filtered_chunks.append(c)
    print(f"过滤后剩余 {len(filtered_chunks)} 个有效文本块（原始 {len(chunks)} 个）")
    chunks = filtered_chunks
    if not chunks:
        print("分割后未生成文本块。")
        # Try loading existing DB if splitting yielded nothing
        print("尝试加载现有向量数据库...")
        vector_db = get_vector_db(None, embeddings, PERSIST_DIR)
        if vector_db:
             print("警告：新加载的文档分割后未产生任何文本块。使用现有数据库。")
             # 使用新的双向量库RAG链
             if chat_history_vector_db:
                 rag_chain = create_dual_retrieval_rag_chain(vector_db, chat_history_vector_db, llm)
             else:
                 # 如果对话历史向量库不可用，使用旧的RAG链
                 rag_chain = create_rag_chain_with_memory(vector_db, llm, None)
             return "警告：文档分割后未产生任何文本块。RAG 链已使用现有数据重新加载。"
        else:
            # No chunks AND no existing DB
            return "错误：文档分割后未产生任何文本块，且无现有数据库。"

    # Step 3: Load or Create/Update vector database
    print("加载或更新向量数据库...")
    # Try loading first, even if we have chunks (in case we want to add to it)
    vector_db_loaded = get_vector_db(None, embeddings, PERSIST_DIR) #None用于加载一个已经存在的数据库 不想新建 

    if vector_db_loaded:
        print(f"向现有向量数据库添加 {len(chunks)} 个块...")
        vector_db = vector_db_loaded # Use the loaded DB
        try:
            # Consider adding only new chunks if implementing duplicate detection later
            print("开始添加文档到向量数据库每次限制最多五千个...")
            safe_add_documents(vector_db, chunks, max_batch_size=5000)
            print("块添加成功。")
            # Persisting might be needed depending on Chroma version/setup, often automatic.
            # vector_db.persist() # Uncomment if persistence issues occur
        except Exception as e:
             print(f"添加文档到 Chroma 时出错: {e}")
             # If adding fails, proceed with the DB as it was before adding
             # 使用新的双向量库RAG链
             if chat_history_vector_db:
                 rag_chain = create_dual_retrieval_rag_chain(vector_db, chat_history_vector_db, llm)
             else:
                 # 如果对话历史向量库不可用，使用旧的RAG链
                 rag_chain = create_rag_chain_with_memory(vector_db, llm, None)
             return f"错误：向向量数据库添加文档时出错: {e}。RAG链可能使用旧数据。"
    else:
        # Database didn't exist or couldn't be loaded, create a new one with the current chunks
        print(f"创建新的向量数据库并添加 {len(chunks)} 个块...")
        try:
            # Call get_vector_db again, this time *with* chunks to trigger creation
            vector_db = get_vector_db(chunks, embeddings, PERSIST_DIR) #用于新建数据库
            if vector_db is None: # Check if creation failed within get_vector_db
                 raise RuntimeError("get_vector_db failed to create a new database.")
            print("新的向量数据库已创建并持久化。")
        except Exception as e:
            print(f"创建新的向量数据库时出错: {e}")
            return f"错误：创建向量数据库失败: {e}"

    if vector_db is None:
         # This should ideally not be reached if error handling above is correct
         return "错误：未能加载或创建向量数据库。"

    # Step 4: Create RAG chain
    print("创建 RAG 链...")
    # 使用新的双向量库RAG链
    if chat_history_vector_db:
        rag_chain = create_dual_retrieval_rag_chain(vector_db, chat_history_vector_db, llm)
    else:
        # 如果对话历史向量库不可用，使用旧的RAG链
        rag_chain = create_rag_chain_with_memory(vector_db, llm, None)
    
    # 保存文档索引
    if incremental:
        save_document_index()
    
    print("索引和 RAG 链已成功更新。")
    return "文档处理完成，索引和 RAG 链已更新。"

def force_rebuild_index():
    """强制全量重建索引（忽略文档索引）"""
    return rebuild_index_and_chain(incremental=False)

def get_document_index_status():
    """获取文档索引状态信息"""
    load_document_index()
    
    if not document_index:
        return "📋 文档索引状态：\n- 暂无已处理的文档记录"
    
    total_files = len(document_index)
    total_size = sum(info.get("size", 0) for info in document_index.values())
    total_size_mb = total_size / (1024 * 1024)
    
    status = f"📋 文档索引状态：\n"
    status += f"- 已处理文档数量：{total_files} 个\n"
    status += f"- 总文件大小：{total_size_mb:.2f} MB\n"
    status += f"- 索引文件：{DOCUMENT_INDEX_FILE}\n\n"
    
    # 列出所有已处理的文件
    status += "📄 已处理文档列表：\n"
    for file_path, info in document_index.items():
        file_name = os.path.basename(file_path)
        file_size_mb = info.get("size", 0) / (1024 * 1024)
        mtime = datetime.fromtimestamp(info.get("mtime", 0)).strftime("%Y-%m-%d %H:%M:%S")
        status += f"- {file_name} ({file_size_mb:.2f} MB, 修改时间: {mtime})\n"
    
    return status

def get_chat_history_status():
    """获取对话历史向量库状态信息"""
    global chat_history_vector_db
    
    if chat_history_vector_db is None:
        return "📋 对话历史向量库状态：\n- 对话历史向量库未初始化"
    
    try:
        # 获取对话历史数量（这里需要根据ChromaDB的API来获取）
        # 由于ChromaDB没有直接的方法获取文档数量，我们通过检索来估算
        test_docs = chat_history_vector_db.similarity_search("", k=1000)
        total_dialogues = len(test_docs)
        
        status = f"📋 对话历史向量库状态：\n"
        status += f"- 对话历史数量：{total_dialogues} 条\n"
        status += f"- 向量库路径：{CHAT_HISTORY_PERSIST_DIR}\n\n"
        
        # 列出最近的对话历史
        if total_dialogues > 0:
            status += "💬 最近的对话历史：\n"
            recent_docs = chat_history_vector_db.similarity_search("", k=min(5, total_dialogues))
            for i, doc in enumerate(recent_docs):
                timestamp = doc.metadata.get("timestamp", "未知时间")
                content_preview = doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
                status += f"- 对话 {i+1} ({timestamp}): {content_preview}\n"
        
        return status
        
    except Exception as e:
        return f"📋 对话历史向量库状态：\n- 获取状态时出错: {e}"

def clear_chat_history_vector_db():
    """清空对话历史向量库"""
    global chat_history_vector_db
    
    try:
        if chat_history_vector_db is not None:
            # 删除对话历史向量库目录
            import shutil
            if os.path.exists(CHAT_HISTORY_PERSIST_DIR):
                shutil.rmtree(CHAT_HISTORY_PERSIST_DIR)
                print(f"已删除对话历史向量库目录: {CHAT_HISTORY_PERSIST_DIR}")
            
            # 重新初始化对话历史向量库
            chat_history_vector_db = initialize_chat_history_vector_db()
            
            # 重新创建RAG链
            if vector_db is not None and chat_history_vector_db is not None:
                rag_chain = create_dual_retrieval_rag_chain(vector_db, chat_history_vector_db, llm)
            
            return "对话历史向量库已清空并重新初始化"
        else:
            return "对话历史向量库未初始化"
            
    except Exception as e:
        return f"清空对话历史向量库时出错: {e}"

# Helper function to list documents in the directory 生成已加载的文档列表 markdown格式化输出  实时更新文档状态
def get_loaded_documents_list():
    """Returns an HTML formatted list of files in DOCUMENTS_DIR."""
    if not os.path.exists(DOCUMENTS_DIR) or not os.listdir(DOCUMENTS_DIR): #存在则函数返回true  如果文档存在，并且里面有东西就不返回内容
        return "当前没有已加载的文档。"
    try:
        files = [f for f in os.listdir(DOCUMENTS_DIR) if os.path.isfile(os.path.join(DOCUMENTS_DIR, f)) and (f.endswith('.pdf') or f.endswith('.docx') or f.endswith('.doc'))]# 符合条件的全部遍历
        if not files: 
            return "当前没有已加载的文档。"
        
        # 生成HTML格式的文档列表
        html_list = "<ul>"
        for file in files:
            html_list += f"<li>{file}</li>"
        html_list += "</ul>"
        return html_list
    except Exception as e:
        print(f"Error listing documents: {e}")
        return "无法列出文档。"


# 9. Function to handle file uploads (Modified to return doc list) 接受上传的文档 保存到文档目录 触发索引重建（类似图书馆进来新的书 重新创建索引） 返回上传状态
def handle_file_upload(file_obj, incremental=True):
    """
    保存上传的文件，触发索引重建，并返回状态和文档列表
    
    参数:
        file_obj: 上传的文件对象
        incremental: 是否启用增量更新模式
    """
    if file_obj is None:
        return "未选择文件。", get_loaded_documents_list() # Return current list even if no file selected

    try:
        # Gradio provides a temporary file path
        temp_file_path = file_obj.name
        file_name = os.path.basename(temp_file_path)
        destination_path = os.path.join(DOCUMENTS_DIR, file_name)

        print(f"将上传的文件从 {temp_file_path} 复制到 {destination_path}")
        # Ensure documents directory exists
        if not os.path.exists(DOCUMENTS_DIR):
            os.makedirs(DOCUMENTS_DIR)
        shutil.copy(temp_file_path, destination_path) # Copy the file

        print(f"文件 {file_name} 上传成功。开始重建索引...")
        status = rebuild_index_and_chain(incremental=incremental)
        final_status = f"文件 '{file_name}' 上传成功。\n{status}"
        # Get updated document list
        doc_list_md = get_loaded_documents_list()
        return final_status, doc_list_md

    except Exception as e:
        print(f"文件上传或处理失败: {e}")
        # Return error and current doc list
        return f"文件上传或处理失败: {e}", get_loaded_documents_list()

def handle_memory_and_query_prep(query_text, current_user_facts): #前者是提问 后者是目前存储的内容  这个函数只会存储要求记住的内容 如果是提问，只会把之前要求记住的内容和提问拼接起来返回回去
    """Handles the memory feature and prepares the full query."""
    # Create a mutable copy of user_facts to modify within this function
    updated_user_facts = list(current_user_facts)

    if "记住" in query_text:
        # 提取记住的内容（去掉"请记住"、"记住"等前缀）
        fact = query_text.replace("请记住", "").replace("记住", "").strip("：:，,。. ")
        if fact:
            updated_user_facts.append(fact)
            # No chat history update here, that's for the UI layer.
            return "", updated_user_facts # Indicate it's a memory command, no query for RAG

    # 拼接所有记忆内容到用户输入前面
    if updated_user_facts: #这部分的memory_prefix都是用户要求记住的内容，可以理解为问题重写了一遍，但至少增加了一部分要求记住的内容
        memory_prefix = "，".join(updated_user_facts)
        full_query = f"请记住：{memory_prefix}。用户提问：{query_text}"
    else:
        full_query = query_text

    return full_query, updated_user_facts


# Updated function to handle query submission for gr.Chatbot 管理聊天历史  显示思考中状态 更新问答 清空输入框
def handle_submit_with_thinking(query_text, chat_history):
    global user_facts, chat_history_vector_db
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
    full_query, updated_user_facts = handle_memory_and_query_prep(query_to_use, user_facts)

    user_facts[:] = updated_user_facts
    if full_query == "":
        if "记住" in query_text:
            # 提取记住的内容（去掉"请记住"、"记住"等前缀）
            fact = query_text.replace("请记住", "").replace("记住", "").strip("：:，,。. ")
            if fact:
                user_facts.append(fact)
                chat_history.append((query_text, f"好的，我已记住：{fact}"))
                yield "", chat_history
                return

    if not query_text or query_text.strip() == "":
        yield "", chat_history
        return
  
    # 拼接所有记忆内容到用户输入前面  这里可以修改一下 比如给一个顺序，没经过几次对话，就删掉之前的内容 防止内容太多 导致超过最大长度
    if user_facts:  
        memory_prefix = "，".join(user_facts)
        full_query = f"请记住：{memory_prefix}。改写后的问题：{query_to_use}" #这个是将用户的提问和回答放在一起用于记忆
    else:
        full_query = query_to_use

    chat_history.append((query_text, "思考中..."))
    yield "", chat_history

    final_response_from_rag = "思考中..."

    for stream_chunk in process_query(full_query):
        final_response_from_rag = stream_chunk
        chat_history[-1] = (query_text, final_response_from_rag) #因为chat_history是一个二元组 这个意思是用后面的内容替代历史对话的最近一次内容
        yield "", chat_history

    if chat_history and chat_history[-1][1] == "思考中...": #如果内容全是思考中就直接pass掉
        pass
 
    # 新增：将本轮对话存入对话历史向量库
    if chat_history_vector_db is not None and query_text.strip() and final_response_from_rag.strip(): #strip去掉开头和结尾的空白字符
        # 组合成一个片段
        dialogue_text = f"用户: {query_text}\nAI: {final_response_from_rag}"
        # 创建 Document 对象
        doc = Document(page_content=dialogue_text, metadata={"type": "chat_history", "timestamp": datetime.now().isoformat()}) #打个标签 表示这个是对话历史
        try: 
            chat_history_vector_db.add_documents([doc]) #将对话历史存储到专门的向量库中
            print("对话已存入对话历史向量库") 
        except Exception as e:
            print(f"存储对话到对话历史向量库失败: {e}")

# 10. 初始化系统函数
def initialize_system():
    """初始化RAG系统"""
    global embeddings, llm, rag_chain, chat_history_vector_db

    print(f"IMPORTANT: Current embedding model is {EMBEDDING_MODEL_PATH}.")
    print(f"If you recently changed the embedding model and encounter dimension mismatch errors,")
    print(f"you MUST manually delete the ChromaDB directory: '{PERSIST_DIR}' and restart.")

    # Ensure documents directory exists at start
    if not os.path.exists(DOCUMENTS_DIR):
        os.makedirs(DOCUMENTS_DIR)
        print(f"创建文档目录: {DOCUMENTS_DIR}")
        print("请将您的 PDF 和 DOCX 文件添加到此目录或使用上传功能。")

    # Initialize embeddings and LLM once
    print("初始化 Embedding 模型...")
    embeddings = initialize_embeddings()

    print("初始化 LLM 客户端...")
    llm = initialize_openai_client()

    # 初始化对话历史向量数据库
    print("初始化对话历史向量数据库...")
    chat_history_vector_db = initialize_chat_history_vector_db()

    # Initial index and chain build
    print("执行初始索引构建...")
    initial_status = rebuild_index_and_chain()
    print(initial_status)
    
    if vector_db is not None:
        # 使用新的双向量库RAG链
        if chat_history_vector_db:
            rag_chain = create_dual_retrieval_rag_chain(vector_db, chat_history_vector_db, llm)
        else:
            # 如果对话历史向量库不可用，使用旧的RAG链
            rag_chain = create_rag_chain_with_memory(vector_db, llm, None)
    else:
        print("警告：向量数据库未初始化，RAG 链可能不可用。")

    return initial_status

# 11. 主函数（用于直接运行）
def main():
    """主函数，用于直接运行RAG系统"""
    initialize_system()
    print("RAG系统初始化完成。")

if __name__ == "__main__":
    main()