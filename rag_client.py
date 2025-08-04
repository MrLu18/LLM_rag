import os
import re
import shutil # Import shutil for file operations
import gradio as gr # Import Gradio
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
import gradio.themes as gr_themes
#加入记忆功能
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import Document
from sentence_transformers import SentenceTransformer, util
import torch
from langchain.embeddings.base import Embeddings



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
EMBEDDING_MODEL_PATH = "model/bge-m3" # 嵌入模型路径 将文本转换为向量    注意这个模型出来的向量都是归一化的
EMBEDDING_DEVICE = "cuda:1" # Or 'cpu' 嵌入模型设备
# VLLM Server details (using OpenAI compatible endpoint)
VLLM_BASE_URL = "http://localhost:7861/v1"  # 使用正确的端口 7861
#VLLM_BASE_URL = "http://172.16.20.193:8000/v1"  
VLLM_API_KEY = "dummy-key" # Required by ChatOpenAI, but VLLM server doesn't usually check it 
VLLM_MODEL_NAME = "/mnt/jrwbxx/LLM/model/qwen3-1.7b"  # 修正模型路径
SIMILARYTY_MODEL = "paraphrase-MiniLM-L6-v2"

# 检索参数 检索的配置 视情况改
CHUNK_SIZE = 512 # Adjusted for bge-m3, which can handle more context  文本块大小
CHUNK_OVERLAP = 50  # Adjusted overlap (approx 20% of CHUNK_SIZE)  文本块重叠大小 这个的目的我个人觉得是确保每个块之间有联系
SEARCH_K = 10 # Retrieve more chunks to increase chances of finding specific sentences  检索到的结果的数量
# --- End Configuration ---

# Global variables
rag_chain = None
vector_db = None
embeddings = None
llm = None
# memory store 这个是短期存储的，重启rag实例就没了
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)
user_facts = []

# model = SentenceTransformer(SIMILARYTY_MODEL,device=EMBEDDING_DEVICE)

def rewrite_question_if_needed(current_question: str, previous_question: str, similarity_threshold=0.65):
    """
    大模型判断当前问题是否需要重写，如果需要，则使用大模型重写一个更合理的问题，否则返回原始问题。
    """

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
        for chunk in llm.stream(rewrite_prompt):
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
def load_documents(directory_path):
    documents = []

    for file in os.listdir(directory_path):
        file_path = os.path.join(directory_path, file)
        try:
            if file.endswith('.pdf'): #将pdf文件解析为多个片段，然后将这些片段放入大的document中
                loader = PyPDFLoader(file_path) #这是一个加载器，读取pdf文件，转化为文档片段（document chunks）
                documents.extend(loader.load()) #loader.load()返回一个列表 都是document对象   然后通过extend将所有的对象放在document中 
            elif file.endswith('.docx') or file.endswith('.doc'):
                loader = Docx2txtLoader(file_path)
                documents.extend(loader.load())
        except Exception as e:
            print(f"警告：无法加载文件 {file}，跳过此文件。错误：{e}")
            continue

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
                all_chunks.append(Document(page_content=cleaned, metadata=metadata))
            else:
                # 长章节再拆块
                sub_chunks = splitter.split_text(cleaned)
                for chunk in sub_chunks:
                    all_chunks.append(Document(page_content=chunk, metadata=metadata))

    return all_chunks

# 3. 初始化HuggingFace嵌入模型 配置gpu加速  返回文本向量化器 
def initialize_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_PATH,
        model_kwargs={'device': EMBEDDING_DEVICE},
    )
class FastBGEEmbedding(Embeddings): #
    def __init__(self, model_name: str = "BAAI/bge-m3", device: str = "cuda"):
        self.model = SentenceTransformer(model_name, device=device)

    def embed_documents(self, texts):
        embeddings = self.model.encode(texts, convert_to_tensor=True, batch_size=32, show_progress_bar=True) #convert_to_tensor=True 将文本转换为tensor  batch_size=32 批量处理  show_progress_bar=True 显示进度条
        normed = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return normed.cpu().tolist()

    def embed_query(self, text):
        emb = self.model.encode([text], convert_to_tensor=True)
        emb = torch.nn.functional.normalize(emb, p=2, dim=1)
        return emb[0].cpu().tolist()
# 4. 创建或加载向量数据库 (Modified) 检测数据库状态  处理模型变更导致的维度问题 支持增量更新文档
def get_vector_db(chunks, embeddings, persist_directory):
    """Creates a new vector DB or loads an existing one."""
    if os.path.exists(persist_directory) and os.listdir(persist_directory): #有现成数据库就加在，没有就看有没有chunk，有就创建 
        print(f"Loading existing vector database from {persist_directory}...")
        try:
            # When loading, ChromaDB will check for dimension compatibility.
            # If EMBEDDING_MODEL_PATH changed leading to a dimension mismatch, this will fail.
            return Chroma(persist_directory=persist_directory, embedding_function=embeddings) #chroma会从目录中找数据文件  并且启用embedding_function去配置
        except Exception as e: #确保embedding模型对的
            print(f"Error loading existing vector database: {e}.")
            print(f"This might be due to a change in the embedding model and a dimension mismatch.")
            print(f"If you changed EMBEDDING_MODEL_PATH, you MUST delete the old database directory: {persist_directory}")
            # If loading fails, proceed as if it doesn't exist, but only create if chunks are given later.
            return None # Indicate loading failed or DB doesn't exist in a usable state
    else:
        # Directory doesn't exist or is empty
        if chunks:
            print(f"Creating new vector database in {persist_directory}...")
            print(f"Creating Chroma DB with {len(chunks)} chunks...")
            try:
                
                vector_db = Chroma.from_documents( #通过from_document把每个chunk转换为embedding并存到数据库
                    documents=chunks,
                    embedding=embeddings,
                    persist_directory=persist_directory #这是保存位置
                )
                print("Vector database created and persisted.")
                return vector_db
            except Exception as e:
                print(f"Error creating new vector database: {e}")
                raise  # Re-raise the exception if creation fails
        else:
            # No chunks provided and DB doesn't exist/is empty - cannot create.
            print(f"Vector database directory {persist_directory} not found or empty, and no chunks provided to create a new one.")
            return None # Indicate DB doesn't exist and cannot be created yet

# 5. 初始化连接到VLLM服务器的ChatOpenAI客户端 (Replaces initialize_llm) 连接VLLM推理服务器 配置模型 返回兼容接口
def initialize_openai_client():
    """Initializes ChatOpenAI client pointing to the VLLM server."""
    print(f"Initializing ChatOpenAI client for VLLM server at {VLLM_BASE_URL}...")
    return ChatOpenAI(
        openai_api_base=VLLM_BASE_URL,
        openai_api_key=VLLM_API_KEY,
        model_name=VLLM_MODEL_NAME
    )

#     return rag_chain
def create_rag_chain_with_memory(vector_db, llm, memory): #创建带有记忆的问答链  这部分的提示词相当于都是封装的 用别人的库 可以尝试自己手写
    retriever = vector_db.as_retriever(search_kwargs={"k": SEARCH_K}) #将向量数据库变为一个检索器，能输入问题返回最相关的k个文本
    # 直接用 ConversationalRetrievalChain，自动管理上下文
    return ConversationalRetrievalChain.from_llm( #封装的类， 可以把检索器返回的文本，当前对话 历史对话拼接成一个prompt   （检索向量库得到最相关的k个，获取memory 拼接prompt LLM调用返回结果，分别对应下面的参数）
        llm=llm, #这个指用哪个模型回答
        retriever=retriever, #这是指用哪个检索器 
        memory=memory,
        return_source_documents=False  # 如果不需要输出检索到的源文档，可设为 False 
    )

# 7. Function to process query using the RAG chain (Modified for Streaming)
def process_query(query):
    """Processes a user query using the RAG chain and streams the answer."""
    global rag_chain, vector_db # Add vector_db to globals accessed here for debugging  这里是表明 这两个是全局变量 可以让函数内部修改函数外部定义的变量（如果没定义这个，函数内是不能修改全局变量） 也就是之前定义过的，如果不声明 会出错  
    if rag_chain is None:
        yield "错误：RAG 链未初始化。"
        return

    # --- For Debugging Retrieval ---
    # Uncomment the block below to see what documents are retrieved by the vector DB
    if vector_db:
        try:
            retrieved_docs = vector_db.similarity_search(query, k=SEARCH_K)
            print(f"\n--- Retrieved Documents for query: '{query}' ---")
            for i, doc in enumerate(retrieved_docs):
                # Attempt to get score if retriever supports it (Chroma's similarity_search_with_score)
                # For basic similarity_search, score might not be directly in metadata.
                # If using retriever.get_relevant_documents(), score might be present.
                score = doc.metadata.get('score', 'N/A') # 先从metadata中取score 没有就返回NA
                if hasattr(doc, 'score'): #  有些document是带score属性  这个相当于是一种兼容性 如果前者没找到 就找这个
                    score = doc.score
                
                print(f"Doc {i+1} (Score: {score}):")
                print(f"Content: {doc.page_content[:500]}...") # Print first 500 chars
                print(f"Metadata: {doc.metadata}")
            print("--- End Retrieved Documents ---\n")
        except Exception as e:
            print(f"Error during debug similarity_search: {e}")
    else:
        print("Vector DB not initialized, skipping debug retrieval.")
    # --- End Debugging Retrieval ---

    try:
        print(f"开始处理流式查询: {query}")

        response_stream = rag_chain.stream({ #rag_chain就是之前创建的包括检索的数据 历史回答 拼接得到的结果 stream可以保证回复是一边生成一边返回   invoke和predict都是一次性返回整段内容
                "question": query,
                                            })

        full_answer = ""
        # Yield chunks as they arrive. Gradio Textbox updates incrementally.
        print("开始流式生成回答...")
        for chunk in response_stream:
            # Check if the 'answer' key exists in the chunk and append it
            answer_part = chunk.get("answer", "") #这边将其拼接起来  如果这样的话 比如改一下 让他们一次性输出
            if answer_part:
                full_answer += answer_part 
                yield full_answer # Yield the progressively built answer

        if not full_answer:
             yield "抱歉，未能生成回答。" # Handle cases where stream completes without answer

        print(f"流式处理完成。最终回答: {full_answer}")

    except Exception as e: # 先简要打印错误信息 然后通过traceback 输出完整的错误栈追踪
        print(f"处理查询时发生错误: {e}")
        import traceback
        traceback.print_exc() # Print stack trace for debugging
        yield f"处理查询时发生错误: {e}"#注意 如果是print前端不会受到然后信息  yield可以

# 8. Function to rebuild the index and RAG chain (Modified to add documents)
def rebuild_index_and_chain(): #全流程索引重建  文档加载-分割-嵌入-存储   
    """Loads documents, creates/updates vector DB by adding new content, and rebuilds the RAG chain."""
    global vector_db, rag_chain, embeddings, llm

    if embeddings is None or llm is None:
        return "错误：Embeddings 或 LLM 未初始化。"

    # Ensure documents directory exists
    if not os.path.exists(DOCUMENTS_DIR):
        os.makedirs(DOCUMENTS_DIR)
        print(f"创建文档目录: {DOCUMENTS_DIR}")

    # Step 1: Load documents
    print("加载文档...")
    documents = load_documents(DOCUMENTS_DIR)
    if not documents:
        print(f"在 {DOCUMENTS_DIR} 中未找到文档。")
        # Try to load existing DB even if no new documents are found
        print("尝试加载现有向量数据库...")
        # Pass None for chunks as we are just trying to load
        vector_db = get_vector_db(None, embeddings, PERSIST_DIR)
        if vector_db:
            print("没有新文档加载，将使用现有的向量数据库。重新创建 RAG 链...")
            rag_chain = create_rag_chain_with_memory(vector_db, llm, memory)

            return "没有找到新文档，已使用现有数据重新加载 RAG 链。"
        else:
            # No documents AND no existing DB
            return "错误：没有文档可加载，且没有现有的向量数据库。"

    # Step 2: Split text
    print("分割文本...")
    chunks = split_documents(documents)
    # 过滤和预处理：只保留非空字符串内容的chunk
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
             rag_chain = create_rag_chain_with_memory(vector_db, llm, memory)# Ensure chain is recreated
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
            vector_db.add_documents(chunks)
            print("块添加成功。")
            # Persisting might be needed depending on Chroma version/setup, often automatic.
            # vector_db.persist() # Uncomment if persistence issues occur
        except Exception as e:
             print(f"添加文档到 Chroma 时出错: {e}")
             # If adding fails, proceed with the DB as it was before adding
             rag_chain = create_rag_chain_with_memory(vector_db, llm, memory)
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
    rag_chain = create_rag_chain_with_memory(vector_db, llm, memory)
    print("索引和 RAG 链已成功更新。")
    return "文档处理完成，索引和 RAG 链已更新。"

# Helper function to list documents in the directory 生成已加载的文档列表 markdown格式化输出  实时更新文档状态
def get_loaded_documents_list():
    """Returns a Markdown formatted list of files in DOCUMENTS_DIR."""
    if not os.path.exists(DOCUMENTS_DIR) or not os.listdir(DOCUMENTS_DIR): #存在则函数返回true  如果文档存在，并且里面有东西就不返回内容
        return "当前没有已加载的文档。"
    try:
        files = [f for f in os.listdir(DOCUMENTS_DIR) if os.path.isfile(os.path.join(DOCUMENTS_DIR, f)) and (f.endswith('.pdf') or f.endswith('.docx') or f.endswith('.doc'))]# 符合条件的全部遍历
        if not files: 
            return "当前没有已加载的文档。"
        markdown_list = "### 当前已加载文档:\n" + "\n".join([f"- {file}" for file in files]) #这里的f只是一种格式化写法  用于{file}
        return markdown_list
    except Exception as e:
        print(f"Error listing documents: {e}")
        return "无法列出文档。"


# 9. Function to handle file uploads (Modified to return doc list) 接受上传的文档 保存到文档目录 触发索引重建（类似图书馆进来新的书 重新创建索引） 返回上传状态
def handle_file_upload(file_obj):
    """Saves the uploaded file, triggers index rebuilding, and returns status and doc list."""
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
        status = rebuild_index_and_chain()
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
    global user_facts
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
 
    # 新增：将本轮对话存入向量库 后序可以添加过滤功能 防止对话污染信息
    if vector_db is not None and query_text.strip() and final_response_from_rag.strip(): #strip去掉开头和结尾的空白字符
        # 组合成一个片段
        dialogue_text = f"用户: {query_text}\nAI: {final_response_from_rag}"
        # 创建 Document 对象
        doc = Document(page_content=dialogue_text, metadata={"type": "chat_history"}) #打个标签 表示这个是对话历史
        try: 
            vector_db.add_documents([doc]) #相当于把历史文档加载进去
            # 可选：print("对话已存入向量库") 
        except Exception as e:
            print(f"存储对话到向量库失败: {e}")

# 10. 主函数 (Modified for Gradio Blocks, Upload, Doc List, Streaming, and Usage Guide)
def main():
    # global embeddings, llm, rag_chain
    global embeddings, llm, rag_chain,memory,vector_db # Declare globals needed

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
    #embeddings = FastBGEEmbedding(model_name=EMBEDDING_MODEL_PATH, device=EMBEDDING_DEVICE)
    embeddings = initialize_embeddings()

    print("初始化 LLM 客户端...")
    llm = initialize_openai_client()

    # Initial index and chain build
    print("执行初始索引构建...")
    initial_status = rebuild_index_and_chain()
    print(initial_status)
    # if rag_chain is None and "错误" in initial_status:
    #     print("无法初始化 RAG 链。请检查文档或配置。退出。")
    if vector_db is not None:
        rag_chain = create_rag_chain_with_memory(vector_db, llm, memory)
    else:
        print("警告：向量数据库未初始化，RAG 链可能不可用。")
        # Optionally, allow Gradio to launch but show an error state
        # return # Or let Gradio launch to show the error

    # Get initial document list
    initial_doc_list = get_loaded_documents_list()

    # --- Custom CSS for ChatGPT-like styling ---
    # Base styling - can be expanded significantly
    custom_css = """
body, .gradio-container { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
.gradio-container { background-color: #F7F7F8; } /* Light background */

/* Chatbot styling */
.gr-chatbot { border: none; box-shadow: 0 2px 10px rgba(0,0,0,0.05); }
.gr-chatbot .message-wrap { box-shadow: none !important; } /* Remove default shadow on messages */
.gr-chatbot .message.user { background-color: #FFFFFF; border: 1px solid #E5E5E5; color: #333; border-radius: 18px; padding: 10px 15px; margin-left: auto; max-width: 70%;}
.gr-chatbot .message.bot { background-color: #F7F7F8; border: 1px solid #E5E5E5; color: #333; border-radius: 18px; padding: 10px 15px; max-width: 70%;}
.gr-chatbot .message.bot.thinking { color: #888; font-style: italic; } /* Style for "Thinking..." */

/* Input area styling */
#chat_input_row { /* Style for the Row containing input and button */
    display: flex !important;
    align-items: center !important; /* Vertically align items (textbox and button) */
    gap: 8px !important; /* Add a small gap between textbox and button */
}
#chat_input_row .gr-textbox textarea { 
    border-radius: 18px !important; 
    border: 1px solid #E0E0E0 !important; 
    padding: 12px 15px !important; 
    font-size: 1rem !important;
    background-color: #FFFFFF !important;
    box-sizing: border-box !important; /* Ensure padding and border are part of the element's total width and height */
    line-height: 1.4 !important; /* Consistent line height */
    min-height: 46px !important; /* Ensure a minimum height, helps with single line consistency */
}
#chat_input_row .gr-button { 
    border-radius: 18px !important; 
    font-weight: 500 !important;
    background-color: #10A37F !important; /* ChatGPT-like green */
    color: white !important; 
    border: none !important;
    min-width: 80px !important;
    font-size: 1rem !important; /* Match textarea font size */
    /* Textarea has 12px padding + 1px border = 13px effective 'outer' space top/bottom. */
    /* Button has no border, so its padding should be 13px top/bottom. */
    padding: 13px 15px !important; 
    box-sizing: border-box !important; /* Ensure padding is part of the element's total width and height */
    line-height: 1.4 !important; /* Match textarea line height */
    height: 46px !important; /* Explicit height to match textarea's typical single-line height */
}
#chat_input_row .gr-button:hover { background-color: #0F8E6C !important; }

/* General Tab Styling */
.tab-nav button { border-radius: 8px 8px 0 0 !important; padding: 10px 15px !important; }
.tab-nav button.selected { background-color: #E0E0E0 !important; border-bottom: 2px solid #10A37F !important;}
.gr-panel { background-color: #FFFFFF; border-radius: 0 0 8px 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); } /* Panel for tab content */
    """

    # --- Gradio Interface using Blocks ---
    print("\n设置 Gradio 界面...") 
    with gr.Blocks(theme=gr_themes.Soft(), css=custom_css) as iface: #blocks是gradio的界面容器 可以添加按钮等各种东西   theme 是设置主题的  css是指加载自定义的美化规则之类的
        gr.Markdown(f"""     
        <div style='text-align: center;'> 
        <h1>耀安科技-煤矿大模型知识问答系统</h1>
        <p>根据已有的文档或您上传的文档提问。</p>
        </div>
        """)  #markdown gradio的组件 用于显示一些html文本   第一行表示文本内容居中 <h1>表示是大标题  <p>表示是副标题 或者是说明文字 （一个就是字体小的）

        with gr.Tab("问答"): #创建了问答页面
            with gr.Column(elem_id="chat-column"): # Added a column for better layout control  创建一个垂直排列的布局容器 并设置对应id
                chatbot_output = gr.Chatbot( #创建一个聊天窗口
                    label="对话窗口",
                    bubble_full_width=False, # Bubbles don't take full width 
                    height=600, # Set a fixed height for the chat area
                    avatar_images=(None, "https://img.icons8.com/fluency/48/chatbot.png"), # User avatar none, bot has a simple icon 左边是用户的 表示无头像 右边是机器人头像
                    latex_delimiters=[ #表示支持latex数学公式显示
                        {"left": "$$", "right": "$$", "display": True},
                        {"left": "$", "right": "$", "display": False}
                    ]
                    # render_markdown=True,  # Explicitly set, default is True
                    # sanitize_html=False    # Test by disabling HTML sanitization
                )
                with gr.Row(elem_id="chat_input_row"): # Row for input textbox and button  在聊天框下面创建横向区域用于输入框和按钮
                    query_input = gr.Textbox( #创建文本输入框 
                        show_label=False,
                        placeholder="在此输入您的问题...",
                        lines=1, # Single line input initially, can expand
                        scale=4 # Textbox takes more space
                    )
                    submit_button = gr.Button("发送", scale=1) # "Send" button

        with gr.Tab("上传与管理文档"): # Renamed tab for clarity
            with gr.Row(): #表示横向布局容器 这里面的内容是从左往右水平排列
                with gr.Column(scale=1): #column表示纵向 接下来纵向表示
                    file_input = gr.File(label="上传 PDF 或 DOCX 文件", file_types=['.pdf', '.docx', '.doc'])
                    upload_button = gr.Button("上传并重建索引")
                    upload_status = gr.Textbox(label="上传状态", interactive=False) #interactive 表示是否可编辑
                with gr.Column(scale=1): #scale表示权重占比 如果两个都是一 则代表平分
                    # Component to display loaded documents
                    loaded_docs_display = gr.Markdown(value=initial_doc_list)

        with gr.Tab("使用教程"): #markdown内容是直接是被显示  #不显示
            gr.Markdown(""" 
            ## 如何使用本 RAG 系统

            **1. 准备文档:**
               - 您可以将包含知识的 PDF 或 Word 文档（.pdf, .docx, .doc）放入程序运行目录下的 `documents` 文件夹中。
               - 程序启动时会自动加载 `documents` 文件夹中的所有支持的文档。

            **2. 上传文档:**
               - 切换到 **上传与管理文档** 标签页。
               - 点击"浏览文件"按钮选择您想要上传的 PDF 或 Word 文档。
               - 点击 **上传并重建索引** 按钮。系统会将文件复制到 `documents` 目录，并更新知识库。
               - 上传和处理需要一些时间，请耐心等待状态更新。
               - 右侧会显示当前 `documents` 目录中已加载的文件列表。

            **3. 提问:**
               - 切换到 **问答** 标签页。
               - 在 **问题** 输入框中输入您想基于文档内容提出的问题。
               - 点击 **提交问题** 按钮或按 Enter 键。
               - 系统将根据文档内容检索相关信息，并使用大语言模型（Qwen 3-8B）生成回答。
               - 回答将在 **回答** 框中以流式方式显示。

            **注意:**
               - 重建索引可能需要一些时间，特别是对于大型文档或大量文档。
               - 回答的质量取决于文档内容的相关性和模型的理解能力。
               - 目前系统每次上传文件后会重新处理 `documents` 目录下的 *所有* 文件。对于非常大的知识库，未来可能需要优化为仅处理新增文件。
            """)


        # --- Event Handlers ---
        # Q&A Submission for Chatbot
        # The `fn` now takes query_input and chatbot_output (history)
        # It returns a tuple: (new_value_for_query_input, new_value_for_chatbot_output)
        submit_button.click( #当触发发生按钮时需要做的事情
            fn=handle_submit_with_thinking,
            inputs=[query_input, chatbot_output],
            outputs=[query_input, chatbot_output] # query_input is cleared, chatbot_output is updated
        ) 
        query_input.submit(  #上面是通过点击实现 这个是通过回车实现
             fn=handle_submit_with_thinking,
             inputs=[query_input, chatbot_output],
             outputs=[query_input, chatbot_output]
        )

        # File Upload and Rebuild
        upload_button.click( 
            fn=handle_file_upload,
            inputs=file_input,
            outputs=[upload_status, loaded_docs_display] # Update both status and doc list
        )

    print("启动 Gradio 界面...")
    # Launch the interface
    iface.launch(server_name="0.0.0.0") # Listen on all interfaces

if __name__ == "__main__":
    main()