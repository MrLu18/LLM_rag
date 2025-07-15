import os
import shutil
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.chains import ConversationalRetrievalChain
#from langchain.schema import Document
from langchain_openai import ChatOpenAI
from rag_new.config import (
    DOCUMENTS_DIR,
    PERSIST_DIR,
    EMBEDDING_MODEL_PATH,
    EMBEDDING_DEVICE,
    VLLM_BASE_URL,
    VLLM_API_KEY,
    VLLM_MODEL_NAME,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    SEARCH_K
)
from langchain.memory import ConversationBufferWindowMemory
from transformers import AutoTokenizer
# Global variables
rag_chain = None
vector_db = None
embeddings = None
llm = None

# Tokenizer配置（如有需要可更换为你的模型名）
TOKENIZER_MODEL = "/mnt/jrwbxx/LLM/model/qwen3-1.7b" #这个记得随着模型而变
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL)

MAX_TURNS = 3
MAX_TOKENS = 1024

memory = ConversationBufferWindowMemory(
    memory_key="chat_history",
    return_messages=True,
    k=MAX_TURNS
)
user_facts = []



def trim_memory_tokens(memory_obj): #此函数要求对话历史不超过三轮，总token不超过1024 一个超过就裁剪最远的对话，但是最终会至少保存一轮对话
    """裁剪memory_obj中的历史，使其token数不超过MAX_TOKENS"""
    history = memory_obj.chat_memory.messages
    # 只保留最近MAX_TURNS轮
    if len(history) > MAX_TURNS * 2:
        history = history[-MAX_TURNS*2:]
    # 拼接成字符串
    history_text = ""
    for msg in history:  
        if hasattr(msg, "content"): #如果历史对话有content属性 则直接用 没有就转换为字符串 hasattr是一个内置函数 检查对象有没有对应的属性
            history_text += msg.content + "\n"
        else:
            history_text += str(msg) + "\n"
    input_ids = tokenizer(history_text, return_tensors="pt").input_ids #用tokenizer将历史文本转换为tokenID 
    while input_ids.shape[1] > MAX_TOKENS and len(history) > 2: #确保至少有一个对话历史 
        history = history[2:]
        history_text = ""
        for msg in history:
            if hasattr(msg, "content"):
                history_text += msg.content + "\n"
            else:
                history_text += str(msg) + "\n"
        input_ids = tokenizer(history_text, return_tensors="pt").input_ids
    memory_obj.chat_memory.messages = history


def handle_memory_and_query_prep(query_text, current_user_facts):
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
    if updated_user_facts:
        memory_prefix = "，".join(updated_user_facts)
        full_query = f"请记住：{memory_prefix}。用户提问：{query_text}"
    else:
        full_query = query_text

    return full_query, updated_user_facts

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

# 2. 文本分割  创建出适合嵌入模型的小文本块  按照分隔符切割，每块长度不超过chunk_size 允许相邻的有chunk_overlap的字符重叠 最终返回切好的小块列表
def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len, #这个len不是变量 而是将len函数  传入
        separators=[ 
            "\n\n",  # Split by double newlines (paragraphs)
            "\n",    # Split by single newlines
            ". ",    # Split by period followed by space (ensure space to avoid splitting mid-sentence e.g. Mr. Smith)
            "? ",    # Split by question mark followed by space
            "! ",    # Split by exclamation mark followed by space
            "。 ",   # Chinese period followed by space (if applicable)
            "？ ",   # Chinese question mark followed by space (if applicable)
            "！ ",   # Chinese exclamation mark followed by space (if applicable)
            "。\n",  # Chinese period followed by newline
            "？\n",  # Chinese question mark followed by newline
            "！\n",  # Chinese exclamation mark followed by newline
            " ",     # Split by space as a fallback
            ""       # Finally, split by character if no other separator is found
        ],
        is_separator_regex=False
    )
    return text_splitter.split_documents(documents)

# 3. 初始化HuggingFace嵌入模型 配置gpu加速  返回文本向量化器 
def initialize_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_PATH,
        model_kwargs={'device': EMBEDDING_DEVICE}
    )

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
            # If loading fails, proceed as if it doesn\'t exist, but only create if chunks are given later.
            return None # Indicate loading failed or DB doesn't exist in a usable state
    else:
        # Directory doesn't exist or is empty
        if chunks:
            print(f"Creating new vector database in {persist_directory}...")
            print(f"Creating Chroma DB with {len(chunks)} chunks...")
            try:
                vector_db_new = Chroma.from_documents(
                    documents=chunks,
                    embedding=embeddings,
                    persist_directory=persist_directory
                )
                print("Vector database created and persisted.")
                return vector_db_new
            except Exception as e:
                print(f"Error creating new vector database: {e}")
                raise  # 继续抛出原始异常
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

def create_rag_chain_with_memory(vector_db_arg, llm_arg, memory_arg): #创建带有记忆的问答链  这部分的提示词相当于都是封装的 用别人的库 可以尝试自己手写
    retriever = vector_db_arg.as_retriever(search_kwargs={"k": SEARCH_K}) #将向量数据库变为一个检索器，能输入问题返回最相关的k个文本
    # 直接用 ConversationalRetrievalChain，自动管理上下文
    return ConversationalRetrievalChain.from_llm( #封装的类， 可以把检索器返回的文本，当前对话 历史对话拼接成一个prompt   （检索向量库得到最相关的k个，获取memory 拼接prompt LLM调用返回结果，分别对应下面的参数）
        llm=llm_arg, #这个指用哪个模型回答
        retriever=retriever, #这是指用哪个检索器 
        memory=memory_arg,
        return_source_documents=False  # 如果不需要输出检索到的源文档，可设为 False 
    )

# 7. Function to process query using the RAG chain (Modified for Streaming)
def process_query(query):
    """Processes a user query using the RAG chain and streams the answer."""
    global rag_chain, vector_db # Add vector_db to globals accessed here for debugging  这里是表明 这两个是全局变量 可以让函数内部修改函数外部定义的变量（如果没定义这个，函数内是不能修改全局变量） 也就是之前定义过的，如果不声明 会出错  
    # 在每次处理前裁剪记忆
    trim_memory_tokens(memory)
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

        # Directly stream from the RAG chain runnable
        # The input format for create_retrieval_chain is typically {"input": query}
        # The output chunks often contain 'answer' and 'context' keys
        # response_stream = rag_chain.stream({"input": query})
        response_stream = rag_chain.stream({
                "question": query,
                                            })

        full_answer = ""
        # Yield chunks as they arrive. Gradio Textbox updates incrementally.
        print("开始流式生成回答...")
        for chunk in response_stream:
            # Check if the 'answer' key exists in the chunk and append it
            answer_part = chunk.get("answer", "")
            if answer_part:
                full_answer += answer_part 
                # Debugging output
                # print(f"Raw answer_part from LLM: '{answer_part}'")
                # print(f"Yielding to Gradio: '{full_answer}'")
                yield full_answer # Yield the progressively built answer

        if not full_answer:
             yield "抱歉，未能生成回答。" # Handle cases where stream completes without answer

        print(f"流式处理完成。最终回答: {full_answer}")

    except Exception as e:
        print(f"处理查询时发生错误: {e}")
        import traceback
        traceback.print_exc() # Print stack trace for debugging
        yield f"处理查询时发生错误: {e}"

# 8. Function to rebuild the index and RAG chain (Modified to add documents)
def rebuild_index_and_chain():
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
        vector_db_loaded = get_vector_db(None, embeddings, PERSIST_DIR)
        if vector_db_loaded:
            vector_db = vector_db_loaded
            print("没有新文档加载，将使用现有的向量数据库。重新创建 RAG 链...")
            rag_chain = create_rag_chain_with_memory(vector_db, llm, memory)

            return "没有找到新文档，已使用现有数据重新加载 RAG 链。"
        else:
            # No documents AND no existing DB
            return "错误：没有文档可加载，且没有现有的向量数据库。"

    # Step 2: Split text
    print("分割文本...")
    chunks = split_documents(documents)
    if not chunks:
        print("分割后未生成文本块。")
        # Try loading existing DB if splitting yielded nothing
        print("尝试加载现有向量数据库...")
        vector_db_loaded = get_vector_db(None, embeddings, PERSIST_DIR)
        if vector_db_loaded:
             vector_db = vector_db_loaded
             print("警告：新加载的文档分割后未产生任何文本块。使用现有数据库。")
             rag_chain = create_rag_chain_with_memory(vector_db, llm, memory)# Ensure chain is recreated
             return "警告：文档分割后未产生任何文本块。RAG 链已使用现有数据重新加载。"
        else:
            # No chunks AND no existing DB
            return "错误：文档分割后未产生任何文本块，且无现有数据库。"

    # Step 3: Load or Create/Update vector database
    print("加载或更新向量数据库...")
    # Try loading first, even if we have chunks (in case we want to add to it)
    vector_db_loaded = get_vector_db(None, embeddings, PERSIST_DIR)

    if vector_db_loaded:
        print(f"向现有向量数据库添加 {len(chunks)} 个块...")
        vector_db = vector_db_loaded # Use the loaded DB
        try:
            vector_db.add_documents(chunks)
            print("块添加成功。")
            # Persisting might be needed depending on Chroma version/setup, often automatic.
            # vector_db.persist() # Uncomment if persistence issues occur
        except Exception as e:
            print(f"添加文档到 Chroma 时出错: {e}")
            print("使用二分法递归定位出错文档...")
            # 二分法递归定位出错chunk
            def bisect_add(chunks, add_func):
                if not chunks:
                    return
                try:
                    add_func(chunks)
                except Exception as e_inner:
                    if len(chunks) == 1:
                        chunk = chunks[0]
                        doc_name = getattr(chunk, 'metadata', {}).get('source', None) or str(chunk)[:100]
                        print(f"出错chunk: {doc_name}, 错误: {e_inner}")
                    else:
                        mid = len(chunks) // 2
                        bisect_add(chunks[:mid], add_func)
                        bisect_add(chunks[mid:], add_func)
            bisect_add(chunks, lambda cs: vector_db.add_documents(cs))
            rag_chain = create_rag_chain_with_memory(vector_db, llm, memory)
            return f"错误：向向量数据库添加文档时出错: {e}。RAG链可能使用旧数据。"
    else:
        # Database didn't exist or couldn't be loaded, create a new one with the current chunks
        print(f"创建新的向量数据库并添加 {len(chunks)} 个块...")
        try:
            # Call get_vector_db again, this time *with* chunks to trigger creation
            newly_created_vector_db = get_vector_db(chunks, embeddings, PERSIST_DIR) #用于新建数据库
            if newly_created_vector_db is None:
                 raise RuntimeError("get_vector_db failed to create a new database.")
            vector_db = newly_created_vector_db # Assign the newly created DB to the global variable
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
    if not os.path.exists(DOCUMENTS_DIR) or not os.listdir(DOCUMENTS_DIR):
        return "当前没有已加载的文档。"
    try:
        files = [f for f in os.listdir(DOCUMENTS_DIR) if os.path.isfile(os.path.join(DOCUMENTS_DIR, f)) and (f.endswith('.pdf') or f.endswith('.docx') or f.endswith('.doc'))]
        if not files: 
            return "当前没有已加载的文档。"
        markdown_list = "### 当前已加载文档:\n" + "\n".join([f"- {file}" for file in files])
        return markdown_list
    except Exception as e:
        print(f"Error listing documents: {e}")
        return "无法列出文档。"

# 9. Function to handle file uploads (Modified to return doc list)
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


def detect_and_remove_duplicates():
    """
    检测并清除向量库中的重复内容
    返回检测结果和清理状态
    """
    global vector_db
    
    if vector_db is None:
        return "错误：向量数据库未初始化"
    
    try:
        print("开始检测向量库中的重复内容...")
        
        # 获取向量库中的所有文档
        all_docs = vector_db.get()
        if not all_docs or not all_docs['documents']:
            return "向量库为空，无需检测重复内容"
        
        documents = all_docs['documents']
        metadatas = all_docs['metadatas']
        ids = all_docs['ids']
        
        print(f"向量库中共有 {len(documents)} 个文档块")
        
        # 检测重复内容
        seen_contents = {}
        duplicate_ids = []
        duplicate_info = []
        
        for i, (doc_id, content, metadata) in enumerate(zip(ids, documents, metadatas)):
            # 清理内容用于比较（去除多余空格和换行）
            cleaned_content = ' '.join(content.strip().split())
            
            if cleaned_content in seen_contents:
                # 发现重复
                original_id = seen_contents[cleaned_content]
                duplicate_ids.append(doc_id)
                duplicate_info.append({
                    'duplicate_id': doc_id,
                    'original_id': original_id,
                    'content_preview': content[:100] + '...' if len(content) > 100 else content,
                    'source': metadata.get('source', 'unknown') if metadata else 'unknown'
                })
                print(f"发现重复内容 - ID: {doc_id}, 原始ID: {original_id}")
            else:
                seen_contents[cleaned_content] = doc_id
        
        if not duplicate_ids:
            return "未发现重复内容"
        
        # 删除重复内容
        print(f"开始删除 {len(duplicate_ids)} 个重复文档...")
        vector_db.delete(ids=duplicate_ids)
        
        # 重新创建RAG链
        global rag_chain, llm, memory
        if llm is not None:
            rag_chain = create_rag_chain_with_memory(vector_db, llm, memory)
        
        # 生成详细报告
        report = f"重复内容清理完成！\n"
        report += f"- 检测到 {len(duplicate_ids)} 个重复文档\n"
        report += f"- 已删除重复内容，保留原始文档\n"
        report += f"- 当前向量库剩余 {len(documents) - len(duplicate_ids)} 个文档块\n\n"
        
        report += "重复内容详情：\n"
        for i, dup in enumerate(duplicate_info, 1):
            report += f"{i}. 重复ID: {dup['duplicate_id']}\n"
            report += f"   原始ID: {dup['original_id']}\n"
            report += f"   来源: {dup['source']}\n"
            report += f"   内容预览: {dup['content_preview']}\n\n"
        
        print("重复内容清理完成")
        return report
        
    except Exception as e:
        error_msg = f"检测重复内容时发生错误: {e}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        return error_msg


def analyze_vector_db_content():
    """
    分析向量库内容，提供统计信息
    """
    global vector_db
    
    if vector_db is None:
        return "错误：向量数据库未初始化"
    
    try:
        # 获取向量库中的所有文档
        all_docs = vector_db.get()
        if not all_docs or not all_docs['documents']:
            return "向量库为空"
        
        documents = all_docs['documents']
        metadatas = all_docs['metadatas']
        ids = all_docs['ids']
        
        # 统计信息
        total_chunks = len(documents)
        total_chars = sum(len(doc) for doc in documents)
        avg_chunk_length = total_chars / total_chunks if total_chunks > 0 else 0
        
        # 按来源文件统计
        source_stats = {}
        for metadata in metadatas:
            if metadata and 'source' in metadata:
                source = metadata['source']
                source_stats[source] = source_stats.get(source, 0) + 1
        
        # 检测潜在重复
        content_hashes = {}
        potential_duplicates = 0
        
        for content in documents:
            cleaned_content = ' '.join(content.strip().split())
            if cleaned_content in content_hashes:
                potential_duplicates += 1
            else:
                content_hashes[cleaned_content] = 1
        
        # 生成报告
        report = f"## 向量库内容分析报告\n\n"
        report += f"**基本信息：**\n"
        report += f"- 总文档块数: {total_chunks}\n"
        report += f"- 总字符数: {total_chars:,}\n"
        report += f"- 平均块长度: {avg_chunk_length:.1f} 字符\n"
        report += f"- 潜在重复块: {potential_duplicates}\n\n"
        
        if source_stats:
            report += f"**按来源文件统计：**\n"
            for source, count in sorted(source_stats.items(), key=lambda x: x[1], reverse=True):
                report += f"- {os.path.basename(source)}: {count} 块\n"
        
        if potential_duplicates > 0:
            report += f"\n**注意：** 发现 {potential_duplicates} 个潜在重复内容，建议运行重复内容清理。"
        
        return report
        
    except Exception as e:
        error_msg = f"分析向量库内容时发生错误: {e}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        return error_msg


def clear_vector_db():
    """
    清空整个向量库
    """
    global vector_db, rag_chain
    
    if vector_db is None:
        return "错误：向量数据库未初始化"
    
    try:
        # 获取所有文档ID
        all_docs = vector_db.get()
        if not all_docs or not all_docs['ids']:
            return "向量库已经是空的"
        
        ids_to_delete = all_docs['ids']
        print(f"开始清空向量库，删除 {len(ids_to_delete)} 个文档...")
        
        # 删除所有文档
        vector_db.delete(ids=ids_to_delete)
        
        # 重置RAG链
        rag_chain = None
        
        return f"向量库已清空，删除了 {len(ids_to_delete)} 个文档块"
        
    except Exception as e:
        error_msg = f"清空向量库时发生错误: {e}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        return error_msg 