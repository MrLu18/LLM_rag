import gradio as gr # Import Gradio
# Import a Gradio theme
import gradio.themes as gr_themes
import os
from sentence_transformers import SentenceTransformer, util
# Import functions and configurations from rag_core.py and config.py
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
1. 保持原意不变
2. 消除可能的歧义
3. 使问题更加明确和具体
4. 只需要改写问题，不需要然后说明
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

#     initial_doc_list = rag_new.rag_core.get_loaded_documents_list()

#     # --- Custom CSS for ChatGPT-like styling ---
#     # Base styling - can be expanded significantly
#     custom_css = """
# body, .gradio-container { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
# .gradio-container { background-color: #F7F7F8; } /* Light background */

# /* Chatbot styling */
# .gr-chatbot { border: none; box-shadow: 0 2px 10px rgba(0,0,0,0.05); }
# .gr-chatbot .message-wrap { box-shadow: none !important; } /* Remove default shadow on messages */
# .gr-chatbot .message.user { background-color: #FFFFFF; border: 1px solid #E5E5E5; color: #333; border-radius: 18px; padding: 10px 15px; margin-left: auto; max-width: 70%;}
# .gr-chatbot .message.bot { background-color: #F7F7F8; border: 1px solid #E5E5E5; color: #333; border-radius: 18px; padding: 10px 15px; max-width: 70%;}
# .gr-chatbot .message.bot.thinking { color: #888; font-style: italic; } /* Style for "Thinking..." */

# /* Input area styling */
# #chat_input_row { /* Style for the Row containing input and button */
#     display: flex !important;
#     align-items: center !important; /* Vertically align items (textbox and button) */
#     gap: 8px !important; /* Add a small gap between textbox and button */
# }
# #chat_input_row .gr-textbox textarea { 
#     border-radius: 18px !important; 
#     border: 1px solid #E0E0E0 !important; 
#     padding: 12px 15px !important; 
#     font-size: 1rem !important;
#     background-color: #FFFFFF !important;
#     box-sizing: border-box !important; /* Ensure padding and border are part of the element's total width and height */
#     line-height: 1.4 !important; /* Consistent line height */
#     min-height: 46px !important; /* Ensure a minimum height, helps with single line consistency */
# }
# #chat_input_row .gr-button { 
#     border-radius: 18px !important; 
#     font-weight: 500 !important;
#     background-color: #10A37F !important; /* ChatGPT-like green */
#     color: white !important; 
#     border: none !important;
#     min-width: 80px !important;
#     font-size: 1rem !important; /* Match textarea font size */
#     /* Textarea has 12px padding + 1px border = 13px effective 'outer' space top/bottom. */
#     /* Button has no border, so its padding should be 13px top/bottom. */
#     padding: 13px 15px !important; 
#     box-sizing: border-box !important; /* Ensure padding is part of the element's total width and height */
#     line-height: 1.4 !important; /* Match textarea line height */
#     height: 46px !important; /* Explicit height to match textarea's typical single-line height */
# }
# #chat_input_row .gr-button:hover { background-color: #0F8E6C !important; }

# /* General Tab Styling */
# .tab-nav button { border-radius: 8px 8px 0 0 !important; padding: 10px 15px !important; }
# .tab-nav button.selected { background-color: #E0E0E0 !important; border-bottom: 2px solid #10A37F !important;}
# .gr-panel { background-color: #FFFFFF; border-radius: 0 0 8px 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); } /* Panel for tab content */
#     """

    # --- Gradio Interface using Blocks ---
    # print("\n设置 Gradio 界面...") 
    # with gr.Blocks(theme=gr_themes.Soft(), css=custom_css) as iface: #blocks是gradio的界面容器 可以添加按钮等各种东西   theme 是设置主题的  css是指加载自定义的美化规则之类的
    #     gr.Markdown(f"""     
    #     <div style='text-align: center;'> 
    #     <h1>耀安科技-煤矿大模型知识问答系统</h1>
    #     <p>根据已有的文档或您上传的文档提问。</p>
    #     </div>
    #     """
    #     )

    #     with gr.Tab("问答"): #创建了问答页面
    #         with gr.Column(elem_id="chat-column"): # Added a column for better layout control  创建一个垂直排列的布局容器 并设置对应id
    #             chatbot_output = gr.Chatbot( #创建一个聊天窗口
    #                 label="对话窗口",
    #                 bubble_full_width=False, # Bubbles don't take full width 
    #                 height=600, # Set a fixed height for the chat area
    #                 avatar_images=(None, "https://img.icons8.com/fluency/48/chatbot.png"), # User avatar none, bot has a simple icon 左边是用户的 表示无头像 右边是机器人头像
    #                 latex_delimiters=[ #表示支持latex数学公式显示
    #                     {"left": "$$", "right": "$$", "display": True},
    #                     {"left": "$", "right": "$", "display": False}
    #                 ]
    #                 # render_markdown=True,  # Explicitly set, default is True
    #                 # sanitize_html=False    # Test by disabling HTML sanitization
    #             )
    #             with gr.Row(elem_id="chat_input_row"): # Row for input textbox and button  在聊天框下面创建横向区域用于输入框和按钮
    #                 query_input = gr.Textbox( #创建文本输入框 
    #                     show_label=False,
    #                     placeholder="在此输入您的问题...",
    #                     lines=1, # Single line input initially, can expand
    #                     scale=4 # Textbox takes more space
    #                 )
    #                 submit_button = gr.Button("发送", scale=1) # "Send" button

    #     with gr.Tab("上传与管理文档"): # Renamed tab for clarity
    #         with gr.Row(): #表示横向布局容器 这里面的内容是从左往右水平排列
    #             with gr.Column(scale=1): #column表示纵向 接下来纵向表示
    #                 file_input = gr.File(label="上传 PDF 或 DOCX 文件", file_types=['.pdf', '.docx', '.doc'])
    #                 upload_button = gr.Button("上传并重建索引")
    #                 upload_status = gr.Textbox(label="上传状态", interactive=False)
    #             with gr.Column(scale=1): #scale表示权重占比 如果两个都是一 则代表平分
    #                 # Component to display loaded documents
    #                 loaded_docs_display = gr.Markdown(value=initial_doc_list)

    #     with gr.Tab("使用教程"): #markdown内容是直接是被显示  #不显示
    #         gr.Markdown(""" 
    #         ## 如何使用本 RAG 系统

    #         **1. 准备文档:**
    #            - 您可以将包含知识的 PDF 或 Word 文档（.pdf, .docx, .doc）放入程序运行目录下的 `documents` 文件夹中。
    #            - 程序启动时会自动加载 `documents` 文件夹中的所有支持的文档。

    #         **2. 上传文档:**
    #            - 切换到 **上传与管理文档** 标签页。
    #            - 点击"浏览文件"按钮选择您想要上传的 PDF 或 Word 文档。
    #            - 点击 **上传并重建索引** 按钮。系统会将文件复制到 `documents` 目录，并更新知识库。
    #            - 上传和处理需要一些时间，请耐心等待状态更新。
    #            - 右侧会显示当前 `documents` 目录中已加载的文件列表。

    #         **3. 提问:**
    #            - 切换到 **问答** 标签页。
    #            - 在 **问题** 输入框中输入您想基于文档内容提出的问题。
    #            - 点击 **提交问题** 按钮或按 Enter 键。
    #            - 系统将根据文档内容检索相关信息，并使用大语言模型（Qwen 3-8B）生成回答。
    #            - 回答将在 **回答** 框中以流式方式显示。

    #         **注意:**
    #            - 重建索引可能需要一些时间，特别是对于大型文档或大量文档。
    #            - 回答的质量取决于文档内容的相关性和模型的理解能力。
    #            - 目前系统每次上传文件后会重新处理 `documents` 目录下的 *所有* 文件。对于非常大的知识库，未来可能需要优化为仅处理新增文件。
    #         """
    #         )

    #     with gr.Tab("向量库管理"):
    #         with gr.Row():
    #             with gr.Column(scale=1):
    #                 gr.Markdown("### 向量库操作")
    #                 analyze_button = gr.Button("分析向量库内容", variant="secondary")
    #                 detect_duplicates_button = gr.Button("检测并清除重复内容", variant="secondary")
    #                 clear_db_button = gr.Button("清空向量库", variant="stop")
                    
    #                 gr.Markdown("### 操作说明")
    #                 gr.Markdown("""
    #                 **分析向量库内容**: 查看向量库的统计信息，包括文档块数量、字符数、来源文件分布等。
                    
    #                 **检测并清除重复内容**: 自动检测并删除向量库中的重复文档块，保留原始内容。
                    
    #                 **清空向量库**: 删除向量库中的所有内容（谨慎使用）。
    #                 """)
                    
    #             with gr.Column(scale=2):
    #                 vector_db_status = gr.Textbox(
    #                     label="向量库状态", 
    #                     interactive=False, 
    #                     lines=15,
    #                     placeholder="点击上方按钮查看向量库信息..."
    #                 )


        # --- Event Handlers ---
        # Q&A Submission for Chatbot
        # The `fn` now takes query_input and chatbot_output (history)
        # It returns a tuple: (new_value_for_query_input, new_value_for_chatbot_output)
        # submit_button.click(
        #     fn=handle_submit_with_thinking,
        #     inputs=[query_input, chatbot_output],
        #     outputs=[query_input, chatbot_output]
        # ) 
        # query_input.submit(
        #      fn=handle_submit_with_thinking,
        #      inputs=[query_input, chatbot_output],
        #      outputs=[query_input, chatbot_output]
        # )

        # File Upload and Rebuild
        # upload_button.click(
        #     fn=rag_new.rag_core.handle_file_upload,
        #     inputs=file_input,
        #     outputs=[upload_status, loaded_docs_display]
        # )
        
        #这个功能也不需要
        # Vector Database Management
        # analyze_button.click(
        #     fn=rag_new.rag_core.analyze_vector_db_content,
        #     inputs=[],
        #     outputs=vector_db_status
        # )
        
        # detect_duplicates_button.click(
        #     fn=rag_new.rag_core.detect_and_remove_duplicates,
        #     inputs=[],
        #     outputs=vector_db_status
        # )
        
        # clear_db_button.click(
        #     fn=rag_new.rag_core.clear_vector_db,
        #     inputs=[],
        #     outputs=vector_db_status
        # )

    # print("启动 Gradio 界面...")
    # # Launch the interface
    # iface.launch(server_name="0.0.0.0") # Listen on all interfaces

# if __name__ == "__main__":
#     main()