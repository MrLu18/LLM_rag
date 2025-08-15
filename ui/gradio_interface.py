import os
import gradio as gr
import gradio.themes as gr_themes
from datetime import datetime
import shutil
import sys

# 添加父目录到路径，以便导入rag_client
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入RAG核心功能
from rag_client import (
    rebuild_index_and_chain, 
    force_rebuild_index, 
    get_document_index_status,
    get_chat_history_status,
    clear_chat_history_vector_db,
    get_loaded_documents_list,
    handle_submit_with_thinking,
    handle_file_upload,
    initialize_system
)

def create_gradio_interface():
    """创建Gradio界面"""
    
    # 初始化系统
    print("初始化RAG系统...")
    initial_status = initialize_system()
    print(initial_status)
    
    # 获取初始文档列表
    initial_doc_list = get_loaded_documents_list()
    
    # --- 自定义CSS样式 ---
    custom_css = """
body, .gradio-container { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
.gradio-container { background-color: #F7F7F8; }

/* Chatbot styling .gr-chatbot 是gradio聊天组件的外层容器 这些是聊天组件的样式，不需要后续调用 控件需要调用   首先通过theme获得官方的样式，然后通过css覆盖官方的一些样式 使用自定义的样式，这部分就属于直接覆盖官方的一些内容*/
.gr-chatbot { border: none; box-shadow: 0 2px 10px rgba(0,0,0,0.05); }   /* 外层容器样式 去边框  后者参数分别是 水平位移 垂直位移 模糊半径（值越大越模糊） 颜色  参数顺序是 水平（必须有） 垂直（必须有） 模糊 扩展 颜色  */
.gr-chatbot .message-wrap { box-shadow: none !important; }
.gr-chatbot .message.user { background-color: #FFFFFF; border: 1px solid #E5E5E5; color: #333; border-radius: 18px; padding: 10px 15px; margin-left: auto; max-width: 70%;} /*margin-left: auto 是让用户的消息靠右对齐 下面是机器人消息*/
.gr-chatbot .message.bot { background-color: #F7F7F8; border: 1px solid #E5E5E5; color: #333; border-radius: 18px; padding: 10px 15px; max-width: 70%;}
.gr-chatbot .message.bot.thinking { color: #888; font-style: italic; } /* 机器人思考状态下 字体颜色是灰色 斜体*/

/* Input area styling */
#chat_input_row {  /*注意 #开头的id选择器 一个页面 一个id理论只可以使用一次 优先级高 .开头的是类选择器 优先级低 但是可以多个类选择器同时使用*/
    display: flex !important;
    align-items: center !important;
    gap: 8px !important; /*表示两个组件之间的间距  这里就是输入框和按钮直接的间距*/
}
#chat_input_row .gr-textbox textarea {  /*这个就是输入框的样式*/
    border-radius: 18px !important; 
    border: 1px solid #E0E0E0 !important; 
    padding: 12px 15px !important; 
    font-size: 1rem !important;
    background-color: #FFFFFF !important;
    box-sizing: border-box !important;
    line-height: 1.4 !important;
    min-height: 46px !important;
}
#chat_input_row .gr-button {   /*这个就是按钮的样式*/
    border-radius: 18px !important; 
    font-weight: 500 !important;
    background-color: #10A37F !important;
    color: white !important; 
    border: none !important;
    min-width: 80px !important;
    font-size: 1rem !important;
    padding: 13px 15px !important; 
    box-sizing: border-box !important;
    line-height: 1.4 !important;
    height: 46px !important;
}
#chat_input_row .gr-button:hover { background-color: #0F8E6C !important; }

/* 文档列表滚动样式 */
.document-list-container {
    max-height: 800px !important;
    overflow-y: auto !important; /*overflow-y: auto 是内容超出显示滚动条 scroll 始终显示滚动条 hidden 超出部分被裁掉 不显示 visible（默认） 超出部分直接显示 （不裁剪，不滚动）*/
    border: 1px solid #E0E0E0 !important;
    border-radius: 8px !important;
    padding: 15px !important;
    background-color: #FAFAFA !important;
    box-shadow: inset 0 1px 3px rgba(0,0,0,0.1) !important;/*注意 insert表明内影，没有则默认外影*/
}

.document-list-container::-webkit-scrollbar {  /*::-webkit-scrollbar 是webkit内核浏览器的滚动条样式 是滚动条的样式*/
    width: 10px !important;
}

.document-list-container::-webkit-scrollbar-track { /*::-webkit-scrollbar-track 是滚动条的轨道样式 */
    background: #F5F5F5 !important;
    border-radius: 5px !important;
    margin: 2px !important;
}

.document-list-container::-webkit-scrollbar-thumb { /*::-webkit-scrollbar-thumb 是滚动条的滑块样式 */
    background: #D0D0D0 !important;
    border-radius: 5px !important;
    border: 1px solid #F5F5F5 !important;
}

.document-list-container::-webkit-scrollbar-thumb:hover { /*::-webkit-scrollbar-thumb:hover  鼠标悬浮时*/
    background: #B0B0B0 !important;
}

.document-list-container::-webkit-scrollbar-thumb:active { /*::-webkit-scrollbar-thumb:active  鼠标点击时*/
    background: #909090 !important;
}

/* 文档列表项样式 */
.document-list-container ul { /*ul 是无序列表样式*/
    margin: 0 !important;
    padding-left: 20px !important;
    list-style-type: disc !important; /*disc 是圆点样式 强制使用实心圆点 也就是列表前面有圆点 实际效果应该就是文档名称前有圆点*/
}

.document-list-container li { /*li 是列表项样式  和ul的关系就是 一个ul可以包含多个li*/
    margin-bottom: 10px !important;
    line-height: 1.5 !important; /*line-height 是行高 1.5 表示行高是字体高度的1.5倍*/
    color: #333 !important;
    font-size: 14px !important;
    word-break: break-word !important; /*word-break: break-word 是单词换行 防止单词过长 强制换行*/
}

.document-list-container li:hover {
    color: #10A37F !important;
    transition: color 0.2s ease !important;
}
    """

    # --- Gradio界面 ---
    print("\n设置 Gradio 界面...") 
    with gr.Blocks(theme=gr_themes.Soft(), css=custom_css) as iface:
        gr.Markdown(f"""     
        <div style='text-align: center;'> 
        <h1>耀安科技-煤矿大模型知识问答系统</h1>
        <p>根据已有的文档或您上传的文档提问。</p>
        </div>
        """)

        # 使用Tab组件  Tab组件是gradio的标签组件 可以用来切换不同的页面 这里就是问答和文档管理两个页面
        with gr.Tabs() as tabs:
            with gr.Tab("问答"):
                chatbot_output = gr.Chatbot(
                    label="对话窗口",
                    height=600,
                    avatar_images=("1.jpg", "2.jpg"),#第一个参数是用户头像 第二个是机器人头像
                    latex_delimiters=[
                        {"left": "$$", "right": "$$", "display": True}, #这个表示 $$ $$包裹的内容作为公式单独一行显示
                        {"left": "$", "right": "$", "display": False}
                    ]
                )
                with gr.Row(elem_id="chat_input_row"):
                    query_input = gr.Textbox(
                        show_label=False,
                        placeholder="在此输入您的问题...",
                        lines=1,
                        scale=4
                    )
                    submit_button = gr.Button("发送", scale=1)

            with gr.Tab("文档管理"):
                with gr.Row():
                    with gr.Column(scale=1):
                        file_input = gr.File(label="上传 PDF 或 DOCX 文件", file_types=['.pdf', '.docx', '.doc'])
                        
                        # 增量更新选项
                        with gr.Row():
                            incremental_checkbox = gr.Checkbox(label="启用增量更新", value=True, info="只处理新文件或已修改的文件") #value表示默认勾选 info是鼠标悬停显示的信息 但实际上是直接显示的小字 不需要停留
                        
                        with gr.Row():
                            upload_button = gr.Button("上传并重建索引", variant="primary") #variant表示按钮风格 primary是主按钮 secondary是次按钮
                            force_rebuild_button = gr.Button("强制全量重建", variant="secondary")
                        
                        upload_status = gr.Textbox(label="上传状态", interactive=False)
                        
                        # 文档索引状态
                        index_status_button = gr.Button("查看文档索引状态")
                        index_status_display = gr.Textbox(label="文档索引状态", interactive=False, lines=10)
                        
                    with gr.Column(scale=1):
                        # 显示已加载文档
                        gr.Markdown("### 当前已加载文档:")
                        loaded_docs_display = gr.HTML(
                            value=f'<div class="document-list-container">{initial_doc_list}</div>'
                        )
                
                # 对话历史管理区域
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 对话历史管理")
                        chat_history_status_button = gr.Button("查看对话历史状态")
                        clear_chat_history_button = gr.Button("清空对话历史", variant="secondary")
                        chat_history_status_display = gr.Textbox(label="对话历史状态", interactive=False, lines=8)

        # --- 事件处理器 ---
        
        # 问答提交
        submit_button.click(
            fn=handle_submit_with_thinking,
            inputs=[query_input, chatbot_output],
            outputs=[query_input, chatbot_output]
        ) 
        query_input.submit(
             fn=handle_submit_with_thinking,
             inputs=[query_input, chatbot_output],
             outputs=[query_input, chatbot_output]
        )

        # 文件上传和重建
        upload_button.click( 
            fn=lambda file_obj, incremental: (
                handle_file_upload(file_obj, incremental)[0],
                f'<div class="document-list-container">{handle_file_upload(file_obj, incremental)[1]}</div>'
            ),
            inputs=[file_input, incremental_checkbox],
            outputs=[upload_status, loaded_docs_display]
        )
        
        # 强制重建按钮
        force_rebuild_button.click(
            fn=lambda: (force_rebuild_index(), f'<div class="document-list-container">{get_loaded_documents_list()}</div>'),
            inputs=[],
            outputs=[upload_status, loaded_docs_display]
        )
        
        # 文档索引状态按钮
        index_status_button.click(
            fn=get_document_index_status,
            inputs=[],
            outputs=[index_status_display]
        )
        
        # 对话历史管理按钮
        chat_history_status_button.click(
            fn=get_chat_history_status,
            inputs=[],
            outputs=[chat_history_status_display]
        )
        
        clear_chat_history_button.click(
            fn=clear_chat_history_vector_db,
            inputs=[],
            outputs=[chat_history_status_display]
        )

    return iface

def launch_interface():
    """启动Gradio界面"""
    iface = create_gradio_interface()   
    print("启动 Gradio 界面...")
    
    # 强制初始化队列锁 - 最彻底的修复方案
    print("强制修复队列锁...")
    try:
        import asyncio
        
        # 确保有事件循环
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            print(f"创建新事件循环: {loop}")
        
        # 如果队列存在但锁为空，强制创建锁
        if hasattr(iface, '_queue') and iface._queue is not None:
            queue = iface._queue
            
            # 强制创建所有必需的锁
            if not hasattr(queue, 'pending_message_lock') or queue.pending_message_lock is None:
                queue.pending_message_lock = asyncio.Lock()
                print("创建 pending_message_lock")
            
            if not hasattr(queue, 'delete_lock') or queue.delete_lock is None:
                queue.delete_lock = asyncio.Lock()
                print("创建 delete_lock")
            
            # 确保其他必要属性存在
            if not hasattr(queue, 'max_thread_count'):
                queue.max_thread_count = 40
                print("设置 max_thread_count = 40")
            
            print("队列锁修复完成")
        else:
            print("队列不存在，跳过修复")
    
    except Exception as e:
        print(f"队列锁修复失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 使用标准启动方式
    iface.launch(
        server_name="0.0.0.0", 
        server_port=7862
    )

if __name__ == "__main__":
    launch_interface() 