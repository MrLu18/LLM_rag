import os
import sys
import threading
import time

# 添加父目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ui.login_interface import launch_login_interface
from ui.gradio_interface import launch_interface

def main():
    """主启动函数"""
    print("=== 耀安科技-煤矿大模型知识问答系统 ===")
    print("正在启动系统...")
    
    # 先在主线程中初始化系统，确保全局变量正确设置
    print("初始化RAG系统...")
    from rag_client import initialize_system
    initialize_system()
    print("RAG系统初始化完成")
    
    # 启动对话界面（后台运行）
    print("启动对话界面 (端口: 7862)...")
    chat_thread = threading.Thread(target=launch_interface, daemon=True)
    chat_thread.start()
    
    # 等待对话界面启动
    print("等待对话界面启动...")
    time.sleep(5)  # 增加等待时间
    
    # 启动登录界面（前台运行）
    print("启动登录界面 (端口: 7863)...")
    print("默认用户账号：")
    print("- 用户名: admin, 密码: admin123")
    print("- 用户名: user, 密码: user123")
    print("- 用户名: test, 密码: test123")
    print("\n访问地址:")
    print("- 登录界面: http://localhost:7863")
    print("- 对话界面: http://localhost:7862")
    
    launch_login_interface()

if __name__ == "__main__":
    main() 