import os
import gradio as gr
import gradio.themes as gr_themes
import sys

# 添加父目录到路径，以便导入rag_client
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) #_file_表示当前文件路径 abspath是保证变为绝对路径  dirname是获取父目录 调用两次就是获取父目录的父目录

# 导入RAG核心功能
# from rag_client import (
#     rebuild_index_and_chain,
#     force_rebuild_index, 
#     get_document_index_status,
#     get_chat_history_status,
#     clear_chat_history_vector_db,
#     get_loaded_documents_list,
#     handle_submit_with_thinking,
#     handle_file_upload,
#     initialize_system
# )

# 简单的用户管理（实际项目中应该使用数据库）
USERS = {
    "admin": "admin123",
    "user": "user123",
    "test": "test123"
}

def authenticate_user(username, password):
    """验证用户登录"""
    if username in USERS and USERS[username] == password:
        return True, f"欢迎回来，{username}！"
    else:
        return False, "用户名或密码错误，请重试。"

def create_login_interface():
    """创建登录界面"""
    
    # 自定义CSS样式 - 模仿图片中的现代设计
    custom_css = """
    body, .gradio-container {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        margin: 0;
        padding: 0;
        min-height: 100vh;
    }
    
    .login-container {
        display: flex;
        min-height: 100vh;
        align-items: center;
        justify-content: center;
        padding: 20px;
        overflow: hidden; /* 隐藏容器的滚动条 */
    }
    
    .login-left {
        flex: 1; /* 权重 1 表示占据剩余空间 也就是左边占满剩余空间 左右布局的时候 左边自动扩展 */
        display: flex; /* 弹性布局 也就是可以随意拉伸 */
        align-items: center; /* 垂直居中 */
        justify-content: center; /* 水平居中 */
        padding: 40px;
        position: relative;
        min-width: 500px; /* 确保左侧有足够的最小宽度 */
        overflow: visible; /* 确保内容不会被裁剪 */
    }
    
    .ai-visual {
        position: relative;
        width: 600px;  /* 进一步增加宽度，确保所有圆圈都能显示 */
        height: 100vh; /* 进一步增加高度，确保所有圆圈都能显示 */
        display: flex;
        align-items: center;
        justify-content: center;
        overflow: visible; /* 确保内容不会被裁剪 */
    }
    /* 生成一个200*200的白色圆，中间可以显示文字  也就是最内部的圆*/
    .ai-circle {   
        width: 200px;
        height: 200px;
        background: white;
        border-radius: 50%; /* 圆角半径设置为百分之五十 直接变成正圆 如果没有border-radius 那么就是直角 正方形 border-radius: 10px 20px 30px 40px; 这样设置就是四个角分别不同圆角 */
        display: flex;     
        align-items: center;  /* 垂直居中内容 */
        justify-content: center; /* 水平居中内容 */
        font-size: 48px;       /* 字体大小 */
        font-weight: bold;     /* 字体加粗 */
        color: #667eea;
        box-shadow: 0 20px 40px rgba(0,0,0,0.1); /* 阴影效果 */
        position: relative;
        z-index: 2; /* 层级 */
    }
    
    .ai-rings {
        position: absolute; /* 绝对定位 也就是在上面圆基础上的一个圆 */
        top: 50%; /* 顶部距离父元素的50% */
        left: 50%; /* 左边距离父元素的50% */
        transform: translate(-50%, -50%); /* 移动 也就是移动到父元素的中心 */
        width: 300px;  /* 增加圆圈大小 */
        height: 300px; /* 增加圆圈大小 */
        border-radius: 50%;
        border: 2px solid rgba(102, 126, 234, 0.5); /* 边框 颜色 透明度 */
        animation: rotate 10s linear infinite; /* 动画 10s 匀速 无限循环 如果将其化成圆 那么就看不出来在旋转 */
    }
    
    .ai-rings::before {  /* 伪元素 也就是在ai-rings的基础上再添加一个元素 ::before和::after是在.ai-rings的基础上创建的额外内容 */
        content: '';  /* 会在这个圈上显示的文本内容 */
        position: absolute; /* 绝对定位 由于是绑定在ai-rings上 所以相当于可以随意移动 */
        top: -20px; /* 顶部距离父元素的-20px 设置负值 相当于比基础的ai-rings大20px */
        left: -20px; /* 左边距离父元素的-20px */
        right: -20px; /* 右边距离父元素的-20px */
        bottom: -20px;
        border-radius: 50%;
        border: 2px solid rgba(118, 75, 162, 0.2);
        animation: rotate 8s linear infinite reverse;
    }
    
    .ai-rings::after {
        content: '';
        position: absolute;
        top: -40px;  /* 增加外层圆圈的大小 */
        left: -40px; /* 增加外层圆圈的大小 */
        right: -40px; /* 增加外层圆圈的大小 */
        bottom: -40px; /* 增加外层圆圈的大小 */
        border-radius: 50%;
        border: 2px solid rgba(102, 126, 234, 0.1);
        animation: rotate 12s linear infinite;
    }
    
    @keyframes rotate {
        from { transform: translate(-50%, -50%) rotate(0deg); } /* 从0度开始旋转 from后面是起始状态 也可以通过0 % 50 % 100 % 来表示  translate就是让元素向左上位移百分之五十*/
        to { transform: translate(-50%, -50%) rotate(360deg); }
    }
    
    .login-right {
        flex: 1;
        max-width: 400px;  /* 最大宽度 也就是最多只能到400px 如果内容太多 那么就会自动换行 */
        background: white;
        border-radius: 20px;
        padding: 40px;
        box-shadow: 0 20px 40px rgba(0,0,0,0.1);
        margin-left: 40px;
    }
    
    .login-title { /* 登录标题 下面几个都是标题的样式*/
        text-align: center; /* 水平居中 */
        margin-bottom: 30px; /* 下边距 也就是和下面内容隔开30px */
    }
    
    .login-title h1 {
        color: #333;
        font-size: 28px;
        font-weight: 600;
        margin-bottom: 8px;
    }
    
    .login-title p {
        color: #666;
        font-size: 16px;
        margin: 0;
    }
    
    .form-group {  /* 先定义了一个整体式样 下面的label input都是在这个基础上给的样式*/
        margin-bottom: 20px; /* 下边距 也就是和下面内容隔开20px 也就是在这个基础上定义的样式，如果没有修改这个属性 其都有和下面内容隔开20px的功能*/
    }
    
    .form-group label {
        display: block; /* 块级元素 会独占一行 会和下面内容隔开 如果display: inline 那么就是行内元素 不会独占一行 不会和下面内容隔开*/
        color: #333;
        font-weight: 500;
        margin-bottom: 8px;
        font-size: 14px;
    }
    
    .form-group input {
        width: 100%; /*表示这个部件允许占满父容器*/
        padding: 12px 16px; /* 前者是上下 后者是左右*/
        border: 2px solid #e1e5e9;  /* 边框 2像素 粗细 实线 dashed是虚线 dotted是点线 颜色  */
        border-radius: 10px;
        font-size: 16px;
        transition: border-color 0.3s ease; /* 过渡效果 0.3秒 平滑 也就是当border-color发生变化时 会有0.3秒的过渡效果  用户点击输入框 边框颜色会渐变*/
        box-sizing: border-box; /* 设置浏览器计算元素宽高的方式  这里的定义是这个部件的宽度会包含边框和内边距 padding和margin 如果box-sizing: content-box; 那么就是不包含边框和内边距 */
    }
    
    .form-group input:focus {  /* :focus是伪类 表示当用户点击输入框时 会应用这个样式   focus实际是元素获取焦点的时候触发 也就是用户输入 点击等等操作 在注意到这个部分的时候就会触发*/
        outline: none; /* 去掉默认的聚焦边框*/
        border-color: #667eea; /* 边框颜色 也就是用户点击输入框时 边框颜色会变成#667eea*/
    }
    
    .checkbox-group {
        display: flex;
        align-items: center;
        margin-bottom: 20px;
    }
    
    .checkbox-group input[type="checkbox"] {  /* 这个是选择器 表示选择type为checkbox的input元素  输入还有其他类型 比如text password email等 这个只针对这一个类型*/
        width: auto; /* 让复选框的（也就是这个输入类型）由浏览器根据默认外观决定 不进行强行拉伸和压缩   复选框是可以打钩的小方框*/ 
        margin-right: 8px;
    }
    
    .checkbox-group label {
        margin: 0;
        font-size: 14px;
        color: #666;
    }
    
    .login-btn {
        width: 100%;
        padding: 14px; /* 一个就表示上下左右都这个设定 两个则前者上下 后者左右*/
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); /* 渐变背景 从#667eea到#764ba2 也就是从蓝色到紫色*/
        color: white; /* 文字颜色*/
        border: none; /* 边框 没有*/
        border-radius: 10px;
        font-size: 16px;
        font-weight: 600; /* 字体加粗 后面的数字表示程度*/
        cursor: pointer; /* 鼠标悬停时 会变成小手*/
        transition: transform 0.2s ease;
    }
    
    .login-btn:hover {  /* :hover是伪类 表示当用户鼠标悬停时 会应用这个样式  ：active是鼠标按下未松开时触发的效果*/
        transform: translateY(-2px);  /* 向上移动2px Y是垂直方向 X是水平方向 Z表示深度移动 3D x，y表示同时水平垂直*/
    }

    
    .error-message {  /* 错误信息 也就是用户输入错误时 会应用这个样式*/
        color: #e74c3c;
        text-align: center;
        margin-top: 10px;
        font-size: 14px;
    }
    
    .success-message {
        color: #27ae60;
        text-align: center;
        margin-top: 10px;
        font-size: 14px;
    }
    
    /* 隐藏默认的Gradio样式 */
    .gradio-container {
        background: transparent !important; /* 背景透明 也就是不显示背景 transparent是透明化*/
    }
    
    .main {
        background: transparent !important;
    }
    
    /* 隐藏滚动条 */
    ::-webkit-scrollbar {
        display: none; /* 隐藏Webkit浏览器的滚动条  chrome Safari edge */
    }
    
    * {
        scrollbar-width: none; /* Firefox */
        -ms-overflow-style: none; /* IE and Edge */
    }
    """

    def login_function(username, password, auto_login):
        """登录处理函数"""
        if not username or not password:
            return "请输入用户名和密码", False, ""
        
        success, message = authenticate_user(username, password)
        if success:
            print(f"用户 {username} 登录成功！")
            print("正在打开RAG系统主界面...")
            # 返回成功消息和跳转HTML代码
            redirect_html = f"""
            <div style="text-align: center; padding: 20px;">  <!--  注意这部分是html注释和css的不一样 视觉提示部分 也就是用户登录成功后 会显示这个提示   div是一个容器 我觉得就是用来设置一些属性的  style是设置的行内样式-->
                <h3 style="color: #27ae60;">{message}</h3> <!--显示一个标题 h是标题作用 h3表示三级标题 颜色按照定义的绿色来，message变量会被实际内容取代-->
                <p>正在跳转到对话界面...</p>    <!-- p是段落作用 显示普通文本 >文本</ 这个应该是固定的格式 -->
                <a href="http://localhost:7862" style="  <!-- a是超链接标签 这是一个链接 超链接按钮 也就是用户点击后会跳转到对话界面 这里面的内容会显示在链接上-->
                    display: inline-block;
                    padding: 10px 20px;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    text-decoration: none;  <!-- 去掉下划线-->
                    border-radius: 8px;
                    font-weight: bold;
                    margin: 10px;
                ">点击进入对话界面</a>
            </div>
            <iframe src="http://localhost:7862" style="display:none;" onload="window.location.href='http://localhost:7862';"></iframe> <!--src是要加载的目标地址 display表明inframe不显示在页面上 onload是当页面加载完成后 会执行这个js代码 自动跳转到这个地址-->
            <script>
                // 立即尝试跳转
                window.location.href = 'http://localhost:7862'; // window.location.href 会直接改变浏览器地址到指定的url  直接跳转 和前面两个不一样 
            </script>
            """
            return f"{message} 正在跳转到主界面...", True, redirect_html
        else:
            return message, False, ""

    # 创建登录界面
    with gr.Blocks(theme=gr_themes.Soft(), css=custom_css) as login_interface:  #创建一个登录界面 主题是gr_themes.Soft() 也就是软主题 这个主题是官方提供的  css是自定义的css样式  as login_interface 是给这个界面起一个名字 方便后面调用
        with gr.Row(elem_classes="login-container"): #用于水平布局 将子元素排成一行  elem_classes是给这个元素起一个名字 方便后面调用  这个类名和前面的自定义形式相互对应  前面定义的属性这边会用到 elem_classes是关键字
            # 左侧AI视觉元素
            with gr.Column(elem_classes="login-left"): #通过gr.Column要求这个容器下的 内容都是垂直布局  下面的gr.HTML是对显示内容的定义  
                gr.HTML("""
                <div class="ai-visual">
                    <div class="ai-rings"></div>
                    <div class="ai-circle">AI</div>
                    <!-- 添加更多装饰性圆圈 -->
                    <div class="ai-rings" style="width: 250px; height: 250px; border-color: rgba(102, 126, 234, 0.3); animation: rotate 15s linear infinite reverse;"></div>   <!-- 虽然用了之前定义好的类 但是style会覆盖之前的属性，这个优先级高-->
                    <div class="ai-rings" style="width: 180px; height: 180px; border-color: rgba(118, 75, 162, 0.4); animation: rotate 20s linear infinite;"></div>
                </div>
                """)
            
            # 右侧登录表单
            with gr.Column(elem_classes="login-right"):
                with gr.Column(elem_classes="login-title"):
                    gr.HTML("<h1>欢迎回来!</h1>")
                    gr.HTML("<p>请登录您的账号</p>")
                
                with gr.Column():
                    username_input = gr.Textbox(
                        label="用户名", #输入框上方的标签
                        placeholder="请输入用户名",  #提示内容 用户没有输入的时候显示
                        elem_classes="form-group" #给这个组件添加css类 让这个部分按照之前定义的属性来显示
                    )
                    
                    password_input = gr.Textbox(
                        label="密码",
                        placeholder="请输入密码",
                        type="password", #密码类型 也就是输入的时候会显示为小圆点
                        elem_classes="form-group" 
                    )
                    
                    auto_login_checkbox = gr.Checkbox(  #这个是复选框组件 这个复选框就是一个小框 可以打钩 也可以取消打钩
                        label="自动登录",
                        elem_classes="checkbox-group"
                    )
                    
                    login_button = gr.Button(
                        "登录", #这个是按钮上显示的文本
                        elem_classes="login-btn"
                    )
                    
                    message_output = gr.Textbox(   #这部分应该就是给登录成功之后， 出现的按钮准备的 
                        label="",
                        interactive=False, #不允许用户编辑
                        visible=False #默认隐藏
                    )
                    
                    # 用于处理跳转的HTML组件  这个和前面的一样 也是一个组件（可以理解为按钮） 目前的逻辑是 这个按钮会直接占据一部分区域 默认是false就不会占据，login_function函数调用之后会给其赋值 显示对应的内容
                    redirect_html = gr.HTML(visible=True) #可以理解这个代码只是先占据位置 根据判断结果给内容
                    

                
                # 登录事件处理
                login_button.click(
                    fn=login_function,
                    inputs=[username_input, password_input, auto_login_checkbox],
                    outputs=[message_output, message_output, redirect_html] #注意这部分的逻辑，如果登录错误，返回的是空，redirect_html也不会显示东西 
                )

    return login_interface


def launch_login_interface():
    """启动登录界面"""
    login_interface = create_login_interface()
    print("启动登录界面...")
    print("默认用户账号：")
    print("- 用户名: admin, 密码: admin123")
    print("- 用户名: user, 密码: user123")
    print("- 用户名: test, 密码: test123")
    login_interface.launch(server_name="0.0.0.0", server_port=7863)

if __name__ == "__main__":
    launch_login_interface() 