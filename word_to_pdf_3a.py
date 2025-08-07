import os
import comtypes.client
import signal
from pathlib import Path

# 全局变量，用于强制退出时释放资源
_word_app = None

def handle_exit(signum, frame):
    """处理 Ctrl+C 信号"""
    print("\n正在强制终止并清理 Word 进程...")
    if _word_app is not None:
        _word_app.Quit()
    os._exit(1)  # 强制退出

def word_to_pdf_a3a(input_dir, output_dir):
    global _word_app
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    _word_app = comtypes.client.CreateObject("Word.Application")
    _word_app.Visible = False
    
    try:
        # 注册 Ctrl+C 处理
        signal.signal(signal.SIGINT, handle_exit)
        
        for root, _, files in os.walk(input_dir):
            for filename in files:
                if not filename.lower().endswith(('.doc', '.docx')):
                    continue
                
                input_path = os.path.join(root, filename)
                rel_path = os.path.relpath(root, input_dir) #记录目录结构 然后还原
                output_subdir = os.path.join(output_dir, rel_path)
                os.makedirs(output_subdir, exist_ok=True)
                output_path = os.path.join(output_subdir, f"{Path(filename).stem}.pdf")
                
                print(f"正在转换: {input_path} -> {output_path}")
                
                try:
                    doc = _word_app.Documents.Open(input_path)
                    doc.ExportAsFixedFormat(
                        OutputFileName=output_path,
                        ExportFormat=17,
                        OpenAfterExport=False,
                        OptimizeFor=0,
                        Range=0,
                        Item=0,
                        IncludeDocProps=True,
                        KeepIRM=True,
                        CreateBookmarks=1,
                        DocStructureTags=True,
                        BitmapMissingFonts=True,
                        UseISO19005_1=True
                    )
                    doc.Close(False)
                except Exception as e:
                    print(f"❌ 转换失败: {filename} - {str(e)}")
    
    finally:
        if _word_app is not None:
            _word_app.Quit()
            _word_app = None
            print("Word 进程已释放")

if __name__ == "__main__":
    input_directory = r"D:\python_program\LLM_RAG\夏兰兰"
    output_directory = r"D:\python_program\LLM_RAG\document"
    
    try:
        word_to_pdf_a3a(input_directory, output_directory)
    except KeyboardInterrupt:
        print("用户主动终止")
    except Exception as e:
        print(f"全局错误: {str(e)}")