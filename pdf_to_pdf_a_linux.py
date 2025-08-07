#sudo apt install ghostscript  # Ubuntu/Debian  #需要安装这个
import os
import subprocess
import signal
from pathlib import Path

# 全局变量控制终止
_should_exit = False

def signal_handler(signum, frame):
    """处理 Ctrl+C 信号"""
    global _should_exit
    print("\n用户请求终止，正在清理...")
    _should_exit = True

def convert_to_pdfa(input_path, output_path, pdfa_version="3"):
    """
    将单个PDF转换为PDF/A-3A格式
    :param input_path: 输入PDF路径
    :param output_path: 输出PDF路径
    :param pdfa_version: PDF/A版本
    """
    cmd = [
        "gs",
        "-dPDFA",
        f"-dPDFACompatibilityPolicy={pdfa_version}",
        "-dBATCH",
        "-dNOPAUSE",
        "-sProcessColorModel=DeviceRGB",
        "-sDEVICE=pdfwrite",
        f"-sOutputFile={output_path}",
        input_path
    ]
    subprocess.run(cmd, check=True)

def batch_convert_pdfa(input_dir, output_dir, pdfa_version="3"):
    """
    批量转换目录中的所有PDF到PDF/A-3A
    :param input_dir: 输入目录
    :param output_dir: 输出目录
    :param pdfa_version: PDF/A版本
    """
    global _should_exit
    
    # 注册信号处理器
    signal.signal(signal.SIGINT, signal_handler)  #SIGINT代表ctrl c
    
    # 确保输出目录存在
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 递归遍历输入目录
    for root, _, files in os.walk(input_dir):
        for filename in files:
            if _should_exit:
                print("⏹️ 转换已终止")
                return
            
            if not filename.lower().endswith('.pdf'):
                continue
                
            # 构建输入输出路径
            input_path = os.path.join(root, filename)
            rel_path = os.path.relpath(root, input_dir)
            output_subdir = os.path.join(output_dir, rel_path)
            os.makedirs(output_subdir, exist_ok=True)
            output_path = os.path.join(output_subdir, filename)
            
            print(f"🔄 正在转换: {input_path} -> {output_path}")
            
            try:
                convert_to_pdfa(input_path, output_path, pdfa_version)
                print(f"✅ 转换成功: {filename}")
            except subprocess.CalledProcessError as e:
                print(f"❌ 转换失败: {filename} (Ghostscript错误: {e})")
            except Exception as e:
                print(f"❌ 未知错误: {filename} - {str(e)}")

if __name__ == "__main__":
    input_directory = r"D:\input_pdfs"  # 替换为输入目录
    output_directory = r"D:\output_pdfa"  # 替换为输出目录
    
    try:
        print("=== PDF批量转换PDF/A-3A ===")
        print(f"输入目录: {input_directory}")
        print(f"输出目录: {output_directory}")
        print("按 Ctrl+C 可随时终止")
        print("=" * 40)
        
        batch_convert_pdfa(input_directory, output_directory, pdfa_version="3")
        print("🎉 所有文件转换完成！")
    except KeyboardInterrupt:
        print("\n用户手动终止")
    except Exception as e:
        print(f"全局错误: {str(e)}")