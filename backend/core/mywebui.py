import os
import sys
# import threading # Removed unused import
import time
import gradio as gr

# 确保可以找到 backend 目录下的模块
# 假设项目根目录是 /home/jbang/codebase/VideoTrans
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# === 添加原始 indextts 库的路径到 sys.path ===
# 这对于解决原始库内部的 'from indextts...' 导入错误至关重要
indextts_module_path = os.path.join(project_root, 'backend', 'models', 'IndexTTS')
indextts_inner_path = os.path.join(indextts_module_path, 'indextts')
if indextts_module_path not in sys.path:
    sys.path.append(indextts_module_path) # 添加包含 indextts 的目录
if indextts_inner_path not in sys.path:
    sys.path.append(indextts_inner_path) # 添加 indextts 目录本身

# 现在可以安全地导入我们自定义的类，它会间接导入原始库
from backend.core.my_index_tts import MyIndexTTS
# 如果需要 i18n 或其他 utils，也需要正确导入
# from backend.models.IndexTTS.tools.i18n.i18n import I18nAuto # 调整路径
# from backend.models.IndexTTS.utils.webui_utils import next_page, prev_page # 调整路径

# --- 全局设置和初始化 ---

# i18n = I18nAuto(language="zh_CN") # 如果需要国际化

# 输出和提示音目录，相对于项目根目录
OUTPUT_DIR = os.path.join(project_root, "outputs", "my_webui")
PROMPT_DIR = os.path.join(project_root, "prompts", "my_webui")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PROMPT_DIR, exist_ok=True)

print(f"Output directory: {OUTPUT_DIR}")
print(f"Prompt directory: {PROMPT_DIR}")

# 初始化 TTS 实例 (重命名为 tts)
try:
    print("Initializing MyIndexTTS...")
    tts = MyIndexTTS() # 重命名实例
    print("MyIndexTTS initialized successfully.")
except FileNotFoundError as e:
    print(f"Error initializing MyIndexTTS: {e}")
    tts = None
except Exception as e:
    print(f"An unexpected error occurred during MyIndexTTS initialization: {e}")
    import traceback
    traceback.print_exc()
    tts = None

# --- 辅助推理函数 (模仿原始 webui.py) ---
def infer(voice, text, output_path=None):
    """调用 tts 实例进行推理"""
    if not tts:
        raise gr.Error("TTS model is not initialized.")
    if not voice:
        raise gr.Error("Please provide a reference audio.")
    if not text:
        raise gr.Error("Please enter the target text.")

    if not output_path:
        # 在 OUTPUT_DIR 中生成路径
        output_filename = f"output_{int(time.time())}.wav"
        output_path = os.path.join(OUTPUT_DIR, output_filename)

    print(f"Inferring with prompt: {voice}, text: '{text}' -> {output_path}")
    try:
        tts.infer(voice, text, output_path)
        print("Inference successful.")
        return output_path
    except Exception as e:
        print(f"Error during inference: {e}")
        import traceback
        traceback.print_exc()
        raise gr.Error(f"Error during audio generation: {e}")

# --- Gradio 回调函数 (模仿原始 webui.py) ---
def gen_single(prompt, text):
    """Gradio 回调，调用 infer 并更新输出"""
    try:
        output_audio_path = infer(prompt, text)
        return gr.update(value=output_audio_path, visible=True)
    except gr.Error as e:
        # 如果 infer 函数内部抛出 Gradio 错误，直接显示
        raise e
    except Exception as e:
        # 其他意外错误
        raise gr.Error(f"An unexpected error occurred: {e}")

def handle_prompt_upload(uploaded_file_path):
    """处理上传，简单地启用按钮"""
    # 按钮初始状态由 tts 是否成功加载决定
    return gr.update(interactive=(tts is not None and uploaded_file_path is not None))

# --- Gradio 界面布局 (简化标签) ---
with gr.Blocks() as demo:
    gr.HTML('''
    <div style="text-align: center; margin-bottom: 20px;">
        <h1>IndexTTS Custom WebUI</h1>
    </div>
    ''')

    with gr.Tab("音频生成"):
        with gr.Row():
            with gr.Column(scale=1):
                prompt_audio = gr.Audio(
                    label="参考音频",
                    sources=["upload", "microphone"],
                    type="filepath",
                )
            with gr.Column(scale=2):
                input_text_single = gr.Textbox(
                    label="目标文本",
                    placeholder="输入文本...",
                    lines=4,
                )
        with gr.Row():
             # 初始交互状态取决于 tts 是否加载成功
            gen_button = gr.Button("生成语音", variant="primary", interactive=(tts is not None))
        with gr.Row():
            output_audio = gr.Audio(
                label="生成结果",
                visible=False,
            )

    # --- 事件处理 ---
    prompt_audio.change(handle_prompt_upload, inputs=[prompt_audio], outputs=[gen_button])

    gen_button.click(
        gen_single, # 使用 gen_single 回调
        inputs=[prompt_audio, input_text_single],
        outputs=[output_audio]
    )

    if tts is None:
        gr.Markdown("**错误：TTS 模型未能成功初始化。请检查控制台日志。**")

# --- 启动 Gradio 应用 ---
if __name__ == "__main__":
    print("Launching Gradio WebUI...")
    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=7861,
        allowed_paths=[OUTPUT_DIR]
    )
    print("Gradio WebUI started.") 