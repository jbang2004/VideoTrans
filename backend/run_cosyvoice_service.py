import logging
import sys
import os
from pathlib import Path
import argparse

current_dir = Path(__file__).parent

def setup_system_paths():
    system_paths = [
        str(current_dir / 'models' / 'CosyVoice'),
        str(current_dir / 'models' / 'CosyVoice' / 'third_party' / 'Matcha-TTS')
    ]
    for path in system_paths:
        if path not in sys.path and os.path.exists(path):
            sys.path.append(path)
            logging.info(f"Added {path} to system path")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default='models/CosyVoice/pretrained_models/CosyVoice2-0.5B', help='base directory containing model folders')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='service host')
    parser.add_argument('--port', type=int, default=50052, help='service port')
    args = parser.parse_args()
    
    # 设置系统路径
    setup_system_paths()

    # 必须在导入任何使用顶层 "cosyvoice" 模块之前注册别名
    import models.CosyVoice.cosyvoice
    sys.modules["cosyvoice"] = models.CosyVoice.cosyvoice

    # 现在再导入服务模块，确保其内部使用 "cosyvoice" 时能正确找到
    from services.cosyvoice.service import serve
    serve(args) 