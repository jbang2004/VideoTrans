import logging
import json
from pathlib import Path
from typing import List, Dict, Any
import asyncio
from utils.worker_decorators import handle_errors

logger = logging.getLogger(__name__)

class SentenceLogger:
    """句子日志记录器"""
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self._lock = asyncio.Lock()
    
    def _format_sentence(self, sentence: Dict[str, Any]) -> Dict[str, Any]:
        return {
            'id': getattr(sentence, 'sentence_id', -1),
            'start_time': getattr(sentence, 'start', 0),
            'end_time': getattr(sentence, 'end', 0),
            'text': getattr(sentence, 'text', ''),
            'translation': getattr(sentence, 'translation', ''),
            'duration': getattr(sentence, 'duration', 0),
            'speaker_id': getattr(sentence, 'speaker_id', 0),
            'speaker_similarity': getattr(sentence, 'speaker_similarity', 0),
            'speaker_embedding': (
                getattr(sentence, 'speaker_embedding', []).tolist() 
                if hasattr(getattr(sentence, 'speaker_embedding', []), 'tolist') 
                else getattr(sentence, 'speaker_embedding', [])
            )
        }
    
    @handle_errors(logger)
    async def save_sentences(self, sentences: List[Dict[str, Any]], output_path: Path, task_id: str) -> None:
        async with self._lock:
            try:
                formatted_sentences = [self._format_sentence(s) for s in sentences]
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(formatted_sentences, f, ensure_ascii=False, indent=2)
                
                self.logger.debug(f"已保存 {len(sentences)} 个句子到 {output_path}")
            except Exception as e:
                self.logger.error(f"保存句子信息失败: {e}")
                raise
